import numpy as np
import torch
from datasets import load_dataset
from evaluate import load
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    pipeline,
)


# 1. 


GLUE_TASKS = [
    "cola",
    "mnli",
    "mnli-mm",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb",
    "wnli",
]

task = "cola"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16

# Maps each GLUE task to the dataset column name(s) containing the sentence(s)
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def main():
    # 2. 
    
    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("nyu-mll/glue", actual_task)
    metric = load("glue", actual_task)

    print(dataset)
    print("Example row:", dataset["train"][0])

    # 
    # 3. 
    # 
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    sentence1_key, sentence2_key = task_to_keys[task]

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True)
        return tokenizer(
            examples[sentence1_key], examples[sentence2_key], truncation=True
        )

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    # -----------------------------------------------------------------------
    # 4. Load the pretrained model
    # -----------------------------------------------------------------------
    if task == "stsb":
        num_labels = 1
    elif task.startswith("mnli"):
        num_labels = 3
    else:
        num_labels = 2

    # Optional: gives cleaner label names in outputs.
    # If using a task other than CoLA, update these to match your labels!
    id2label = {0: "Invalid", 1: "Valid"}
    label2id = {val: key for key, val in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=num_labels, id2label=id2label, label2id=label2id
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # -----------------------------------------------------------------------
    # 5. Define the evaluation metric function
    # -----------------------------------------------------------------------
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    # -----------------------------------------------------------------------
    # 6. Configure training and fine-tune the model
    # -----------------------------------------------------------------------
    validation_key = (
        "validation_mismatched"
        if task == "mnli-mm"
        else "validation_matched"
        if task == "mnli"
        else "validation"
    )

    model_name = model_checkpoint.split("/")[-1]
    push_to_hub_model_id = f"{model_name}-finetuned-{task}"

    metric_name = (
        "pearson"
        if task == "stsb"
        else "matthews_correlation"
        if task == "cola"
        else "accuracy"
    )

    num_epochs = 3

    args = TrainingArguments(
        output_dir="./text_classification_model_save",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        push_to_hub=True,
        hub_model_id=push_to_hub_model_id,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Print final evaluation results
    print("Final evaluation results:", trainer.evaluate())

    # -----------------------------------------------------------------------
    # 7. Push the fine-tuned model to the Hugging Face Hub
    # -----------------------------------------------------------------------
    trainer.push_to_hub()

    # -----------------------------------------------------------------------
    # 8. Run inference with the fine-tuned model
    # -----------------------------------------------------------------------
    # Replace with your own username/model name once pushed above.
    hub_model_id = f"YOUR-USERNAME/{push_to_hub_model_id}"

    inference_model = AutoModelForSequenceClassification.from_pretrained(hub_model_id)
    inference_tokenizer = AutoTokenizer.from_pretrained(hub_model_id)

    sentences = [
        "The judge told the jurors to think carefully.",
        "The judge told that the jurors to think carefully.",
    ]

    tokenized = inference_tokenizer(sentences, return_tensors="pt", padding="longest")
    with torch.no_grad():
        outputs = inference_model(**tokenized).logits

    predicted_classes = np.argmax(outputs.numpy(), axis=1)
    predicted_labels = [
        inference_model.config.id2label[c] for c in predicted_classes
    ]
    print("Predicted labels:", predicted_labels)

    # Equivalent, one-line version using the Pipeline API
    classifier = pipeline("text-classification", hub_model_id)
    print(classifier(sentences))


if __name__ == "__main__":
    main()
