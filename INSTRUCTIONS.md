# Fine-Tuning DistilBERT on GLUE (CoLA) with PyTorch & Hugging Face

This project fine-tunes **DistilBERT** on the **CoLA** task from the **GLUE benchmark** —
a binary text classification task that determines whether a sentence is
grammatically acceptable or not — using PyTorch and the Hugging Face
`Trainer` API.

The trained model is pushed to the Hugging Face Hub and can be used for
inference on new sentences.

## What this project does

1. Loads the CoLA dataset (part of GLUE) from the Hugging Face Hub.
2. Tokenizes the sentences using a pretrained DistilBERT tokenizer.
3. Fine-tunes `distilbert-base-uncased` with a classification head on top.
4. Evaluates the model using the Matthews Correlation Coefficient (MCC),
   the standard metric for CoLA.
5. Pushes the fine-tuned model to the Hugging Face Hub.
6. Runs inference on example sentences, both manually and via the
   Hugging Face `pipeline` API.

## Requirements

- Python 3.9+
- A Hugging Face account (free) — required to push the trained model
- (Recommended) a GPU — training on CPU works but is much slower

Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup

1. **Clone this repository**

   ```bash
   git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
   cd YOUR-REPO-NAME
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Log in to Hugging Face**

   You need an access token from https://huggingface.co/settings/tokens
   (a "Write" token, since the script pushes the model to your account).

   ```bash
   huggingface-cli login
   ```

   Or, if running inside a notebook:

   ```python
   from huggingface_hub import notebook_login
   notebook_login()
   ```

4. **(Optional) Install Git-LFS**

   Needed if you want the Hub push to work smoothly with large model files.

   ```bash
   # Debian/Ubuntu
   sudo apt install git-lfs
   git lfs install
   git config --global user.email "you@example.com"
   git config --global user.name "Your Name"
   ```

## Running the script

```bash
python train_glue_classifier.py
```

Before running, open `train_glue_classifier.py` and update this line near
the bottom to point to your own Hugging Face username once the model has
been pushed:

```python
hub_model_id = f"YOUR-USERNAME/{push_to_hub_model_id}"
```

## Changing the task or base model

Both are configurable at the top of the script:

```python
task = "cola"                          # any task in GLUE_TASKS
model_checkpoint = "distilbert-base-uncased"  # any Hub checkpoint with a classification head
batch_size = 16
```

Supported GLUE tasks: `cola`, `mnli`, `mnli-mm`, `mrpc`, `qnli`, `qqp`,
`rte`, `sst2`, `stsb`, `wnli`.

## Understanding the results

After training, the script prints the final evaluation metrics, e.g.:

```
Loss: 0.6073
Matthews Correlation: 0.5450
```

**Matthews Correlation Coefficient (MCC)** ranges from -1 to +1:

| Score        | Interpretation          |
|--------------|--------------------------|
| 0.6 – 1.0    | Excellent                |
| 0.4 – 0.6    | Good                     |
| 0.2 – 0.4    | Weak                     |
| ~0.0         | No better than random    |
| Negative     | Worse than random        |

MCC is used instead of plain accuracy because CoLA's classes are
imbalanced, and MCC can't be inflated by a model that just predicts the
majority class every time.

## Example output

```
Predicted labels: ['Valid', 'Invalid']
```

for the input sentences:

```
"The judge told the jurors to think carefully."          -> grammatically valid
"The judge told that the jurors to think carefully."     -> grammatically invalid
```

## License

This project is released under the Apache 2.0 License, consistent with
the base `distilbert-base-uncased` model license.
