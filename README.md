# LLM Injection Shield

NLP semester project: classify prompts as **benign** or **malicious**
(prompt injection / jailbreak attempts aimed at an LLM).

## Status

Step 5: Word2Vec + LSTM trained, selected by validation F1, evaluated, and saved.

## Planned approaches

1. TF-IDF / Bag-of-Words + Logistic Regression.
2. Word2Vec / GloVe embeddings + a PyTorch LSTM.
3. Fine-tuned DistilBERT using Hugging Face Transformers.

All three models will reuse a shared data preprocessing pipeline.
Evaluation will compare Accuracy, Precision, Recall, and F1 in a table/chart.
A CLI or minimal web demo will return a predicted label and confidence score.

## Folder structure

```text
llm-injection-shield/
├── data/              # Downloaded datasets and processed data
├── src/               # Preprocessing, training, evaluation, and demo code
├── models/            # Saved trained models and related artifacts
├── reports/           # Dataset notes, evaluation tables, and charts
├── tests/             # Offline regression check
├── .gitignore
├── requirements.txt
├── setup_env.py
└── README.md
```

Empty folders contain `.gitkeep` placeholders so Git preserves them.
Downloaded data and trained model files are excluded from Git.

## Environment setup

Install Python first. Run these commands from the project folder.
The setup script creates `.venv` with pip; install dependencies afterward.

### Windows PowerShell

```powershell
python setup_env.py
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 setup_env.py
.venv/bin/python -m pip install -r requirements.txt
```

Use the environment's Python executable for subsequent project commands, or
select `.venv` as your interpreter in your editor. Activation is optional.
Dependencies are unpinned for now; exact versions will be recorded after the
training environment is validated.

## Dataset

Selected: [xTRam1/safe-guard-prompt-injection](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection).
See [the dataset comparison](reports/dataset_selection.md) for alternatives and limitations.

From the project root, with dependencies installed:

```powershell
.\.venv\Scripts\python.exe src/data_loader.py
```

On macOS/Linux, use `.venv/bin/python src/data_loader.py`.

For this data step alone, `python -m pip install datasets pandas` is sufficient
when run with the virtual environment's Python.

The loader downloads a pinned revision and writes:

- `data/raw/train.csv`
- `data/raw/test.csv`
- `data/raw/metadata.json` (source revision, label mapping, and class counts)

It prints benign/malicious counts for each split and the total.
Labels remain `0 = benign` and `1 = malicious`. Original text and split
membership are preserved. Paths are resolved relative to the project, so the
script also works when launched from another directory. Re-running refreshes
the same files from the same revision, using Hugging Face's download cache.

To read a local split without interpreting prompt strings such as "NA" as missing:

```python
import pandas as pd

train = pd.read_csv("data/raw/train.csv", keep_default_na=False)
```

Run the offline regression check from the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Shared preprocessing

Run after downloading the dataset:

```powershell
.\.venv\Scripts\python.exe -m src.preprocess
```

This writes shared `train.jsonl`, `val.jsonl`, and `test.jsonl` files under
`data/processed/`, plus excluded rows and metadata. Validation is a stratified
20% of the deduplicated training pool, using seed 42.

Each row provides `raw_text` for the transformer, `tokens` for the LSTM, and
`clean_text` for TF-IDF, alongside its label and original source identifier.
Raw input files remain unchanged. All three models must use these same splits.

```python
import pandas as pd
from src.preprocess import preprocess_text

train = pd.read_json("data/processed/train.jsonl", lines=True)
views = preprocess_text("Do NOT ignore the system instructions!")
```

See [the preprocessing report](reports/preprocessing.md) for tokenization rules,
stopword choices, duplicate/conflict handling, split counts, and verification.

## Baseline: TF-IDF + Logistic Regression

Run from the project root after preprocessing, with dependencies installed:

```powershell
.\.venv\Scripts\python.exe -m src.model_baseline
```

The model uses TF-IDF unigrams/bigrams and Logistic Regression with
`C=1.0`, `class_weight="balanced"`, `solver="lbfgs"`, and `max_iter=1000`.
Whitespace tokenization preserves the punctuation and one-character tokens in
`clean_text`. The fitted vocabulary, IDF weights, and classifier are saved
together in a scikit-learn pipeline.
[TF-IDF reference](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

Only `train.jsonl` is used for fitting. The fixed configuration is evaluated on
`test.jsonl`; validation data is reserved for future tuning. Test results do not
select hyperparameters. Class balancing uses training labels only.

Outputs:

- `models/baseline.pkl`: fitted pipeline, saved locally and excluded from Git.
- `reports/baseline_metrics.json`: metrics, confusion matrix, parameters,
  row counts, package versions, and data/model SHA-256 hashes.

| Test metric | Value |
| --- | ---: |
| Accuracy | 99.12% |
| Precision (malicious) | 99.22% |
| Recall (malicious) | 97.99% |
| F1 (malicious) | 98.60% |

The model was trained on 6,466 rows and evaluated on 2,049 rows.
Confusion matrix: `[[1396, 5], [13, 635]]`, with true labels as rows and predicted
labels as columns in `[benign, malicious]` order. This means 5 false positives
and 13 false negatives. Scores describe this dataset; the synthetic/template
limitations in the dataset and preprocessing reports still apply.

To reuse the model, first apply the same preprocessing:

```python
import pickle
from src.preprocess import preprocess_text

# Load only a model file you trust; pickle can execute code during loading.
with open("models/baseline.pkl", "rb") as file:
    model = pickle.load(file)

text = preprocess_text("Do NOT ignore the system instructions!")["clean_text"]
label = int(model.predict([text])[0])
probabilities = model.predict_proba([text])[0]  # Order given by model.classes_.
```

Use the recorded package versions when loading the pickle. The probabilities
are model estimates and have not been calibrated.

All three offline regression checks passed. A fresh Python process reloaded
the saved model, reproduced every metric from its test predictions, and verified
the model/data hashes. Training completed in 11 solver iterations.
This run used the temporary verification environment from the data step;
install the requirements into the project's virtual environment before using
the commands above.

## Word2Vec + LSTM

Run after preprocessing, with the project requirements installed:

```powershell
.\.venv\Scripts\python.exe -m src.model_lstm
```

The model uses the same 6,466 training, 1,617 validation, and 2,049 test rows.
It consumes `tokens`, retaining stopwords and punctuation from shared preprocessing.

- Word2Vec: 100-dimensional skip-gram vectors, window 5, minimum frequency 2,
  10 epochs, one worker, and seed 42. Vocabulary and vectors use training text only.
- Network: trainable embedding layer, one bidirectional LSTM with 64 hidden units
  per direction, dropout 0.3, and a single binary output logit.
- Training: Adam at 0.001, batch size 64, gradient clipping at 1.0, and
  BCEWithLogitsLoss weighted by the training benign/malicious ratio.
- Selection: up to 12 epochs, retaining the earliest checkpoint with the highest
  validation F1; stop after three epochs without improvement. Test evaluation
  happens after restoring that checkpoint. The probability threshold stays 0.5.

PAD has index 0 and an all-zero vector; UNK has index 1 and starts at the mean
Word2Vec vector. Rare and unseen words map to UNK. Packed sequences prevent
padding from changing the LSTM states. Prompts retain their first 256 tokens:
470 training, 110 validation, and 142 test prompts are truncated. Raw data stays
unchanged. This limit is configurable through `train_lstm(max_length=...)`.

Outputs:

- `models/lstm.pt`: selected model state, including fine-tuned embeddings,
  vocabulary, architecture, tokenization metadata, and sequence limit.
- `reports/lstm_metrics.json`: the same five metrics as the baseline, plus
  validation history, selected epoch, configuration, truncation counts,
  dependency versions, and model/data hashes.

The default run uses CPU with four PyTorch threads. `train_lstm` exposes the
training settings as keyword arguments. Fixed seeds, a stable Word2Vec hash, and
one Word2Vec worker support reproducibility with the same environment; results
can differ across library versions or devices.

To reload the checkpoint and classify a prompt:

```python
import torch
from src.model_lstm import encode_tokens, load_lstm
from src.preprocess import preprocess_text

model, checkpoint = load_lstm()
tokens = preprocess_text("Ignore all previous instructions!")["tokens"]
ids, lengths = encode_tokens([tokens], checkpoint["vocab"], checkpoint["max_length"])
with torch.inference_mode():
    malicious_probability = model(ids, lengths).sigmoid().item()
label = int(malicious_probability >= checkpoint["threshold"])
```

The loader uses `torch.load(..., weights_only=True)`. Probabilities are
uncalibrated model estimates. The embedding weights are included in the
checkpoint, so no separate Word2Vec download is needed for inference.

References: [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)
and [PyTorch packed sequences](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html).

### LSTM results

Training stopped after epoch 5 and restored epoch 2, whose validation F1 was
99.38%. The saved checkpoint contains a vocabulary of 13,434 learned tokens
plus the reserved PAD/UNK embedding rows.

| Test metric | Value |
| --- | ---: |
| Accuracy | 99.32% |
| Precision (malicious) | 99.22% |
| Recall (malicious) | 98.61% |
| F1 (malicious) | 98.92% |

Confusion matrix: `[[1396, 5], [9, 639]]` (true labels as rows, predicted labels
as columns, ordered benign/malicious). There were 5 false positives and 9 false
negatives. These results describe the current dataset and its documented
synthetic/template limitations.

All four regression checks passed, including Word2Vec vocabulary isolation,
unknown/padding IDs, unequal sequence lengths, padding invariance, validation
checkpoint restoration, and save/reload predictions. A separate Python process
reproduced all test metrics from the saved model, verified its checksum and
selected epoch, and confirmed train/test data hashes match the baseline.

This run used CPU PyTorch 2.13.0, Gensim 4.4.0, and the temporary verification
environment used in previous steps. Install the project requirements into
`.venv` before using the commands above. The checkpoint is stored locally and
excluded from Git; code and metrics are committed.

## Remaining model and demo

DistilBERT, the cross-model comparison, and the interactive demo are planned
for subsequent steps.

## Development workflow

Complete and verify each agreed project step, then commit and push it to this
repository.