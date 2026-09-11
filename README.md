# LLM Injection Shield

NLP semester project: classify prompts as **benign** or **malicious**
(prompt injection / jailbreak attempts aimed at an LLM).

## Status

Step 4: TF-IDF + Logistic Regression baseline trained, evaluated, and saved.

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

## Remaining models and demo

LSTM, DistilBERT, the cross-model comparison, and the interactive demo are planned
for subsequent steps.

## Development workflow

Complete and verify each agreed project step, then commit and push it to this
repository.