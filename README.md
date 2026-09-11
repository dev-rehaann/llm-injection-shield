# LLM Injection Shield

NLP semester project: classify prompts as **benign** or **malicious**
(prompt injection / jailbreak attempts aimed at an LLM).

## Status

Step 2: dataset downloader and class balance reporting. No model code has been added.

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

## Training, evaluation, and demo

TBD in subsequent steps.

## Development workflow

Complete and verify each agreed project step, then commit and push it to this
repository.