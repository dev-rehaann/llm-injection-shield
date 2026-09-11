# LLM Injection Shield

NLP semester project: classify prompts as **benign** or **malicious**
(prompt injection / jailbreak attempts aimed at an LLM).

## Status

Step 1: project skeleton only. No model code has been added.

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
├── reports/           # Evaluation tables and charts
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

TBD: confirm the Hugging Face dataset name, label mapping, and
train/validation/test split in the data step.

## Training, evaluation, and demo

TBD in subsequent steps.

## Development workflow

Complete and verify each agreed project step, then commit and push it to this
repository.