"""Train and evaluate the TF-IDF + Logistic Regression baseline.

Run from the project root: python -m src.model_baseline
The saved pipeline expects clean_text produced by src.preprocess.
"""

import hashlib
import json
import pickle
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from src.data_loader import PROJECT_ROOT, class_counts
from src.preprocess import PROCESSED_DIR, SEED

MODEL_PATH = PROJECT_ROOT / "models" / "baseline.pkl"
METRICS_PATH = PROJECT_ROOT / "reports" / "baseline_metrics.json"


def load_split(path):
    """Validate processed inputs without silently coercing string labels."""
    frame = pd.read_json(path, lines=True, dtype=False)
    if not {"clean_text", "label"}.issubset(frame.columns):
        raise ValueError(f"{path}: expected clean_text and label columns")
    class_counts(frame[["clean_text", "label"]].rename(columns={"clean_text": "text"}), str(path))
    return frame


def evaluate_predictions(y_true, y_pred):
    """Binary metrics: malicious=1; matrix rows=true, columns=predicted."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def train_baseline(processed_dir=PROCESSED_DIR, model_path=MODEL_PATH, metrics_path=METRICS_PATH):
    """Fit on training rows only, evaluate once on test rows, then save artifacts."""
    processed_dir = Path(processed_dir)
    model_path, metrics_path = Path(model_path), Path(metrics_path)
    train = load_split(processed_dir / "train.jsonl")
    test = load_split(processed_dir / "test.jsonl")
    if set(train["label"]) != {0, 1}:
        raise ValueError("Training data must contain both benign and malicious classes")
    if set(train["clean_text"]) & set(test["clean_text"]):
        raise ValueError("Train/test overlap detected; regenerate the processed splits")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=str.split, token_pattern=None, lowercase=False, ngram_range=(1, 2),
        )),
        ("classifier", LogisticRegression(
            C=1.0, solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=SEED,
        )),
    ])
    # A single pipeline keeps the fitted vocabulary/IDF and classifier together.
    # str.split preserves punctuation and one-character tokens from preprocessing.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(train["clean_text"], train["label"])

    predictions = model.predict(test["clean_text"])
    metrics = evaluate_predictions(test["label"], predictions)
    metrics.update({
        "model": "tfidf_logistic_regression",
        "evaluation_split": "test",
        "positive_label": 1,
        "label_mapping": {"0": "benign", "1": "malicious"},
        "confusion_matrix_layout": "rows=true, columns=predicted; labels=[0,1]; [[TN,FP],[FN,TP]]",
        "train_samples": len(train),
        "test_samples": len(test),
        "vocabulary_size": len(model.named_steps["tfidf"].vocabulary_),
        "hyperparameters": {
            "ngram_range": [1, 2], "C": 1.0, "solver": "lbfgs",
            "class_weight": "balanced", "max_iter": 1000, "random_state": SEED,
        },
        "input_column": "clean_text",
        "data_sha256": {
            split: hashlib.sha256((processed_dir / f"{split}.jsonl").read_bytes()).hexdigest()
            for split in ("train", "test")
        },
        "versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("scikit-learn", "numpy", "scipy", "pandas")},
        },
    })

    # Only publish a model after fitting and evaluation complete successfully.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = model_path.with_suffix(".pkl.tmp")
    with temporary_model.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metrics["model_sha256"] = hashlib.sha256(temporary_model.read_bytes()).hexdigest()
    temporary_metrics = metrics_path.with_suffix(".json.tmp")
    temporary_metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    temporary_model.replace(model_path)
    temporary_metrics.replace(metrics_path)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {model_path.resolve()}")
    print(f"Saved metrics: {metrics_path.resolve()}")
    return model, metrics


if __name__ == "__main__":
    train_baseline()
