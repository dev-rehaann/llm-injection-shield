"""Fine-tune DistilBERT with Trainer: python -m src.model_transformer.

Uses the existing raw_text splits. The pretrained tokenizer handles normalization;
the classical models' stopword removal must not be applied here.
"""

import hashlib
import json
import math
import platform
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data_loader import PROJECT_ROOT, class_counts
from src.model_baseline import evaluate_predictions, load_split
from src.preprocess import PROCESSED_DIR, SEED

MODEL_ID = "distilbert/distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
MODEL_DIR = PROJECT_ROOT / "models" / "transformer"
METRICS_PATH = PROJECT_ROOT / "reports" / "transformer_metrics.json"
CACHE_DIR = PROJECT_ROOT / "models" / "hf_cache"


def tokenize_split(frame, tokenizer, max_length):
    """Keep the original text intact until the pretrained tokenizer sees it."""
    dataset = Dataset.from_dict({
        "text": frame["raw_text"].tolist(), "labels": frame["label"].tolist(),
    })
    # ponytail: retain the first max_length subwords; use chunking for long-document detection.
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True, remove_columns=["text"],
    )


def compute_metrics(prediction):
    """Trainer needs scalar metrics; the final report also includes the matrix."""
    metrics = evaluate_predictions(
        prediction.label_ids, prediction.predictions.argmax(axis=-1),
    )
    return {name: value for name, value in metrics.items() if name != "confusion_matrix"}


def train_transformer(
    processed_dir=PROCESSED_DIR, model_dir=MODEL_DIR, metrics_path=METRICS_PATH,
    model_name=MODEL_ID, revision=MODEL_REVISION, epochs=3, batch_size=8,
    max_length=256, learning_rate=2e-5,
    seed=SEED, cpu_threads=4,
):
    """Fine-tune all weights, select by validation F1, and evaluate test once.

    Pass a local model directory and revision=None for offline experiments.
    Trainer automatically uses CUDA when available, otherwise CPU.
    """
    for name, value in {
        "epochs": epochs, "batch_size": batch_size, "max_length": max_length,
        "cpu_threads": cpu_threads,
    }.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if max_length < 3:
        raise ValueError("max_length must leave room for content and special tokens")

    processed_dir, model_dir, metrics_path = map(Path, (processed_dir, model_dir, metrics_path))
    frames = {
        split: load_split(processed_dir / f"{split}.jsonl")
        for split in ("train", "val", "test")
    }
    for split, frame in frames.items():
        if "raw_text" not in frame:
            raise ValueError(f"{split}: expected a raw_text column")
        class_counts(frame[["raw_text", "label"]].rename(columns={"raw_text": "text"}), split)
        if set(frame["label"]) != {0, 1}:
            raise ValueError(f"{split}: both classes must be present")
    for left, right in combinations(frames, 2):
        if set(frames[left]["clean_text"]) & set(frames[right]["clean_text"]):
            raise ValueError(f"{left}/{right} overlap detected; regenerate processed splits")

    set_seed(seed)
    torch.set_num_threads(cpu_threads)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, cache_dir=CACHE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=revision, cache_dir=CACHE_DIR, num_labels=2,
        id2label={0: "benign", 1: "malicious"}, label2id={"benign": 0, "malicious": 1},
    )
    if max_length > model.config.max_position_embeddings:
        raise ValueError("max_length exceeds the model's positional embedding limit")
    # Save this limit with the tokenizer so inference uses the training limit too.
    tokenizer.model_max_length = max_length
    datasets = {
        split: tokenize_split(frame, tokenizer, max_length)
        for split, frame in frames.items()
    }

    args = TrainingArguments(
        output_dir=str(model_dir.with_name(model_dir.name + "_checkpoints")),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate, weight_decay=0.01,
        lr_scheduler_type="linear", warmup_steps=0.1, max_grad_norm=1.0,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
        save_total_limit=1, save_only_model=True,
        # Similar lengths in a batch reduce padding work, especially on CPU.
        train_sampling_strategy="group_by_length",
        optim="adamw_torch", seed=seed, data_seed=seed,
        dataloader_num_workers=0, dataloader_pin_memory=torch.cuda.is_available(),
        logging_steps=20, disable_tqdm=True, report_to="none",
    )
    trainer = Trainer(
        model=model, args=args, processing_class=tokenizer,
        train_dataset=datasets["train"], eval_dataset=datasets["val"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    print(f"Fine-tuning {model_name} on {args.device}; {len(frames['train'])} training rows", flush=True)
    trainer.train()

    # Trainer restores the best validation checkpoint before this test prediction.
    prediction = trainer.predict(datasets["test"])
    metrics = evaluate_predictions(prediction.label_ids, prediction.predictions.argmax(axis=-1))
    history = [entry for entry in trainer.state.log_history if "eval_f1" in entry]
    best_epoch = max(history, key=lambda entry: entry["eval_f1"])["epoch"]
    metrics.update({
        "model": f"{model.config.model_type}_sequence_classification",
        "pretrained_model": str(model_name), "pretrained_revision": revision,
        "evaluation_split": "test", "positive_label": 1,
        "label_mapping": {"0": "benign", "1": "malicious"},
        "confusion_matrix_layout": "rows=true, columns=predicted; labels=[0,1]; [[TN,FP],[FN,TP]]",
        **{f"{split}_samples": len(frame) for split, frame in frames.items()},
        "best_epoch": best_epoch, "best_val_f1": float(trainer.state.best_metric),
        "history": history, "input_column": "raw_text",
        "hyperparameters": {
            "epochs": epochs, "batch_size": batch_size,
            "max_length": max_length, "learning_rate": learning_rate,
            "weight_decay": 0.01, "warmup_ratio": 0.1, "lr_scheduler_type": "linear",
            "max_grad_norm": 1.0, "optimizer": "adamw_torch",
            "train_sampling_strategy": "group_by_length", "loss": "cross_entropy",
            "seed": seed, "cpu_threads": cpu_threads, "device": str(args.device),
            "all_weights_trainable": True,
        },
        "data_sha256": {
            split: hashlib.sha256((processed_dir / f"{split}.jsonl").read_bytes()).hexdigest()
            for split in frames
        },
        "versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("torch", "transformers", "datasets", "accelerate", "scikit-learn")},
        },
    })
    # Hugging Face saves config + safetensors + tokenizer for standalone reload.
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(model_dir)
    metrics["model_files_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(model_dir.iterdir()) if path.is_file()
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_metrics = metrics_path.with_suffix(".json.tmp")
    temporary_metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    temporary_metrics.replace(metrics_path)
    print(json.dumps({key: metrics[key] for key in (
        "accuracy", "precision", "recall", "f1", "confusion_matrix",
    )}, indent=2))
    print(f"Saved model: {model_dir.resolve()}\nSaved metrics: {metrics_path.resolve()}", flush=True)
    return trainer, metrics


if __name__ == "__main__":
    train_transformer()
