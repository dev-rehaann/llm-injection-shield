"""Download the fixed study dataset without changing its text, labels, or splits."""

import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_ID = "xTRam1/safe-guard-prompt-injection"
# Pin the source so future downloads reproduce this semester's experiments.
REVISION = "a3a877d608f37b7d20d9945671902df895ecdb46"
LABEL_NAMES = {0: "benign", 1: "malicious"}


def class_counts(frame, split):
    """Validate the source schema and count labels without cleaning the data."""
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError(f"{split}: expected text and label columns")
    if frame.empty:
        raise ValueError(f"{split}: empty split")
    if not frame["text"].map(lambda text: isinstance(text, str) and bool(text.strip())).all():
        raise ValueError(f"{split}: text must contain non-empty strings")
    if not pd.api.types.is_integer_dtype(frame["label"]) or not frame["label"].isin(LABEL_NAMES).all():
        raise ValueError(f"{split}: labels must be integers 0 (benign) or 1 (malicious)")

    counts = frame["label"].value_counts()
    return {name: int(counts.get(label, 0)) for label, name in LABEL_NAMES.items()}


def download_data(raw_dir=RAW_DIR):
    """Save UTF-8 CSVs and provenance; return a per-split class balance table."""
    raw_dir = Path(raw_dir)
    dataset = load_dataset(
        DATASET_ID,
        name="default",
        revision=REVISION,
        cache_dir=str(raw_dir.parent / "hf_cache"),
    )
    if set(dataset) != {"train", "test"}:
        raise ValueError("Expected the published train and test splits")

    frames = {split: dataset[split].to_pandas() for split in ("train", "test")}
    # Validate every split before writing files; never silently drop bad rows.
    counts = {split: class_counts(frame, split) for split, frame in frames.items()}
    balance = pd.DataFrame.from_dict(counts, orient="index")
    balance.loc["total"] = balance.sum()
    balance["total"] = balance["benign"] + balance["malicious"]
    balance.index.name = "split"

    raw_dir.mkdir(parents=True, exist_ok=True)
    for split, frame in frames.items():
        # Replace each file only after its full write succeeds.
        temporary = raw_dir / f"{split}.csv.tmp"
        frame.to_csv(temporary, index=False, encoding="utf-8")
        temporary.replace(raw_dir / f"{split}.csv")

    metadata = {
        "dataset_id": DATASET_ID,
        "revision": REVISION,
        "label_mapping": LABEL_NAMES,
        "class_counts": counts,
        "files": ["train.csv", "test.csv"],
        "text_processing": "none; source text and published splits preserved",
    }
    temporary = raw_dir / "metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(raw_dir / "metadata.json")

    print(f"Dataset: {DATASET_ID}\nRevision: {REVISION}")
    print(balance.to_string())
    print(f"Saved raw data to: {raw_dir.resolve()}")
    return balance


if __name__ == "__main__":
    download_data()
