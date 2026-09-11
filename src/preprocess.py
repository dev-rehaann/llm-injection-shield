"""Create common splits and model-specific text views without changing raw files.

Run from the project root: python -m src.preprocess
"""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import PROJECT_ROOT, RAW_DIR, class_counts

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SEED = 42
# A small English list avoids removing task cues such as "system", "all",
# "must", "not", and "never". LSTM tokens retain even these stopwords.
STOPWORDS = frozenset(
    "a an the am is are was were be been being of to in on at for with by as".split()
)
# Keep contractions together; retain punctuation/symbols as separate tokens.
TOKEN_PATTERN = r"\w+(?:['’]\w+)*|[^\w\s]"


def preprocess_text(text):
    """Return baseline text, LSTM tokens, and untouched transformer input."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    tokens = re.findall(TOKEN_PATTERN, text.lower())
    filtered = [token for token in tokens if token not in STOPWORDS]
    # Avoid an empty baseline input for prompts consisting only of stopwords.
    clean_text = " ".join(filtered or tokens)
    return {"raw_text": text, "clean_text": clean_text, "tokens": tokens}


def prepare_splits(raw_dir=RAW_DIR, output_dir=PROCESSED_DIR, val_size=0.2, seed=SEED):
    """Clean duplicate groups, split training data, and save aligned JSONL files."""
    if isinstance(val_size, bool) or not 0 < val_size < 1:
        raise ValueError("val_size must be a fraction between 0 and 1")
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    parts, source_hashes = [], {}
    # Put test first so deduplication keeps held-out copies rather than training copies.
    for split in ("test", "train"):
        path = raw_dir / f"{split}.csv"
        frame = pd.read_csv(path, keep_default_na=False)
        class_counts(frame, split)  # Reuse the loader's schema/label validation.
        source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        views = pd.DataFrame(frame["text"].map(preprocess_text).tolist())
        views["label"] = frame["label"].to_numpy()
        views["source_id"] = [f"{split}:{row}" for row in range(len(frame))]
        views["source_split"] = split
        parts.append(views)
    all_rows = pd.concat(parts, ignore_index=True)

    # Use the baseline representation as the duplicate key. It also collapses
    # case, whitespace, punctuation spacing, and our stopword-only differences.
    # ponytail: exact normalized matches only; add near-duplicate grouping if
    # a later template-similarity audit shows material leakage.
    conflicts = all_rows.groupby("clean_text")["label"].transform("nunique") > 1
    excluded_conflicts = all_rows.loc[conflicts].assign(reason="conflicting_labels")
    remaining = all_rows.loc[~conflicts]
    duplicates = remaining.duplicated("clean_text", keep="first")
    excluded_duplicates = remaining.loc[duplicates].assign(reason="duplicate")
    excluded = pd.concat([excluded_conflicts, excluded_duplicates], ignore_index=True)
    unique = remaining.loc[~duplicates]

    training_pool = unique.loc[unique["source_split"] == "train"]
    test = unique.loc[unique["source_split"] == "test"]
    if set(training_pool["label"]) != {0, 1} or set(test["label"]) != {0, 1}:
        raise ValueError("Both classes must remain in the training pool and test split")
    try:
        train, val = train_test_split(
            training_pool, test_size=val_size, stratify=training_pool["label"],
            random_state=seed,
        )
    except ValueError as error:
        raise ValueError("Not enough examples for the requested stratified split") from error
    splits = {"train": train, "val": val, "test": test}
    if any(set(frame["label"]) != {0, 1} for frame in splits.values()):
        raise ValueError("Both classes must be represented in every output split")

    # source_id fixes output ordering; all models must use these exact rows.
    splits = {
        name: frame.sort_values("source_id").reset_index(drop=True)
        for name, frame in splits.items()
    }
    counts = {
        name: class_counts(frame.rename(columns={"raw_text": "text"}), name)
        for name, frame in splits.items()
    }
    metadata = {
        "seed": seed,
        "validation_fraction_of_clean_training_pool": val_size,
        "source_sha256": source_hashes,
        "label_mapping": {"0": "benign", "1": "malicious"},
        "class_counts": counts,
        "input_rows": len(all_rows),
        "conflicting_groups": int(excluded_conflicts["clean_text"].nunique()),
        "excluded_conflicting_rows": len(excluded_conflicts),
        "excluded_duplicate_rows": len(excluded_duplicates),
        "retained_rows": sum(len(frame) for frame in splits.values()),
        "duplicate_key": "clean_text; conflicting groups excluded, test copies preferred",
        "token_pattern": TOKEN_PATTERN,
        "stopwords": sorted(STOPWORDS),
        "views": {
            "raw_text": "unchanged source text for the transformer tokenizer",
            "tokens": "lowercase tokens, punctuation and stopwords retained for LSTM",
            "clean_text": "space-joined tokens with conservative stopword removal for TF-IDF",
        },
    }

    # Validate/split everything before saving; finish each write before replacing.
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in {**splits, "excluded": excluded}.items():
        temporary = output_dir / f"{name}.jsonl.tmp"
        frame.to_json(temporary, orient="records", lines=True, force_ascii=False)
        temporary.replace(output_dir / f"{name}.jsonl")
    temporary = output_dir / "metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "metadata.json")

    balance = pd.DataFrame.from_dict(counts, orient="index")
    balance["total"] = balance.sum(axis=1)
    print(balance.to_string())
    print(f"Excluded: {len(excluded_conflicts)} conflicting rows, "
          f"{len(excluded_duplicates)} duplicate rows")
    print(f"Saved processed data to: {output_dir.resolve()}")
    return splits


if __name__ == "__main__":
    prepare_splits()
