"""Train Word2Vec + a PyTorch LSTM: python -m src.model_lstm."""

import hashlib
import json
import platform
import random
import zlib
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from gensim.models import Word2Vec
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

from src.data_loader import PROJECT_ROOT
from src.model_baseline import evaluate_predictions, load_split
from src.preprocess import PROCESSED_DIR, SEED, TOKEN_PATTERN

MODEL_PATH = PROJECT_ROOT / "models" / "lstm.pt"
METRICS_PATH = PROJECT_ROOT / "reports" / "lstm_metrics.json"
PAD_ID, UNK_ID = 0, 1


def stable_hash(word):
    """Avoid Python's process-dependent string hash when initializing Word2Vec."""
    return zlib.crc32(word.encode("utf-8"))


def build_embeddings(sentences, embedding_dim=100, epochs=10, min_count=2, seed=SEED):
    """Learn vocabulary and skip-gram vectors from training sentences only."""
    word2vec = Word2Vec(
        vector_size=embedding_dim, window=5, min_count=min_count, sg=1,
        workers=1, seed=seed, hashfxn=stable_hash,
    )
    word2vec.build_vocab(sentences)
    if not len(word2vec.wv):
        raise ValueError("Word2Vec vocabulary is empty; lower min_count")
    word2vec.train(sentences, total_examples=len(sentences), epochs=epochs)
    vocab = {word: index + 2 for index, word in enumerate(word2vec.wv.index_to_key)}
    weights = np.zeros((len(vocab) + 2, embedding_dim), dtype=np.float32)
    weights[UNK_ID] = word2vec.wv.vectors.mean(axis=0)
    weights[2:] = word2vec.wv.vectors
    return vocab, torch.from_numpy(weights)


def encode_tokens(sentences, vocab, max_length):
    """Map tokens to IDs, truncate on the right, and pad; empty input uses UNK."""
    if type(max_length) is not int or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    ids = torch.full((len(sentences), max_length), PAD_ID, dtype=torch.long)
    lengths = torch.empty(len(sentences), dtype=torch.long)
    for row, tokens in enumerate(sentences):
        sequence = [vocab.get(token, UNK_ID) for token in tokens[:max_length]] or [UNK_ID]
        ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        lengths[row] = len(sequence)
    return ids, lengths


class LSTMClassifier(nn.Module):
    """Trainable Word2Vec embeddings -> bidirectional LSTM -> one binary logit."""

    def __init__(self, embedding_weights, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_weights.clone(), freeze=False, padding_idx=PAD_ID,
        )
        self.lstm = nn.LSTM(
            embedding_weights.shape[1], hidden_dim, batch_first=True, bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, ids, lengths):
        # Packing prevents padding from changing either direction's final state.
        packed = pack_padded_sequence(
            self.embedding(ids), lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        summary = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(summary)).squeeze(1)


@torch.inference_mode()
def predict_probabilities(model, loader, device="cpu"):
    """Return P(malicious) in loader order; evaluation loaders must not shuffle."""
    model.eval()
    probabilities = []
    for ids, lengths, _ in loader:
        logits = model(ids.to(device), lengths)
        probabilities.append(logits.sigmoid().cpu())
    return torch.cat(probabilities).numpy()


def load_lstm(path=MODEL_PATH, device="cpu"):
    """Restore weights and preprocessing metadata without pickling a model class."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = LSTMClassifier(checkpoint["state_dict"]["embedding.weight"], **checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def train_lstm(
    processed_dir=PROCESSED_DIR, model_path=MODEL_PATH, metrics_path=METRICS_PATH,
    epochs=12, batch_size=64, embedding_dim=100, hidden_dim=64, max_length=256,
    learning_rate=0.001, patience=3, word2vec_epochs=10, min_count=2,
    seed=SEED, device="cpu", cpu_threads=4,
):
    """Select a checkpoint by validation F1, then evaluate the untouched test split."""
    for name, value in {
        "epochs": epochs, "batch_size": batch_size, "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim, "max_length": max_length, "patience": patience,
        "word2vec_epochs": word2vec_epochs, "min_count": min_count, "cpu_threads": cpu_threads,
    }.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    processed_dir = Path(processed_dir)
    model_path, metrics_path = Path(model_path), Path(metrics_path)
    frames = {split: load_split(processed_dir / f"{split}.jsonl") for split in ("train", "val", "test")}
    for split, frame in frames.items():
        if "tokens" not in frame or not frame["tokens"].map(
            lambda tokens: isinstance(tokens, list) and bool(tokens)
            and all(isinstance(token, str) and bool(token.strip()) for token in tokens)
        ).all():
            raise ValueError(f"{split}: expected non-empty lists of string tokens")
        if set(frame["label"]) != {0, 1}:
            raise ValueError(f"{split}: both classes must be present")
    for left, right in combinations(frames, 2):
        if set(frames[left]["clean_text"]) & set(frames[right]["clean_text"]):
            raise ValueError(f"{left}/{right} overlap detected; regenerate processed splits")

    print("Training Word2Vec on training tokens only...", flush=True)
    vocab, weights = build_embeddings(
        frames["train"]["tokens"].tolist(), embedding_dim, word2vec_epochs, min_count, seed,
    )
    print(f"Word2Vec vocabulary: {len(vocab)} tokens", flush=True)
    loaders = {}
    for split, frame in frames.items():
        # ponytail: padded tensors live in RAM; use a streaming Dataset for larger corpora.
        ids, lengths = encode_tokens(frame["tokens"].tolist(), vocab, max_length)
        labels = torch.tensor(frame["label"].to_numpy(), dtype=torch.float32)
        loaders[split] = DataLoader(
            TensorDataset(ids, lengths, labels), batch_size=batch_size,
            shuffle=(split == "train"), generator=torch.Generator().manual_seed(seed),
        )

    model_config = {"hidden_dim": hidden_dim, "dropout": 0.3}
    model = LSTMClassifier(weights, **model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    counts = frames["train"]["label"].value_counts()
    positive_weight = float(counts[0] / counts[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    best_f1, best_epoch, stale_epochs = -1.0, 0, 0
    best_state, history = None, []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for ids, lengths, labels in loaders["train"]:
            optimizer.zero_grad()
            loss = criterion(model(ids.to(device), lengths), labels.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; artifacts were not replaced")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
        val_probs = predict_probabilities(model, loaders["val"], device)
        val_metrics = evaluate_predictions(frames["val"]["label"], (val_probs >= 0.5).astype(int))
        train_loss = total_loss / len(frames["train"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_f1": val_metrics["f1"]})
        print(f"Epoch {epoch:02d}: loss={train_loss:.4f}, val_f1={val_metrics['f1']:.4f}", flush=True)
        if val_metrics["f1"] > best_f1:
            best_f1, best_epoch, stale_epochs = val_metrics["f1"], epoch, 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print("Early stopping: validation F1 stopped improving.", flush=True)
                break

    # Restore the best validation checkpoint, rather than evaluating the final epoch.
    model.load_state_dict(best_state)
    test_probs = predict_probabilities(model, loaders["test"], device)
    metrics = evaluate_predictions(frames["test"]["label"], (test_probs >= 0.5).astype(int))
    config = {
        "embedding_dim": embedding_dim, "hidden_dim": hidden_dim, "bidirectional": True,
        "dropout": 0.3, "max_length": max_length, "batch_size": batch_size,
        "epochs": epochs, "patience": patience, "learning_rate": learning_rate,
        "word2vec_epochs": word2vec_epochs, "min_count": min_count, "word2vec_window": 5,
        "word2vec_sg": 1, "word2vec_workers": 1, "trainable_embeddings": True,
        "seed": seed, "device": str(device), "cpu_threads": cpu_threads,
        "positive_weight": positive_weight, "gradient_clip_norm": 1.0,
    }
    versions = {
        "python": platform.python_version(),
        **{name: version(name) for name in ("torch", "gensim", "numpy", "scikit-learn")},
    }
    data_hashes = {
        split: hashlib.sha256((processed_dir / f"{split}.jsonl").read_bytes()).hexdigest()
        for split in frames
    }
    checkpoint = {
        "state_dict": best_state, "vocab": vocab, "model_config": model_config,
        "max_length": max_length, "pad_id": PAD_ID, "unk_id": UNK_ID,
        "label_mapping": {0: "benign", 1: "malicious"}, "threshold": 0.5,
        "preprocessing": {"input_column": "tokens", "lowercase": True, "token_pattern": TOKEN_PATTERN},
        "best_epoch": best_epoch, "best_val_f1": best_f1, "hyperparameters": config,
        "data_sha256": data_hashes, "versions": versions,
    }
    metrics.update({
        "model": "word2vec_bilstm", "evaluation_split": "test", "positive_label": 1,
        "label_mapping": {"0": "benign", "1": "malicious"}, "threshold": 0.5,
        "confusion_matrix_layout": "rows=true, columns=predicted; labels=[0,1]; [[TN,FP],[FN,TP]]",
        **{f"{split}_samples": len(frame) for split, frame in frames.items()},
        "vocabulary_size": len(vocab), "best_epoch": best_epoch, "best_val_f1": best_f1,
        "history": history, "hyperparameters": config, "versions": versions,
        "data_sha256": data_hashes, "input_column": "tokens",
        "truncated_samples": {
            split: int(frame["tokens"].map(len).gt(max_length).sum()) for split, frame in frames.items()
        },
    })
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = model_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary_model)
    metrics["model_sha256"] = hashlib.sha256(temporary_model.read_bytes()).hexdigest()
    temporary_metrics = metrics_path.with_suffix(".json.tmp")
    temporary_metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    temporary_model.replace(model_path)
    temporary_metrics.replace(metrics_path)
    print(json.dumps({key: metrics[key] for key in ("accuracy", "precision", "recall", "f1", "confusion_matrix")}, indent=2))
    print(f"Saved model: {model_path.resolve()}\nSaved metrics: {metrics_path.resolve()}", flush=True)
    return model, metrics


if __name__ == "__main__":
    train_lstm()
