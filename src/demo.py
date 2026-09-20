"""Presentation CLI: python -m src.demo [--model transformer] [--prompt TEXT]."""

import argparse
import math
import pickle
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
NAMES = {
    "baseline": "Baseline (TF-IDF + LR)",
    "lstm": "Word2Vec + BiLSTM",
    "transformer": "DistilBERT",
}
LABELS = {0: "benign", 1: "malicious"}


def load_models(selection="all", models_dir=MODELS_DIR):
    """Load selected local artifacts once, then reuse them for every prompt."""
    if selection not in ("all", *NAMES):
        raise ValueError(f"Unknown model: {selection}")
    selected = list(NAMES) if selection == "all" else [selection]
    paths = {
        "baseline": Path(models_dir) / "baseline.pkl",
        "lstm": Path(models_dir) / "lstm.pt",
        "transformer": Path(models_dir) / "transformer",
    }
    for name in selected:
        if not paths[name].exists():
            raise FileNotFoundError(f"Missing {paths[name]}. Run python -m src.model_{name} first.")

    loaded = {}
    if "baseline" in selected:
        # Pickle is for trusted artifacts produced by this project only.
        with paths["baseline"].open("rb") as handle:
            loaded["baseline"] = pickle.load(handle)
        if set(loaded["baseline"].classes_) != {0, 1}:
            raise ValueError("Baseline must use labels 0=benign and 1=malicious.")
    if any(name in selected for name in ("lstm", "transformer")):
        import torch

        torch.set_num_threads(4)
    if "lstm" in selected:
        from src.model_lstm import load_lstm

        loaded["lstm"] = load_lstm(paths["lstm"], device="cpu")
    if "transformer" in selected:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.utils import logging

        logging.set_verbosity_error()
        logging.disable_progress_bar()
        tokenizer = AutoTokenizer.from_pretrained(paths["transformer"], local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            paths["transformer"], local_files_only=True,
        ).eval()
        if model.config.id2label != LABELS:
            raise ValueError("Transformer must use labels 0=benign and 1=malicious.")
        loaded["transformer"] = (tokenizer, model)
    return loaded


def predict_prompt(text, models):
    """Return each model's label and P(predicted label), not always P(malicious)."""
    from src.preprocess import preprocess_text

    views = preprocess_text(text)
    results = []
    if "baseline" in models:
        model = models["baseline"]
        label = int(model.predict([views["clean_text"]])[0])
        probabilities = model.predict_proba([views["clean_text"]])[0]
        confidence = float(probabilities[list(model.classes_).index(label)])
        results.append({"model": NAMES["baseline"], "prediction": LABELS[label],
                        "confidence": confidence, "note": ""})
    if "lstm" in models:
        import torch
        from src.model_lstm import encode_tokens

        model, checkpoint = models["lstm"]
        limit = checkpoint["max_length"]
        ids, lengths = encode_tokens([views["tokens"]], checkpoint["vocab"], limit)
        with torch.inference_mode():
            malicious_probability = model(ids, lengths).sigmoid().item()
        label = int(malicious_probability >= checkpoint["threshold"])
        results.append({
            "model": NAMES["lstm"], "prediction": LABELS[label],
            "confidence": malicious_probability if label else 1 - malicious_probability,
            "note": f"Only the first {limit} word/punctuation tokens were used."
                    if len(views["tokens"]) > limit else "",
        })
    if "transformer" in models:
        import torch

        tokenizer, model = models["transformer"]
        limit = tokenizer.model_max_length
        # Count subwords to flag truncation; feed the original text to the tokenizer.
        length = len(tokenizer.tokenize(views["raw_text"])) + tokenizer.num_special_tokens_to_add()
        inputs = tokenizer(views["raw_text"], truncation=True, max_length=limit, return_tensors="pt")
        with torch.inference_mode():
            probabilities = model(**inputs).logits.softmax(dim=-1)[0]
        label = int(probabilities.argmax().item())
        results.append({
            "model": NAMES["transformer"], "prediction": LABELS[label],
            "confidence": float(probabilities[label].item()),
            "note": f"Only the first {limit} subword tokens (including special tokens) were used."
                    if length > limit else "",
        })
    for result in results:
        if not math.isfinite(result["confidence"]) or not 0 <= result["confidence"] <= 1:
            raise ValueError(f"{result['model']} returned an invalid probability.")
    return results


def print_results(results):
    """Use plain text so the display works in PowerShell and classroom terminals."""
    print()
    print(f"{'Model':<25} {'Prediction':<12} {'Confidence':>11}")
    print("-" * 50)
    for result in results:
        print(f"{result['model']:<25} {result['prediction']:<12} {result['confidence']:>11.2%}")
    print()
    if len(results) > 1:
        labels = {result["prediction"] for result in results}
        print(f"All {len(results)} models agree: {next(iter(labels))}."
              if len(labels) == 1 else "Models disagree. Review the individual predictions.")
    for result in results:
        if result["note"]:
            print(f"{result['model']}: {result['note']}")
    print("Confidence is an uncalibrated estimate for the predicted label.\n")


def interactive_loop(models):
    """Multiline input: /run submits, /clear resets, /quit exits."""
    print("Paste a prompt (multiple lines and blank lines are supported).")
    print("Type /run on its own line to classify, /clear to reset, or /quit to exit.\n")
    lines = []
    while True:
        try:
            line = input("prompt> " if not lines else "      > ")
        except EOFError:
            if "\n".join(lines).strip():
                print_results(predict_prompt("\n".join(lines), models))
            return
        command = line.strip()
        if command == "/quit":
            return
        if command == "/clear":
            lines.clear()
            print("Prompt cleared.")
        elif command == "/run":
            text = "\n".join(lines)
            if not text.strip():
                print("Please enter a non-empty prompt.")
                continue
            print_results(predict_prompt(text, models))
            lines.clear()
        else:
            lines.append(line)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classify prompts with the trained LLM Injection Shield models.")
    parser.add_argument("--model", choices=("all", *NAMES), default="all",
                        help="default: all; transformer runs only DistilBERT")
    parser.add_argument("--prompt", help="classify one prompt and exit; otherwise use interactive input or stdin")
    args = parser.parse_args(argv)
    text = args.prompt
    interactive = text is None and sys.stdin.isatty()
    if text is None and not interactive:
        text = sys.stdin.read()
    if text is not None and not text.strip():
        parser.error("Please provide a non-empty prompt.")

    print("\nLLM INJECTION SHIELD")
    print("Prompt injection / jailbreak detection")
    print("Loading local models...", flush=True)
    try:
        models = load_models(args.model)
        if interactive:
            interactive_loop(models)
            print("Goodbye.")
        else:
            print_results(predict_prompt(text, models))
        return 0
    except ImportError as error:
        print(f"Dependency error: {error}. Install requirements.txt in your environment.", file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError, pickle.UnpicklingError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nGoodbye.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
