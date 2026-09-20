"""Compare saved test reports: python -m src.compare_models.

This reads metrics only; it does not reload or retrain the classifiers.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures on machines without a graphical display.
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
MODEL_REPORTS = (
    ("Baseline (TF-IDF + LR)", "baseline_metrics.json"),
    ("Word2Vec + BiLSTM", "lstm_metrics.json"),
    ("DistilBERT", "transformer_metrics.json"),
)
METRICS = ("accuracy", "precision", "recall", "f1")


def load_metrics(reports_dir):
    """Reject invalid scores and reports from different test sets."""
    rows = []
    reference = None
    for name, filename in MODEL_REPORTS:
        path = Path(reports_dir) / filename
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError(f"{filename}: expected a JSON object")
        for metric in METRICS:
            value = report.get(metric)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{filename}: {metric} must be a finite number between 0 and 1")
        if report.get("evaluation_split") != "test" or report.get("positive_label") != 1:
            raise ValueError(f"{filename}: expected test metrics with malicious=1")
        count = report.get("test_samples")
        hashes = report.get("data_sha256")
        test_hash = hashes.get("test") if isinstance(hashes, dict) else None
        if type(count) is not int or count < 1 or not isinstance(test_hash, str) or not test_hash:
            raise ValueError(f"{filename}: missing valid test_samples or test SHA-256")
        # Matching hashes identify the exact split, rather than just equal row counts.
        identity = (count, test_hash)
        if reference is not None and identity != reference:
            raise ValueError(f"{filename}: reports must describe the same test split")
        reference = identity
        rows.append((name, filename, report))
    return rows


def compare_models(reports_dir=REPORTS_DIR):
    """Write a Markdown table and a grouped accuracy/F1 chart to reports_dir."""
    reports_dir = Path(reports_dir)
    rows = load_metrics(reports_dir)  # Validate everything before replacing outputs.
    count = rows[0][2]["test_samples"]
    table = [
        "# Model comparison",
        "",
        f"Shared held-out test set: **{count:,} prompts**. "
        "Precision, recall, and F1 use malicious (label 1) as the positive class.",
        "",
        "| Model | Accuracy | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, filename, report in rows:
        scores = " | ".join(f"{report[metric]:.2%}" for metric in METRICS)
        table.append(f"| [{name}]({filename}) | {scores} |")
    table.extend([
        "",
        "![Accuracy and F1 comparison](comparison_chart.png)",
        "",
        "The chart starts at zero; labels show small score differences precisely. "
        "These are descriptive results for this test set, not a statistical significance test.",
        "",
        "Dataset scope and limitations: [dataset notes](dataset_selection.md).",
        "",
    ])

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.subplots_adjust(top=0.80, bottom=0.17, left=0.10, right=0.98)
        positions = list(range(len(rows)))
        width = 0.34
        for metric, label, offset, color in (
            ("accuracy", "Accuracy", -width / 2, "#2166ac"),
            ("f1", "F1 (malicious)", width / 2, "#d97706"),
        ):
            values = [report[metric] * 100 for _, _, report in rows]
            bars = ax.bar([x + offset for x in positions], values, width, label=label, color=color)
            ax.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=4, fontsize=10)
        ax.set_xticks(positions, [name.replace(" (", "\n(") for name, _, _ in rows])
        ax.set_ylabel("Test score")
        ax.set_ylim(0, 108)  # Zero baseline plus room for value labels.
        ax.set_yticks(range(0, 101, 20))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("Prompt injection detection: model comparison", fontsize=16, fontweight="bold")
        fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
                   bbox_to_anchor=(0.5, 0.92), ncol=2, frameon=False)
        fig.text(0.5, 0.035, f"Same held-out test split | {count:,} prompts | Higher is better",
                 ha="center", fontsize=10, color="#555555")
        chart_path = reports_dir / "comparison_chart.png"
        fig.savefig(chart_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    markdown_path = reports_dir / "comparison.md"
    markdown_path.write_text("\n".join(table), encoding="utf-8")
    print(f"Saved table: {markdown_path.resolve()}")
    print(f"Saved chart: {chart_path.resolve()}")
    return markdown_path, chart_path


if __name__ == "__main__":
    compare_models()
