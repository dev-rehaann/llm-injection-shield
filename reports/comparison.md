# Model comparison

Shared held-out test set: **2,049 prompts**. Precision, recall, and F1 use malicious (label 1) as the positive class.

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| [Baseline (TF-IDF + LR)](baseline_metrics.json) | 99.12% | 99.22% | 97.99% | 98.60% |
| [Word2Vec + BiLSTM](lstm_metrics.json) | 99.32% | 99.22% | 98.61% | 98.92% |
| [DistilBERT](transformer_metrics.json) | 99.51% | 99.38% | 99.07% | 99.23% |

![Accuracy and F1 comparison](comparison_chart.png)

The chart starts at zero; labels show small score differences precisely. These are descriptive results for this test set, not a statistical significance test.

Dataset scope and limitations: [dataset notes](dataset_selection.md).
