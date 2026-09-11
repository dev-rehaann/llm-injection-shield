# Dataset selection — Step 2

| Option | Advantages | Limitations |
| --- | --- | --- |
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | 662 examples; simple text/label schema; fast baseline experiments | Small sample for LSTM and transformer comparisons |
| [jackhhao/jailbreak-classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification) | About 1,300 examples in the default balanced configuration; explicit benign/jailbreak labels; collected jailbreak prompts | Limited size; mainly jailbreak detection; default configuration uses the balanced files despite its name |
| [xTRam1/safe-guard-prompt-injection](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection) | 10,296 published rows; text/label schema; existing train/test splits | Includes synthetic attacks; some labels may capture harmful requests or suspicious phrasing instead of a clear injection |

## Choice

Use **xTRam1/safe-guard-prompt-injection** for the initial comparison because it
provides more training examples while remaining small enough for student work.
This is a practical choice, not evidence that its annotations are ground truth.

Pinned revision: `a3a877d608f37b7d20d9945671902df895ecdb46`.
Labels: `0 = benign`, `1 = malicious` (the source's injection/attack category).

The [source card](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection)
describes GPT-3.5-generated attacks and includes jackhhao among its seed sources.
Therefore jackhhao should not be treated as an independent evaluation set without
checking overlap. The card narrative and published split sizes differ; report
counts computed from the pinned download.

## Scope of this step

Save source rows unchanged as UTF-8 CSVs, preserving published train/test membership.
Save dataset identity, revision, label mapping, and counts in `data/raw/metadata.json`.
No deduplication, resampling, normalization, or model training is performed.

Before training, audit duplicates and split overlap, and create validation data
from training data only. Reserve test data for final evaluation. Similar attack
templates and synthetic wording can inflate scores; do not claim broad real-world
coverage from this dataset alone.

The selected dataset card does not declare a license in its metadata. Raw data
remains local and excluded from Git; this repository publishes the downloader.

## Download verification

The loader completed successfully and saved the following source counts:

| Split | Benign (0) | Malicious (1) | Total |
| --- | ---: | ---: | ---: |
| Train | 5,740 | 2,496 | 8,236 |
| Test | 1,410 | 650 | 2,060 |
| Total | 7,150 | 3,146 | 10,296 |

This is about 69.4% benign and 30.6% malicious. Keep the original balance in raw
data; address imbalance during training and report malicious-class precision,
recall, and F1 alongside accuracy.

Validation: the real download succeeded, and the offline regression check passed
for counts, schema errors, missing-class counts, metadata, and CSV round trips.
Tested with Python 3.13.7, datasets 5.0.1, pandas 3.0.3, NumPy 2.4.6,
and PyArrow 24.0.0 in a temporary environment reusing installed scientific
packages. The project's default virtual environment setup remains isolated.
