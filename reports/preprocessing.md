# Shared preprocessing — Step 3

Run from the project root with dependencies installed:

```powershell
.\.venv\Scripts\python.exe -m src.preprocess
```

The pipeline reads `data/raw/train.csv` and `data/raw/test.csv`.
It never modifies those files. Outputs are JSON Lines, one record per line,
so token lists remain real arrays instead of Python strings embedded in CSV.

## Shared representation

Call `preprocess_text(prompt)` from `src.preprocess` for future predictions,
so training and demo inputs use the same rules.

| Field | Intended consumer | Processing |
| --- | --- | --- |
| raw_text | DistilBERT tokenizer | Exact source string, including case, punctuation, and whitespace |
| tokens | Word2Vec/GloVe + LSTM | Lowercase; Unicode word/contraction tokens; punctuation and symbols are individual tokens; stopwords retained |
| clean_text | TF-IDF / Bag-of-Words | Space-joined tokens with conservative English stopword removal |
| label | All three models | 0 = benign; 1 = malicious |
| source_id | Traceability | Original split and zero-based CSV record number, e.g. train:42 |
| source_split | Traceability | Original train or test membership |

The baseline removes only a small fixed list of common function words, defined
in the script. Negations, instruction cues, and words such as "system", "must",
"all", "not", and "never" remain. Stopword-only prompts fall back to their full
token sequence to avoid empty model inputs.

The punctuation rule preserves attack syntax such as angle brackets, hashes,
slashes, and exclamation marks as tokens. For the future TF-IDF model, split
`clean_text` on whitespace and disable the vectorizer's default token pattern;
otherwise its default tokenizer would discard punctuation and one-character
tokens. [scikit-learn feature extraction documentation](https://scikit-learn.org/stable/modules/feature_extraction.html)

Pass `raw_text` directly to the pretrained transformer tokenizer. Its own
normalization, subword tokenization, padding, and truncation belong in the model
step. No transformer tokenizer or embedding vocabulary is fitted here.
[Hugging Face tokenizer documentation](https://huggingface.co/docs/transformers/model_doc/distilbert)

## Deduplication and split policy

1. Validate the source text and binary labels using the loader's existing checks.
2. Group rows by `clean_text`, the representation shared by baseline inputs.
3. Exclude every row in groups that have contradictory labels.
4. Keep one row per remaining group, preferring original test rows.
5. Split only the surviving original training pool: 80% training, 20% validation,
   stratified by label with seed 42.

This yields roughly 64/16/20 overall, with the held-out test pool kept separate.
The cleaned test set has fewer rows than the published test set because conflict
and duplicate removal applies to it too. Every retained test row came from the
original test split.

Grouping by baseline text deliberately merges some case, spacing, and stopword
variants. This keeps identical baseline inputs out of different splits, but does
not guarantee separation of paraphrases or related attack templates. A later
similarity audit may require stricter grouping.

No vocabulary, TF-IDF statistics, or learned transformations use validation/test
data. Fit those on training data only when implementing the models.
[Stratified splitting reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

## Saved results

| Split | Benign | Malicious | Total |
| --- | ---: | ---: | ---: |
| Train | 4,530 | 1,936 | 6,466 |
| Validation | 1,133 | 484 | 1,617 |
| Test | 1,401 | 648 | 2,049 |
| Total | 7,064 | 3,068 | 10,132 |

Five rows from two conflicting groups and 159 further duplicate rows are stored
in `excluded.jsonl` with a reason. No source row disappears without accounting:
10,132 retained + 164 excluded = 10,296 input rows.

`data/processed/` contains `train.jsonl`, `val.jsonl`, `test.jsonl`,
`excluded.jsonl`, and `metadata.json`. Metadata records source CSV SHA-256
hashes, seed, split fraction, counts, token pattern, and stopword list.

## Validation

Both regression checks passed. Tests cover text views, negation and punctuation,
contractions, Unicode and mixed line endings, stopword-only inputs, invalid text,
duplicate precedence, label conflicts, repeatable splitting, original row/text
preservation, and JSONL round trips.

An independent check of the real output verified full row accounting, exact raw
text/label preservation, unchanged source hashes, source split boundaries, no
duplicate baseline strings across retained splits, and pandas JSONL loading.

Verified with Python 3.13.7, scikit-learn 1.9.0, pandas 3.0.3, and NumPy 2.4.6.
These checks used the temporary verification environment from Step 2; the default
project virtual environment still needs its requirements installed.
