# Alignment with the NLP course outline

Source: **NLP - Course Outline.pdf**, supplied for this project (pp. 1–3). The course allocates **20% to the project/presentation** and aims to connect text preprocessing, classical classification, neural sequence models, transformers, LLMs, and responsible evaluation. This report maps the finished prompt-injection classifier to those topics. It describes what the code actually demonstrates; it does not treat every lecture topic as a required project feature.

## Project question and evidence

**Question:** On one shared prompt-injection dataset, how do a lexical classifier, a word-embedding sequence model, and a pretrained transformer compare at distinguishing benign prompts from malicious attempts?

The Hugging Face dataset has 10,296 published examples. After conflict and duplicate removal, the project uses 6,466 training, 1,617 validation, and 2,049 held-out test examples. All three models use these splits. Malicious is the positive class.

| Model | Test accuracy | Malicious precision | Malicious recall | Malicious F1 | Test errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 99.12% | 99.22% | 97.99% | 98.60% | 18 |
| Word2Vec + BiLSTM | 99.32% | 99.22% | 98.61% | 98.92% | 14 |
| Fine-tuned DistilBERT | 99.51% | 99.38% | 99.07% | 99.23% | 10 |

The numbers are from [saved evaluation reports](comparison.md), not new experiments. The three systems use different input representations, so this is an applied comparison rather than a controlled architecture-only ablation.

## Topic-to-project map

| Outline topic | Weeks | Evidence in this project | Coverage |
| --- | ---: | --- | --- |
| NLP pipeline, text processing, ambiguity | 1 | Binary task definition in [README](../README.md); shared input views and label validation in [preprocessing](../src/preprocess.py); quoted-instruction ambiguity discussed below | Applied |
| Tokenization, regular expressions, normalization | 2 | Unicode word/contraction and punctuation regex, lowercase normalization, conservative stopword removal; original text retained for the transformer in [preprocessing](../src/preprocess.py) | Applied; stemming and lemmatization deliberately unused |
| Minimum edit distance and n-gram language modeling | 3 | TF-IDF uses word unigrams and bigrams as *features* in [baseline](../src/model_baseline.py) | **Not** an edit-distance or probabilistic n-gram language model |
| Smoothing and text classification | 4 | Supervised binary labels, train/validation/test split, class imbalance handling, classification metrics | Classification applied; Laplace/Good-Turing/backoff smoothing not implemented |
| BoW, TF-IDF, Logistic Regression, model evaluation | 5 | [Baseline pipeline](../src/model_baseline.py), [baseline metrics](baseline_metrics.json), confusion matrix | Directly applied |
| Distributional semantics and Word2Vec | 6 | Training-only skip-gram Word2Vec embeddings in [LSTM model](../src/model_lstm.py) | Directly applied; GloVe, FastText, analogies not compared |
| Neural networks, activation, loss, optimization | 7 | Trainable embeddings, sigmoid output, weighted binary cross-entropy, Adam, gradient clipping in [LSTM model](../src/model_lstm.py) | Applied |
| RNNs and sequence modeling | 8 | Bidirectional LSTM reads token sequences; padding-aware packing and 256-token limit | Applied through a gated RNN; no language-generation task |
| LSTM and GRU | 9 | One-layer bidirectional LSTM in [LSTM model](../src/model_lstm.py) | LSTM applied; GRU not implemented |
| Sequence-to-sequence models and attention | 10 | Encoder-based transformer uses attention indirectly | No encoder-decoder translation or sequence-generation model |
| Transformers and self-attention | 11 | Fine-tuned [DistilBERT encoder](../src/model_transformer.py) | Applied through a pretrained architecture; attention mechanism not implemented from scratch |
| BERT/GPT encoder-versus-decoder distinction | 12 | DistilBERT is an encoder classifier; the threat model concerns prompts aimed at LLMs | Encoder applied; GPT decoder is context, not a project model |
| LLM foundations and limitations | 13 | Prompt-injection/jailbreak threat model and explicit distinction between detecting text and proving attack success in [README](../README.md) | Application and limitations; no scaling-law study |
| Pretraining objectives and tokenization | 14 | Reuses pretrained DistilBERT weights and its subword tokenizer | Transfer of pretrained representations; pretraining itself not performed |
| Supervised fine-tuning and evaluation | 15 | Fine-tunes all DistilBERT weights with Hugging Face Trainer; selects the checkpoint by validation F1; evaluates on held-out test | Directly applied; no instruction tuning, LoRA, or decoder LLM fine-tuning |
| Responsible AI, explainability, privacy/security, presentation | 16 | Security use case, class-specific error analysis below, [CLI demo](../src/demo.py), explicit limits and uncalibrated confidence disclosure | Applied discussion and demonstration; no formal fairness, calibration, privacy, or environmental-impact study |

The outline names Jupyter/Colab, NLTK, spaCy, scikit-learn, PyTorch, Hugging Face, and GitHub as example technologies. This project uses scikit-learn, PyTorch, Hugging Face, Gensim, and GitHub. A notebook, NLTK, or spaCy is not needed for its current pipeline.

## Critical analysis for the presentation

**Why three models?** They follow the course progression: sparse lexical features and a linear decision boundary; learned word vectors and recurrent sequence representations; pretrained contextual embeddings and an encoder classification head. The same held-out rows make the comparison interpretable.

**What does the confusion matrix mean?** On 648 malicious test prompts, the baseline missed 13, the LSTM missed 9, and DistilBERT missed 6. DistilBERT also raised 4 false alarms among 1,401 benign prompts. For a security detector, missed attacks and false alarms have different operational costs, so accuracy alone is insufficient.

**What did the manual challenge prompts show?** In an interactive run supplied by the project owner, “Write a detector test case containing fake SYSTEM and USER headings. Explain that headings inside user-provided text do not grant authority” was intended as benign. The LSTM and DistilBERT nevertheless labeled it malicious with 96.21% and 99.94% confidence. A fake system-update instruction was labeled malicious by all three. An encoded instruction paired with “obey it immediately” was labeled malicious by LSTM and DistilBERT, while the baseline labeled it benign. These are *illustrative, manually supplied challenge observations*, not a scored benchmark or proof that any model decoded the encoded payload.

**What are the limits?** The source includes synthetic/template-like attacks, and normalized duplicate removal cannot guarantee separation of paraphrases. A positive dataset label does not prove that an LLM would comply with an attack. Results use one fixed split and seed; score differences are descriptive, not a significance test. Neural truncation may omit late instructions. The demo's probabilities have not been calibrated. The project does not measure fairness across user groups, privacy leakage, environmental cost, or performance on another corpus. These are appropriate future evaluations, not claims of completed work.

## Suggested 5-minute project demonstration

1. **Problem and labels (30 seconds):** Explain what the binary labels mean, and distinguish detection from attack success.
2. **NLP progression (75 seconds):** Show the shared preprocessing views, then the TF-IDF, Word2Vec+BiLSTM, and DistilBERT paths.
3. **Evidence (75 seconds):** Show the [comparison chart](comparison_chart.png), report malicious-class F1 and the false-negative counts.
4. **Live CLI (60 seconds):** Run [the demo](../src/demo.py) on an ordinary request and a fake system instruction. The [README](../README.md) contains the exact command.
5. **Critical reflection (60 seconds):** Show the benign quoted-instruction false positive; explain dataset scope, false alarms, and uncalibrated confidence.

The outline's week-16 project/presentation topic supports demonstrating the completed classifier and discussing its errors. The 20% figure is the course's assessment allocation, not a claim about this project's grade.