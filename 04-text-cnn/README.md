# Disaster Tweet Classification

An end-to-end NLP project that predicts whether a tweet refers to a real disaster. It compares a classical machine-learning baseline with a compact deep-learning model using a reproducible validation workflow.

> **Task:** binary text classification (`0` = not disaster-related, `1` = disaster-related)  
> **Dataset:** [Natural Language Processing with Disaster Tweets — Kaggle](https://www.kaggle.com/competitions/nlp-getting-started)

## Models

1. **TF-IDF + Logistic Regression** — a lightweight baseline using unigrams and bigrams.
2. **TextCNN (PyTorch)** — a trainable embedding layer followed by parallel 1D convolutions with different kernel widths, global max pooling, and a binary classification head.

The aim is to compare the models fairly, not to assume that the neural model will perform better.

## Repository structure

```text
.
├── .gitignore
├── LICENSE
├── README.md
├── data/
│   ├── raw/                    # Local Kaggle train.csv and test.csv (not tracked)
│   └── reference/              # Optional local sample_submission.csv (not tracked)
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── artifacts/                  # Generated models, metrics, and plots (not tracked)
├── src/disaster_tweets/
│   ├── data.py                 # CSV validation and leakage-aware splitting
│   ├── metrics.py              # Binary classification metrics
│   ├── model.py                # TextCNN architecture
│   ├── predict.py              # Single-text and CSV inference
│   ├── text.py                 # Tweet-aware normalization and tokenization
│   ├── train.py                # Baseline and neural training workflow
│   ├── training.py             # PyTorch training loop and early stopping
│   └── vocabulary.py           # Training-only vocabulary and encoding
├── tests/
└── pyproject.toml
```

## Dataset setup

1. Download the competition files from the [Kaggle dataset page](https://www.kaggle.com/competitions/nlp-getting-started/data).
2. Accept the competition rules on Kaggle, if prompted.
3. Place the files locally:

```text
data/raw/train.csv
data/raw/test.csv
data/reference/sample_submission.csv   # Optional; only needed as a reference template
```

The training file must contain `id`, `text`, and `target`. The test file must contain `id` and `text`. The optional submission template has `id` and `target` columns.

**Dataset and privacy note:** Kaggle competition CSVs are deliberately excluded from this repository. Tweet text can contain user-generated content and may include incidental personal or contact information. Keep the downloaded files local; do not commit the CSVs, downloaded archives, or derived copies. Follow the competition's terms when downloading and using the data.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate         # Windows
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For a CPU-only PyTorch installation, use the command recommended by the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for your operating system. The default training device is CPU.

## Train and compare the models

Run from this project directory (`04-text-cnn/`):

```bash
python -m disaster_tweets.train --data data/raw/train.csv
```

Useful options:

```bash
python -m disaster_tweets.train --help
python -m disaster_tweets.train --epochs 5 --batch-size 64 --max-length 64 --seed 42
python -m disaster_tweets.train --device cuda  # Requires a compatible CUDA-enabled PyTorch install
```

The workflow:

- validates the input file and target labels;
- creates an approximately 80/20 stratified, group-aware train/validation split (normalized duplicate tweets stay in one partition);
- fits TF-IDF and neural vocabularies on **training texts only**;
- trains both approaches and applies early stopping to TextCNN using validation F1;
- reports Accuracy, Precision, Recall, F1, and ROC-AUC at a probability threshold of 0.50;
- saves models, metrics, and plots under `artifacts/`.

Generated files include `metrics.json`, `training_history.json`, `model_comparison.png`, `textcnn_training_history.png`, `tfidf_logreg.joblib`, and the TextCNN state/configuration/vocabulary files. They are generated locally and are not committed.

## Inference

Classify one example with TextCNN:

```bash
python -m disaster_tweets.predict \
  --model textcnn \
  --text "Wildfire crews are responding to a large fire near the highway." \
  --artifacts-dir artifacts
```

Create a Kaggle-format submission from `test.csv`:

```bash
python -m disaster_tweets.predict \
  --model textcnn \
  --input data/raw/test.csv \
  --output artifacts/submission.csv
```

Use `--model baseline` to run the TF-IDF model instead. The generated submission contains `id` and `target` columns.

## Exploratory analysis

Open `notebooks/01_exploratory_analysis.ipynb` to inspect class balance, text length, missing values, and representative examples before training.

## Evaluation notes and limitations

- The validation split is deterministic (`seed=42` by default), stratified, and group-aware.
- The official competition test labels are not available, so local validation metrics are not Kaggle leaderboard scores.
- The neural vocabulary is fitted on training text only; validation data is not used to build the vocabulary.
- URLs and user mentions are normalized, while hashtags and most punctuation are preserved.
- A model prediction is not emergency intelligence. The classifier can misread figurative language, sarcasm, location context, and breaking-news terminology. Do not use it as the sole basis for real-world decisions.

## Security notes

- Do not commit API keys, passwords, access tokens, private keys, personal contact details, raw tweet CSVs, model weights, or generated artifacts.
- Only load model files that you generated or obtained from a trusted source. In particular, `joblib.load()` deserializes Python objects and must not be used with untrusted files.
- The TextCNN checkpoint is loaded with PyTorch's `weights_only=True` option.

## Tests

```bash
pytest -q
```

## Licence

The project code is released under the MIT License in `LICENSE`. The Kaggle dataset is not included and remains subject to its own source terms.
