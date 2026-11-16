# Disaster Tweet Classification

An end-to-end NLP project that predicts whether a tweet refers to a real disaster. It compares a classical machine-learning baseline with a compact deep-learning model and keeps the validation process reproducible.

> **Task:** binary text classification (`0` = not disaster-related, `1` = disaster-related)
>
> **Dataset:** [Natural Language Processing with Disaster Tweets — Kaggle](https://www.kaggle.com/competitions/nlp-getting-started)

## Why this project?

Short social-media messages are noisy: they contain hashtags, URLs, mentions, abbreviations, and ambiguous language. A word such as *ablaze* can be literal or figurative, so recognizing context matters.

This repository evaluates two approaches on the same stratified validation split:

1. **TF-IDF + Logistic Regression** — a strong, lightweight baseline using unigrams and bigrams.
2. **TextCNN (PyTorch)** — a trainable embedding layer followed by parallel 1D convolutions with different kernel widths, global max pooling, and a binary classification head.

The aim is not to assume that the neural model wins. The validation metrics should determine which approach works better on this dataset.

## Repository structure

```text
.
├── data/
│   ├── raw/                  # Place train.csv and test.csv here (not committed)
│   └── reference/
│       └── sample_submission.csv  # Optional Kaggle submission template
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── artifacts/                # Generated models, metrics, and plots (not committed)
├── src/disaster_tweets/
│   ├── data.py               # CSV validation and stratified splitting
│   ├── metrics.py            # Classification metrics
│   ├── model.py              # TextCNN architecture
│   ├── predict.py            # Single-text and CSV inference
│   ├── text.py               # Tweet-aware normalization/tokenization
│   ├── train.py              # Baseline + neural training workflow
│   ├── training.py           # PyTorch training loop and early stopping
│   └── vocabulary.py         # Training-only vocabulary and encoding
├── tests/
└── pyproject.toml
```

## Dataset setup

1. Download the competition files from the [Kaggle dataset page](https://www.kaggle.com/competitions/nlp-getting-started/data).
2. Accept the competition rules on Kaggle, if prompted.
3. Put the files in `data/raw/`:

```text
data/raw/train.csv
data/raw/test.csv
```

The training file must contain `id`, `text`, and `target`. The test file must contain `id` and `text`. The included `data/reference/sample_submission.csv` is an optional copy of the submission template you supplied; it is not training data and is not required by the training or prediction commands. The competition dataset itself is not redistributed in this repository; follow Kaggle's terms when downloading and using it.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate         # Windows
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For a CPU-only PyTorch installation, use the command recommended by the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for your operating system. The default training device in this project is CPU.

## Train and compare the models

From the repository root:

```bash
python -m disaster_tweets.train --data data/raw/train.csv
```

Useful options:

```bash
python -m disaster_tweets.train --help
python -m disaster_tweets.train --epochs 5 --batch-size 64 --max-length 64 --seed 42
python -m disaster_tweets.train --device cuda  # only with a compatible CUDA-enabled PyTorch install
```

The workflow:

- validates the input file and target labels;
- creates an approximately 80/20 stratified, group-aware train/validation split (identical normalized tweets stay in one partition);
- fits the TF-IDF vocabulary and the neural vocabulary using **training texts only**;
- trains both approaches and applies early stopping to TextCNN using validation F1;
- reports accuracy, precision, recall, F1, and ROC-AUC at a fixed probability threshold of 0.50;
- saves models, metrics, and plots under `artifacts/`.

Generated files include `metrics.json`, `model_comparison.png`, `textcnn_training_history.png`, `tfidf_logreg.joblib`, and the TextCNN state/configuration/vocabulary files. Results are generated locally and are intentionally not fabricated or prefilled in this README.

## Inference

Classify one example with the trained TextCNN:

```bash
python -m disaster_tweets.predict \
  --model textcnn \
  --text "Wildfire crews are responding to a large fire near the highway." \
  --artifacts-dir artifacts
```

Create a Kaggle-format submission file from `test.csv`:

```bash
python -m disaster_tweets.predict \
  --model textcnn \
  --input data/raw/test.csv \
  --output artifacts/submission.csv
```

Use `--model baseline` to run the TF-IDF model instead. The submission contains the required `id` and `target` columns.

## Exploratory analysis

Open `notebooks/01_exploratory_analysis.ipynb` to inspect class balance, text length, missing values, and representative examples before training.

## Evaluation notes and limitations

- The validation split is stratified and deterministic (`seed=42` by default).
- The official competition test labels are not available, so local validation metrics are not Kaggle leaderboard scores.
- The neural vocabulary is fit on training text only; validation data is not used to create the vocabulary.
- URLs and user mentions are normalized, while hashtags and most punctuation are preserved.
- A model prediction is not emergency intelligence. It can misread figurative language, sarcasm, location context, and breaking-news terminology. Do not use it as the sole basis for real-world decisions.

## Tests

```bash
pytest -q
```

## Learning reference and attribution

The task and learning progression were inspired by the NLP material in [aaghamohammadi/deep-learning-youtube](https://github.com/aaghamohammadi/deep-learning-youtube), especially its tokenization, embedding, and NLP project notebooks (`041`–`043`). This repository is an independent implementation: it does not copy the source notebook's code. The original course repository remains the reference for the lesson; the implementation, experiment structure, CLI, tests, and documentation here are separate work.

## License

The original code in this repository is released under the MIT License. The Kaggle dataset is not included and remains subject to its source terms.
