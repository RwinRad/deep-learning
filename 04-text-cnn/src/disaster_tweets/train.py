"""Train TF-IDF + Logistic Regression and TextCNN on the Kaggle tweet dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import load_training_data, stratified_split
from .metrics import classification_metrics
from .text import normalize_tweet
from .training import TrainConfig, train_text_cnn


def _baseline_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_tweet,
                    token_pattern=r"(?u)\b\w+\b|<url>|<user>",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=2_000, random_state=seed)),
        ]
    )


def _save_comparison(metrics: dict[str, dict[str, float]], output_path: Path) -> None:
    metric_names = ["f1", "roc_auc", "precision", "recall"]
    model_names = list(metrics)
    x = np.arange(len(metric_names))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9, 5))
    for index, model_name in enumerate(model_names):
        values = [metrics[model_name][name] for name in metric_names]
        ax.bar(x + (index - (len(model_names) - 1) / 2) * width, values, width, label=model_name)
    ax.set_xticks(x, [name.upper() for name in metric_names])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Validation score")
    ax.set_title("Validation performance comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_history(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    losses = [row["train_loss"] for row in history]
    f1_scores = [row["val_f1"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, losses, marker="o")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[1].plot(epochs, f1_scores, marker="o")
    axes[1].set_title("Validation F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 score")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_training_data(args.data)
    x_train, x_valid, y_train, y_valid = stratified_split(frame, random_state=args.seed)
    print(f"Loaded {len(frame):,} examples | train={len(x_train):,} | validation={len(x_valid):,}")

    # Classical baseline: a strong, lightweight point of comparison.
    print("\nTraining TF-IDF + Logistic Regression...")
    baseline = _baseline_pipeline(args.seed)
    baseline.fit(x_train, y_train)
    baseline_probabilities = baseline.predict_proba(x_valid)[:, 1]
    baseline_metrics = classification_metrics(y_valid, baseline_probabilities)
    joblib.dump(baseline, args.output_dir / "tfidf_logreg.joblib")

    # Neural model: vocabulary is learned from training texts only.
    print("\nTraining TextCNN...")
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
    )
    cnn_metrics, history = train_text_cnn(
        x_train, y_train, x_valid, y_valid, args.output_dir, config, device=args.device
    )

    all_metrics = {"TF-IDF + Logistic Regression": baseline_metrics, "TextCNN": cnn_metrics}
    (args.output_dir / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8"
    )
    (args.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    _save_comparison(all_metrics, args.output_dir / "model_comparison.png")
    _save_history(history, args.output_dir / "textcnn_training_history.png")

    print("\nValidation results (threshold = 0.50):")
    for model_name, model_metrics in all_metrics.items():
        print(f"\n{model_name}")
        for metric_name, value in model_metrics.items():
            print(f"  {metric_name:>9}: {value:.4f}")
    print(f"\nArtifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
