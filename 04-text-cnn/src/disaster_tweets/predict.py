"""Run inference with a trained baseline or TextCNN model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import torch

from .model import TextCNN
from .vocabulary import encode_text


def load_textcnn(artifacts_dir: Path, device: str = "cpu"):
    config = json.loads((artifacts_dir / "textcnn_config.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((artifacts_dir / "vocabulary.json").read_text(encoding="utf-8"))
    model = TextCNN(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        num_filters=config["num_filters"],
        kernel_sizes=tuple(config["kernel_sizes"]),
        dropout=config["dropout"],
    )
    state = torch.load(artifacts_dir / "textcnn_state.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(torch.device(device))
    model.eval()
    return model, vocabulary, config["max_length"]


def predict_textcnn(texts: list[str], artifacts_dir: Path, device: str = "cpu") -> list[float]:
    model, vocabulary, max_length = load_textcnn(artifacts_dir, device)
    token_ids = torch.tensor(
        [encode_text(text, vocabulary, max_length) for text in texts], dtype=torch.long
    )
    with torch.inference_mode():
        logits = model(token_ids.to(device))
        return torch.sigmoid(logits).cpu().numpy().tolist()


def predict_baseline(texts: list[str], artifacts_dir: Path) -> list[float]:
    model = joblib.load(artifacts_dir / "tfidf_logreg.joblib")
    return model.predict_proba(texts)[:, 1].tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--model", choices=["textcnn", "baseline"], default="textcnn")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--text", type=str, help="Classify one text string.")
    parser.add_argument("--input", type=Path, help="Competition test CSV with id and text columns.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/submission.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.text is None) == (args.input is None):
        raise SystemExit("Provide exactly one of --text or --input.")

    predict = predict_textcnn if args.model == "textcnn" else predict_baseline
    if args.text:
        probability = predict([args.text], args.artifacts_dir, args.device) if args.model == "textcnn" else predict([args.text], args.artifacts_dir)
        score = probability[0]
        label = int(score >= 0.5)
        print(f"Prediction: {label} ({'disaster-related' if label else 'not disaster-related'})")
        print(f"Positive-class probability: {score:.4f}")
        return

    frame = pd.read_csv(args.input)
    if not {"id", "text"}.issubset(frame.columns):
        raise ValueError("Input CSV must contain 'id' and 'text' columns.")
    texts = frame["text"].fillna("").astype(str).tolist()
    if args.model == "textcnn":
        probabilities = predict(texts, args.artifacts_dir, args.device)
    else:
        probabilities = predict(texts, args.artifacts_dir)
    submission = pd.DataFrame(
        {"id": frame["id"], "target": [int(value >= 0.5) for value in probabilities]}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {len(submission):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
