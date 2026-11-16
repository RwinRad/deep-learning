"""Training utilities for the TextCNN model."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .metrics import classification_metrics
from .model import TextCNN
from .vocabulary import build_vocabulary, encode_text


@dataclass(frozen=True)
class TrainConfig:
    max_vocab_size: int = 20_000
    min_frequency: int = 2
    max_length: int = 64
    embedding_dim: int = 100
    num_filters: int = 96
    dropout: float = 0.4
    batch_size: int = 64
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 3
    seed: int = 42


class TweetDataset(Dataset):
    """Minimal PyTorch dataset for fixed-length token sequences."""

    def __init__(self, texts: list[str], labels: list[int], vocabulary: dict[str, int], max_length: int):
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have equal lengths.")
        self.features = [encode_text(text, vocabulary, max_length) for text in texts]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.features[index], dtype=torch.long),
            torch.tensor(float(self.labels[index]), dtype=torch.float32),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _predict_probabilities(
    model: TextCNN, loader: DataLoader, device: torch.device
) -> tuple[list[int], list[float]]:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    with torch.inference_mode():
        for token_ids, batch_labels in loader:
            logits = model(token_ids.to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(batch_labels.to(torch.int64).cpu().numpy().tolist())
    return labels, probabilities


def train_text_cnn(
    train_texts: list[str],
    train_labels: list[int],
    valid_texts: list[str],
    valid_labels: list[int],
    output_dir: str | Path,
    config: TrainConfig,
    device: str = "cpu",
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Fit TextCNN with validation-F1 early stopping and persist inference artifacts."""
    if not train_texts or not valid_texts:
        raise ValueError("Training and validation sets must both contain at least one row.")
    if len(train_texts) != len(train_labels) or len(valid_texts) != len(valid_labels):
        raise ValueError("Text and label counts must match.")
    if config.epochs < 1 or config.batch_size < 1 or config.patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive.")

    set_seed(config.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocabulary = build_vocabulary(
        train_texts, max_size=config.max_vocab_size, min_frequency=config.min_frequency
    )
    train_dataset = TweetDataset(train_texts, train_labels, vocabulary, config.max_length)
    valid_dataset = TweetDataset(valid_texts, valid_labels, vocabulary, config.max_length)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator
    )
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)

    device_obj = torch.device(device)
    model = TextCNN(
        vocab_size=len(vocabulary),
        embedding_dim=config.embedding_dim,
        num_filters=config.num_filters,
        dropout=config.dropout,
    ).to(device_obj)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for token_ids, labels in train_loader:
            token_ids = token_ids.to(device_obj)
            labels = labels.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            logits = model(token_ids)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        valid_y, valid_probabilities = _predict_probabilities(model, valid_loader, device_obj)
        valid_metrics = classification_metrics(valid_y, valid_probabilities)
        epoch_record = {
            "epoch": float(epoch),
            "train_loss": float(running_loss / max(seen, 1)),
            **{f"val_{key}": value for key, value in valid_metrics.items()},
        }
        history.append(epoch_record)
        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"loss={epoch_record['train_loss']:.4f} | "
            f"val_f1={valid_metrics['f1']:.4f} | val_auc={valid_metrics['roc_auc']:.4f}"
        )

        if valid_metrics["f1"] > best_f1:
            best_f1 = valid_metrics["f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping: validation F1 did not improve for {config.patience} epochs.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")

    model.load_state_dict(best_state)
    valid_y, valid_probabilities = _predict_probabilities(model, valid_loader, device_obj)
    best_metrics = classification_metrics(valid_y, valid_probabilities)

    torch.save(best_state, output_dir / "textcnn_state.pt")
    (output_dir / "vocabulary.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model_config = {
        "vocab_size": len(vocabulary),
        "embedding_dim": config.embedding_dim,
        "num_filters": config.num_filters,
        "kernel_sizes": [3, 5, 7],
        "dropout": config.dropout,
        "max_length": config.max_length,
        "training_config": asdict(config),
    }
    (output_dir / "textcnn_config.json").write_text(
        json.dumps(model_config, indent=2), encoding="utf-8"
    )
    return best_metrics, history
