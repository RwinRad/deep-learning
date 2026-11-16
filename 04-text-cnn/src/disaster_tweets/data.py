"""Dataset loading, validation, and leakage-safe train/validation splitting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .text import normalize_tweet

REQUIRED_COLUMNS = {"id", "text", "target"}


def load_training_data(path: str | Path) -> pd.DataFrame:
    """Load the official competition training CSV and validate its schema."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Training file not found: {path}. Download train.csv from the Kaggle competition "
            "and place it at data/raw/train.csv."
        )

    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Training CSV is missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["target"] = pd.to_numeric(frame["target"], errors="raise").astype("int64")

    labels = set(frame["target"].unique())
    if not labels.issubset({0, 1}) or len(labels) != 2:
        raise ValueError(f"Expected binary target labels 0 and 1; found {sorted(labels)}")
    if frame.empty:
        raise ValueError("Training CSV contains no rows.")

    return frame


def stratified_split(
    frame: pd.DataFrame,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[str], list[str], list[int], list[int]]:
    """Split texts and labels while preserving class proportions."""
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")

    # Group identical normalized tweets together so duplicates cannot appear in
    # both training and validation sets and inflate validation performance.
    n_splits = round(1 / validation_size)
    if n_splits < 2 or abs((1 / n_splits) - validation_size) > 1e-8:
        raise ValueError("validation_size must be the reciprocal of an integer (e.g. 0.2 or 0.25).")

    groups = frame["text"].map(normalize_tweet)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    train_indices, valid_indices = next(
        splitter.split(frame["text"], frame["target"], groups=groups)
    )
    x_train = frame.iloc[train_indices]["text"].tolist()
    x_valid = frame.iloc[valid_indices]["text"].tolist()
    y_train = frame.iloc[train_indices]["target"].tolist()
    y_valid = frame.iloc[valid_indices]["target"].tolist()
    return x_train, x_valid, y_train, y_valid
