"""A compact multi-kernel TextCNN for binary tweet classification."""

from __future__ import annotations

import torch
from torch import nn


class TextCNN(nn.Module):
    """Embedding + parallel temporal convolutions + max pooling classifier."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        num_filters: int = 96,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must include at least PAD and UNK tokens.")
        if not kernel_sizes or any(size % 2 == 0 for size in kernel_sizes):
            raise ValueError("kernel_sizes must be a non-empty tuple of odd integers.")

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=size,
                padding=size // 2,
            )
            for size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return one unnormalized logit per example (shape: batch_size)."""
        embedded = self.embedding(token_ids).transpose(1, 2)
        pooled = [torch.relu(conv(embedded)).amax(dim=2) for conv in self.convolutions]
        features = self.dropout(torch.cat(pooled, dim=1))
        return self.classifier(features).squeeze(1)
