"""1D-CNN baseline operating on the raw byte sequence.

Inspired by Wang et al. [25] (paper reference). The image-shaped tensor used by
ANDE is flattened back into a 1D byte sequence so the same preprocessed cache
can be reused.
"""

from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    def __init__(self, in_length: int, num_classes: int) -> None:
        super().__init__()
        self.in_length = in_length
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=25, stride=1, padding=12),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=3),
            nn.Conv1d(32, 64, kernel_size=25, stride=1, padding=12),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=3),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_length)
            out_dim = self.features(dummy).flatten(1).shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, image: torch.Tensor, _stat: torch.Tensor | None = None) -> torch.Tensor:
        x = image.flatten(1).unsqueeze(1)  # (B, 1, H*W)
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
