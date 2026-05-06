"""FlowPic baseline (Shapira & Shavitt 2021 [35]).

A FlowPic is a 2-D histogram of (packet length, relative arrival time) within
a time window of a flow. The ANDE paper cites the published numbers but does
not specify the FlowPic configuration. We approximate the original 1500x1500
setting downsampled to 64x64 to keep training tractable.

Status: minimal reference implementation; not part of the default
reproduction matrix because it requires per-packet timing information that
must be re-extracted from pcaps (it is *not* derivable from the cached image
tensors). Hook in by writing a thin extractor in ``preprocess_flowpic.py``
and a Dataset that emits FlowPic histograms instead of byte images.
"""

from __future__ import annotations

import torch
from torch import nn


class FlowPicCNN(nn.Module):
    def __init__(self, num_classes: int, in_size: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=10, stride=1, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=10, stride=1, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_size, in_size)
            out_dim = self.features(dummy).flatten(1).shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))
