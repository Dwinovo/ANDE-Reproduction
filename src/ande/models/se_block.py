"""Squeeze-and-Excitation block (Hu et al. 2018, paper Section IV-A)."""

from __future__ import annotations

import torch
from torch import nn


class SEBlock(nn.Module):
    """Channel-wise attention block.

    Squeeze: global average pool over spatial dims.
    Excitation: two-layer MLP (channels -> channels // reduction -> channels)
    with ReLU and Sigmoid. The output rescales the input channel-wise.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        z = self.avg_pool(x).view(b, c)
        s = self.fc(z).view(b, c, 1, 1)
        return x * s
