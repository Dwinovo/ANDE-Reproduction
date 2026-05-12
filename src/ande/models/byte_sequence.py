"""Byte-sequence models for extended ANDE experiments.

These models deliberately avoid the 2-D image inductive bias used by ANDE's
ResNet branch. They consume the cached byte image tensor, flatten it back to
the original byte order, and model it as a 1-D sequence.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ande.models.ande import STAT_FEATURE_DIM, StatMLP


class _ResidualDilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        padding = dilation * 3
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=7, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=7, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class ByteTCN(nn.Module):
    """Dilated 1-D CNN over the byte sequence, with optional stats fusion."""

    def __init__(
        self,
        num_classes: int,
        channels: int = 96,
        use_stats: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_stats = use_stats
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.blocks = nn.Sequential(
            _ResidualDilatedBlock(channels, dilation=1, dropout=dropout),
            _ResidualDilatedBlock(channels, dilation=2, dropout=dropout),
            _ResidualDilatedBlock(channels, dilation=4, dropout=dropout),
            _ResidualDilatedBlock(channels, dilation=8, dropout=dropout),
        )
        self.stat_backbone = StatMLP() if use_stats else None
        head_in = channels * 2 + (STAT_FEATURE_DIM if use_stats else 0)
        self.classifier = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, image: torch.Tensor, stat: torch.Tensor | None = None) -> torch.Tensor:
        x = image.flatten(1).unsqueeze(1)
        x = self.blocks(self.stem(x))
        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)
        if self.use_stats:
            if stat is None:
                raise ValueError("stat tensor is required when use_stats=True")
            pooled = torch.cat([pooled, self.stat_backbone(stat)], dim=1)
        return self.classifier(pooled)


class ByteSegmentAttention(nn.Module):
    """Selects informative contiguous byte segments with attention pooling.

    The sequence is split into non-overlapping byte patches. A shallow
    Transformer contextualizes segment embeddings, then a learned scorer
    produces attention weights over segments. This is the smallest useful
    version of "adaptive length" for the current cached data format.
    """

    def __init__(
        self,
        num_classes: int,
        segment_len: int = 128,
        d_model: int = 128,
        nhead: int = 4,
        layers: int = 2,
        max_length: int = 8100,
        use_stats: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.segment_len = segment_len
        self.use_stats = use_stats
        self.max_segments = math.ceil(max_length / segment_len)
        self.patch_proj = nn.Linear(segment_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.max_segments, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        self.stat_backbone = StatMLP() if use_stats else None
        head_in = d_model + (STAT_FEATURE_DIM if use_stats else 0)
        self.classifier = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self.last_attention: torch.Tensor | None = None

    def _patchify(self, image: torch.Tensor) -> torch.Tensor:
        x = image.flatten(1)
        pad = (-x.shape[1]) % self.segment_len
        if pad:
            x = F.pad(x, (0, pad))
        return x.unfold(dimension=1, size=self.segment_len, step=self.segment_len)

    def forward(self, image: torch.Tensor, stat: torch.Tensor | None = None) -> torch.Tensor:
        patches = self._patchify(image)
        n_segments = patches.shape[1]
        x = self.patch_proj(patches) + self.pos[:, :n_segments]
        x = self.encoder(x)
        attn = torch.softmax(self.score(x).squeeze(-1), dim=1)
        self.last_attention = attn.detach()
        pooled = torch.sum(x * attn.unsqueeze(-1), dim=1)
        if self.use_stats:
            if stat is None:
                raise ValueError("stat tensor is required when use_stats=True")
            pooled = torch.cat([pooled, self.stat_backbone(stat)], dim=1)
        return self.classifier(pooled)
