"""ANDE composite model (paper Section IV-D, Fig. 9).

Two parallel feature extractors:
  * SE-ResNet-18 over the grayscale image of raw bytes  -> 256-d
  * MLP (26 -> 18 -> 9) over normalised statistics      ->   9-d
The 265-d concatenation is fed into a final MLP head (265 -> 100 -> 30 -> C).
"""

from __future__ import annotations

import torch
from torch import nn

from ande.models.se_resnet import SEResNet18

NUM_STAT_FEATURES = 26
IMG_FEATURE_DIM = 256
STAT_FEATURE_DIM = 9


class StatMLP(nn.Module):
    """26 -> 18 -> 9 with ReLU."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_STAT_FEATURES, 18),
            nn.ReLU(inplace=True),
            nn.Linear(18, STAT_FEATURE_DIM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionHead(nn.Module):
    """265 -> 100 -> 30 -> num_classes."""

    def __init__(self, num_classes: int, in_dim: int = IMG_FEATURE_DIM + STAT_FEATURE_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 30),
            nn.ReLU(inplace=True),
            nn.Linear(30, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ANDE(nn.Module):
    """The full ANDE model: SE-ResNet image branch + MLP stat branch + fusion head."""

    def __init__(self, num_classes: int, use_se: bool = True, se_reduction: int = 16) -> None:
        super().__init__()
        self.use_se = use_se
        self.image_backbone = SEResNet18(
            in_channels=1, num_features=IMG_FEATURE_DIM, use_se=use_se, se_reduction=se_reduction
        )
        self.stat_backbone = StatMLP()
        self.head = FusionHead(num_classes=num_classes)

    def forward(self, image: torch.Tensor, stat: torch.Tensor) -> torch.Tensor:
        f_img = self.image_backbone(image)
        f_stat = self.stat_backbone(stat)
        return self.head(torch.cat([f_img, f_stat], dim=1))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
