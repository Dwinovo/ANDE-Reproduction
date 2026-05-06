"""Plain ResNet-18 baseline (no SE, image branch only).

Equivalent to ANDE's image backbone with ``use_se=False`` followed by a single
Linear classifier (no statistical features and no fusion head). Used to
isolate the contribution of the dual-branch architecture.
"""

from __future__ import annotations

import torch
from torch import nn

from ande.models.se_resnet import SEResNet18


class PlainResNet18(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = SEResNet18(in_channels=1, num_features=256, use_se=False)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, image: torch.Tensor, _stat: torch.Tensor | None = None) -> torch.Tensor:
        return self.classifier(self.backbone(image))
