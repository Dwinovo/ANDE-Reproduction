"""ResNet-18 with SE blocks; channel widths halved per paper Table III.

Channel progression follows the paper:
    conv1   :  1 -> 32          (kernel 7, stride 2, padding 3)
    layer1  : 32 -> 32   stride 1
    layer2  : 32 -> 64   stride 2
    layer3  : 64 -> 128  stride 2
    layer4  : 128 -> 256 stride 2
    avgpool -> flatten 256

When ``use_se=False`` the SE block is replaced with the identity, allowing the
ablation study from paper Table V (ANDE without SE block).
"""

from __future__ import annotations

import torch
from torch import nn

from ande.models.se_block import SEBlock


class _Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SEBasicBlock(nn.Module):
    """ResNet BasicBlock with optional SE module before the residual add."""

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_se: bool = True,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se: nn.Module = SEBlock(out_channels, reduction=se_reduction) if use_se else _Identity()

        if stride != 1 or in_channels != out_channels:
            self.downsample: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = _Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + identity
        return self.relu(out)


class SEResNet18(nn.Module):
    """ResNet-18 backbone matching the paper's channel widths."""

    def __init__(
        self,
        in_channels: int = 1,
        num_features: int = 256,
        use_se: bool = True,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.use_se = use_se
        self.conv1 = nn.Conv2d(
            in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(32, 32, blocks=2, stride=1, use_se=use_se, r=se_reduction)
        self.layer2 = self._make_layer(32, 64, blocks=2, stride=2, use_se=use_se, r=se_reduction)
        self.layer3 = self._make_layer(64, 128, blocks=2, stride=2, use_se=use_se, r=se_reduction)
        self.layer4 = self._make_layer(128, 256, blocks=2, stride=2, use_se=use_se, r=se_reduction)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.num_features = num_features
        if num_features != 256:
            self.proj: nn.Module = nn.Linear(256, num_features)
        else:
            self.proj = _Identity()

        self._init_weights()

    @staticmethod
    def _make_layer(
        in_c: int, out_c: int, blocks: int, stride: int, use_se: bool, r: int
    ) -> nn.Sequential:
        layers: list[nn.Module] = [SEBasicBlock(in_c, out_c, stride=stride, use_se=use_se, se_reduction=r)]
        for _ in range(1, blocks):
            layers.append(SEBasicBlock(out_c, out_c, stride=1, use_se=use_se, se_reduction=r))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5**0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.proj(x)
