import pytest
import torch

from ande.models import ANDE, SEResNet18, count_parameters


@pytest.mark.parametrize("size,hw", [(784, 28), (4096, 64), (8100, 90)])
@pytest.mark.parametrize("use_se", [True, False])
def test_ande_forward(size: int, hw: int, use_se: bool) -> None:
    model = ANDE(num_classes=14, use_se=use_se).eval()
    img = torch.randn(2, 1, hw, hw)
    stat = torch.randn(2, 26)
    with torch.no_grad():
        out = model(img, stat)
    assert out.shape == (2, 14)


def test_ande_two_class() -> None:
    model = ANDE(num_classes=2).eval()
    img = torch.randn(3, 1, 28, 28)
    stat = torch.randn(3, 26)
    with torch.no_grad():
        out = model(img, stat)
    assert out.shape == (3, 2)


def test_se_resnet_features_dim() -> None:
    backbone = SEResNet18(in_channels=1, num_features=256).eval()
    with torch.no_grad():
        feat = backbone(torch.randn(1, 1, 28, 28))
    assert feat.shape == (1, 256)


def test_param_count_reasonable() -> None:
    n = count_parameters(ANDE(num_classes=14))
    # Paper claims a lightweight model; 1M-5M is the expected ballpark.
    assert 1_000_000 < n < 5_000_000, n


def test_use_se_flag_changes_param_count() -> None:
    n_with = count_parameters(ANDE(num_classes=14, use_se=True))
    n_without = count_parameters(ANDE(num_classes=14, use_se=False))
    assert n_with > n_without
