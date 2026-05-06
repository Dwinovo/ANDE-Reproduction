import torch

from ande.models.se_block import SEBlock


def test_se_block_preserves_shape():
    se = SEBlock(channels=64).eval()
    x = torch.randn(4, 64, 8, 8)
    with torch.no_grad():
        y = se(x)
    assert y.shape == x.shape


def test_se_block_scales_into_zero_one_range_after_sigmoid():
    se = SEBlock(channels=32).eval()
    x = torch.ones(2, 32, 4, 4)
    with torch.no_grad():
        y = se(x)
    # sigmoid output * 1 stays in (0, 1)
    assert y.min().item() >= 0.0
    assert y.max().item() <= 1.0


def test_se_block_handles_unit_channel():
    se = SEBlock(channels=8, reduction=16).eval()
    x = torch.randn(1, 8, 16, 16)
    with torch.no_grad():
        y = se(x)
    assert y.shape == x.shape
