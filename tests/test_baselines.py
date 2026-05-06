import numpy as np
import pandas as pd
import pytest
import torch

from ande.baselines.cnn1d import CNN1D
from ande.baselines.flowpic import FlowPicCNN
from ande.baselines.hierarchical import fit_hierarchy, predict_hierarchy
from ande.baselines.plain_resnet import PlainResNet18
from ande.data.preprocess_stats import FEATURE_ORDER


@pytest.mark.parametrize("size,hw", [(784, 28), (4096, 64), (8100, 90)])
def test_cnn1d_forward(size: int, hw: int) -> None:
    model = CNN1D(in_length=size, num_classes=14).eval()
    img = torch.randn(2, 1, hw, hw)
    with torch.no_grad():
        out = model(img)
    assert out.shape == (2, 14)


def test_plain_resnet_forward() -> None:
    model = PlainResNet18(num_classes=14).eval()
    img = torch.randn(2, 1, 28, 28)
    with torch.no_grad():
        out = model(img)
    assert out.shape == (2, 14)


def test_flowpic_forward() -> None:
    model = FlowPicCNN(num_classes=14, in_size=64).eval()
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 14)


def test_hierarchical_fits_and_predicts() -> None:
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame(rng.random((n, len(FEATURE_ORDER))), columns=list(FEATURE_ORDER))
    # Construct labels so each of the 14 classes has at least a few samples.
    df["label_14cls"] = np.tile(np.arange(14), n // 14 + 1)[:n]
    model = fit_hierarchy(df, seed=0)
    pred = predict_hierarchy(model, df[list(FEATURE_ORDER)].to_numpy())
    assert pred.shape == (n,)
    assert pred.min() >= 0 and pred.max() < 14
