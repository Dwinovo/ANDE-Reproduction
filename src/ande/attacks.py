"""Test-time traffic perturbations for robustness experiments.

The functions here operate on cached model inputs rather than raw pcaps. They
are intended as controlled proxies for protocol-layer evasion pressure:
padding changes the byte sequence, delay changes timing statistics, and
traffic shaping smooths size/payload statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from ande.data.preprocess_stats import FEATURE_ORDER

_FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_ORDER)}
_DELAY_FEATURES = (
    "Duration_window_flow",
    "Avg_deltas_time",
    "Min_deltas_time",
    "Max_deltas_time",
    "StDev_deltas_time",
)
_SHAPE_FEATURES = (
    "Avg_Pkts_length",
    "Min_Pkts_length",
    "Max_Pkts_length",
    "StDev_Pkts_length",
    "Avg_payload",
    "Min_payload",
    "Max_payload",
    "StDev_payload",
    "Avg_small_loadings_pkt",
)


@dataclass(frozen=True)
class AttackSpec:
    name: str
    level: str = "medium"


def _level_value(level: str, low: float, medium: float, high: float) -> float:
    values = {"low": low, "medium": medium, "high": high}
    if level not in values:
        raise ValueError(f"level must be one of {tuple(values)}, got {level!r}")
    return values[level]


def _shift_with_padding(flat: torch.Tensor, n: int, mode: str) -> torch.Tensor:
    if n <= 0:
        return flat
    n = min(n, flat.numel())
    if mode == "zero":
        prefix = torch.zeros(n, dtype=flat.dtype, device=flat.device)
    elif mode == "random":
        prefix = torch.rand(n, dtype=flat.dtype, device=flat.device)
    else:
        raise ValueError(mode)
    return torch.cat([prefix, flat[:-n]], dim=0)


def perturb_image(image: torch.Tensor, spec: AttackSpec) -> torch.Tensor:
    """Return a perturbed copy of ``image``.

    The model input is normalized to [0, 1]. Padding is inserted at the front
    and the tail is truncated, approximating inserted bytes before the most
    discriminative prefix while keeping the cached fixed length unchanged.
    """
    if spec.name in {"clean", "none"}:
        return image
    out = image.clone()
    if spec.name in {"zero_padding", "random_padding", "padding"}:
        frac = _level_value(spec.level, low=0.05, medium=0.15, high=0.30)
        flat = out.flatten()
        n = int(round(flat.numel() * frac))
        mode = "random" if spec.name == "random_padding" else "zero"
        return _shift_with_padding(flat, n, mode).view_as(out)
    return out


def perturb_stat(stat: torch.Tensor, spec: AttackSpec) -> torch.Tensor:
    """Return a perturbed copy of the normalized 26-d statistic vector."""
    if spec.name in {"clean", "none"}:
        return stat
    out = stat.clone()
    if spec.name == "random_delay":
        scale = _level_value(spec.level, low=0.35, medium=0.75, high=1.25)
        for name in _DELAY_FEATURES:
            out[_FEATURE_INDEX[name]] = out[_FEATURE_INDEX[name]] + scale
    elif spec.name == "traffic_shaping":
        shrink = _level_value(spec.level, low=0.75, medium=0.50, high=0.25)
        for name in _SHAPE_FEATURES:
            idx = _FEATURE_INDEX[name]
            out[idx] = out[idx] * shrink
    elif spec.name in {"combined", "adaptive_evasion"}:
        out = perturb_stat(out, AttackSpec("random_delay", spec.level))
        out = perturb_stat(out, AttackSpec("traffic_shaping", spec.level))
    return out


class PerturbedDataset(Dataset):
    """Dataset wrapper that applies one deterministic perturbation on access."""

    def __init__(self, base: Dataset, spec: AttackSpec) -> None:
        self.base = base
        self.spec = spec

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, stat, label = self.base[idx]
        spec = self.spec
        if spec.name in {"combined", "adaptive_evasion"}:
            image = perturb_image(image, AttackSpec("random_padding", spec.level))
        else:
            image = perturb_image(image, spec)
        stat = perturb_stat(stat, spec)
        return image, stat, label
