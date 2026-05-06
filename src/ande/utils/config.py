"""Tiny YAML-backed config loader. We avoid hydra to keep dependencies simple."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataCfg:
    size: int = 8100  # 784 / 4096 / 8100
    task: str = "behavior14"  # binary2 / behavior14
    split_ratio: float = 0.8
    batch_size: int = 64
    num_workers: int = 4
    manifest_raw: str = "data/manifest_raw.parquet"
    manifest_stats: str = "data/manifest_stats.parquet"


@dataclass
class ModelCfg:
    name: str = "ande"  # ande / resnet18 / cnn1d / dt / rf / xgb
    use_se: bool = True
    se_reduction: int = 16


@dataclass
class TrainCfg:
    epochs: int = 50
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    scheduler: str = "step"  # step / cosine / none
    step_size: int = 10
    gamma: float = 0.5
    early_stop_patience: int = 10


@dataclass
class Config:
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    seed: int = 42
    run_name: str = "ande_default"
    out_dir: str = "outputs"
    checkpoints_dir: str = "checkpoints"
    runs_dir: str = "runs"


def _merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            out[k] = _merge(base.get(k), v) if k in base else v
        return out
    return override


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    raw = raw or {}
    return Config(
        data=DataCfg(**_merge({}, raw.get("data", {}))),
        model=ModelCfg(**_merge({}, raw.get("model", {}))),
        train=TrainCfg(**_merge({}, raw.get("train", {}))),
        seed=raw.get("seed", 42),
        run_name=raw.get("run_name", "ande_default"),
        out_dir=raw.get("out_dir", "outputs"),
        checkpoints_dir=raw.get("checkpoints_dir", "checkpoints"),
        runs_dir=raw.get("runs_dir", "runs"),
    )
