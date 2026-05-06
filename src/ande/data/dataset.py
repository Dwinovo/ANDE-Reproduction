"""Dataset that joins per-session image tensors with the 26-d statistical features.

The two preprocessing scripts produce two parquet manifests:

  * data/manifest_raw.parquet    - one row per session (image_<size> column)
  * data/manifest_stats.parquet  - one row per pcap (26 features + pcap_src)

We join on ``pcap_src`` so every session inherits the statistical feature
vector of its source capture. Stratified 8:2 split is performed at the session
level (not the pcap level) to mirror the paper's evaluation protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from ande.data.preprocess_stats import FEATURE_ORDER

VALID_TASKS = ("binary2", "behavior14")


@dataclass
class SplitManifest:
    train: pd.DataFrame
    test: pd.DataFrame
    label_col: str
    num_classes: int


def _label_col(task: str) -> tuple[str, int]:
    if task == "binary2":
        return "label_2cls", 2
    if task == "behavior14":
        return "label_14cls", 14
    raise ValueError(f"task must be one of {VALID_TASKS}, got {task!r}")


def load_joined_manifest(
    manifest_raw: str | Path,
    manifest_stats: str | Path,
    size: int,
    task: str,
) -> tuple[pd.DataFrame, str, int]:
    raw_df = pd.read_parquet(manifest_raw)
    stat_df = pd.read_parquet(manifest_stats)
    if f"image_{size}" not in raw_df.columns:
        raise KeyError(f"manifest_raw lacks image_{size} column")
    label_col, num_classes = _label_col(task)
    joined = raw_df.merge(stat_df, on="pcap_src", how="inner", suffixes=("", "_stat"))
    keep = ["session_id", "pcap_src", label_col, f"image_{size}", *FEATURE_ORDER]
    joined = joined[keep].dropna()
    return joined, label_col, num_classes


VALID_SPLIT_GRANULARITY = ("session", "pcap")


def _stratified_pcap_split(
    pcap_label: pd.DataFrame, label_col: str, train_ratio: float, seed: int
) -> tuple[set[str], set[str]]:
    """Per-class stratified pcap split that guarantees at least 1 pcap in
    each of train and test for every class that has >= 2 pcaps.

    sklearn's ``train_test_split(stratify=...)`` is unreliable for classes
    with only 2-3 samples (it can put both pcaps in train when the math
    rounds that way). We do it manually instead.
    """
    rng = np.random.default_rng(seed)
    train: list[str] = []
    test: list[str] = []
    for cls in sorted(pcap_label[label_col].unique()):
        pcaps = pcap_label.loc[pcap_label[label_col] == cls, "pcap_src"].tolist()
        n = len(pcaps)
        if n < 2:
            # Singleton class — cannot be split. Drop from both halves.
            continue
        rng.shuffle(pcaps)
        n_test = max(1, int(round(n * (1.0 - train_ratio))))
        n_test = min(n - 1, n_test)  # leave at least 1 for train
        test.extend(pcaps[:n_test])
        train.extend(pcaps[n_test:])
    return set(train), set(test)


def stratified_split(
    df: pd.DataFrame,
    label_col: str,
    train_ratio: float,
    seed: int,
    split_at: str = "pcap",
) -> SplitManifest:
    """Stratified train/test split.

    ``split_at='session'`` is the naive choice but causes pcap-level
    leakage: the 26-d statistical features are computed per pcap, so when
    sessions from the same pcap end up in both train and test the features
    are duplicated and tree models can perfectly memorise pcap -> label.

    ``split_at='pcap'`` (default) splits at the pcap level first, then
    expands to all sessions of the chosen pcaps. The 26-d stats vector for
    a given pcap therefore appears either entirely in train or entirely in
    test, never both.
    """
    if split_at not in VALID_SPLIT_GRANULARITY:
        raise ValueError(f"split_at must be one of {VALID_SPLIT_GRANULARITY}, got {split_at!r}")

    num_classes = int(df[label_col].max()) + 1

    if split_at == "session":
        train_df, test_df = train_test_split(
            df,
            train_size=train_ratio,
            stratify=df[label_col],
            random_state=seed,
        )
    else:  # pcap-level split
        if "pcap_src" not in df.columns:
            raise KeyError("pcap_src column required for pcap-level split")
        # One label per pcap (verified by Algorithm 2's per-pcap output).
        pcap_label = df.groupby("pcap_src", as_index=False)[label_col].first()
        train_pcaps, test_pcaps = _stratified_pcap_split(
            pcap_label, label_col, train_ratio, seed
        )
        train_df = df[df["pcap_src"].isin(train_pcaps)]
        test_df = df[df["pcap_src"].isin(test_pcaps)]

    return SplitManifest(
        train=train_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
        label_col=label_col,
        num_classes=num_classes,
    )


class ANDEDataset(Dataset):
    """Returns ``(image, stat_vec, label)`` per session."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str | Path,
        size: int,
        label_col: str,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.size = size
        self.image_col = f"image_{size}"
        self.label_col = label_col
        # Pre-stack stats for speed
        self.stats = self.df[list(FEATURE_ORDER)].to_numpy(dtype=np.float32)
        self.labels = self.df[label_col].to_numpy(dtype=np.int64)
        self.image_paths = self.df[self.image_col].tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        img = np.load(self.data_root / self.image_paths[idx])
        # Normalise to [0, 1]; SE-ResNet expects (1, H, W) float
        img = img.astype(np.float32) / 255.0
        return (
            torch.from_numpy(img).unsqueeze(0),
            torch.from_numpy(self.stats[idx]),
            int(self.labels[idx]),
        )
