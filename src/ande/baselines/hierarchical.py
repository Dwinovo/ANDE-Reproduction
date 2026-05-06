"""Hierarchical Classifier baseline (Hu et al. 2020 [26]).

Three-level decision cascade over the 26-d statistical features:

    L0:  binary  Tor vs non-Tor
    L1a: 7-way activity classifier on Tor samples
    L1b: 7-way activity classifier on non-Tor samples

Each node is a Random Forest (the original paper uses RF as well). At
inference time L0's prediction routes the sample to L1a or L1b. The final
14-class prediction is reconstructed from the (is_tor, activity) pair.

This is intentionally simpler than Hu et al.'s six-classifier tree because the
ANDE paper only cites the headline number; we keep the cascade structure but
drop the dataset-specific intermediate node that distinguishes anonymity-tool
families (we already restrict to Tor in this reproduction).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ande.data.dataset import load_joined_manifest, stratified_split
from ande.data.preprocess_stats import FEATURE_ORDER
from ande.metrics import compute_metrics
from ande.utils.config import load_config
from ande.utils.seed import seed_all

LOG = logging.getLogger(__name__)


def _xy(df: pd.DataFrame, label_col: str) -> tuple[np.ndarray, np.ndarray]:
    return df[list(FEATURE_ORDER)].to_numpy(dtype=np.float32), df[label_col].to_numpy(dtype=np.int64)


def fit_hierarchy(df: pd.DataFrame, seed: int) -> dict:
    x = df[list(FEATURE_ORDER)].to_numpy(dtype=np.float32)
    y14 = df["label_14cls"].to_numpy(dtype=np.int64)
    is_tor = (y14 % 2 == 1).astype(np.int64)
    activity = y14 // 2  # 0..6

    root = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
    root.fit(x, is_tor)

    tor_mask = is_tor == 1
    nontor_mask = ~tor_mask
    leaves: dict[int, RandomForestClassifier] = {}
    for branch, mask in [(1, tor_mask), (0, nontor_mask)]:
        if mask.sum() == 0:
            continue
        leaf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
        leaf.fit(x[mask], activity[mask])
        leaves[branch] = leaf
    return {"root": root, "leaves": leaves}


def predict_hierarchy(model: dict, x: np.ndarray) -> np.ndarray:
    is_tor_pred = model["root"].predict(x)
    activity_pred = np.zeros_like(is_tor_pred)
    for branch, leaf in model["leaves"].items():
        m = is_tor_pred == branch
        if m.any():
            activity_pred[m] = leaf.predict(x[m])
    return activity_pred * 2 + is_tor_pred


def run(config_path: str) -> dict:
    cfg = load_config(config_path)
    if cfg.data.task != "behavior14":
        raise ValueError("hierarchical baseline is defined for the 14-class task only")
    seed_all(cfg.seed)

    df, label_col, num_classes = load_joined_manifest(
        cfg.data.manifest_raw, cfg.data.manifest_stats, cfg.data.size, cfg.data.task
    )
    # ensure both labels are present in the joined manifest
    if "label_14cls" not in df.columns:
        raise KeyError("manifest_raw must expose label_14cls for the hierarchical baseline")
    split = stratified_split(df, label_col, cfg.data.split_ratio, cfg.seed)
    x_tr, _ = _xy(split.train, label_col)
    x_te, y_te = _xy(split.test, label_col)
    model = fit_hierarchy(split.train, cfg.seed)
    pred = predict_hierarchy(model, x_te)
    metrics = compute_metrics(y_te, pred, num_classes)

    out = Path(cfg.out_dir) / f"baseline_hierarchical_{cfg.run_name}"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "hierarchical",
        "config": config_path,
        "num_classes": num_classes,
        **metrics.to_dict(),
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    LOG.info("metrics: %s", metrics.to_dict())
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
