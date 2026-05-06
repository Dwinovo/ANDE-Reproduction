"""Classical ML baselines: DT, RF, XGB on the 26-d statistical features.

These are run from the same joined manifest used by ANDE so the train/test
split matches exactly. Image columns are simply ignored.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from ande.data.dataset import load_joined_manifest, stratified_split
from ande.data.preprocess_stats import FEATURE_ORDER
from ande.metrics import compute_metrics
from ande.utils.config import load_config
from ande.utils.seed import seed_all

LOG = logging.getLogger(__name__)


def _xy(df: pd.DataFrame, label_col: str) -> tuple[np.ndarray, np.ndarray]:
    return df[list(FEATURE_ORDER)].to_numpy(dtype=np.float32), df[label_col].to_numpy(dtype=np.int64)


def _build(name: str, num_classes: int, seed: int):
    if name == "dt":
        return DecisionTreeClassifier(random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
    if name == "xgb":
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob" if num_classes > 2 else "binary:logistic",
            num_class=num_classes if num_classes > 2 else None,
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            eval_metric="mlogloss" if num_classes > 2 else "logloss",
        )
    raise ValueError(name)


def run(config_path: str, model_name: str) -> dict:
    cfg = load_config(config_path)
    seed_all(cfg.seed)

    df, label_col, num_classes = load_joined_manifest(
        cfg.data.manifest_raw, cfg.data.manifest_stats, cfg.data.size, cfg.data.task
    )
    split = stratified_split(df, label_col, cfg.data.split_ratio, cfg.seed)
    x_tr, y_tr = _xy(split.train, label_col)
    x_te, y_te = _xy(split.test, label_col)

    clf = _build(model_name, num_classes, cfg.seed)
    LOG.info("fit %s on %d samples", model_name, len(x_tr))
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    metrics = compute_metrics(y_te, pred, num_classes)

    out = Path(cfg.out_dir) / f"baseline_{model_name}_{cfg.run_name}"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": model_name,
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
    parser.add_argument("--model", choices=["dt", "rf", "xgb"], required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    run(args.config, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
