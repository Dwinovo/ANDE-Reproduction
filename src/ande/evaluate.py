"""Standalone evaluation: load a checkpoint, run on the held-out test set, dump
metrics and a confusion-matrix plot.

Usage::

    uv run python -m ande.evaluate --config configs/ande_8100_14cls.yaml --ckpt checkpoints/<run>/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from ande.data.dataset import ANDEDataset, load_joined_manifest, stratified_split
from ande.metrics import compute_metrics
from ande.models import ANDE
from ande.utils.config import load_config
from ande.utils.seed import seed_all

LOG = logging.getLogger(__name__)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    plt.figure(figsize=(max(6, num_classes * 0.6), max(5, num_classes * 0.5)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title(f"Confusion matrix ({num_classes} classes)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run(config_path: str, ckpt_path: str) -> dict:
    cfg = load_config(config_path)
    seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, label_col, num_classes = load_joined_manifest(
        cfg.data.manifest_raw, cfg.data.manifest_stats, cfg.data.size, cfg.data.task
    )
    split = stratified_split(df, label_col, cfg.data.split_ratio, cfg.seed, cfg.data.split_at)
    data_root = Path(cfg.data.manifest_raw).parent
    test_ds = ANDEDataset(split.test, data_root, cfg.data.size, label_col)
    loader = DataLoader(test_ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=2)

    model = ANDE(
        num_classes=num_classes,
        use_se=cfg.model.use_se,
        se_reduction=cfg.model.se_reduction,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for image, stat, label in loader:
            image, stat = image.to(device), stat.to(device)
            logits = model(image, stat)
            y_true.extend(label.numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    metrics = compute_metrics(y_true_arr, y_pred_arr, num_classes)

    out = Path(cfg.out_dir) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    plot_confusion(y_true_arr, y_pred_arr, num_classes, out / "confusion_matrix.png")

    payload = {
        "checkpoint": ckpt_path,
        "num_classes": num_classes,
        **metrics.to_dict(),
    }
    (out / "eval.json").write_text(json.dumps(payload, indent=2))
    LOG.info("metrics: %s", metrics.to_dict())
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    run(args.config, args.ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
