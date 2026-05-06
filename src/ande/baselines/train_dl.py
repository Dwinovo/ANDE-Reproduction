"""Train deep-learning baselines (CNN-1D / plain ResNet-18) using the same
data pipeline + training loop as ANDE. The model takes ``(image, stat)`` but
the stat tensor is ignored, so the dataset can be shared.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ande.baselines.cnn1d import CNN1D
from ande.baselines.plain_resnet import PlainResNet18
from ande.data.dataset import ANDEDataset, load_joined_manifest, stratified_split
from ande.metrics import compute_metrics
from ande.utils.config import load_config
from ande.utils.seed import seed_all

LOG = logging.getLogger(__name__)


def _build_model(name: str, size: int, num_classes: int) -> nn.Module:
    if name == "cnn1d":
        return CNN1D(in_length=size, num_classes=num_classes)
    if name == "resnet18":
        return PlainResNet18(num_classes=num_classes)
    raise ValueError(name)


def _train_eval(
    model: nn.Module, train_loader, test_loader, device, epochs: int, lr: float, num_classes: int
):
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

    best_acc = 0.0
    best_state: dict | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        for image, stat, label in train_loader:
            image, stat, label = image.to(device), stat.to(device), label.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(image, stat), label)
            loss.backward()
            opt.step()
        scheduler.step()

        model.eval()
        y_t: list[int] = []
        y_p: list[int] = []
        with torch.no_grad():
            for image, stat, label in test_loader:
                image, stat = image.to(device), stat.to(device)
                logits = model(image, stat)
                y_t.extend(label.numpy().tolist())
                y_p.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        m = compute_metrics(np.array(y_t), np.array(y_p), num_classes)
        LOG.info("ep=%d acc=%.4f f1=%.4f", epoch, m.accuracy, m.f1)
        if m.accuracy > best_acc:
            best_acc = m.accuracy
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    y_t = []
    y_p = []
    with torch.no_grad():
        for image, stat, label in test_loader:
            image, stat = image.to(device), stat.to(device)
            logits = model(image, stat)
            y_t.extend(label.numpy().tolist())
            y_p.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    return compute_metrics(np.array(y_t), np.array(y_p), num_classes), y_t, y_p


def run(config_path: str, model_name: str) -> dict:
    cfg = load_config(config_path)
    seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, label_col, num_classes = load_joined_manifest(
        cfg.data.manifest_raw, cfg.data.manifest_stats, cfg.data.size, cfg.data.task
    )
    split = stratified_split(df, label_col, cfg.data.split_ratio, cfg.seed, cfg.data.split_at)
    data_root = Path(cfg.data.manifest_raw).parent
    train_ds = ANDEDataset(split.train, data_root, cfg.data.size, label_col)
    test_ds = ANDEDataset(split.test, data_root, cfg.data.size, label_col)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers
    )
    model = _build_model(model_name, cfg.data.size, num_classes).to(device)
    metrics, y_t, y_p = _train_eval(
        model, train_loader, test_loader, device, cfg.train.epochs, cfg.train.lr, num_classes
    )

    out = Path(cfg.out_dir) / f"baseline_{model_name}_{cfg.run_name}"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": model_name,
        "config": config_path,
        "num_classes": num_classes,
        **metrics.to_dict(),
        "config_dump": asdict(cfg),
        "y_true": y_t,
        "y_pred": y_p,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    LOG.info("metrics: %s", metrics.to_dict())
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=["cnn1d", "resnet18"], required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    run(args.config, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
