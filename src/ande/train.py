"""End-to-end training entry point for the ANDE model.

Usage::

    uv run python -m ande.train --config configs/ande_8100_14cls.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ande.data.dataset import ANDEDataset, load_joined_manifest, stratified_split
from ande.metrics import compute_metrics
from ande.models import ANDE
from ande.utils.config import Config, load_config
from ande.utils.seed import seed_all

LOG = logging.getLogger(__name__)


def _build_optimizer(params, cfg: Config) -> optim.Optimizer:
    if cfg.train.optimizer == "adam":
        return optim.Adam(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    if cfg.train.optimizer == "sgd":
        return optim.SGD(params, lr=cfg.train.lr, momentum=0.9, weight_decay=cfg.train.weight_decay)
    raise ValueError(cfg.train.optimizer)


def _build_scheduler(opt: optim.Optimizer, cfg: Config):
    if cfg.train.scheduler == "step":
        return optim.lr_scheduler.StepLR(opt, step_size=cfg.train.step_size, gamma=cfg.train.gamma)
    if cfg.train.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.epochs)
    return None


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running = 0.0
    n = 0
    for image, stat, label in tqdm(loader, leave=False, desc="train"):
        image, stat, label = image.to(device), stat.to(device), label.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(image, stat)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        running += loss.item() * label.size(0)
        n += label.size(0)
    return running / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
):
    model.eval()
    running = 0.0
    n = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for image, stat, label in tqdm(loader, leave=False, desc="eval"):
        image, stat, label = image.to(device), stat.to(device), label.to(device)
        logits = model(image, stat)
        loss = criterion(logits, label)
        running += loss.item() * label.size(0)
        n += label.size(0)
        y_true.extend(label.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    import numpy as np

    metrics = compute_metrics(np.array(y_true), np.array(y_pred), num_classes)
    return running / max(n, 1), metrics, y_true, y_pred


def run(config_path: str) -> dict:
    cfg = load_config(config_path)
    seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("device=%s", device)

    df, label_col, num_classes = load_joined_manifest(
        cfg.data.manifest_raw, cfg.data.manifest_stats, cfg.data.size, cfg.data.task
    )
    split = stratified_split(df, label_col, cfg.data.split_ratio, cfg.seed, cfg.data.split_at)
    LOG.info("split: train=%d test=%d classes=%d", len(split.train), len(split.test), num_classes)

    data_root = Path(cfg.data.manifest_raw).parent
    train_ds = ANDEDataset(split.train, data_root, cfg.data.size, label_col)
    test_ds = ANDEDataset(split.test, data_root, cfg.data.size, label_col)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = ANDE(
        num_classes=num_classes,
        use_se=cfg.model.use_se,
        se_reduction=cfg.model.se_reduction,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model.parameters(), cfg)
    scheduler = _build_scheduler(optimizer, cfg)

    out = Path(cfg.out_dir) / cfg.run_name
    ckpt = Path(cfg.checkpoints_dir) / cfg.run_name
    runs = Path(cfg.runs_dir) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(runs)

    best_acc = 0.0
    patience = 0
    history: list[dict] = []
    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, _, _ = evaluate(
            model, test_loader, criterion, device, num_classes
        )
        if scheduler is not None:
            scheduler.step()
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        for k, v in val_metrics.to_dict().items():
            writer.add_scalar(f"val/{k}", v, epoch)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics.to_dict(),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        LOG.info("ep=%d train=%.4f val=%.4f acc=%.4f f1=%.4f",
                 epoch, train_loss, val_loss, val_metrics.accuracy, val_metrics.f1)

        if val_metrics.accuracy > best_acc:
            best_acc = val_metrics.accuracy
            patience = 0
            torch.save(model.state_dict(), ckpt / "best.pt")
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                LOG.info("early-stop at epoch %d", epoch)
                break

    # Reload best weights for final reporting
    model.load_state_dict(torch.load(ckpt / "best.pt", map_location=device))
    final_loss, final_metrics, y_true, y_pred = evaluate(
        model, test_loader, criterion, device, num_classes
    )
    final = {
        "config": config_path,
        "run_name": cfg.run_name,
        "num_classes": num_classes,
        "test_loss": final_loss,
        **final_metrics.to_dict(),
        "history": history,
        "y_true": y_true,
        "y_pred": y_pred,
        "config_dump": asdict(cfg),
    }
    (out / "results.json").write_text(json.dumps(final, indent=2))
    writer.close()
    return final


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
