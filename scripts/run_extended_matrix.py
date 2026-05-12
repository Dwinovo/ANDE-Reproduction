"""Run extended exploratory experiments.

Phase A defaults are intentionally small:
    8100-byte sessions, 14-class task, seed 42, a few sequence/adaptive models,
    clean evaluation plus medium-strength evasion probes.

Examples:
    PYTHONPATH=src python scripts/run_extended_matrix.py
    PYTHONPATH=src python scripts/run_extended_matrix.py --models ande byte_tcn segment_attention_fusion
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ande.attacks import AttackSpec, PerturbedDataset
from ande.baselines.cnn1d import CNN1D
from ande.baselines.plain_resnet import PlainResNet18
from ande.data.dataset import ANDEDataset, load_joined_manifest, stratified_split
from ande.metrics import Metrics, compute_metrics
from ande.models import ANDE
from ande.models.byte_sequence import ByteSegmentAttention, ByteTCN
from ande.utils.seed import seed_all

LOG = logging.getLogger(__name__)


def _build_model(name: str, size: int, num_classes: int, segment_len: int) -> nn.Module:
    if name == "ande":
        return ANDE(num_classes=num_classes, use_se=True)
    if name == "ande_no_se":
        return ANDE(num_classes=num_classes, use_se=False)
    if name == "resnet18":
        return PlainResNet18(num_classes=num_classes)
    if name == "cnn1d":
        return CNN1D(in_length=size, num_classes=num_classes)
    if name == "byte_tcn":
        return ByteTCN(num_classes=num_classes, use_stats=False)
    if name == "byte_tcn_fusion":
        return ByteTCN(num_classes=num_classes, use_stats=True)
    if name == "segment_attention":
        return ByteSegmentAttention(
            num_classes=num_classes,
            max_length=size,
            segment_len=segment_len,
            use_stats=False,
        )
    if name == "segment_attention_fusion":
        return ByteSegmentAttention(
            num_classes=num_classes,
            max_length=size,
            segment_len=segment_len,
            use_stats=True,
        )
    raise ValueError(f"unknown model {name!r}")


def _attack_from_token(token: str) -> AttackSpec:
    if ":" in token:
        name, level = token.split(":", 1)
    else:
        name, level = token, "medium"
    return AttackSpec(name=name, level=level)


def _loader(ds, batch_size: int, workers: int, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> tuple[Metrics, list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for image, stat, label in tqdm(loader, leave=False, desc="eval"):
        image, stat = image.to(device), stat.to(device)
        logits = model(image, stat)
        y_true.extend(label.numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    return compute_metrics(y_true_arr, y_pred_arr, num_classes), y_true, y_pred


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    epochs: int,
    lr: float,
    patience_limit: int,
) -> tuple[nn.Module, list[dict]]:
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    best_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for image, stat, label in tqdm(train_loader, leave=False, desc=f"train ep{epoch}"):
            image, stat, label = image.to(device), stat.to(device), label.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(image, stat), label)
            loss.backward()
            opt.step()
            running += loss.item() * label.size(0)
            n += label.size(0)
        scheduler.step()

        metrics, _, _ = evaluate_model(model, val_loader, device, num_classes)
        row = {
            "epoch": epoch,
            "train_loss": running / max(n, 1),
            **metrics.to_dict(),
            "lr": opt.param_groups[0]["lr"],
        }
        history.append(row)
        LOG.info("ep=%d train=%.4f acc=%.4f f1=%.4f", epoch, row["train_loss"], metrics.accuracy, metrics.f1)

        if metrics.accuracy > best_acc:
            best_acc = metrics.accuracy
            patience = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= patience_limit:
                LOG.info("early-stop at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def attention_summary(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    segment_len: int,
    max_batches: int = 8,
) -> dict | None:
    if not hasattr(model, "last_attention"):
        return None
    model.eval()
    chunks: list[torch.Tensor] = []
    for i, (image, stat, _label) in enumerate(loader):
        if i >= max_batches:
            break
        _ = model(image.to(device), stat.to(device))
        attn = getattr(model, "last_attention", None)
        if attn is not None:
            chunks.append(attn.detach().cpu())
    if not chunks:
        return None
    mean_attn = torch.cat(chunks, dim=0).mean(dim=0).numpy()
    top = mean_attn.argsort()[::-1][:8]
    return {
        "segment_len": segment_len,
        "mean_attention": mean_attn.tolist(),
        "top_segments": [
            {
                "segment": int(idx),
                "byte_start": int(idx * segment_len),
                "byte_end": int((idx + 1) * segment_len),
                "weight": float(mean_attn[idx]),
            }
            for idx in top
        ],
    }


def run_one(args: argparse.Namespace, model_name: str) -> dict:
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, label_col, num_classes = load_joined_manifest(
        args.manifest_raw, args.manifest_stats, args.size, args.task
    )
    split = stratified_split(df, label_col, args.split_ratio, args.seed, args.split_at)
    data_root = Path(args.manifest_raw).parent
    train_ds = ANDEDataset(split.train, data_root, args.size, label_col)
    test_ds = ANDEDataset(split.test, data_root, args.size, label_col)
    train_loader = _loader(train_ds, args.batch_size, args.workers, True, device)
    clean_loader = _loader(test_ds, args.batch_size, args.workers, False, device)

    model = _build_model(model_name, args.size, num_classes, args.segment_len).to(device)
    out_dir = Path(args.out_dir) / f"extended_{model_name}_{args.size}_{args.task}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, history = train_model(
        model,
        train_loader,
        clean_loader,
        device,
        num_classes,
        args.epochs,
        args.lr,
        args.patience,
    )
    train_s = time.time() - t0
    torch.save(model.state_dict(), out_dir / "best.pt")

    evals: dict[str, dict] = {}
    clean_metrics, y_true, y_pred = evaluate_model(model, clean_loader, device, num_classes)
    evals["clean"] = clean_metrics.to_dict()

    for spec in [_attack_from_token(token) for token in args.attacks]:
        attack_ds = PerturbedDataset(test_ds, spec)
        attack_loader = _loader(attack_ds, args.batch_size, args.workers, False, device)
        metrics, _, _ = evaluate_model(model, attack_loader, device, num_classes)
        evals[f"{spec.name}:{spec.level}"] = metrics.to_dict()

    summary = attention_summary(model, clean_loader, device, segment_len=args.segment_len)
    payload = {
        "method": model_name,
        "size": args.size,
        "task": args.task,
        "seed": args.seed,
        "num_classes": num_classes,
        "split": {"train": len(split.train), "test": len(split.test), "split_at": args.split_at},
        "train_seconds": train_s,
        "history": history,
        "evaluations": evals,
        "attention_summary": summary,
        "y_true": y_true,
        "y_pred": y_pred,
        "args": vars(args),
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    LOG.info("%s clean=%s", model_name, evals["clean"])
    return payload


def write_summary_csv(rows: list[dict], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "size",
                "task",
                "seed",
                "condition",
                "accuracy",
                "precision",
                "f1",
                "recall",
                "fpr",
                "train_seconds",
            ],
        )
        writer.writeheader()
        for row in rows:
            for condition, metrics in row["evaluations"].items():
                writer.writerow(
                    {
                        "method": row["method"],
                        "size": row["size"],
                        "task": row["task"],
                        "seed": row["seed"],
                        "condition": condition,
                        **metrics,
                        "train_seconds": row["train_seconds"],
                    }
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["cnn1d", "byte_tcn", "byte_tcn_fusion", "segment_attention_fusion"],
    )
    parser.add_argument("--size", type=int, default=8100)
    parser.add_argument("--task", default="behavior14", choices=["binary2", "behavior14"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--split-at", default="session", choices=["session", "pcap"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--segment-len", type=int, default=128)
    parser.add_argument("--manifest-raw", default="data/manifest_raw.parquet")
    parser.add_argument("--manifest-stats", default="data/manifest_stats.parquet")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--summary-csv", default="docs/results/extended_phaseA.csv")
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=[
            "random_padding:medium",
            "random_delay:medium",
            "traffic_shaping:medium",
            "combined:medium",
        ],
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    rows = [run_one(args, model_name) for model_name in args.models]
    write_summary_csv(rows, Path(args.summary_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
