"""Generate the four figures used in docs/reproduction_report.md.

  1. training_curves.png    -- loss + val accuracy + lr schedule across epochs
  2. confusion_matrix.png   -- 14x14 confusion matrix on the held-out test set
  3. per_class_metrics.png  -- bar chart of precision / recall / f1 per class
  4. sample_images.png      -- one 90x90 raw-byte image per 14-class label

All figures read from outputs/ande_8100_14cls/results.json + manifest_raw.parquet.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
# Use the 8100 / 14-class ANDE run from the matrix as the canonical "main" run.
# Falls back to the older single-run path if the matrix output is absent.
_CANDIDATES = [
    REPO_ROOT / "outputs" / "ande_8100_behavior14_seed42" / "results.json",
    REPO_ROOT / "outputs" / "ande_8100_14cls" / "results.json",
]
RESULTS = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])
MANIFEST = REPO_ROOT / "data" / "manifest_raw.parquet"
DATA_ROOT = REPO_ROOT / "data"
OUT = REPO_ROOT / "docs" / "figures"

# Order: activity_idx * 2 + is_tor  (NonTor=0, Tor=1)
CLASS_NAMES = [
    "browsing-NonTor", "browsing-Tor",
    "chat-NonTor", "chat-Tor",
    "email-NonTor", "email-Tor",
    "ft-NonTor", "ft-Tor",
    "p2p-NonTor", "p2p-Tor",
    "streaming-NonTor", "streaming-Tor",
    "voip-NonTor", "voip-Tor",
]


def _load_results() -> dict:
    return json.loads(RESULTS.read_text())


def fig_training_curves() -> Path:
    data = _load_results()
    hist = pd.DataFrame(data["history"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(hist["epoch"], hist["train_loss"], label="train", linewidth=2)
    ax.plot(hist["epoch"], hist["val_loss"], label="val", linewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training / validation loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(hist["epoch"], hist["accuracy"], label="accuracy", color="tab:green", linewidth=2)
    ax.plot(hist["epoch"], hist["f1"], label="F1", color="tab:purple", linewidth=2, linestyle="--")
    ax.axhline(0.9820, color="tab:red", linewidth=1, linestyle=":", label="paper Acc 0.9820")
    best_ep = int(hist.iloc[hist["accuracy"].idxmax()]["epoch"])
    best_acc = hist["accuracy"].max()
    ax.scatter([best_ep], [best_acc], color="tab:red", zorder=5, s=80,
               label=f"best ep={best_ep} acc={best_acc:.4f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("metric")
    ax.set_title("Validation accuracy / F1")
    ax.set_ylim(0.96, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.suptitle(
        f"ANDE 8100B / 14-class on RTX 5090, pcap-level split "
        f"(early-stopped at ep {len(hist)})",
        y=1.02,
    )
    fig.tight_layout()
    out = OUT / "training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_confusion_matrix() -> Path:
    data = _load_results()
    y_true = np.array(data["y_true"])
    y_pred = np.array(data["y_pred"])
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=False,
        ax=axes[0],
        annot_kws={"size": 9},
    )
    axes[0].set_xlabel("predicted")
    axes[0].set_ylabel("true")
    axes[0].set_title(f"Confusion matrix (raw counts, n={len(y_true):,})")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=True,
        ax=axes[1],
        annot_kws={"size": 8},
        vmin=0,
        vmax=1,
    )
    axes[1].set_xlabel("predicted")
    axes[1].set_ylabel("true")
    axes[1].set_title("Confusion matrix (row-normalised)")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    fig.tight_layout()
    out = OUT / "confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_per_class_metrics() -> Path:
    data = _load_results()
    y_true = np.array(data["y_true"])
    y_pred = np.array(data["y_pred"])
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )

    df = pd.DataFrame({
        "class": CLASS_NAMES,
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": support,
    })

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    x = np.arange(len(CLASS_NAMES))
    width = 0.27
    axes[0].bar(x - width, p, width, label="precision", color="tab:blue")
    axes[0].bar(x, r, width, label="recall", color="tab:green")
    axes[0].bar(x + width, f1, width, label="F1", color="tab:purple")
    axes[0].axhline(1.0, color="black", linewidth=0.5, alpha=0.3)
    axes[0].set_ylabel("score")
    axes[0].set_ylim(0.5, 1.02)
    axes[0].set_title("Per-class precision / recall / F1 (test set)")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(loc="lower right")
    for i, v in enumerate(f1):
        axes[0].text(x[i] + width, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)

    axes[1].bar(x, support, color="tab:gray")
    axes[1].set_ylabel("test samples")
    axes[1].set_yscale("log")
    axes[1].grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(support):
        axes[1].text(x[i], v * 1.15, f"{int(v)}", ha="center", fontsize=8)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASS_NAMES, rotation=45, ha="right")

    fig.tight_layout()
    out = OUT / "per_class_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_matrix_overview() -> Path:
    """Bar chart: accuracy by (method, size) for both tasks."""
    csv = REPO_ROOT / "docs" / "results" / "results_long.csv"
    if not csv.exists():
        raise FileNotFoundError(csv)
    df = pd.read_csv(csv)
    df = df.dropna(subset=["size", "task", "method"])
    df["size"] = df["size"].astype(int)

    method_order = ["dt", "rf", "xgb", "cnn1d", "resnet18", "ande_no_se", "ande"]
    method_color = {
        "dt": "#9467bd", "rf": "#8c564b", "xgb": "#e377c2",
        "cnn1d": "#1f77b4", "resnet18": "#17becf",
        "ande_no_se": "#ff7f0e", "ande": "#2ca02c",
    }
    sizes = [784, 4096, 8100]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for ax, task, paper_line, ylim in zip(
        axes,
        ["behavior14", "binary2"],
        [0.9820, None],
        [(0.5, 1.02), (0.99, 1.005)],
        strict=True,
    ):
        sub = df[df["task"] == task]
        # average across seeds
        agg = sub.groupby(["method", "size"], as_index=False)["accuracy"].mean()
        x = np.arange(len(method_order))
        width = 0.27
        for i, sz in enumerate(sizes):
            sz_acc = []
            for m in method_order:
                row = agg[(agg["method"] == m) & (agg["size"] == sz)]
                sz_acc.append(row["accuracy"].iloc[0] if not row.empty else np.nan)
            offset = (i - 1) * width
            colors = [method_color[m] for m in method_order]
            ax.bar(
                x + offset, sz_acc, width, label=f"size={sz}",
                color=colors, alpha=0.55 + 0.225 * i,
                edgecolor="black", linewidth=0.4,
            )
            for j, v in enumerate(sz_acc):
                if not np.isnan(v):
                    ax.text(x[j] + offset, v + 0.005, f"{v:.3f}",
                            ha="center", fontsize=6.5, rotation=90)
        if paper_line is not None:
            ax.axhline(paper_line, color="red", linewidth=1, linestyle=":",
                       label=f"paper Acc {paper_line}")
        ax.set_ylim(*ylim)
        ax.set_ylabel("accuracy")
        ax.set_title(f"task = {task}")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(method_order, rotation=0)
        ax.legend(loc="lower left", ncol=4, fontsize=8)
    fig.suptitle(
        "42-experiment matrix: accuracy by method x size x task "
        "(pcap-level split, RTX 5090)",
        y=1.005,
    )
    fig.tight_layout()
    out = OUT / "matrix_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_sample_images() -> Path:
    """One 90x90 raw-byte image per 14-class label."""
    df = pd.read_parquet(MANIFEST)
    fig, axes = plt.subplots(2, 7, figsize=(15, 5))
    rng = np.random.default_rng(0)
    for label_id in range(14):
        sub = df[df["label_14cls"] == label_id]
        if sub.empty:
            ax = axes[label_id // 7, label_id % 7]
            ax.text(0.5, 0.5, "no samples", ha="center", va="center")
            ax.set_title(CLASS_NAMES[label_id], fontsize=9)
            ax.axis("off")
            continue
        # pick a session with reasonable byte coverage (lots of non-zero bytes)
        # to avoid mostly-black tiny captures
        candidates = sub.sample(min(20, len(sub)), random_state=int(label_id))
        best = None
        best_nonzero = -1
        for _, row in candidates.iterrows():
            arr = np.load(DATA_ROOT / row["image_8100"])
            nz = (arr > 0).mean()
            if nz > best_nonzero:
                best_nonzero = nz
                best = arr
        ax = axes[label_id // 7, label_id % 7]
        ax.imshow(best, cmap="gray", vmin=0, vmax=255)
        ax.set_title(CLASS_NAMES[label_id], fontsize=9)
        ax.axis("off")

    fig.suptitle("Raw-byte session images (90x90) - one sample per 14-class label",
                 y=1.02)
    fig.tight_layout()
    out = OUT / "sample_images.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [
        ("training_curves", fig_training_curves),
        ("confusion_matrix", fig_confusion_matrix),
        ("per_class_metrics", fig_per_class_metrics),
        ("sample_images", fig_sample_images),
        ("matrix_overview", fig_matrix_overview),
    ]:
        path = fn()
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
