"""Aggregate every outputs/<run>/results.json into Table V / Table VI markdown.

Each results.json should contain at least:
    method, num_classes, accuracy, precision, f1, recall, fpr, config_dump.data.size,
    config_dump.data.task, seed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

METRIC_COLS = ["accuracy", "precision", "f1", "recall", "fpr"]


def collect(out_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for results in out_dir.glob("**/results.json"):
        try:
            data = json.loads(results.read_text())
        except json.JSONDecodeError:
            continue
        cfg = data.get("config_dump", {})
        size = cfg.get("data", {}).get("size")
        task = cfg.get("data", {}).get("task")
        method = data.get("method") or cfg.get("model", {}).get("name", "ande")
        if cfg.get("model", {}).get("name") == "ande" and not cfg.get("model", {}).get("use_se", True):
            method = "ande_no_se"
        rows.append(
            {
                "method": method,
                "size": size,
                "task": task,
                "seed": data.get("config_dump", {}).get("seed", 0),
                **{k: data.get(k) for k in METRIC_COLS},
            }
        )
    return pd.DataFrame(rows)


def render_table(df: pd.DataFrame, size: int, task: str) -> str:
    sub = df[(df["size"] == size) & (df["task"] == task)]
    if sub.empty:
        return f"_no runs found for size={size} task={task}_\n"
    grouped = sub.groupby("method")[METRIC_COLS].agg(["mean", "std"]).fillna(0)
    out = [f"### size={size}  task={task}\n"]
    header = "| method | " + " | ".join(METRIC_COLS) + " |"
    sep = "|" + "|".join(["---"] * (len(METRIC_COLS) + 1)) + "|"
    out += [header, sep]
    for method, row in grouped.iterrows():
        cells = [method]
        for col in METRIC_COLS:
            cells.append(f"{row[(col, 'mean')]:.4f} ± {row[(col, 'std')]:.4f}")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--target", type=Path, default=Path("docs/results"))
    args = parser.parse_args(argv)

    df = collect(args.out_dir)
    if df.empty:
        print(f"no results.json found under {args.out_dir}")
        return 1
    args.target.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.target / "results_long.csv", index=False)

    sections: dict[str, list[str]] = defaultdict(list)
    for size in sorted(df["size"].dropna().unique()):
        for task in sorted(df["task"].dropna().unique()):
            sections[task].append(render_table(df, int(size), task))

    for task, blocks in sections.items():
        body = f"# {task}\n\n" + "\n".join(blocks)
        (args.target / f"table_{task}.md").write_text(body, encoding="utf-8")

    print(f"wrote {args.target}/results_long.csv and {len(sections)} table_*.md files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
