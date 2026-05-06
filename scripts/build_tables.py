"""Aggregate every outputs/<run>/results.json into Table V / Table VI markdown.

Each results.json should contain at least:
    method, num_classes, accuracy, precision, f1, recall, fpr, config_dump.data.size,
    config_dump.data.task, seed.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

METRIC_COLS = ["accuracy", "precision", "f1", "recall", "fpr"]


_RUN_NAME_RE = re.compile(
    r"^(?P<method>[a-z0-9_]+?)_(?P<size>784|4096|8100)_(?P<task>binary2|behavior14)(?:_seed(?P<seed>\d+))?$"
)


def _fallback_from_run_name(run_name: str) -> dict | None:
    """Parse method/size/task/seed from run_name like 'ande_nose_8100_behavior14_seed42'."""
    if not run_name:
        return None
    m = _RUN_NAME_RE.match(run_name)
    if not m:
        return None
    return {
        "method": m.group("method"),
        "size": int(m.group("size")),
        "task": m.group("task"),
        "seed": int(m.group("seed") or 0),
    }


def _fallback_from_config_path(config_path: str | None) -> dict | None:
    if not config_path:
        return None
    p = Path(config_path)
    if not p.exists():
        return None
    try:
        cfg = yaml.safe_load(p.read_text())
    except Exception:  # pragma: no cover
        return None
    data = cfg.get("data", {}) if cfg else {}
    return {
        "method": cfg.get("model", {}).get("name", "ande"),
        "size": data.get("size"),
        "task": data.get("task"),
        "seed": cfg.get("seed", 0),
    }


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
        seed = cfg.get("seed", 0)

        # ANDE without SE distinguishing
        if cfg.get("model", {}).get("name") == "ande" and not cfg.get("model", {}).get("use_se", True):
            method = "ande_no_se"

        # Fallbacks for results.json that lack config_dump (older ML baseline writes).
        if size is None or task is None:
            run_name = results.parent.name
            # baseline outputs are nested under 'baseline_<model>_<run_name>'
            for prefix in ("baseline_dt_", "baseline_rf_", "baseline_xgb_",
                           "baseline_cnn1d_", "baseline_resnet18_", "baseline_hierarchical_"):
                if run_name.startswith(prefix):
                    run_name = run_name[len(prefix):]
                    break
            fb = _fallback_from_run_name(run_name) or _fallback_from_config_path(data.get("config"))
            if fb:
                size = size if size is not None else fb["size"]
                task = task if task is not None else fb["task"]
                seed = seed or fb["seed"]
                if method in ("ande", None):
                    method = fb["method"]

        # Normalise method names: ande_nose -> ande_no_se to match config-dump output
        if method == "ande_nose":
            method = "ande_no_se"

        rows.append(
            {
                "method": method,
                "size": size,
                "task": task,
                "seed": seed,
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
