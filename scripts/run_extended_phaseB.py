"""Run the scaled Phase B extended matrix.

Default matrix:
    models: ande, byte_tcn, byte_tcn_fusion
    sizes : 784, 4096, 8100
    seeds : 42, 43, 44
    task  : behavior14

The script is resumable: if an output results.json already exists it is
skipped, then all matching JSON files are collected into one CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def output_dir(out_root: Path, model: str, size: int, task: str, seed: int) -> Path:
    return out_root / f"extended_{model}_{size}_{task}_seed{seed}"


def run_one(args: argparse.Namespace, model: str, size: int, seed: int) -> tuple[str, float, int]:
    out = output_dir(Path(args.out_dir), model, size, args.task, seed)
    if (out / "results.json").exists():
        print(f"SKIP {model} size={size} seed={seed}", flush=True)
        return f"{model}_{size}_seed{seed}", 0.0, 0

    cmd = [
        sys.executable,
        "scripts/run_extended_matrix.py",
        "--models",
        model,
        "--size",
        str(size),
        "--task",
        args.task,
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--out-dir",
        args.out_dir,
        "--summary-csv",
        str(Path(args.target).with_name("_extended_phaseB_last.csv")),
        "--attacks",
        *args.attacks,
    ]
    print(f"RUN  {model} size={size} seed={seed}", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONPATH": args.pythonpath}
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=args.repo, env=env)
    dt = time.time() - t0
    print(f"DONE {model} size={size} seed={seed} rc={proc.returncode} dt={dt:.1f}s", flush=True)
    return f"{model}_{size}_seed{seed}", dt, proc.returncode


def collect(args: argparse.Namespace) -> int:
    rows: list[dict] = []
    out_root = Path(args.out_dir)
    for model in args.models:
        for size in args.sizes:
            for seed in args.seeds:
                results = output_dir(out_root, model, size, args.task, seed) / "results.json"
                if not results.exists():
                    print(f"MISSING {results}", flush=True)
                    continue
                data = json.loads(results.read_text())
                for condition, metrics in data["evaluations"].items():
                    rows.append(
                        {
                            "method": data["method"],
                            "size": data["size"],
                            "task": data["task"],
                            "seed": data["seed"],
                            "condition": condition,
                            **metrics,
                            "train_seconds": data["train_seconds"],
                        }
                    )

    target = Path(args.target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
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
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {target} rows={len(rows)}", flush=True)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--models", nargs="+", default=["ande", "byte_tcn", "byte_tcn_fusion"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[784, 4096, 8100])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--task", default="behavior14", choices=["binary2", "behavior14"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--target", default="docs/results/extended_phaseB.csv")
    parser.add_argument("--pythonpath", default="src")
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
    args = parser.parse_args(argv)

    summary: list[dict] = []
    for size in args.sizes:
        for seed in args.seeds:
            for model in args.models:
                run_name, seconds, rc = run_one(args, model, size, seed)
                summary.append({"run": run_name, "seconds": seconds, "returncode": rc})
                if rc != 0:
                    Path(args.target).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
                    return rc
    collect(args)
    Path(args.target).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
