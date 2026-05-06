#!/usr/bin/env bash
# Run the full reproduction matrix on a Linux GPU server.
#
# Usage:   bash scripts/run_all.sh [SEEDS]
# Default: SEEDS="42"
#
# Pre-requisites:
#   * data/raw/ populated
#   * data/manifest_raw.parquet  (uv run python -m ande.data.preprocess_raw  ...)
#   * data/manifest_stats.parquet (uv run python -m ande.data.preprocess_stats ...)
#
# Each run writes outputs/<run_name>/results.json which build_tables.py later
# aggregates into a Table V/VI markdown.

set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${1:-"42"}
SIZES=(784 4096 8100)
TASKS=(behavior14 binary2)

run_ml () {
    local size=$1 task=$2 model=$3 seed=$4
    local cfg="configs/_runtime_${size}_${task}_${model}_${seed}.yaml"
    cat > "$cfg" <<YAML
run_name: ${model}_${size}_${task}_seed${seed}
seed: ${seed}
data: { size: ${size}, task: ${task}, manifest_raw: data/manifest_raw.parquet, manifest_stats: data/manifest_stats.parquet }
model: { name: ${model} }
train: { epochs: 1 }
YAML
    uv run python -m ande.baselines.ml --config "$cfg" --model "$model"
}

run_dl_baseline () {
    local size=$1 task=$2 model=$3 seed=$4
    local cfg="configs/_runtime_${size}_${task}_${model}_${seed}.yaml"
    cat > "$cfg" <<YAML
run_name: ${model}_${size}_${task}_seed${seed}
seed: ${seed}
data: { size: ${size}, task: ${task}, batch_size: 64, num_workers: 4, manifest_raw: data/manifest_raw.parquet, manifest_stats: data/manifest_stats.parquet }
model: { name: ${model} }
train: { epochs: 50, optimizer: adam, lr: 0.001, scheduler: step, step_size: 10, gamma: 0.5, early_stop_patience: 10 }
YAML
    uv run python -m ande.baselines.train_dl --config "$cfg" --model "$model"
}

run_ande () {
    local size=$1 task=$2 use_se=$3 seed=$4
    local tag=$([ "$use_se" = "true" ] && echo "ande" || echo "ande_nose")
    local cfg="configs/_runtime_${size}_${task}_${tag}_${seed}.yaml"
    cat > "$cfg" <<YAML
run_name: ${tag}_${size}_${task}_seed${seed}
seed: ${seed}
data: { size: ${size}, task: ${task}, batch_size: 64, num_workers: 4, manifest_raw: data/manifest_raw.parquet, manifest_stats: data/manifest_stats.parquet }
model: { name: ande, use_se: ${use_se}, se_reduction: 16 }
train: { epochs: 50, optimizer: adam, lr: 0.001, scheduler: step, step_size: 10, gamma: 0.5, early_stop_patience: 10 }
YAML
    uv run python -m ande.train --config "$cfg"
}

for seed in $SEEDS; do
    for size in "${SIZES[@]}"; do
        for task in "${TASKS[@]}"; do
            for ml in dt rf xgb; do
                run_ml "$size" "$task" "$ml" "$seed"
            done
            run_dl_baseline "$size" "$task" cnn1d "$seed"
            run_dl_baseline "$size" "$task" resnet18 "$seed"
            run_ande "$size" "$task" true "$seed"
            run_ande "$size" "$task" false "$seed"
        done
    done
done

uv run python scripts/build_tables.py --out-dir outputs --target docs/results
