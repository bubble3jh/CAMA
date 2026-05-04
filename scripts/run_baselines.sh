#!/usr/bin/env bash
# Run all in-tree baselines (ZS, TENT, RoTTA, RPL, SAR, TPT, VTE, BATCLIP) on a given dataset.
# Usage: bash scripts/run_baselines.sh {cifar10_c | cifar100_c | imagenet_c}
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:-cifar10_c}"
case "$DATASET" in
    cifar10_c|cifar100_c|imagenet_c) ;;
    *) echo "Unknown dataset: $DATASET" >&2; exit 1 ;;
esac

OUT_ROOT="${OUT_ROOT:-outputs/baselines/${DATASET}}"
mkdir -p "$OUT_ROOT"

# (cfg-name, output-subdir) — one entry per paper-table baseline.
BASELINES=(
    "source:zs"
    "tent:tent"
    "rotta:rotta"
    "rpl:rpl"
    "sar:sar"
    "tpt:tpt"
    "vte:vte"
    "ours:batclip"
)

for entry in "${BASELINES[@]}"; do
    name="${entry%%:*}"
    out="${entry##*:}"
    cfg="cfgs/${DATASET}/${name}.yaml"
    if [[ ! -f "$cfg" ]]; then
        echo "[skip] $cfg not found"
        continue
    fi
    save="${OUT_ROOT}/${out}"
    mkdir -p "$save"
    echo "[run] ${name} → ${save}"
    python test_time.py --cfg "$cfg" DATA_DIR ./data SAVE_DIR "$save"
done
