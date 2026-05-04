#!/usr/bin/env bash
# Reproduce the CIFAR-10-C row of the main table.
# Expected mean online accuracy over 15 corruptions at severity 5: ≈ 79.49%.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-outputs/cama_c10}"
mkdir -p "$OUT"

python main_cama.py \
    --dataset cifar10_c \
    --phase main \
    --output-dir "$OUT" \
    --alpha 0.05 \
    --beta 0.1 \
    --cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data SAVE_DIR "$OUT"
