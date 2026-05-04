#!/usr/bin/env bash
# Reproduce the CIFAR-100-C row of the main table.
# Expected mean online accuracy over 15 corruptions at severity 5: ≈ 48.55%.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-outputs/cama_c100}"
mkdir -p "$OUT"

python main_cama.py \
    --dataset cifar100_c \
    --phase main \
    --wd 0.0 \
    --output-dir "$OUT" \
    --alpha 0.05 \
    --beta 0.1 \
    --cfg cfgs/cifar100_c/ours.yaml DATA_DIR ./data SAVE_DIR "$OUT"
