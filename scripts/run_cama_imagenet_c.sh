#!/usr/bin/env bash
# Reproduce the ImageNet-C row of the main table.
# Expected mean online accuracy over 15 corruptions at severity 5: ≈ 36.26%.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-outputs/cama_inc}"
mkdir -p "$OUT"

python main_cama.py \
    --dataset imagenet_c \
    --output-dir "$OUT" \
    --alpha 0.05 \
    --beta 0.1 \
    --cfg cfgs/imagenet_c/ours.yaml DATA_DIR ./data SAVE_DIR "$OUT"
