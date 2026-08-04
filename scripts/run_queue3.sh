#!/usr/bin/env bash
# Stage 3: finish the rollout-fine-tuning curriculum and add the probabilistic row.
set -u
PY=${PY:-.venv/bin/python}
cd "$(dirname "$0")/.."
run() { echo "=== $1 @ $(date -u +%H:%M:%S) ==="; shift; $PY "$@" 2>&1 | tail -4; }

run unet_ft4 scripts/train.py --config configs/unet.yaml --run-name unet_ft4 \
  --finetune-rollout 4 --max-steps 1200 \
  --init-ckpt artifacts/checkpoints/unet_ft2/best.pt --auto-resume
run unet_ft4_eval scripts/evaluate.py --ckpt artifacts/checkpoints/unet_ft4/best.pt \
  --name unet_ft4
run vit_ft2 scripts/train.py --config configs/vit.yaml --run-name vit_ft2 \
  --finetune-rollout 2 --max-steps 2000 \
  --init-ckpt artifacts/checkpoints/vit/best.pt --auto-resume
run vit_ft2_eval scripts/evaluate.py --ckpt artifacts/checkpoints/vit_ft2/best.pt \
  --name vit_ft2
echo "=== stage 3 complete @ $(date -u +%H:%M:%S) ==="
