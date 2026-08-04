#!/usr/bin/env bash
# Overnight program. Goal: meet ambition tier 1 -- beat the published 5.625 deg
# anchor (Rasp & Thuerey 2021 ResNet, z500 RMSE 268/499 at 3/5 days). Our
# best so far is 447/675, and the gap is training budget, not architecture.
#
# Every stage is --auto-resume, so re-running this script after any
# interruption continues each stage exactly where it stopped. Stages run
# strictly one at a time: on 4 cores, two concurrent jobs each run >2x slower.
set -u
PY=${PY:-.venv/bin/python}
cd "$(dirname "$0")/.."
run() { echo "=== $1 @ $(date -u +%H:%M:%S) ==="; shift; $PY "$@" 2>&1 | tail -4; }

# 1. Long run of the best family (~4 h budget), then the rollout curriculum on
#    top of it -- fine-tuning was worth 15-17% on the short run.
run unet_long      scripts/train.py --config configs/unet_long.yaml --auto-resume
run unet_long_eval scripts/evaluate.py --ckpt artifacts/checkpoints/unet_long/best.pt \
  --name unet_long
run unet_long_ft2  scripts/train.py --config configs/unet_long.yaml \
  --run-name unet_long_ft2 --finetune-rollout 2 --max-steps 4000 \
  --init-ckpt artifacts/checkpoints/unet_long/best.pt --auto-resume
run unet_long_ft2_eval scripts/evaluate.py \
  --ckpt artifacts/checkpoints/unet_long_ft2/best.pt --name unet_long_ft2
run unet_long_ft4  scripts/train.py --config configs/unet_long.yaml \
  --run-name unet_long_ft4 --finetune-rollout 4 --max-steps 2000 \
  --init-ckpt artifacts/checkpoints/unet_long_ft2/best.pt --auto-resume
run unet_long_ft4_eval scripts/evaluate.py \
  --ckpt artifacts/checkpoints/unet_long_ft4/best.pt --name unet_long_ft4

# 2. The open scientific question from finding #2: does the transformer
#    overtake the CNN once it is no longer starved of compute?
run vit_long       scripts/train.py --config configs/vit_long.yaml --auto-resume
run vit_long_eval  scripts/evaluate.py --ckpt artifacts/checkpoints/vit_long/best.pt \
  --name vit_long

# 3. Capacity vs budget: does a 4x wider U-Net help, or is length what binds?
run unet_wide      scripts/train.py --config configs/unet_wide.yaml --auto-resume
run unet_wide_eval scripts/evaluate.py --ckpt artifacts/checkpoints/unet_wide/best.pt \
  --name unet_wide

# 4. Refresh the report with everything that landed.
run report scripts/make_report.py

echo "=== overnight complete @ $(date -u +%H:%M:%S) ==="
