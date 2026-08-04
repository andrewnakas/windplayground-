#!/usr/bin/env bash
# Stage 2 queue: the "push it further" work, ordered by value per CPU-hour.
# Everything auto-resumes, and jobs run strictly one at a time -- on 4 cores,
# two concurrent jobs each run more than twice as slow.
set -u
PY=${PY:-.venv/bin/python}
cd "$(dirname "$0")/.."

run() { echo "=== $1 @ $(date -u +%H:%M:%S) ==="; shift; $PY "$@" 2>&1 | tail -4; }

# 1. GraphCast corrector: the primary route to beating a frontier model on
#    wind at our evaluated resolution. Trains on GraphCast's 2018 forecasts.
run corrector scripts/train_corrector.py --competitor graphcast --steps 6000 \
  --wind-weight 2.0 --time-budget-hours 1.0
run corrector_eval scripts/evaluate.py --competitor graphcast_2020 \
  --corrector-ckpt artifacts/checkpoints/graphcast_corrector/best.pt \
  --name graphcast_corrected

# 2. Multi-model blend (numpy least squares, no training).
run blend scripts/blend.py --members graphcast pangu hres

# 3. Rollout fine-tuning of the best from-scratch model.
run unet_ft2 scripts/train.py --config configs/unet.yaml --run-name unet_ft2 \
  --finetune-rollout 2 --max-steps 2500 \
  --init-ckpt artifacts/checkpoints/unet/best.pt --auto-resume
run unet_ft2_eval scripts/evaluate.py --ckpt artifacts/checkpoints/unet_ft2/best.pt \
  --name unet_ft2

echo "=== stage 2 complete @ $(date -u +%H:%M:%S) ==="
