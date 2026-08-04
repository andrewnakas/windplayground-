#!/usr/bin/env bash
# Stage 4: fairness of the architecture comparison.
# The U-Net got ~9500 steps while AFNO got 5000 and the GNN 3000, and only
# U-Net/ViT received the rollout curriculum. This evens that out.
set -u
PY=${PY:-.venv/bin/python}
cd "$(dirname "$0")/.."
run() { echo "=== $1 @ $(date -u +%H:%M:%S) ==="; shift; $PY "$@" 2>&1 | tail -4; }

# AFNO rollout fine-tune, completing the K=2 curriculum across all families
run afno_ft2 scripts/train.py --config configs/afno.yaml --run-name afno_ft2 \
  --finetune-rollout 2 --max-steps 2000 \
  --init-ckpt artifacts/checkpoints/afno/best.pt --auto-resume
run afno_ft2_eval scripts/evaluate.py --ckpt artifacts/checkpoints/afno_ft2/best.pt \
  --name afno_ft2

# Continue the GNN to 6000 steps (warm cosine restart from step 3000).
run graph_long scripts/train.py --config configs/graph.yaml --max-steps 6000 --auto-resume
run graph_long_eval scripts/evaluate.py --ckpt artifacts/checkpoints/graph/best.pt \
  --name graph_6k

echo "=== stage 4 complete @ $(date -u +%H:%M:%S) ==="
