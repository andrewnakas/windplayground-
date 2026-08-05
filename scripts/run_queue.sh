#!/usr/bin/env bash
# Sequential training queue. Every stage uses --auto-resume, so re-running this
# script after an interruption continues each stage where it stopped rather
# than restarting it.
set -u
PY=${PY:-.venv/bin/python}
cd "$(dirname "$0")/.."

run() {
  local name=$1; shift
  echo "=== $name @ $(date -u +%H:%M:%S) ==="
  $PY "$@" 2>&1 | tail -3
}

# from-scratch architectures
run afno  scripts/train.py --config configs/afno.yaml  --auto-resume
run graph scripts/train.py --config configs/graph.yaml --auto-resume

# rollout fine-tuning (GraphCast/Stormer's medium-range stabilizer):
# K=2 from the converged 1-step weights, then K=4 from the K=2 weights
for base in vit unet; do
  if [ -f "artifacts/checkpoints/$base/best.pt" ]; then
    run "${base}_ft2" scripts/train.py --config "configs/$base.yaml" \
      --run-name "${base}_ft2" --finetune-rollout 2 --max-steps 2500 \
      --init-ckpt "artifacts/checkpoints/$base/best.pt" --auto-resume
    run "${base}_ft4" scripts/train.py --config "configs/$base.yaml" \
      --run-name "${base}_ft4" --finetune-rollout 4 --max-steps 1200 \
      --init-ckpt "artifacts/checkpoints/${base}_ft2/best.pt" --auto-resume
  fi
done

echo "=== queue complete @ $(date -u +%H:%M:%S) ==="
