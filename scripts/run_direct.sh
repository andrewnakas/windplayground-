#!/usr/bin/env bash
# Direct lead-time models: the biggest structural gap to the published
# 5.625 deg anchor. Rasp & Thuerey 2021 reach z500 268 at 3 days with a model
# trained to predict 72h in ONE shot; our 6h model rolls out 12 times and
# accumulates error. Same architecture and budget here, only the target
# changes, so the comparison isolates that one design choice.
set -u
PY=${PY:-.venv/bin/python}
cd "${WINDML_REPO:?set by scripts/supervise.sh}"
run() { echo "=== $1 @ $(date -u +%H:%M:%S) ==="; shift; $PY "$@" 2>&1 | tail -4; }
run_eval() {
  local label=$1 name=$2; shift 2
  if [ -f "artifacts/results/${name}_test.csv" ]; then
    echo "=== $label: already evaluated, skipping ==="; return 0
  fi
  run "$label" "$@"
}

for H in 72 120; do
  run "direct${H}" scripts/train.py --config configs/unet_long.yaml \
    --run-name "direct${H}" --direct-lead-h "$H" --max-steps 20000 \
    --time-budget-hours 2.0 --auto-resume
  run_eval "direct${H}_eval" "direct${H}" scripts/evaluate.py \
    --ckpt "artifacts/checkpoints/direct${H}/best.pt" --name "direct${H}"
done
echo "=== direct-lead runs complete @ $(date -u +%H:%M:%S) ==="
