#!/usr/bin/env bash
# The literature-anchor attempt. Combines the three things that differ from
# Rasp & Thuerey 2021 (z500 268 at 3 days): vertical structure in the inputs,
# direct prediction of the target lead, and a wider network.
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

# the anchor attempt runs first: loss focused on z500/t850 (their direct model
# optimizes those two, our 20-channel average sent ~1/20th of the signal to z500)
run anchor72 scripts/train.py --config configs/anchor72.yaml \
  --run-name anchor72 --auto-resume
run_eval anchor72_eval anchor72 scripts/evaluate.py \
  --ckpt artifacts/checkpoints/anchor72/best.pt --name anchor72

run levels72 scripts/train.py --config configs/unet_levels.yaml \
  --run-name levels72 --auto-resume
run_eval levels72_eval levels72 scripts/evaluate.py \
  --ckpt artifacts/checkpoints/levels72/best.pt --name levels72

# same recipe aimed at 5 days, the other lead the leaderboard reports
run levels120 scripts/train.py --config configs/unet_levels.yaml \
  --run-name levels120 --direct-lead-h 120 --max-steps 18000 \
  --time-budget-hours 2.5 --auto-resume
run_eval levels120_eval levels120 scripts/evaluate.py \
  --ckpt artifacts/checkpoints/levels120/best.pt --name levels120

echo "=== levels runs complete @ $(date -u +%H:%M:%S) ==="
