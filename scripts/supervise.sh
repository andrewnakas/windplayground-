#!/usr/bin/env bash
# Relaunch the training program if it is not running.
#
# Idempotent by construction: every stage inside run_all.sh uses --auto-resume
# and finished stages exit immediately, so calling this while training is
# healthy is a no-op. Call it on every loop wakeup -- a container reclaim then
# costs one wakeup interval instead of a whole night (which is exactly what it
# cost on 2026-08-04: reclaimed 11:25, noticed 18:30).
set -u
cd "$(dirname "$0")/.."
if pgrep -f "scripts/train\.py|scripts/evaluate\.py" > /dev/null; then
  echo "training already running: $(pgrep -af 'scripts/train\.py' | head -1 | cut -c1-90)"
  exit 0
fi
echo "no training running -- (re)launching run_all.sh @ $(date -u +%H:%M:%S)"
nohup bash scripts/run_all.sh >> artifacts/checkpoints/run_all.out 2>&1 &
disown
sleep 5
pgrep -f "scripts/train\.py" > /dev/null && echo "relaunched OK" || echo "WARNING: did not start"
