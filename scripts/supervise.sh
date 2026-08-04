#!/usr/bin/env bash
# Relaunch the training program if it is not running.
#
# Idempotent by construction: every training stage uses --auto-resume, finished
# stages exit immediately, and finished evaluations are skipped by result-file
# check. Calling this while training is healthy is a no-op. Call it on every
# loop wakeup -- a container reclaim then costs one wakeup interval instead of
# a whole night (which is what it cost on 2026-08-04: reclaimed 11:25, noticed
# 18:30).
#
# The queue runs from a SNAPSHOT of the scripts, not from the working tree.
# bash reads a script incrementally by byte offset, so editing scripts/*.sh
# while they run makes the live shell resume mid-token -- that corrupted a run
# on 2026-08-04 ("cripts/train.py: No such file or directory"). Snapshotting
# makes the working tree safe to edit at any time.
set -u
export WINDML_REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$(dirname "$0")/.."

if pgrep -f "scripts/train\.py|scripts/evaluate\.py" > /dev/null; then
  echo "training already running: $(pgrep -af 'scripts/(train|evaluate)\.py' | head -1 | cut -c1-90)"
  exit 0
fi

SNAP=artifacts/checkpoints/_running
rm -rf "$SNAP"
mkdir -p "$SNAP"
cp scripts/run_all.sh scripts/run_queue4.sh scripts/run_overnight.sh scripts/run_direct.sh scripts/run_levels.sh "$SNAP"/

echo "no training running -- (re)launching from snapshot @ $(date -u +%H:%M:%S)"
nohup bash "$SNAP/run_all.sh" >> artifacts/checkpoints/run_all.out 2>&1 &
disown
sleep 8
if pgrep -f "scripts/train\.py|scripts/evaluate\.py" > /dev/null; then
  echo "relaunched OK"
else
  echo "WARNING: nothing started -- check artifacts/checkpoints/run_all.out"
fi
