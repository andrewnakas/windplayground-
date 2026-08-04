#!/usr/bin/env bash
# Run the remaining queues back to back, unattended. Safe to re-run: every
# stage inside auto-resumes, and finished stages exit immediately.
set -u
cd "$(dirname "$0")/.."
bash scripts/run_queue4.sh
bash scripts/run_overnight.sh
