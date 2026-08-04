#!/usr/bin/env bash
# Run the remaining queues back to back, unattended. Safe to re-run: every
# stage inside auto-resumes, and finished stages exit immediately.
set -u
# run from the repo root even when launched from a snapshot copy
cd "${WINDML_REPO:?set by scripts/supervise.sh}"
bash "$(dirname "$0")/run_queue4.sh"
bash "$(dirname "$0")/run_overnight.sh"
