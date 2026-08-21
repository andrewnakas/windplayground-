#!/bin/zsh
# WeatherNext refresh, scheduled from the Mac (launchd) — the one live member
# CI cannot fetch, because the BigQuery export needs this machine's Google
# credentials. Runs in its OWN clone so it never touches the working checkout,
# commits to main only when the data actually moved, and lets pages.yml deploy.
#
# Cost note: each export bills the tree-sixty project a few cents (~3 GB
# scanned); at four runs a day that is a handful of dollars a month.
#
# Install:  scripts/wn_refresh_local.sh --install   (writes + loads the
#           launchd job at 03:50/09:50/15:50/21:50 UTC, ~90 min after each
#           CI refresh so --match-live finds the fresh AIFS/GFS cycle)
set -euo pipefail

REPO_URL="https://github.com/andrewnakas/windplayground-.git"
WORK="${WINDML_CRON_DIR:-$HOME/.windplayground-cron}"
CLONE="$WORK/windplayground-"
VENV="$WORK/venv"
PLIST="$HOME/Library/LaunchAgents/com.windplayground.weathernext.plist"
LOG="$HOME/Library/Logs/windplayground-wn.log"
export PATH="/opt/homebrew/bin:/usr/bin:/bin:$PATH"

if [[ "${1:-}" == "--install" ]]; then
  mkdir -p "$WORK" "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
  [[ -d "$CLONE/.git" ]] || git clone -q "$REPO_URL" "$CLONE"
  [[ -x "$VENV/bin/python" ]] || {
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet google-cloud-bigquery pandas pyyaml
  }
  # 21:50/03:50/09:50/15:50 MDT == 03:50/09:50/15:50/21:50 UTC (launchd runs
  # on local time; a DST shift skews this an hour, which --match-live absorbs)
  cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.windplayground.weathernext</string>
  <key>ProgramArguments</key>
  <array><string>/bin/zsh</string>
    <string>$CLONE/scripts/wn_refresh_local.sh</string></array>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Hour</key><integer>21</integer><key>Minute</key><integer>50</integer></dict>
    <dict><key>Hour</key><integer>3</integer><key>Minute</key><integer>50</integer></dict>
    <dict><key>Hour</key><integer>9</integer><key>Minute</key><integer>50</integer></dict>
    <dict><key>Hour</key><integer>15</integer><key>Minute</key><integer>50</integer></dict>
  </array>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict></plist>
PLIST
  launchctl unload "$PLIST" 2>/dev/null || true
  launchctl load "$PLIST"
  echo "installed: $(launchctl list | grep com.windplayground.weathernext || true)"
  exit 0
fi

echo "== $(date -u +%FT%TZ) WeatherNext refresh"
cd "$CLONE"

# one run at a time; a stale lock older than 3 h is a crashed run, not a live one
LOCK="$WORK/refresh.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  if [[ -n "$(find "$LOCK" -maxdepth 0 -mmin +180 2>/dev/null)" ]]; then
    rmdir "$LOCK"; mkdir "$LOCK"
  else
    echo "another refresh is running; skipping"; exit 0
  fi
fi
trap 'rmdir "$LOCK"' EXIT

git checkout -q main
git fetch -q origin
git reset -q --hard origin/main   # this clone holds no work of its own

"$VENV/bin/python" scripts/wn_export_local.py --export --project tree-sixty \
  --table tree-sixty.weathernext_2_mean.weathernext_2_0_0_mean --match-live
python3 scripts/blend_live.py
PYTHONPATH=src python3 scripts/build_site.py

git add docs
if git diff --cached --quiet; then
  echo "no change -- WeatherNext already on this cycle"
  exit 0
fi
INIT=$(python3 -c "import json;m=json.load(open('docs/data/manifest.json'));print(next(s['init_time'] for s in m['sources'] if s['id']=='weathernext_live'))")
git commit -q -m "WeatherNext refresh: init $INIT"
git push -q origin main
git push -q origin main:claude/wind-forecast-ml-research-veu6d5 || true
echo "pushed init $INIT"
