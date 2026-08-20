# Runbook — adding WeatherNext 2 from your own machine

Everything else in this project runs in a cloud container. This one job cannot,
for a boring reason: the container has no Google credentials and no way to
obtain any. Your Mac already has working `gcloud` credentials, so running the
export there means **no service-account key is ever created, copied, pasted or
rotated**. The output is six JSON files totalling well under 1 MB, which you
commit like any other change.

Read this top to bottom before starting — step 1 is free, step 3 bills your
project, and step 2 is the one that tells you whether step 3 will work.

## What you are adding, and why it matters

The viewer already carries a live multi-model mean of **ECMWF AIFS** (operational
ML) and **NOAA GFS** (operational physics). WeatherNext 2 is the DeepMind
GraphCast/GenCast lineage productised, and it is the frontier-grade member that
blend is otherwise missing — GenCast, GraphCast, Pangu and FuXi have no live
public feed at all (probed; see the note in `scripts/fetch_dynamical.py`).

## 0. Get the repo

```bash
git clone https://github.com/andrewnakas/windplayground-.git
cd windplayground-
git checkout claude/wind-forecast-ml-research-veu6d5      # not main
pip install google-cloud-bigquery                        # the exporter needs nothing else
gcloud auth application-default login
```

The exporter is deliberately self-contained — it imports only the standard
library and `google-cloud-bigquery`, so you do **not** need the project's
training environment (torch, xarray, zarr) to run it. You do need it for step 4.

## 1. Probe — metadata only, costs nothing

```bash
python scripts/wn_export_local.py --probe --project treesixty \
    --impersonate 631486859154-compute@developer.gserviceaccount.com
```

**The first line is the principal it is acting as.** That line exists because
"no datasets visible" has three causes that look identical, and knowing who did
the looking eliminates one of them immediately:

| what you see | what it means |
|---|---|
| datasets and a nested schema listing | good, go to step 2 |
| `permission denied as <you>` | the grant is on the service account and you are not it — that is what `--impersonate` is for, and it needs `roles/iam.serviceAccountTokenCreator` on that account, granted to you |
| `could not mint a token for <SA>` | impersonation itself is not set up; same role, same fix |
| nothing matching `'weathernext'` | most likely not subscribed yet — add it from the **Analytics Hub** listing, which links the dataset into your project |

A service-account email is a **grantee identifier, not a secret**. It is safe to
paste anywhere; it authenticates nothing on its own.

Expect the schema listing to show a nested layout: `init_time`, `geography`,
`geography_polygon`, and a REPEATED `forecast` record containing `hours`,
`10m_u_component_of_wind`, `10m_v_component_of_wind` and friends. The probe
recurses into that record — if it prints `forecast RECORD` and nothing beneath
it, tell the session, because that means the flattening is not working and
everything downstream is guesswork.

## 2. Export — dry-run first, then six files

```bash
python scripts/wn_export_local.py --export --project treesixty \
    --impersonate 631486859154-compute@developer.gserviceaccount.com \
    --table treesixty.<dataset>.weathernext_2_0_0_mean \
    --match-live
```

Use the `<dataset>` name step 1 printed.

Three things this does on purpose:

- **`_mean`, not the 64-member table.** Both exist. The 64-member table scans
  roughly 64× as much to produce a field that would only be averaged anyway.
  The script warns if you name it.
- **`--match-live`** pins the export to the init AIFS and GFS already carry.
  `blend_live.py` refuses to average members from different cycles — correctly,
  since that is not an ensemble — so taking WeatherNext's newest init and hoping
  it lines up is how the export succeeds and the blend then fails.
- **One query, all six leads.** It prints the dry-run cost before running
  anything and refuses above `--max-gb` (default 50) unless you pass `--force`.
  **Your project is billed**, so read that number rather than skipping past it.

If `--match-live` says the existing members are on a cycle WeatherNext does not
have, re-fetch them onto a shared one rather than relaxing the check:

```bash
python -m venv .venv-dyn && .venv-dyn/bin/pip install "zarr>=3" "xarray>=2025.1" numpy aiohttp
.venv-dyn/bin/python scripts/fetch_dynamical.py --dataset aifs
.venv-dyn/bin/python scripts/fetch_dynamical.py --dataset gfs
```

## 3. Blend and rebuild

```bash
python scripts/blend_live.py        # refuses on grid or init mismatch, by design
python scripts/build_site.py
```

`blend_live.py` picks WeatherNext up automatically as a third `kind: "live"`
source — there is no list to add it to. It will also fail if the shared init is
more than 18 h old, which is a stopped refresh rather than a data error; the
message says which.

**Look at the map before you push.** Open `docs/viewer/index.html`, pick "Live
multi-model mean" and confirm particles are moving over land as well as ocean.
A previous fix in this project was shipped without looking at the render and was
aimed at the wrong bug entirely.

## 4. Publish

```bash
git add viewer/data docs
git commit -m "weathernext live"
git push -u origin claude/wind-forecast-ml-research-veu6d5
```

## 5. While you are here — deploy the site

The GitHub Pages site has served **2026-08-06** content for twelve days. The
cloud session cannot fix this: pushing to `main` is blocked there by a
permission classifier. From your Mac it is a clean fast-forward — `git diff
HEAD...origin/main` is empty, because main's commits are merges of this branch:

```bash
git push origin HEAD:main
```

Also worth one click while you are in GitHub: **Actions → "Refresh live wind" →
Run workflow.** That workflow is registered, active, on the default branch, with
valid YAML and a valid cron — and GitHub has queued it **zero** times since it
was created. A successful manual run would prove the workflow itself is fine and
narrow the problem to scheduling. The cloud session cannot test it either; its
GitHub token lacks `actions:write`.
