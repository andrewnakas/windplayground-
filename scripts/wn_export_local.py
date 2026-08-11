"""Export Google WeatherNext into the live wind blend — run this on YOUR machine.

Why here and not in the Claude container: that container cannot open a
connection into your computer (no ssh binary, egress is an HTTPS-CONNECT proxy,
and a home machine behind NAT could not accept one anyway), so the data has to
travel outward. The good version of that is this script, because your machine
already has working gcloud credentials — **no service-account key is created,
copied, pasted or rotated.** It reads Application Default Credentials, writes
six small JSON files, and you commit them.

    gcloud auth application-default login          # if you have not already
    pip install google-cloud-bigquery
    python scripts/wn_export_local.py --probe   --project treesixty
    python scripts/wn_export_local.py --export  --project treesixty
    git add viewer/data && git commit -m "weathernext live" && git push

Output is ~0.1 MB per lead, so under 1 MB total. It lands in `viewer/data/` in
exactly the layout `scripts/fetch_dynamical.py` produces, which means
`scripts/blend_live.py` picks WeatherNext up as another live member with no
further changes — and its grid and init-time guards then apply automatically, so
a mismatch fails loudly rather than quietly averaging the wrong thing.

Two things this deliberately will not do:

**It will not guess your schema.** I do not know WeatherNext 2's column names.
Discovery matches candidates case-insensitively, *prints the mapping it chose*,
and exits naming the missing field if anything is unmatched or ambiguous.
Guessing identifiers is what cost this project four wasted runs on a filename
assumption.

**It will not run an expensive query without telling you.** Every query is
dry-run first and the cost printed; anything over --max-gb needs --force. It is
your project being billed.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

OUT_DIR = pathlib.Path("viewer/data")
SOURCE_ID = "weathernext_live"
LABEL = "Google WeatherNext (frontier ML) — live"
LEADS = [0, 24, 48, 72, 96, 120]

# The viewer grid the other live sources use. blend_live.py requires an exact
# match, so these are fixed, not tunable.
NX, NY = 180, 90
DX = DY = 2.0
LAT0, LON0 = 89.125, 0.875

# Candidate column names, best guess first. Discovery reports what it picked.
CANDIDATES = {
    "init": ["init_time", "initialization_time", "forecast_reference_time",
             "reference_time", "time", "run_time", "base_time"],
    "lead": ["lead_time", "forecast_hour", "step", "lead", "forecast_period",
             "hours", "valid_delta"],
    "member": ["ensemble_member", "member", "realization", "number",
               "ensemble", "sample_id"],
    "u10": ["10m_u_component_of_wind", "u_component_of_wind_10m", "wind_u_10m",
            "u10", "eastward_wind_10m", "10u"],
    "v10": ["10m_v_component_of_wind", "v_component_of_wind_10m", "wind_v_10m",
            "v10", "northward_wind_10m", "10v"],
    "lat": ["latitude", "lat"],
    "lon": ["longitude", "lon", "long"],
}
# member is optional: a deterministic product has none, and that is fine
OPTIONAL = {"member"}


def discover(schema_names: list[str]) -> dict[str, str]:
    """Map our roles onto real column names, or say exactly what is missing."""
    lower = {n.lower(): n for n in schema_names}
    found, missing, ambiguous = {}, [], []
    for role, cands in CANDIDATES.items():
        hit = next((lower[c] for c in cands if c in lower), None)
        if hit is None:
            # a looser pass: any column containing the strongest token
            token = {"u10": "u_comp", "v10": "v_comp"}.get(role, role)
            near = [n for n in schema_names if token in n.lower()]
            if len(near) == 1:
                hit = near[0]
            elif len(near) > 1:
                ambiguous.append(f"{role}: {near}")
        if hit:
            found[role] = hit
        elif role not in OPTIONAL:
            missing.append(role)

    print("windml column mapping:")
    for role in CANDIDATES:
        print(f"  {role:8s} -> {found.get(role, '(none)')}")
    if ambiguous:
        sys.exit("ambiguous columns, refusing to guess:\n  " + "\n  ".join(ambiguous))
    if missing:
        sys.exit(f"could not find columns for: {missing}\n"
                 f"available: {schema_names}\n"
                 f"add the right name to CANDIDATES and re-run.")
    return found


def velocity_records(u, v, ref_time: str, lead: int) -> list:
    """grib2json layout, north-first row-major — identical to fetch_dynamical.py."""
    def header(param):
        return {"parameterUnit": "m.s-1", "parameterNumber": param,
                "parameterNumberName": "U-component_of_wind" if param == 2
                                       else "V-component_of_wind",
                "parameterCategory": 2, "nx": NX, "ny": NY,
                "lo1": LON0, "la1": LAT0,
                "lo2": LON0 + DX * (NX - 1), "la2": LAT0 - DY * (NY - 1),
                "dx": DX, "dy": DY,
                "refTime": ref_time, "forecastTime": lead}
    return [{"header": header(2), "data": [round(float(x), 3) for x in u]},
            {"header": header(3), "data": [round(float(x), 3) for x in v]}]


def grid_sql(cols: dict, table: str, init: str, lead_expr: str) -> str:
    """Aggregate to the 180x90 grid IN BigQuery, so only the small field returns.

    Binning server-side is the whole cost argument: a global 0.25-degree field is
    ~4M points per variable per lead, and we want 16,200. The GROUP BY happens
    where the data lives.
    """
    return f"""
WITH binned AS (
  SELECT
    CAST(FLOOR(({LAT0} - {cols['lat']}) / {DY}) AS INT64) AS gy,
    MOD(CAST(FLOOR(({cols['lon']} - {LON0}) / {DX}) AS INT64) + {NX}, {NX}) AS gx,
    {cols['u10']} AS u, {cols['v10']} AS v
  FROM `{table}`
  WHERE {cols['init']} = TIMESTAMP('{init}')
    AND {lead_expr}
)
SELECT gy, gx, AVG(u) AS u, AVG(v) AS v
FROM binned
WHERE gy BETWEEN 0 AND {NY - 1}
GROUP BY gy, gx
ORDER BY gy, gx
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--probe", action="store_true")
    p.add_argument("--export", action="store_true")
    p.add_argument("--project", required=True)
    p.add_argument("--table", default=None, help="fully-qualified; from --probe")
    p.add_argument("--pattern", default="weathernext")
    p.add_argument("--max-gb", type=float, default=50.0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", default=str(OUT_DIR))
    a = p.parse_args()
    if not (a.probe or a.export):
        sys.exit("pick --probe or --export (probe first)")

    try:
        from google.cloud import bigquery
    except ImportError:
        sys.exit("pip install google-cloud-bigquery")
    bq = bigquery.Client(project=a.project)

    if a.probe:
        found = False
        for ds in bq.list_datasets():
            if a.pattern.lower() not in ds.dataset_id.lower():
                continue
            found = True
            print(f"windml dataset={ds.dataset_id}")
            for t in bq.list_tables(ds.reference):
                full = f"{ds.project}.{ds.dataset_id}.{t.table_id}"
                print(f"windml   table={full}")
                ref = bq.get_table(full)
                print(f"windml     rows={ref.num_rows:,} "
                      f"partitioned_on={getattr(ref.time_partitioning, 'field', None)}")
                for f in ref.schema:
                    print(f"windml     {f.name:36s} {f.field_type}")
        if not found:
            print(f"windml nothing matching {a.pattern!r}. WeatherNext may be "
                  f"shared as a linked dataset -- check the BigQuery console for "
                  f"its exact project.dataset and pass --table directly.")
        return

    if not a.table:
        sys.exit("--table required for --export; run --probe first")

    ref = bq.get_table(a.table)
    cols = discover([f.name for f in ref.schema])

    init_row = list(bq.query(
        f"SELECT MAX({cols['init']}) AS m FROM `{a.table}`").result())[0]
    init = init_row["m"]
    if init is None:
        sys.exit("no rows / no init time found")
    init_s = init.strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"windml latest init = {init_s}")

    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for lead in LEADS:
        # lead may be an INTERVAL, an INT of hours, or a timestamp delta; the
        # two common encodings are handled and anything else fails loudly
        ftype = next(f.field_type for f in ref.schema if f.name == cols["lead"])
        if ftype in ("INTEGER", "INT64"):
            lead_expr = f"{cols['lead']} = {lead}"
        elif ftype in ("INTERVAL",):
            lead_expr = f"{cols['lead']} = INTERVAL {lead} HOUR"
        else:
            sys.exit(f"unhandled lead column type {ftype} on {cols['lead']}; "
                     f"inspect it and extend lead_expr rather than guessing")

        sql = grid_sql(cols, a.table, init_s, lead_expr)
        job = bq.query(sql, job_config=bigquery.QueryJobConfig(
            dry_run=True, use_query_cache=False))
        gb = job.total_bytes_processed / 1e9
        print(f"windml lead={lead:3d}h dry_run={gb:.2f} GB", flush=True)
        if gb > a.max_gb and not a.force:
            sys.exit(f"would scan {gb:.1f} GB (> --max-gb {a.max_gb}). "
                     f"Re-run with --force if that is acceptable; it bills "
                     f"your project.")

        rows = list(bq.query(sql).result())
        u = [math.nan] * (NX * NY)
        v = [math.nan] * (NX * NY)
        for r in rows:
            i = r["gy"] * NX + r["gx"]
            u[i], v[i] = r["u"], r["v"]
        if any(math.isnan(x) for x in u):
            n = sum(math.isnan(x) for x in u)
            sys.exit(f"{n} of {NX*NY} grid cells empty at lead {lead}; the bin "
                     f"expressions do not match this table's lat/lon convention")
        (out_dir / f"{SOURCE_ID}_latest_{lead:03d}.json").write_text(
            json.dumps(velocity_records(u, v, init_s, lead), separators=(",", ":")))
        written.append(lead)

    man_path = out_dir / "manifest.json"
    man = json.loads(man_path.read_text())
    entry = {"id": SOURCE_ID, "label": LABEL, "kind": "live",
             "inits": ["latest"], "leads": written, "init_time": init_s}
    man["sources"] = [s for s in man["sources"] if s["id"] != SOURCE_ID] + [entry]
    man_path.write_text(json.dumps(man, indent=2))
    print(f"windml wrote leads={written} init={init_s}")
    print("windml next: git add viewer/data && git commit && git push")


if __name__ == "__main__":
    main()
