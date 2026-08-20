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
    python scripts/wn_export_local.py --probe  --project treesixty
    python scripts/wn_export_local.py --export --project treesixty \
        --table treesixty.<dataset>.weathernext_2_0_0_mean --match-live
    python scripts/blend_live.py && python scripts/build_site.py
    git add viewer/data docs && git commit -m "weathernext live" && git push

**If WeatherNext was granted to a service account, add `--impersonate`.** A
service-account email is not a credential -- it is the principal a grant
attaches to -- so a grant on `SA@project.iam.gserviceaccount.com` does nothing
for your user login, and the resulting 403 looks exactly like "the access did
not work". Acting as the granted account needs
`roles/iam.serviceAccountTokenCreator` on it:

    python scripts/wn_export_local.py --probe --project treesixty \
        --impersonate 631486859154-compute@developer.gserviceaccount.com

Every run prints the principal it is acting as before anything else, so "no
datasets visible" is a diagnosable answer rather than a dead end.

Output is ~0.1 MB per lead, so under 1 MB total. It lands in `viewer/data/` in
exactly the layout `scripts/fetch_dynamical.py` produces, which means
`scripts/blend_live.py` picks WeatherNext up as another live member with no
further changes — and its grid and init-time guards then apply automatically, so
a mismatch fails loudly rather than quietly averaging the wrong thing.

**The schema is nested, and that is handled.** WeatherNext does not store flat
columns: position is a GEOGRAPHY point (`geography`, alongside a
`geography_polygon` companion) and every variable hangs off a REPEATED RECORD
called `forecast`, whose subfields are `hours`, `time`,
`10m_u_component_of_wind`, `10m_v_component_of_wind`, ... So the query UNNESTs
`forecast`, reads latitude and longitude with ST_Y/ST_X, and backticks every
identifier -- `10m_u_component_of_wind` starts with a digit, which is a syntax
error unquoted. Discovery still *prints the mapping it chose* and still exits
naming the field if anything is unmatched or ambiguous, because the layout was
read from Google's docs, not from your table.

Prefer the ensemble-MEAN table (`....weathernext_2_0_0_mean`). The 64-member
table scans roughly 64x as much to produce a field this script would only
average anyway.

**All six leads come back in ONE query.** With the variables inside a repeated
record, a query per lead re-scans the whole array each time and bills six times
over for a single cycle of wind.

**Init times must agree or the blend refuses**, so `--match-live` pins the
export to the init the other live members already carry, rather than taking
WeatherNext's newest and hoping. `--init` pins it explicitly.

**It will not run an expensive query without telling you.** The query is
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


# --- who are we acting as? -------------------------------------------------
#
# A service-account email is not a credential; it is the principal a grant
# attaches to. "WeatherNext was granted to 631486859154-compute@..." means the
# *service account* can read it -- not necessarily the user login that
# `gcloud auth application-default login` left behind. Running as the wrong
# principal fails exactly like having no access at all, so --probe prints who
# it is before it prints what it can see.

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def adc_hint(exc) -> str:
    return ("no Google credentials on this machine. Run:\n\n"
            "    gcloud auth application-default login\n\n"
            "No service-account key is needed -- your own login is enough, and\n"
            "--impersonate covers the case where the grant is on a service\n"
            f"account rather than on you.\n\n({exc})")


def build_credentials(impersonate: str | None):
    """ADC, or ADC impersonating a service account. No key file either way."""
    import google.auth
    try:
        source, _ = google.auth.default(scopes=SCOPES)
    except Exception as exc:
        if type(exc).__name__ == "DefaultCredentialsError":
            sys.exit(adc_hint(exc))
        raise
    if not impersonate:
        return source, None
    from google.auth import impersonated_credentials
    return impersonated_credentials.Credentials(
        source_credentials=source,
        target_principal=impersonate,
        target_scopes=SCOPES,
    ), impersonate


def principal_of(creds, bq=None) -> str:
    """Best available answer to 'who am I', by local inspection first."""
    for attr in ("service_account_email", "signer_email", "account"):
        value = getattr(creds, attr, None)
        if isinstance(value, str) and "@" in value:
            return value
    # A user ADC token carries no email locally, but a dry run costs nothing
    # and BigQuery answers authoritatively.
    if bq is not None:
        try:
            from google.cloud import bigquery
            job = bq.query("SELECT 1", job_config=bigquery.QueryJobConfig(
                dry_run=True, use_query_cache=False))
            if job.user_email:
                return job.user_email
        except Exception:  # noqa: BLE001,S110 -- a nicety, never a gate
            pass
    return "unknown (user ADC carries no email; try `gcloud auth list`)"


def denied_message(principal: str, impersonate: str | None, exc) -> str:
    """Name the likely cause, because 403 here has two very different ones."""
    if impersonate:
        return (
            f"permission denied while acting as {principal}\n  {exc}\n\n"
            f"Impersonation itself worked, so this is the data grant rather\n"
            f"than the principal: {impersonate} needs roles/bigquery.dataViewer\n"
            f"on the WeatherNext dataset and roles/bigquery.jobUser on the\n"
            f"project being billed.")
    return (
        f"permission denied as {principal}\n  {exc}\n\n"
        f"WeatherNext access is normally granted to a SERVICE ACCOUNT, and this\n"
        f"ran as a user login -- so the likely cause is the wrong principal, not\n"
        f"missing access. Re-run against the granted account:\n\n"
        f"    ... --impersonate SERVICE_ACCOUNT_EMAIL\n\n"
        f"which needs roles/iam.serviceAccountTokenCreator on it, granted to you:\n\n"
        f"    gcloud iam service-accounts add-iam-policy-binding SA_EMAIL \\\n"
        f"      --member=user:YOUR_EMAIL \\\n"
        f"      --role=roles/iam.serviceAccountTokenCreator")


def guarded(fn, principal: str, impersonate: str | None):
    """Run fn(), turning the two IAM failure modes into their explanations."""
    try:
        return fn()
    except Exception as exc:
        name = type(exc).__name__
        if name in ("Forbidden", "PermissionDenied") or "403" in str(exc):
            sys.exit(denied_message(principal, impersonate, exc))
        if name == "RefreshError" and impersonate:
            sys.exit(
                f"could not mint a token for {impersonate}:\n  {exc}\n\n"
                f"This is the impersonation grant, not WeatherNext. You need\n"
                f"roles/iam.serviceAccountTokenCreator on {impersonate}:\n\n"
                f"    gcloud iam service-accounts add-iam-policy-binding "
                f"{impersonate} \\\n"
                f"      --member=user:YOUR_EMAIL \\\n"
                f"      --role=roles/iam.serviceAccountTokenCreator")
        raise


def flatten_schema(fields, prefix: str = "") -> list[tuple[str, str, str]]:
    """[(dotted_path, type, mode)] INCLUDING the subfields of nested records.

    WeatherNext hangs every variable off a single repeated RECORD called
    `forecast`, so a schema listing that stops at the top level prints
    "forecast RECORD" and hides the wind entirely. That is worse than useless:
    --probe exists precisely so nothing downstream has to be guessed, and a
    probe that cannot see the columns cannot do that job.
    """
    out = []
    for f in fields:
        path = f"{prefix}{f.name}"
        out.append((path, f.field_type, f.mode or "NULLABLE"))
        if f.field_type in ("RECORD", "STRUCT") and getattr(f, "fields", None):
            out.extend(flatten_schema(f.fields, path + "."))
    return out


def _leaf(path: str) -> str:
    return path.rsplit(".", 1)[-1].lower()


def discover(schema) -> dict:
    """Map our roles onto real columns -- flat OR nested -- or say what is missing.

    Returns a layout, not just names, because where a column lives changes the
    SQL as much as what it is called:

        {"kind": "flat"|"nested", "unnest": None|"forecast",
         "geo": None|"geography", "cols": {role: dotted_path}}

    `schema` takes either (path, type, mode) triples from flatten_schema() or a
    bare list of column names for the simple flat case.

    Two things it will not do, both learned the hard way. It will not pick
    between two plausible candidates, and it will not silently accept a table
    where the wind is missing -- it names the role and stops.
    """
    fields = [(n, "STRING", "NULLABLE") if isinstance(n, str) else tuple(n)
              for n in schema]
    paths = [f[0] for f in fields]
    types = {f[0]: f[1] for f in fields}
    modes = {f[0]: f[2] for f in fields}

    found, missing, ambiguous = {}, [], []
    for role, cands in CANDIDATES.items():
        hit, role_ambiguous = None, False
        for c in cands:
            hits = [p for p in paths if _leaf(p) == c]
            if len(hits) == 1:
                hit = hits[0]
                break
            if len(hits) > 1:
                ambiguous.append(f"{role}: {hits}")
                role_ambiguous = True
                break
        if hit is None and not role_ambiguous:
            # looser pass: any column whose leaf carries the strongest token
            token = {"u10": "u_comp", "v10": "v_comp"}.get(role, role)
            near = [p for p in paths if token in _leaf(p)]
            if len(near) == 1:
                hit = near[0]
            elif len(near) > 1:
                ambiguous.append(f"{role}: {near}")
        if hit:
            found[role] = hit
        elif role not in OPTIONAL:
            missing.append(role)

    # No lat/lon pair? WeatherNext stores position as a GEOGRAPHY point, and
    # ST_X/ST_Y read it. Exclude the polygon companion column explicitly rather
    # than by ordering, so a schema change cannot silently select the wrong one.
    geo = None
    if "lat" in missing or "lon" in missing:
        cands = [p for p in paths
                 if types.get(p) == "GEOGRAPHY" and "polygon" not in p.lower()]
        if len(cands) == 1:
            geo = cands[0]
            missing = [m for m in missing if m not in ("lat", "lon")]
            found.pop("lat", None)
            found.pop("lon", None)
        elif len(cands) > 1:
            ambiguous.append(f"lat/lon: several GEOGRAPHY columns {cands}")

    # Everything nested must hang off ONE repeated record; two would need two
    # UNNESTs and a join whose semantics are not obvious enough to guess at.
    nested = sorted({p.split(".", 1)[0] for p in found.values() if "." in p})
    if len(nested) > 1:
        ambiguous.append(f"columns span several nested records {nested}")
    unnest = nested[0] if nested else None
    if unnest and modes.get(unnest) != "REPEATED":
        ambiguous.append(f"{unnest} holds our columns but is not REPEATED "
                         f"(mode={modes.get(unnest)}); UNNEST would be wrong")
    if any(p.count(".") > 1 for p in found.values()):
        ambiguous.append("a column is nested more than one level deep; "
                         "extend grid_sql rather than guessing at the path")

    print("windml column mapping:")
    for role in CANDIDATES:
        if role in ("lat", "lon") and geo:
            print(f"  {role:8s} -> {'ST_Y' if role == 'lat' else 'ST_X'}({geo})")
        else:
            print(f"  {role:8s} -> {found.get(role, '(none)')}")
    if unnest:
        print(f"  layout   -> nested, UNNEST({unnest})")

    if ambiguous:
        sys.exit("ambiguous columns, refusing to guess:\n  " + "\n  ".join(ambiguous))
    if missing:
        sys.exit(f"could not find columns for: {missing}\n"
                 f"available: {paths}\n"
                 f"add the right name to CANDIDATES and re-run.")
    return {"kind": "nested" if unnest else "flat", "unnest": unnest,
            "geo": geo, "cols": found}


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


def col_ref(layout: dict, role: str, t1: str = "t1", t2: str = "t2") -> str:
    """Qualified, BACKTICKED reference to a discovered column.

    The backticks are not decoration: WeatherNext's wind columns are called
    `10m_u_component_of_wind`, and an identifier starting with a digit is a
    syntax error in BigQuery unless quoted.
    """
    path = layout["cols"][role]
    if "." in path:
        return f"{t2}.`{path.split('.', 1)[1]}`"
    return f"{t1}.`{path}`"


def grid_sql(layout: dict, table: str, init: str, leads: list[int],
             lead_type: str = "INTEGER") -> str:
    """Bin to the 180x90 grid IN BigQuery, every lead in ONE query.

    Two cost arguments, both about not moving data. Binning server-side is the
    first: a global 0.25-degree field is ~1M points per lead and we want 16,200,
    so the GROUP BY belongs where the data lives. Asking for all six leads at
    once is the second -- with the variables inside a repeated record, a
    per-lead query re-scans the whole array each time and bills six times over
    for one cycle's worth of wind.
    """
    t1, t2 = "t1", "t2"
    lead_e, init_e = col_ref(layout, "lead", t1, t2), col_ref(layout, "init", t1, t2)
    if lead_type in ("INTEGER", "INT64"):
        lead_filter = f"{lead_e} IN ({', '.join(str(h) for h in leads)})"
    elif lead_type == "INTERVAL":
        lead_filter = " OR ".join(f"{lead_e} = INTERVAL {h} HOUR" for h in leads)
        lead_filter = f"({lead_filter})"
    else:
        sys.exit(f"unhandled lead column type {lead_type}; inspect it and "
                 f"extend grid_sql rather than guessing")

    if layout["geo"]:
        lat_e, lon_e = f"ST_Y({t1}.`{layout['geo']}`)", f"ST_X({t1}.`{layout['geo']}`)"
    else:
        lat_e, lon_e = col_ref(layout, "lat", t1, t2), col_ref(layout, "lon", t1, t2)

    frm = f"FROM `{table}` AS {t1}"
    if layout["unnest"]:
        frm += f",\n       UNNEST({t1}.`{layout['unnest']}`) AS {t2}"

    # ST_X returns -180..180 while the grid starts at 0.875, so gx is wrapped
    # rather than clamped: FLOOR of a negative offset gives a negative index and
    # the + NX before MOD carries it round the date line.
    return f"""
WITH binned AS (
  SELECT
    {lead_e} AS lead_h,
    CAST(FLOOR(({LAT0} - {lat_e}) / {DY}) AS INT64) AS gy,
    MOD(CAST(FLOOR(({lon_e} - {LON0}) / {DX}) AS INT64) + {NX}, {NX}) AS gx,
    {col_ref(layout, 'u10', t1, t2)} AS u,
    {col_ref(layout, 'v10', t1, t2)} AS v
  {frm}
  WHERE {init_e} = TIMESTAMP('{init}')
    AND {lead_filter}
)
SELECT lead_h, gy, gx, AVG(u) AS u, AVG(v) AS v
FROM binned
WHERE gy BETWEEN 0 AND {NY - 1}
GROUP BY lead_h, gy, gx
ORDER BY lead_h, gy, gx
"""


def live_init(out_dir: pathlib.Path) -> str:
    """The init the OTHER live members already carry.

    blend_live.py refuses to average members whose refTime differs, and that
    refusal is right -- a 00Z field averaged with an 18Z one is not an ensemble.
    So the useful default is not WeatherNext's newest init, it is the one that
    will actually blend. Taking MAX() and hoping is how the export succeeds and
    the blend then refuses, with nothing pointing at why.
    """
    man = json.loads((out_dir / "manifest.json").read_text())
    inits = {s["id"]: s.get("init_time") for s in man.get("sources", [])
             if s.get("kind") == "live" and s["id"] not in (SOURCE_ID, "live_blend")}
    have = {v for v in inits.values() if v}
    if not have:
        sys.exit("--match-live: no other live source carries an init_time; "
                 "run scripts/fetch_dynamical.py first, or pass --init")
    if len(have) > 1:
        sys.exit(f"--match-live: the existing live members disagree {inits}. "
                 f"Re-fetch them onto one cycle before adding a third.")
    return have.pop()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--probe", action="store_true")
    p.add_argument("--export", action="store_true")
    p.add_argument("--project", required=True)
    p.add_argument("--table", default=None,
                   help="fully-qualified; from --probe. Prefer the ensemble-MEAN "
                        "table (…weathernext_2_0_0_mean): the 64-member table "
                        "scans ~64x as much for a field we would only average.")
    p.add_argument("--pattern", default="weathernext")
    p.add_argument("--max-gb", type=float, default=50.0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--leads", nargs="+", type=int, default=LEADS)
    p.add_argument("--init", default=None,
                   help="pin the init (e.g. 2026-08-18T00:00:00Z); default is "
                        "the newest one the table has")
    p.add_argument("--match-live", action="store_true",
                   help="pin to the init the other live members already carry, "
                        "so blend_live.py can actually average them")
    p.add_argument("--impersonate", default=None, metavar="SA_EMAIL",
                   help="act as this service account instead of your own login "
                        "-- use it when WeatherNext was granted to a service "
                        "account rather than to you. Needs "
                        "roles/iam.serviceAccountTokenCreator on it.")
    a = p.parse_args()
    if not (a.probe or a.export):
        sys.exit("pick --probe or --export (probe first)")
    if a.init and a.match_live:
        sys.exit("--init and --match-live both set; pick one")

    try:
        from google.cloud import bigquery
    except ImportError:
        sys.exit("pip install google-cloud-bigquery")
    creds, impersonating = build_credentials(a.impersonate)
    bq = bigquery.Client(project=a.project, credentials=creds)

    # First line of output, always: a listing that finds nothing is only
    # interpretable once you know which principal did the looking.
    principal = principal_of(creds, bq)
    print(f"windml acting as {principal}"
          + (" (impersonated from your login)" if impersonating else "")
          + f", billing project {bq.project}")

    if a.probe:
        found = False
        for ds in guarded(lambda: list(bq.list_datasets()), principal, impersonating):
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
                for path, ftype, mode in flatten_schema(ref.schema):
                    star = " REPEATED" if mode == "REPEATED" else ""
                    print(f"windml     {path:44s} {ftype}{star}")
        if not found:
            print(f"windml nothing matching {a.pattern!r} is visible to "
                  f"{principal}. Three causes look identical here: WeatherNext "
                  f"may not be subscribed yet (add it from the Analytics Hub "
                  f"listing), it may be linked under another project (pass "
                  f"--table directly), or the grant may be on a service account "
                  f"rather than on this principal (re-run with --impersonate).")
        return

    if not a.table:
        sys.exit("--table required for --export; run --probe first")
    if a.table.endswith("weathernext_2_0_0"):
        print("windml NOTE: that is the 64-member table. Every member gets "
              "averaged into one field anyway, so …_mean costs ~64x less for "
              "the same output. Continuing, since you named it explicitly.")

    ref = guarded(lambda: bq.get_table(a.table), principal, impersonating)
    schema = flatten_schema(ref.schema)
    layout = discover(schema)
    types = {path: ftype for path, ftype, _ in schema}
    init_col = col_ref(layout, "init")

    # Recent inits, printed. This is the cheapest query in the run and it turns
    # "the init you asked for is not there" into a list of the ones that are.
    recent = guarded(lambda: [r["i"] for r in bq.query(
        f"SELECT DISTINCT {init_col} AS i FROM `{a.table}` AS t1 "
        f"ORDER BY i DESC LIMIT 8").result()], principal, impersonating)
    if not recent:
        sys.exit("no rows / no init time found")
    available = [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in recent]
    print(f"windml recent inits: {', '.join(available)}")

    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.match_live:
        init_s = live_init(out_dir)
        print(f"windml matching the existing live members at {init_s}")
    else:
        init_s = a.init or available[0]
    if init_s not in available:
        sys.exit(f"init {init_s} is not among the 8 most recent inits above.\n"
                 f"If you are matching the live members, re-fetch them onto a "
                 f"cycle WeatherNext also has (scripts/fetch_dynamical.py), "
                 f"rather than blending across cycles.")

    lead_type = types.get(layout["cols"]["lead"], "INTEGER")
    sql = grid_sql(layout, a.table, init_s, list(a.leads), lead_type)

    job = bq.query(sql, job_config=bigquery.QueryJobConfig(
        dry_run=True, use_query_cache=False))
    gb = job.total_bytes_processed / 1e9
    print(f"windml dry run: {gb:.2f} GB for all {len(a.leads)} leads", flush=True)
    if gb > a.max_gb and not a.force:
        sys.exit(f"would scan {gb:.1f} GB (> --max-gb {a.max_gb}). "
                 f"Re-run with --force if that is acceptable; it bills "
                 f"your project.")

    rows = guarded(lambda: list(bq.query(sql).result()), principal, impersonating)
    print(f"windml returned {len(rows):,} rows")

    fields = {}
    for r in rows:
        u, v = fields.setdefault(int(r["lead_h"]),
                                 ([math.nan] * (NX * NY), [math.nan] * (NX * NY)))
        i = r["gy"] * NX + r["gx"]
        u[i], v[i] = r["u"], r["v"]

    written = []
    for lead in sorted(fields):
        u, v = fields[lead]
        gaps = sum(math.isnan(x) for x in u)
        if gaps:
            sys.exit(f"{gaps} of {NX*NY} grid cells empty at lead {lead}h; the "
                     f"bin expressions do not match this table's lat/lon "
                     f"convention -- check ST_X/ST_Y against the probe output "
                     f"rather than filling the holes.")
        (out_dir / f"{SOURCE_ID}_latest_{lead:03d}.json").write_text(
            json.dumps(velocity_records(u, v, init_s, lead), separators=(",", ":")))
        written.append(lead)
    missing_leads = [h for h in a.leads if h not in written]
    if missing_leads:
        print(f"windml WARNING: no rows at leads {missing_leads}; they are not "
              f"written, so blend_live.py will simply skip them")

    man_path = out_dir / "manifest.json"
    man = json.loads(man_path.read_text())
    entry = {"id": SOURCE_ID, "label": LABEL, "kind": "live",
             "inits": ["latest"], "leads": written, "init_time": init_s}
    man["sources"] = [s for s in man["sources"] if s["id"] != SOURCE_ID] + [entry]
    man_path.write_text(json.dumps(man, indent=2))
    print(f"windml wrote leads={written} init={init_s}")
    print("windml next: python scripts/blend_live.py && python scripts/build_site.py")


if __name__ == "__main__":
    main()
