"""Pull Google WeatherNext into the live wind blend, via BigQuery.

WeatherNext is the DeepMind GraphCast/GenCast lineage productized, and
WeatherNext 2 is their newer ensemble model. It is the frontier-grade live
member this project could not otherwise get: GenCast, GraphCast, Pangu and FuXi
have no live public feed (probed 2026-08-10 -- see the note in
scripts/fetch_dynamical.py), and the one that does, ECMWF IFS ENS, costs ~37 GB
per init through dynamical.org's chunk layout.

Output matches scripts/fetch_dynamical.py exactly -- leaflet-velocity JSON at
`weathernext_live_latest_<lead>.json` on the same 180x90 2-degree grid -- so
scripts/blend_live.py picks it up as another `kind: "live"` source with no
changes. Its grid and init-time guards then apply automatically.

Three modes, in the order they should be used:

    python scripts/fetch_weathernext.py --probe      # what is actually there
    python scripts/fetch_weathernext.py --estimate   # what would it cost
    python scripts/fetch_weathernext.py --fetch      # do it

**--probe first, always.** I do not know WeatherNext 2's dataset, table or
column names, and hardcoding a guess is precisely the mistake that cost four
Kaggle runs on the CMIP filename format. Probe reads metadata only, costs
nothing, and prints the real names to build the query from.

**--estimate before --fetch.** WeatherNext tables are large and a careless query
scans terabytes against the caller's project. Every query is dry-run first and
the fetch refuses above --max-gb unless overridden. Filters on init and lead go
first so partition and cluster pruning actually happens.

Credentials: standard Google application-default lookup, so either
GOOGLE_APPLICATION_CREDENTIALS pointing at a service-account JSON, or
GOOGLE_APPLICATION_CREDENTIALS_JSON holding the body (written to a temp file
here). Needs BigQuery Job User on the project plus read on the dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile

OUT_DIR = pathlib.Path("viewer/data")
SOURCE_ID = "weathernext_live"
LABEL = "Google WeatherNext (frontier ML) — live"
LEADS = [0, 24, 48, 72, 96, 120]
# the viewer grid the other live sources already use; blend_live.py requires a
# match, so these are not free parameters
NX, NY = 180, 90
DX = DY = 2.0
LAT0, LON0 = 89.125, 0.875


def credentials_hint() -> str:
    return (
        "No Google credentials found. Set one of:\n"
        "  GOOGLE_APPLICATION_CREDENTIALS      = /path/to/service-account.json\n"
        "  GOOGLE_APPLICATION_CREDENTIALS_JSON = <the JSON body>\n"
        "with BigQuery Job User on the project and read on the WeatherNext "
        "dataset. As an environment secret it takes effect in a NEW session."
    )


def ensure_credentials() -> None:
    """Accept the JSON body as well as a path -- environment secrets carry bodies."""
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    body = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not body:
        sys.exit(credentials_hint())
    try:
        json.loads(body)
    except ValueError:
        sys.exit("GOOGLE_APPLICATION_CREDENTIALS_JSON is not valid JSON")
    fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    fh.write(body)
    fh.close()
    os.chmod(fh.name, 0o600)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = fh.name


def client(project: str | None):
    try:
        from google.cloud import bigquery
    except ImportError:
        sys.exit("pip install google-cloud-bigquery  (not installed here yet)")
    ensure_credentials()
    return bigquery.Client(project=project) if project else bigquery.Client()


def probe(project: str | None, pattern: str) -> None:
    """Metadata only: which datasets/tables/columns exist, and the newest init.

    Prints rather than returns, because the point is to put the real names in
    front of a human before any query is written against them.
    """
    bq = client(project)
    print(f"windml project={bq.project}", flush=True)

    hits = []
    for ds in bq.list_datasets():
        did = ds.dataset_id
        if pattern.lower() not in did.lower():
            continue
        print(f"windml dataset={did}", flush=True)
        for tbl in bq.list_tables(ds.reference):
            hits.append(f"{did}.{tbl.table_id}")
            print(f"windml   table={tbl.table_id}", flush=True)

    if not hits:
        print(f"windml no dataset matching {pattern!r} is visible to these "
              f"credentials. WeatherNext may be shared as a linked dataset from "
              f"another project -- pass --project or --dataset explicitly.",
              flush=True)
        return

    ref = bq.get_table(hits[0])
    print(f"windml schema of {hits[0]} ({ref.num_rows:,} rows):", flush=True)
    for f in ref.schema:
        print(f"windml   {f.name:38s} {f.field_type}"
              f"{' REPEATED' if f.mode == 'REPEATED' else ''}", flush=True)


def dry_run(bq, sql: str) -> float:
    """Bytes this query would bill, in GB, without running it."""
    from google.cloud import bigquery
    cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    job = bq.query(sql, job_config=cfg)
    return job.total_bytes_processed / 1e9


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--probe", action="store_true")
    p.add_argument("--estimate", action="store_true")
    p.add_argument("--fetch", action="store_true")
    p.add_argument("--project", default=os.environ.get("WINDML_GCP_PROJECT"))
    p.add_argument("--pattern", default="weathernext",
                   help="substring used to find the dataset during --probe")
    p.add_argument("--table", default=None,
                   help="fully-qualified table, once --probe has told you its name")
    p.add_argument("--max-gb", type=float, default=50.0,
                   help="refuse a query that would bill more than this")
    p.add_argument("--force", action="store_true",
                   help="run even above --max-gb (say why in the commit)")
    p.add_argument("--out", default=str(OUT_DIR))
    a = p.parse_args()

    if not (a.probe or a.estimate or a.fetch):
        sys.exit("pick one of --probe / --estimate / --fetch (probe first)")

    if a.probe:
        probe(a.project, a.pattern)
        return

    if not a.table:
        sys.exit("--table is required for --estimate/--fetch; run --probe first "
                 "to learn the real name rather than guessing it")

    # The query is written only once --probe has shown the column names, so it
    # lives here as a template with the discovered identifiers substituted in
    # rather than as a guess baked into the file.
    sys.exit(
        "The extraction query is intentionally not written yet.\n\n"
        "--probe has to run against real credentials first and print the actual\n"
        "column names (init/valid time, lead, ensemble member, the 10 m wind\n"
        "components) and the partitioning. Writing SQL against guessed columns\n"
        "is how the CMIP prep burned four runs on a filename assumption.\n\n"
        "Run:  python scripts/fetch_weathernext.py --probe --project treesixty\n"
        "and the query gets written to match what it reports."
    )


if __name__ == "__main__":
    main()
