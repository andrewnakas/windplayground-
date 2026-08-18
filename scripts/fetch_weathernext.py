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
    return f"{credentials_hint()}\n\n({exc})"


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
        except Exception:      # noqa: S110 -- identity is a nicety, never a gate
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


def client(project: str | None, impersonate: str | None = None):
    """Returns (client, principal, impersonating) -- who matters as much as what."""
    try:
        from google.cloud import bigquery
    except ImportError:
        sys.exit("pip install google-cloud-bigquery  (not installed here yet)")
    ensure_credentials()
    creds, impersonating = build_credentials(impersonate)
    bq = (bigquery.Client(project=project, credentials=creds) if project
          else bigquery.Client(credentials=creds))
    return bq, principal_of(creds, bq), impersonating


def probe(project: str | None, pattern: str, impersonate: str | None = None) -> None:
    """Metadata only: which datasets/tables/columns exist, and the newest init.

    Prints rather than returns, because the point is to put the real names in
    front of a human before any query is written against them -- starting with
    the principal, since an empty listing means nothing until you know who did
    the looking.
    """
    bq, principal, impersonating = client(project, impersonate)
    print(f"windml acting as {principal}"
          + (" (impersonated)" if impersonating else "")
          + f", project={bq.project}", flush=True)

    hits = []
    for ds in guarded(lambda: list(bq.list_datasets()), principal, impersonating):
        did = ds.dataset_id
        if pattern.lower() not in did.lower():
            continue
        print(f"windml dataset={did}", flush=True)
        for tbl in bq.list_tables(ds.reference):
            hits.append(f"{did}.{tbl.table_id}")
            print(f"windml   table={tbl.table_id}", flush=True)

    if not hits:
        print(f"windml no dataset matching {pattern!r} is visible to "
              f"{principal}. Two different causes look identical here: it may be "
              f"shared as a linked dataset from another project (pass --project "
              f"or --table explicitly), or the grant may be on a service account "
              f"rather than on this principal (re-run with --impersonate).",
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
    p.add_argument("--impersonate", default=None, metavar="SA_EMAIL",
                   help="act as this service account -- use it when WeatherNext "
                        "was granted to a service account rather than to the "
                        "principal these credentials carry. Needs "
                        "roles/iam.serviceAccountTokenCreator on it.")
    a = p.parse_args()

    if not (a.probe or a.estimate or a.fetch):
        sys.exit("pick one of --probe / --estimate / --fetch (probe first)")

    if a.probe:
        probe(a.project, a.pattern, a.impersonate)
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
        "      (add --impersonate SA_EMAIL if the grant is on a service account)\n"
        "and the query gets written to match what it reports."
    )


if __name__ == "__main__":
    main()
