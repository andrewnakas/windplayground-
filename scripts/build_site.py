"""Build the docs/ site for GitHub Pages from the repo's markdown and results.

Renders REPORT.md and RESULTS.md to styled HTML and copies the wind viewer's
code files to the docs root, where the viewer IS the landing page. Data is not
copied: docs/data is its single home, written directly by the export/fetch
scripts, and this build must never delete it.

    python scripts/build_site.py

Publishing is a separate, manual step: the repo is private, and GitHub Pages
would serve the site at a PUBLIC url. Nothing here makes anything public.
"""
from __future__ import annotations

import json
import html
from datetime import datetime, timezone
import re
import shutil

import pandas as pd

from windml.config import ARTIFACTS, REPO_ROOT

DOCS = REPO_ROOT / "docs"
RESULTS_DIR = ARTIFACTS / "results"

CSS = """
:root{--bg:#0b1020;--panel:#141b30;--line:#26304d;--text:#e8ecf7;--muted:#93a0c0;--accent:#58a6ff;--good:#4ade80}
@media (prefers-color-scheme: light){:root{--bg:#f7f9fc;--panel:#fff;--line:#dde3ee;--text:#131722;--muted:#5b6478;--accent:#0b62d0;--good:#137a3d}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:16px/1.65 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:32px 20px 80px}
nav{position:sticky;top:0;background:color-mix(in srgb,var(--bg) 92%,transparent);backdrop-filter:blur(8px);border-bottom:1px solid var(--line);z-index:10}
nav .wrap{padding:12px 20px;display:flex;gap:18px;align-items:center;flex-wrap:wrap}
nav a{color:var(--muted);text-decoration:none;font-size:14px}
nav a:hover,nav a.on{color:var(--accent)}
nav .brand{font-weight:700;color:var(--text);margin-right:auto}
h1{font-size:30px;line-height:1.25;margin:.4em 0 .3em}
h2{font-size:22px;margin:1.8em 0 .5em;padding-top:.3em;border-top:1px solid var(--line)}
h3{font-size:17px;margin:1.4em 0 .4em}
a{color:var(--accent)}
code{background:var(--panel);padding:.15em .4em;border-radius:4px;font-size:.9em}
pre{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;overflow-x:auto}
pre code{background:none;padding:0}
table{border-collapse:collapse;width:100%;margin:1em 0;font-size:14px;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{background:var(--panel);position:sticky;top:0}
tbody tr:nth-child(odd){background:color-mix(in srgb,var(--panel) 45%,transparent)}
blockquote{border-left:3px solid var(--accent);margin:1em 0;padding:.1em 1em;color:var(--muted)}
img{max-width:100%;border-radius:10px;border:1px solid var(--line)}
.lede{color:var(--muted);font-size:17px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;margin:1.5em 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
.card .k{font-size:12px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}
.card .v{font-size:26px;font-weight:700;font-variant-numeric:tabular-nums;margin:.15em 0}
.card .s{font-size:13px;color:var(--muted)}
.win{color:var(--good)}
.cta{display:inline-block;background:var(--accent);color:#05070f;font-weight:600;padding:10px 18px;border-radius:9px;text-decoration:none;margin:.4em .5em .4em 0}
.note{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:8px;padding:12px 16px;font-size:14px;color:var(--muted);margin:1.2em 0}
footer{margin-top:60px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}
"""

NAV = """<nav><div class="wrap">
<span class="brand">windplayground</span>
<a href="{p}index.html">Wind map</a>
<a href="{p}report.html">Research</a>
<a href="{p}results.html">Results</a>
<a href="https://github.com/andrewnakas/windplayground-">GitHub</a>
</div></nav>"""

# The viewer's code files, copied verbatim to the docs root. The map IS the
# landing page; data/ and vendor/ sit beside it so its relative fetches work
# unchanged. Listed explicitly so a stray file in viewer/ never ships.
VIEWER_FILES = ("index.html", "styles.css", "app.js", "raster.js", "units.js",
                "meteogram.js")

REDIRECT_STUB = ("<!doctype html><meta charset=utf-8>"
                 "<meta http-equiv=refresh content='0;url=../'>"
                 "<a href='../'>The wind viewer moved to the site root.</a>")


def md_to_html(md: str) -> str:
    """Small markdown renderer: enough for our reports, no dependency."""
    out, lines, i = [], md.split("\n"), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):                      # fenced code
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(html.escape(lines[i])); i += 1
            out.append("<pre><code>" + "\n".join(buf) + "</code></pre>")
        elif ln.startswith("|") and i + 1 < len(lines) and set(lines[i + 1]) <= set("|-: "):
            head = [c.strip() for c in ln.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")]); i += 1
            th = "".join(f"<th>{inline(c)}</th>" for c in head)
            tb = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
                         for r in rows)
            out.append(f"<table><thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table>")
            continue
        elif re.match(r"^#{1,4} ", ln):
            n = len(ln) - len(ln.lstrip("#"))
            out.append(f"<h{n}>{inline(ln[n:].strip())}</h{n}>")
        elif ln.startswith("> "):
            out.append(f"<blockquote>{inline(ln[2:])}</blockquote>")
        elif re.match(r"^[-*] ", ln):
            items = []
            while i < len(lines) and re.match(r"^[-*] ", lines[i]):
                items.append(f"<li>{inline(lines[i][2:])}</li>"); i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif re.match(r"^\d+\. ", ln):
            items = []
            while i < len(lines) and re.match(r"^\d+\. ", lines[i]):
                text = re.sub(r"^\d+\. ", "", lines[i])
                items.append("<li>" + inline(text) + "</li>")
                i += 1
            out.append("<ol>" + "".join(items) + "</ol>")
            continue
        elif ln.strip() in ("---", "***"):
            out.append("<hr>")
        elif ln.strip():
            buf = []
            while i < len(lines) and lines[i].strip() and not re.match(
                    r"^(#{1,4} |[-*] |\d+\. |\||```|> )", lines[i]):
                buf.append(lines[i]); i += 1
            out.append("<p>" + inline(" ".join(buf)) + "</p>")
            continue
        i += 1
    return "\n".join(out)


def inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2">', s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", s)
    return s


def page(title: str, body: str, depth: int = 0) -> str:
    p = "../" * depth
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
            f"{NAV.format(p=p)}<div class=wrap>{body}"
            f"<footer>Generated by <code>scripts/build_site.py</code>. "
            f"Scores are latitude-weighted RMSE at 64&times;32 against ERA5. Test year "
            f"is 2020, except the RT2021 recreation, which uses that paper's "
            f"2017&ndash;2018 so its numbers are comparable to theirs.</footer>"
            f"</div></body></html>")


def score(model: str, variable: str, lead: int) -> float | None:
    f = RESULTS_DIR / f"{model}_test.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    row = d[(d.variable == variable) & (d.lead_h == lead)]
    if not len(row) or pd.isna(row.rmse.iloc[0]) or row.n_inits.iloc[0] == 0:
        return None
    return float(row.rmse.iloc[0])


def rt2021_score(variable: str, lead: int = 72) -> float | None:
    """The faithful RT2021 recreation's score, from the Kaggle eval kernel.

    Kept separate from score(): that reads artifacts/results/<model>_test.csv,
    which the local harness writes on the 2020 split, whereas this model is
    scored on the paper's 2017-2018 years by a kernel running where the data
    lives. Mixing the two files would silently compare different test periods.
    """
    path = ARTIFACTS / "results" / "rt2021_72h_scores.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        for entry in data.get("by_lead", []):
            if entry.get("lead_h") == lead:
                return float(entry["rmse"][variable])
    except (ValueError, KeyError, TypeError):
        return None
    return None


# Leads the local harness scores at; metrics.json arrays align to this list,
# with null where a model has no forecast (12-hourly models on odd leads,
# direct-72h models everywhere but 72).
LEAD_HOURS = list(range(6, 121, 6))
METRIC_VARS = ("wind_speed", "u10", "v10")

PROVENANCE = ("Latitude-weighted RMSE/ACC vs ERA5 on the 64x32 grid, 2020 test "
              "year. Live operational runs are NOT verified here: no truth "
              "exists yet for today's dates.")


def viewer_csv_stem(source_id: str) -> str | None:
    """Map a viewer source id to its results CSV (competitors carry _2020)."""
    for stem in (source_id, f"{source_id}_2020"):
        if (RESULTS_DIR / f"{stem}_test.csv").exists():
            return stem
    return None


def export_metrics() -> dict:
    """Write docs/data/metrics.json: every score the viewer shows.

    The viewer must never hard-code a number; it reads this file, and this
    file reads artifacts/results/, so the map cannot drift from the scores.
    """
    ids = {"persistence", "avg5"}          # sparkline reference curves
    man_path = DOCS / "data" / "manifest.json"
    if man_path.exists():
        ids |= {s["id"] for s in json.loads(man_path.read_text())["sources"]}

    models = {}
    for sid in sorted(ids):
        stem = viewer_csv_stem(sid)
        if stem is None:
            continue                       # live sources land here, by design
        d = pd.read_csv(RESULTS_DIR / f"{stem}_test.csv")
        entry: dict = {"csv": stem}
        for var in METRIC_VARS:
            dv = d[(d.variable == var) & (d.n_inits > 0)].dropna(subset=["rmse"])
            by_lead = {int(r.lead_h): r for r in dv.itertuples()}
            if not by_lead:
                continue
            entry[var] = {
                "rmse": [round(float(by_lead[h].rmse), 4) if h in by_lead
                         else None for h in LEAD_HOURS],
                "acc": [round(float(by_lead[h].acc), 4)
                        if h in by_lead and pd.notna(by_lead[h].acc)
                        else None for h in LEAD_HOURS],
            }
        if len(entry) > 1:
            models[sid] = entry

    sharpness: dict = {}
    sf = RESULTS_DIR / "sharpness.csv"
    if sf.exists():
        for r in pd.read_csv(sf).itertuples():
            sid = "avg4" if r.model == "avg4 (mean of 4)" else re.sub(r"_2020$", "", r.model)
            sharpness.setdefault(sid, {})[str(int(r.lead_h))] = {
                "ws_spec_ratio": round(float(r.ws_spec_ratio), 4),
                "ws_p95_bias": round(float(r.ws_p95_bias), 4),
                "cf_rmse": round(float(r.cf_rmse), 5),
            }

    best = score("avg5", "wind_speed", 72)
    single = [v for v in (score("graphcast_2020", "wind_speed", 72),
                          score("fuxi_2020", "wind_speed", 72)) if v is not None]
    ref = min(single) if single else None
    out = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": PROVENANCE,
        "lead_hours": LEAD_HOURS,
        "models": models,
        "sharpness": sharpness,
        "headline": {
            "best_blend_rmse72": round(best, 4) if best else None,
            "best_single_rmse72": round(ref, 4) if ref else None,
            "our_best_rmse72": (lambda v: round(v, 4) if v else None)(
                score("unet_long_ft4", "wind_speed", 72)),
        },
    }
    (DOCS / "data").mkdir(exist_ok=True)
    (DOCS / "data" / "metrics.json").write_text(json.dumps(out, separators=(",", ":")))
    return out


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    for src, out, title in [("REPORT.md", "report.html", "Research — windplayground"),
                            ("RESULTS.md", "results.html", "Results — windplayground")]:
        md = (REPO_ROOT / src).read_text()
        md = md.replace("artifacts/figures/", "figures/")
        (DOCS / out).write_text(page(title, md_to_html(md)))

    (DOCS / "figures").mkdir(exist_ok=True)
    for png in (ARTIFACTS / "figures").glob("*.png"):
        shutil.copy2(png, DOCS / "figures" / png.name)

    export_metrics()

    # The viewer is the landing page: copy its code files to the docs root.
    # docs/data is NOT touched here -- it is the data's only home, written by
    # the export/fetch scripts, and deleting it would take the site's fields
    # (and the live history's bounded diffs) with it.
    viewer = REPO_ROOT / "viewer"
    for name in VIEWER_FILES:
        f = viewer / name
        if f.exists():
            shutil.copy2(f, DOCS / name)
    vendor_dst = DOCS / "vendor"
    if vendor_dst.exists():
        shutil.rmtree(vendor_dst)
    shutil.copytree(viewer / "vendor", vendor_dst)

    # The viewer used to live at /viewer/; keep old links working.
    old = DOCS / "viewer"
    if old.exists():
        shutil.rmtree(old)
    old.mkdir()
    (old / "index.html").write_text(REDIRECT_STUB)

    (DOCS / ".nojekyll").touch()   # keep Jekyll from eating vendor/ and data/

    total = sum(f.stat().st_size for f in DOCS.rglob("*") if f.is_file())
    print(f"docs/ built: {total/1e6:.1f} MB")
    # Pages Source is "GitHub Actions"; .github/workflows/pages.yml runs on
    # pushes to main that touch docs/ and publishes this directory. It has to
    # live on main -- the github-pages environment only authorizes deployments
    # whose ref is the default branch, so the same workflow on a side branch is
    # rejected before its first step.
    print("To publish: commit docs/, then merge this branch to main and push.")
    print("  the pages.yml workflow deploys docs/ on every push to main")
    print("  live at https://andrewnakas.github.io/windplayground-/")


if __name__ == "__main__":
    main()
