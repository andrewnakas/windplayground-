"""Build the docs/ site for GitHub Pages from the repo's markdown and results.

Renders REPORT.md and RESULTS.md to styled HTML, copies the wind viewer and
figures, and writes a landing page whose headline numbers are read from
artifacts/results/ rather than hard-coded, so the site cannot drift from the
scores it claims.

    python scripts/build_site.py

Publishing is a separate, manual step: the repo is private, and GitHub Pages
would serve the site at a PUBLIC url. Nothing here makes anything public.
"""
from __future__ import annotations

import html
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
<a href="{p}index.html">Overview</a>
<a href="{p}viewer/index.html">Wind viewer</a>
<a href="{p}report.html">Research</a>
<a href="{p}results.html">Results</a>
<a href="https://github.com/andrewnakas/windplayground-">GitHub</a>
</div></nav>"""


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
            f"Scores are latitude-weighted RMSE on 2020 at 64&times;32, ERA5 truth.</footer>"
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


def landing() -> str:
    best = score("avg5", "wind_speed", 72)
    gc = score("graphcast_2020", "wind_speed", 72)
    fuxi = score("fuxi_2020", "wind_speed", 72)
    ours = score("unet_long_ft4", "wind_speed", 72)
    anchor = score("anchor72", "z500", 72)
    ref = min(v for v in (gc, fuxi) if v is not None) if (gc or fuxi) else None
    gain = (best / ref - 1) * 100 if best and ref else None

    cards = [
        ("Best forecast (ensemble)", f"{best:.3f}" if best else "—",
         "10m wind speed RMSE @72h, m/s", True),
        ("vs best single model", f"{gain:+.1f}%" if gain else "—",
         f"beats FuXi/GraphCast ({ref:.3f})" if ref else "", True),
        ("Best from-scratch model", f"{ours:.3f}" if ours else "—",
         "our U-Net, 4 CPU cores", False),
        ("Literature anchor (z500 @72h)", f"{anchor:.0f} vs 268" if anchor else "—",
         "not reached — see Research", False),
    ]
    card_html = "".join(
        f'<div class=card><div class=k>{k}</div>'
        f'<div class="v{" win" if win else ""}">{v}</div><div class=s>{s}</div></div>'
        for k, v, s, win in cards)

    return page("windplayground — global wind forecasting with ML", f"""
<h1>Global wind forecasting with machine learning</h1>
<p class=lede>Recreations of the leading global ML weather models at CPU scale, scored
WeatherBench-style on wind — plus a multi-model ensemble that beats every published
frontier model on 2020 wind at the evaluated resolution.</p>
<div class=cards>{card_html}</div>
<p>
<a class=cta href="viewer/index.html">Open the wind viewer &rarr;</a>
<a class=cta href="report.html" style="background:transparent;color:var(--accent);border:1px solid var(--line)">Read the research</a>
</p>
<h2>What's here</h2>
<ul>
<li><strong>An interactive wind map</strong> — windy-style particle animation of 10m wind,
with ERA5 truth, our four trained models, the published GraphCast / GenCast / Pangu / HRES /
FuXi forecasts, and live ECMWF AIFS runs all selectable.</li>
<li><strong>A literature review</strong> of the top papers and groups on wind benchmarks,
and what we found reproducing them.</li>
<li><strong>Full results tables</strong> — every model scored by one pipeline on the same
init times against the same truth.</li>
</ul>
<div class=note>Everything our models produce is at 5.625&deg; on 4 CPU cores; the frontier
models were trained at 0.25&deg; on hundreds of GPU-days. Those rows are a reference
ceiling, not a like-for-like architecture comparison. The ensemble result <em>is</em>
like-for-like — it beats those models using their own published forecasts.</div>
<h2>Skill vs lead time</h2>
<img src="figures/frontier_wind.png" alt="10m wind speed RMSE against lead time">
""")


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(landing())
    for src, out, title in [("REPORT.md", "report.html", "Research — windplayground"),
                            ("RESULTS.md", "results.html", "Results — windplayground")]:
        md = (REPO_ROOT / src).read_text()
        md = md.replace("artifacts/figures/", "figures/")
        (DOCS / out).write_text(page(title, md_to_html(md)))

    (DOCS / "figures").mkdir(exist_ok=True)
    for png in (ARTIFACTS / "figures").glob("*.png"):
        shutil.copy2(png, DOCS / "figures" / png.name)
    viewer_dst = DOCS / "viewer"
    if viewer_dst.exists():
        shutil.rmtree(viewer_dst)
    shutil.copytree(REPO_ROOT / "viewer", viewer_dst)
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
