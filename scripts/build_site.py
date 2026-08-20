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

import json
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


def landing() -> str:
    best = score("avg5", "wind_speed", 72)
    gc = score("graphcast_2020", "wind_speed", 72)
    fuxi = score("fuxi_2020", "wind_speed", 72)
    ours = score("unet_long_ft4", "wind_speed", 72)
    # Best from-scratch z500, whichever model holds it -- naming one hardcodes
    # a result. anchor72 held it at 387.4, then resnet72_big at 378.5, then the
    # faithful RT2021 recreation at 306.7, which is the first to pass the
    # ERA5-only anchor. A hardcoded card would still be showing 387.
    anchor_runs = [(score(m, "z500", 72), m) for m in
                   ("anchor72", "levels72", "resnet72", "resnet72_big")]
    anchor_hits = [(v, m) for v, m in anchor_runs if v is not None]
    rt = rt2021_score("z500")
    if rt is not None:
        anchor_hits.append((rt, "RT2021 recreation"))
    anchor, anchor_model = min(anchor_hits) if anchor_hits else (None, None)
    beat = anchor is not None and anchor < 314.0
    ref = min(v for v in (gc, fuxi) if v is not None) if (gc or fuxi) else None
    gain = (best / ref - 1) * 100 if best and ref else None

    cards = [
        ("Best forecast (ensemble)", f"{best:.3f}" if best else "—",
         "10m wind speed RMSE @72h, m/s", True),
        ("vs best single model", f"{gain:+.1f}%" if gain else "—",
         f"beats FuXi/GraphCast ({ref:.3f})" if ref else "", True),
        ("Best from-scratch model", f"{ours:.3f}" if ours else "—",
         "our U-Net, 4 CPU cores", False),
        # 314 is Rasp & Thuerey's ERA5-only figure; their 268 is the
        # CMIP6-pretrained one and is not what an ERA5-only model should be
        # measured against
        ("Literature anchor (z500 @72h)", f"{anchor:.0f} vs 314" if anchor else "—",
         (f"beats the ERA5-only anchor — {anchor_model}" if beat else
          f"not reached — {anchor_model}, ERA5-only anchor") if anchor else
         "not reached — see Research", beat),
    ]
    stat_html = "".join(
        f'<div class=stat><div class="v{" win" if win else ""}">{v}</div>'
        f'<div class=k>{k}</div></div>'
        for k, v, _s, win in cards)

    # The map IS the page. It was previously one click behind a wall of text,
    # which buried the only part of this project that is worth just looking at.
    # An iframe rather than promoting viewer/index.html to the site root:
    # the viewer fetches data/ and vendor/ by RELATIVE path, so moving it means
    # moving both trees and rewriting those fetches -- real breakage risk for a
    # cosmetic gain. This keeps the viewer self-contained and still linkable at
    # /viewer/.
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content='width=device-width,initial-scale=1'>
<title>windplayground — live global wind</title>
<style>
{CSS}
html,body{{height:100%;overflow:hidden}}
.app{{display:flex;flex-direction:column;height:100vh}}
.topbar{{flex:0 0 auto;display:flex;align-items:center;gap:18px;flex-wrap:wrap;
  padding:10px 18px;background:var(--panel);border-bottom:1px solid var(--line)}}
.topbar .brand{{font-weight:700;letter-spacing:.02em;white-space:nowrap}}
.topbar .tag{{color:var(--muted);font-size:13px;white-space:nowrap}}
.stats{{display:flex;gap:16px;margin-left:auto;flex-wrap:wrap}}
.stat{{text-align:right;line-height:1.15}}
.stat .v{{font-size:17px;font-weight:700;font-variant-numeric:tabular-nums}}
.stat .k{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}}
.stat .v.win{{color:var(--accent)}}
.links{{display:flex;gap:10px;white-space:nowrap}}
.links a{{font-size:13px;padding:6px 11px;border:1px solid var(--line);
  border-radius:8px;color:var(--text);text-decoration:none}}
.links a:hover{{border-color:var(--accent);color:var(--accent)}}
frame-wrap{{flex:1 1 auto;min-height:0}}
iframe{{width:100%;height:100%;border:0;display:block}}
@media (max-width:760px){{.stats{{display:none}}}}
</style></head><body>
<div class=app>
  <div class=topbar>
    <span class=brand>windplayground</span>
    <span class=tag>10&thinsp;m wind &middot; live operational runs, frontier models,
      and every model we trained</span>
    <div class=stats>{stat_html}</div>
    <div class=links>
      <a href="report.html">Research</a>
      <a href="results.html">Results</a>
      <a href="https://github.com/andrewnakas/windplayground-">GitHub</a>
    </div>
  </div>
  <frame-wrap style="display:block"><iframe src="viewer/index.html"
    title="Interactive global wind map" loading="eager"></iframe></frame-wrap>
</div>
</body></html>
"""


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
