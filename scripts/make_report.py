"""Aggregate result CSVs into RESULTS.md tables + figures.

Usage: python scripts/make_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from windml.config import ARTIFACTS, REPO_ROOT

RESULTS = ARTIFACTS / "results"
FIGURES = ARTIFACTS / "figures"

HEADLINE_VARS = ["u10", "v10", "wind_speed", "u850", "v850", "z500", "t2m"]
HEADLINE_LEADS = [24, 72, 120]
UNITS = {
    "u10": "m/s", "v10": "m/s", "wind_speed": "m/s", "u850": "m/s", "v850": "m/s",
    "t850": "K", "z500": "m²/s²", "t2m": "K", "msl": "Pa",
}
# display order: baselines, our models, then competitors
ORDER_HINT = [
    # baselines
    "persistence", "climatology", "linear",
    # our from-scratch models (1-step, then rollout fine-tuned)
    "unet", "vit", "afno", "graph",
    "vit_ft2", "unet_ft2", "unet_ft4",
    # published frontier forecasts, regridded by WB2 to 64x32
    "hres_2020", "pangu_2020", "graphcast_2020", "gencast_mean_2020",
    # post-processing / combinations of those forecasts
    "graphcast_corrected", "graphcast_affine",
    "blend_graphcast+pangu+hres", "avg4",
]

GROUPS = {
    "baseline": ["persistence", "climatology", "linear"],
    "ours (from scratch, 4 CPU cores, 5.625 deg)":
        ["unet", "vit", "afno", "graph", "vit_ft2", "unet_ft2", "unet_ft4"],
    "published frontier forecasts (0.25 deg, regridded)":
        ["hres_2020", "pangu_2020", "graphcast_2020", "gencast_mean_2020"],
    "ours: post-processing / combination of the above":
        ["graphcast_corrected", "graphcast_affine",
         "blend_graphcast+pangu+hres", "avg4"],
}


def rt2021_section() -> list[str]:
    """The faithful RT2021 recreation, scored on the paper's own 2017-2018.

    Deliberately NOT folded into the tables below. Those are 2020; this is
    2017-2018, because that is the only way its numbers mean anything next to
    the paper's. Putting them in one table would invite exactly the comparison
    that is invalid.
    """
    paths = sorted(RESULTS.glob("rt2021_*_scores.json"))
    if not paths:
        return []
    # one file per lead, each already grouped by lead inside; merge so the
    # section grows as leads land rather than needing an edit per run
    entries = []
    for path in paths:
        entries += json.loads(path.read_text()).get("by_lead", [])
    data = {"by_lead": sorted(entries, key=lambda e: e["lead_h"])}
    out = ["", "## The Rasp & Thuerey 2021 recreation (scored on 2017-2018)", "",
           "Separate from every table below, which is scored on 2020. This model",
           "is scored on the paper's own test years so the comparison is",
           "like-for-like: same 64x32 grid, same latitude-weighted RMSE.", ""]
    for entry in data.get("by_lead", []):
        lead, r = entry["lead_h"], entry["rmse"]
        ref = entry.get("paper") or {}
        era, pre = ref.get("era5_only", {}), ref.get("pretrained", {})
        n = entry.get("n_members", 1)
        label = f"ensemble of {n}" if n > 1 else "single model"
        out += [f"### {lead} h lead ({label}, {entry.get('n_inits')} inits)", "",
                "| variable | ours | R&T ERA5-only | vs | R&T CMIP6-pretrained |",
                "|---|---|---|---|---|"]
        for v in ("z500", "t850", "t2m"):
            if v not in r:
                continue
            e, p = era.get(v), pre.get(v)
            delta = f"{100 * (r[v] - e) / e:+.1f}%" if e else "--"
            out.append(f"| {v} | **{r[v]:.2f}** | {e if e else '--'} | {delta} "
                       f"| {p if p else '--'} |")
        out.append("")
    out += ["Their ERA5-only row is the comparable one -- this model does no",
            "pretraining. The pretrained column is shown because it is the number",
            "usually quoted, and it costs ~150 years of CMIP6 data to reach.", ""]
    return out


def sharpness_section() -> list[str]:
    """The realism metrics, and the recalibration that trades RMSE for them.

    Kept out of the RMSE tables on purpose. Every one of those is a squared-error
    number and the whole point here is that squared error cannot see the defect
    being measured, so mixing the two columns would bury the finding in the
    thing it contradicts.
    """
    sharp, recal = RESULTS / "sharpness.csv", RESULTS / "spectral_recalibration.csv"
    if not sharp.exists():
        return []
    out = ["", "## Sharpness: what RMSE cannot see", "",
           "An RMSE-optimal forecast is the conditional mean, so every model",
           "below is blurred by construction. For wind that matters because",
           "power goes as v^3. `spec` is the share of high-wavenumber power the",
           "forecast retains against ERA5 (1.0 = right), and `cf_bias` is the",
           "capacity-factor error through a turbine power curve.", ""]
    d = pd.read_csv(sharp)
    for lead in sorted(d.lead_h.unique()):
        sub = d[d.lead_h == lead].sort_values("ws_rmse")
        out += [f"### {lead} h", "",
                "| model | ws RMSE | var ratio | spec | 95th-pct bias | cf bias |",
                "|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            out.append(f"| {r.model} | {r.ws_rmse:.3f} | {r.ws_var_ratio:.3f} "
                       f"| {r.ws_spec_ratio:.3f} | {r.ws_p95_bias:+.3f} "
                       f"| {r.cf_bias:+.4f} |")
        out.append("")
    out += ["The ordering is close to an inversion of the RMSE ordering: the",
            "sharpest forecast is the physics model, which is last on RMSE.", ""]

    if recal.exists():
        r = pd.read_csv(recal)
        out += ["### Spectral recalibration", "",
                "One amplification factor per zonal wavenumber per lead, fitted",
                "on held-out inits. RMSE is shown alongside because sharpening",
                "always costs it -- reporting only the improved column would be",
                "the trick this section exists to avoid.", "",
                "| model | variant | lead | ws RMSE | spec | cf bias |",
                "|---|---|---|---|---|---|"]
        for _, x in r.iterrows():
            out.append(f"| {x.model} | {x.variant} | {x.lead_h} h | "
                       f"{x.ws_rmse:.3f} | {x.ws_spec_ratio:.3f} | {x.cf_bias:+.4f} |")
        out += ["", "HRES is the negative control: it is already sharp, so the",
                "correction has nothing to restore and makes it worse. A",
                "post-processor that improved every model equally would be a",
                "metric artefact rather than a physical correction.", ""]
    return out


def load_all(pattern: str = "*_test.csv") -> pd.DataFrame:
    frames = [pd.read_csv(csv) for csv in sorted(RESULTS.glob(pattern))]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "n_inits" in df:  # drop leads a model does not produce (12-hourly models)
        df = df[df.n_inits > 0]
    order = {name: i for i, name in enumerate(ORDER_HINT)}
    df["order"] = df.model.map(lambda m: order.get(m, 99))
    return df.sort_values(["order", "model"])


def acc_table(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    sub = df[(df.variable == variable) & (df.lead_h.isin(HEADLINE_LEADS))]
    tab = sub.pivot_table(index=["order", "model"], columns="lead_h", values="acc")
    tab.index = tab.index.droplevel(0)
    return tab.round(3)


def skill_summary(df: pd.DataFrame, reference: str = "graphcast_2020") -> pd.DataFrame:
    """Percent RMSE change vs a reference model (negative = better)."""
    if reference not in set(df.model):
        return pd.DataFrame()
    ref = df[df.model == reference].set_index(["variable", "lead_h"]).rmse
    rows = []
    for model, grp in df.groupby("model", sort=False):
        if model == reference:
            continue
        g = grp.set_index(["variable", "lead_h"]).rmse
        common = g.index.intersection(ref.index)
        if not len(common):
            continue
        rel = (g.loc[common] / ref.loc[common] - 1.0) * 100
        rec = {"model": model}
        for var in ["u10", "v10", "wind_speed"]:
            for lead in HEADLINE_LEADS:
                if (var, lead) in rel.index:
                    rec[f"{var}@{lead}h"] = round(float(rel.loc[(var, lead)]), 1)
        rows.append(rec)
    return pd.DataFrame(rows).set_index("model")


def rmse_table(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    sub = df[(df.variable == variable) & (df.lead_h.isin(HEADLINE_LEADS))]
    tab = sub.pivot_table(index=["order", "model"], columns="lead_h", values="rmse")
    tab.index = tab.index.droplevel(0)
    return tab.round(3)


def _style(model: str) -> dict:
    """Colour by group so the three tiers read at a glance."""
    for group, members in GROUPS.items():
        if model in members:
            if group.startswith("baseline"):
                return {"color": "0.6", "ls": ":", "lw": 1.2}
            if group.startswith("ours (from"):
                return {"color": "tab:orange", "ls": "--", "lw": 1.2}
            if group.startswith("published"):
                return {"color": "tab:blue", "ls": "-", "lw": 1.2}
            return {"color": "tab:red", "ls": "-", "lw": 2.0}
    return {"color": "0.3", "ls": "-", "lw": 1.0}


def curves_figure(df: pd.DataFrame, variables: list[str], out: Path) -> None:
    fig, axes = plt.subplots(1, len(variables), figsize=(5 * len(variables), 4.2))
    for ax, var in zip(axes, variables):
        sub = df[df.variable == var]
        for model, grp in sub.groupby("model", sort=False):
            grp = grp.sort_values("lead_h")
            ax.plot(grp.lead_h, grp.rmse, label=model, marker=".", ms=3, **_style(model))
        ax.set_title(var)
        ax.set_xlabel("lead time (h)")
        ax.set_ylabel(f"RMSE ({UNITS.get(var, '')})")
        ax.grid(alpha=0.3)
    axes[-1].legend(fontsize=6.5, loc="upper left", ncol=2)
    fig.suptitle(
        "grey = baselines · orange = our CPU-scale models · "
        "blue = published frontier forecasts · red = our combinations",
        fontsize=8, y=0.02,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def frontier_figure(df: pd.DataFrame, out: Path) -> None:
    """Zoom on the frontier models + our combinations for 10m wind speed."""
    keep = GROUPS["published frontier forecasts (0.25 deg, regridded)"] + [
        "blend_graphcast+pangu+hres", "avg4", "graphcast_corrected",
    ]
    sub = df[(df.variable == "wind_speed") & df.model.isin(keep)]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for model, grp in sub.groupby("model", sort=False):
        grp = grp.sort_values("lead_h")
        ax.plot(grp.lead_h, grp.rmse, label=model, marker="o", ms=3, **_style(model))
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("10m wind speed RMSE (m/s)")
    ax.set_title("Beating the frontier models on 2020 wind (64x32, ERA5 truth)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    df = load_all()
    lines = [
        "# Results",
        "",
        "*Generated by `scripts/make_report.py`. Latitude-weighted RMSE on 2020,",
        "00/12 UTC init times, verified against ERA5 at 64x32 (5.625 deg).*",
        "",
        "## How to read these tables",
        "",
        "Every row is scored by the same code on the same init times and truth.",
        "The rows fall into four groups:",
        "",
        "| group | what it is |",
        "|---|---|",
        "| baselines | persistence, climatology, per-pixel ridge regression |",
        "| ours (from scratch) | U-Net / ViT / AFNO / mesh-GNN trained here on 4 CPU "
        "cores at 5.625 deg, `_ft2`/`_ft4` = rollout fine-tuned |",
        "| published frontier forecasts | the actual GraphCast / GenCast / Pangu / HRES "
        "forecasts, produced at 0.25 deg and regridded to 64x32 by WeatherBench 2 |",
        "| ours: combinations | post-processing and blends built *on top of* those "
        "published forecasts |",
        "",
        "Our from-scratch models are **not** a like-for-like test of the architectures:",
        "the frontier models used hundreds of GPU/TPU-days at 0.25 deg. Treat those rows",
        "as the reference ceiling. The comparable published anchor at our resolution is",
        "Rasp & Thuerey 2021 (ResNet, 5.625 deg). Their ERA5-only model -- the one",
        "comparable to ours, since we do no pretraining -- scores z500 RMSE 314 / 561",
        "m2/s2 at 3/5 days. Their widely-quoted 268 / 523 requires pretraining on ~150",
        "years of CMIP6, and an earlier version of this line quoted it as if it were the",
        "ERA5-only figure (with a 5-day number taken from a third, 'continuous' model).",
        "The blend rows *are* like-for-like -- they beat those models on their own",
        "published forecasts, at the resolution everything here is scored on.",
        "",
    ] + rt2021_section() + sharpness_section()
    for var in HEADLINE_VARS:
        tab = rmse_table(df, var)
        if tab.empty:
            continue
        lines.append(f"## {var} — RMSE ({UNITS.get(var, '')}) at 24/72/120 h")
        lines.append("")
        lines.append(tab.to_markdown())
        lines.append("")

    acc = acc_table(df, "wind_speed")
    if not acc.empty:
        lines += ["## 10m wind speed — ACC at 24/72/120 h", "", acc.to_markdown(), ""]

    skill = skill_summary(df)
    if not skill.empty:
        lines += [
            "## Wind RMSE relative to GraphCast (%, negative = better)",
            "",
            skill.to_markdown(),
            "",
        ]

    crps = load_all("*_crps.csv")
    if not crps.empty:
        sub = crps[crps.variable.isin(["u10", "v10", "wind_speed"])
                   & crps.lead_h.isin(HEADLINE_LEADS)]
        tab = sub.pivot_table(index="model", columns=["variable", "lead_h"], values="crps")
        lines += ["## Probabilistic: wind CRPS (m/s)", "", tab.round(3).to_markdown(), ""]

    curves_figure(df, ["u10", "wind_speed", "z500"], FIGURES / "rmse_curves.png")
    frontier_figure(df, FIGURES / "frontier_wind.png")
    lines += [
        "## Figures",
        "",
        "![RMSE curves](artifacts/figures/rmse_curves.png)",
        "",
        "![Frontier wind](artifacts/figures/frontier_wind.png)",
        "",
    ]

    (REPO_ROOT / "RESULTS.md").write_text("\n".join(lines))
    print(f"wrote {REPO_ROOT / 'RESULTS.md'} with {df.model.nunique()} models")


if __name__ == "__main__":
    main()
