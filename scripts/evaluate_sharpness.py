"""Score every forecast set on the sharpness diagnostics, not just RMSE.

The blend already beats each frontier model on RMSE (z500 98.4 against
GraphCast's 112.9). This asks the question RMSE cannot: are any of these
forecasts actually *sharp*, or are they all the conditional mean?

They are all trained or tuned to minimise squared error, so the expectation is
that every one of them is under-dispersed, increasingly so with lead time, and
that averaging them -- which is what makes the blend win on RMSE -- makes the
blur strictly worse. If that holds, "beats GraphCast by 12%" and "is a worse
description of the wind field" are both true of the same forecast, and only one
of them is currently reported.

    python scripts/evaluate_sharpness.py
    python scripts/evaluate_sharpness.py --models graphcast_2020 avg4 --leads 24 72

Writes artifacts/results/sharpness.csv.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from windml.config import ARTIFACTS, CHANNELS, DataConfig
from windml.data.build_cache import load_statics, load_years
from windml.data.competitors import CompetitorForecaster, competitor_path
from windml.eval.metrics import (
    capacity_factor_error,
    extreme_scores,
    rmse,
    spectral_ratio,
    variance_ratio,
    wind_speed,
)
from windml.utils.grid import latitude_weights

U10, V10 = CHANNELS.index("u10"), CHANNELS.index("v10")
Z500 = CHANNELS.index("z500")
OUT = ARTIFACTS / "results" / "sharpness.csv"

# the members the blend averages; kept in one place so the blend scored here is
# the same object the RMSE tables describe
BLEND_MEMBERS = ["graphcast_2020", "pangu_2020", "hres_2020", "gencast_mean_2020"]
DEFAULT_MODELS = BLEND_MEMBERS + ["fuxi_2020", "avg4"]


def gather(cfg, name, year, inits, leads_idx):
    """(n_lead, n_init, C, H, W) forecasts, or None if not cached."""
    if not competitor_path(cfg, name, year).exists():
        return None
    fc = CompetitorForecaster(cfg, name, year)
    kmax = max(leads_idx)
    out = []
    for t0 in inits:
        f = fc.forecast(t0, kmax)
        out.append(np.stack([f[k - 1] for k in leads_idx]))
    return np.stack(out, axis=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--leads", nargs="+", type=int, default=[24, 72, 120])
    p.add_argument("--init-stride", type=int, default=8,
                   help="every Nth init; the spectrum is stable long before "
                        "all 1460 of them")
    p.add_argument("--year", type=int, default=2020)
    a = p.parse_args()

    cfg = DataConfig()
    statics = load_statics(cfg)
    lat_w = latitude_weights(statics["latitude"])
    truth = np.asarray(load_years(cfg, [a.year]), dtype=np.float32)

    leads_idx = [ld // 6 for ld in a.leads]
    kmax = max(leads_idx)
    inits = np.arange(0, truth.shape[0] - kmax, a.init_stride)
    print(f"windml inits={len(inits)} leads={a.leads} year={a.year}")

    cache = {}
    for name in set(a.models) | set(BLEND_MEMBERS):
        if name.startswith("avg"):
            continue
        g = gather(cfg, name, a.year, inits, leads_idx)
        if g is None:
            print(f"skip {name}: not cached")
            continue
        cache[name] = g

    rows = []
    for name in a.models:
        if name.startswith("avg"):
            have = [cache[m] for m in BLEND_MEMBERS if m in cache]
            if len(have) < 2:
                print(f"skip {name}: need >=2 cached members")
                continue
            fc = np.mean(have, axis=0)
            label = f"{name} (mean of {len(have)})"
        elif name in cache:
            fc, label = cache[name], name
        else:
            continue

        for li, lead in enumerate(a.leads):
            k = leads_idx[li]
            tgt = truth[inits + k]
            pf, pt = fc[li], tgt
            ok = np.isfinite(pf).all(axis=(1, 2, 3))
            if not ok.any():
                continue
            pf, pt = pf[ok], pt[ok]

            ws_p = wind_speed(pf[:, U10], pf[:, V10])
            ws_t = wind_speed(pt[:, U10], pt[:, V10])
            ex = extreme_scores(ws_p, ws_t, lat_w, 95.0)
            cf = capacity_factor_error(ws_p, ws_t, lat_w)
            rows.append({
                "model": label, "lead_h": lead, "n_inits": int(ok.sum()),
                "ws_rmse": round(float(rmse(ws_p, ws_t, lat_w)), 4),
                "ws_var_ratio": round(variance_ratio(ws_p, ws_t, lat_w), 4),
                "ws_spec_ratio": round(spectral_ratio(ws_p, ws_t, lat_w), 4),
                "ws_p95_rmse": round(ex["rmse"], 4),
                "ws_p95_bias": round(ex["bias"], 4),
                "cf_bias": round(cf["bias"], 5),
                "cf_rmse": round(cf["rmse"], 5),
                "z500_var_ratio": round(
                    variance_ratio(pf[:, Z500], pt[:, Z500], lat_w), 4),
                "z500_spec_ratio": round(
                    spectral_ratio(pf[:, Z500], pt[:, Z500], lat_w), 4),
            })
            print(f"windml {label:28s} @{lead:3d}h  ws_rmse={rows[-1]['ws_rmse']:.3f}  "
                  f"var={rows[-1]['ws_var_ratio']:.3f}  "
                  f"spec={rows[-1]['ws_spec_ratio']:.3f}  "
                  f"p95_bias={rows[-1]['ws_p95_bias']:+.3f}  "
                  f"cf_bias={rows[-1]['cf_bias']:+.4f}", flush=True)

    if not rows:
        raise SystemExit("no models scored -- are the competitor caches present?")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
