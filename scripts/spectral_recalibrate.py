"""Put back the small scales the ML forecasts smooth away, and price the trade.

The sharpness scan found a clean inversion: the better a forecast's wind RMSE,
the blurrier it is. At 120 h our blend leads on RMSE (1.392 against HRES's
1.670) while retaining only 71% of the high-wavenumber power and
under-predicting capacity factor by 2.2 points, where HRES retains 96% and is
unbiased to 0.1 points. Every ML model sits between them; the physics model is
the sharp one.

That is not a bug in any of those models -- it is what minimising squared error
buys. It does mean a wind-energy user gets a systematically low forecast from
the model that "wins".

The fix is deliberately tiny: one amplification factor per zonal wavenumber per
lead, fitted so the forecast's mean power spectrum matches ERA5's. Around 30
numbers per lead, no network, nothing that can hallucinate structure -- it can
only restore amplitude the forecast already misplaced. k=0 is pinned to 1 so the
field mean is untouched.

Out-of-sample by construction: factors are fitted on the FIRST half of 2020 and
every number reported comes from the second half. (2018 would be a cleaner
split still, but only 2020 competitor forecasts are cached and this container
has ~3 GB of disk, so a temporal split within the year is the honest option
available rather than a silently in-sample fit.)

RMSE is expected to get WORSE -- adding variance back always costs squared
error. The output table shows both columns for exactly that reason.

    python scripts/spectral_recalibrate.py
    python scripts/spectral_recalibrate.py --models avg4 graphcast_2020
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
OUT = ARTIFACTS / "results" / "spectral_recalibration.csv"
BLEND_MEMBERS = ["graphcast_2020", "pangu_2020", "hres_2020", "gencast_mean_2020"]


def fit_amplification(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """sqrt(truth power / forecast power) per zonal wavenumber.

    Averaged over time and latitude, so it is a single spectral shape per lead
    rather than anything spatially resolved -- the deficit measured is a
    property of the model's effective resolution, not of any one location.
    """
    fp = np.abs(np.fft.rfft(pred, axis=-1)) ** 2
    ft = np.abs(np.fft.rfft(truth, axis=-1)) ** 2
    a = np.sqrt(ft.mean(axis=(0, 1)) / np.maximum(fp.mean(axis=(0, 1)), 1e-20))
    a[0] = 1.0          # never touch the zonal mean
    return a


def apply_amplification(field: np.ndarray, a: np.ndarray) -> np.ndarray:
    f = np.fft.rfft(field, axis=-1)
    return np.fft.irfft(f * a, n=field.shape[-1], axis=-1)


def gather(cfg, name, year, inits, k):
    if not competitor_path(cfg, name, year).exists():
        return None
    fc = CompetitorForecaster(cfg, name, year)
    return np.stack([fc.forecast(t0, k)[k - 1] for t0 in inits])


def score(pred_u, pred_v, truth_u, truth_v, lat_w) -> dict:
    ws_p, ws_t = wind_speed(pred_u, pred_v), wind_speed(truth_u, truth_v)
    ex = extreme_scores(ws_p, ws_t, lat_w, 95.0)
    cf = capacity_factor_error(ws_p, ws_t, lat_w)
    return {
        "ws_rmse": float(rmse(ws_p, ws_t, lat_w)),
        "ws_var_ratio": variance_ratio(ws_p, ws_t, lat_w),
        "ws_spec_ratio": spectral_ratio(ws_p, ws_t, lat_w),
        "ws_p95_bias": ex["bias"],
        "cf_bias": cf["bias"],
        "cf_rmse": cf["rmse"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["avg4", "graphcast_2020", "hres_2020"])
    p.add_argument("--leads", nargs="+", type=int, default=[72, 120])
    p.add_argument("--init-stride", type=int, default=6)
    p.add_argument("--year", type=int, default=2020)
    a = p.parse_args()

    cfg = DataConfig()
    lat_w = latitude_weights(load_statics(cfg)["latitude"])
    truth = np.asarray(load_years(cfg, [a.year]), dtype=np.float32)

    rows = []
    for lead in a.leads:
        k = lead // 6
        inits = np.arange(0, truth.shape[0] - k, a.init_stride)
        half = len(inits) // 2
        fit_i, test_i = inits[:half], inits[half:]
        print(f"\nwindml lead={lead}h fit_inits={len(fit_i)} test_inits={len(test_i)} "
              f"(fit = first half of {a.year}, test = second half)", flush=True)

        members = {}
        for m in BLEND_MEMBERS + [x for x in a.models if not x.startswith("avg")]:
            if m in members:
                continue
            g_fit, g_test = gather(cfg, m, a.year, fit_i, k), gather(cfg, m, a.year, test_i, k)
            if g_fit is None:
                continue
            members[m] = (g_fit, g_test)

        for name in a.models:
            if name.startswith("avg"):
                have = [members[m] for m in BLEND_MEMBERS if m in members]
                if len(have) < 2:
                    continue
                f_fit = np.mean([h[0] for h in have], axis=0)
                f_test = np.mean([h[1] for h in have], axis=0)
                label = f"{name} (mean of {len(have)})"
            elif name in members:
                f_fit, f_test = members[name]
                label = name
            else:
                continue

            t_fit, t_test = truth[fit_i + k], truth[test_i + k]
            ok = np.isfinite(f_test).all(axis=(1, 2, 3))
            okf = np.isfinite(f_fit).all(axis=(1, 2, 3))
            if not ok.any() or not okf.any():
                continue

            au = fit_amplification(f_fit[okf][:, U10], t_fit[okf][:, U10])
            av = fit_amplification(f_fit[okf][:, V10], t_fit[okf][:, V10])

            base = score(f_test[ok][:, U10], f_test[ok][:, V10],
                         t_test[ok][:, U10], t_test[ok][:, V10], lat_w)
            corr = score(apply_amplification(f_test[ok][:, U10], au),
                         apply_amplification(f_test[ok][:, V10], av),
                         t_test[ok][:, U10], t_test[ok][:, V10], lat_w)

            for tag, s in (("raw", base), ("spectral", corr)):
                rows.append({"model": label, "variant": tag, "lead_h": lead,
                             "n_inits": int(ok.sum()),
                             **{kk: round(vv, 5) for kk, vv in s.items()}})
            print(f"windml {label:24s} @{lead}h", flush=True)
            print(f"windml   raw      rmse={base['ws_rmse']:.3f} "
                  f"spec={base['ws_spec_ratio']:.3f} cf_bias={base['cf_bias']:+.4f} "
                  f"p95={base['ws_p95_bias']:+.3f}", flush=True)
            print(f"windml   spectral rmse={corr['ws_rmse']:.3f} "
                  f"spec={corr['ws_spec_ratio']:.3f} cf_bias={corr['cf_bias']:+.4f} "
                  f"p95={corr['ws_p95_bias']:+.3f}", flush=True)
            print(f"windml   trade    rmse {100*(corr['ws_rmse']/base['ws_rmse']-1):+.1f}%  "
                  f"cf_bias {abs(corr['cf_bias'])/max(abs(base['cf_bias']),1e-9):.2f}x "
                  f"of raw", flush=True)

    if not rows:
        raise SystemExit("nothing scored")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
