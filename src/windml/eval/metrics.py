"""WeatherBench-style metrics: latitude-weighted RMSE, ACC, and CRPS.

All fields are (..., H, W) with H the latitude axis; weights come from
windml.utils.grid.latitude_weights (mean 1 over the grid).
"""
from __future__ import annotations

import numpy as np


def weighted_mean(x: np.ndarray, lat_w: np.ndarray) -> np.ndarray:
    """Mean over the trailing (H, W) axes, weighted by latitude."""
    return (x * lat_w[:, None]).mean(axis=(-2, -1))


def rmse(pred: np.ndarray, truth: np.ndarray, lat_w: np.ndarray) -> np.ndarray:
    """Latitude-weighted RMSE over (H, W), then averaged over leading axes.

    WeatherBench convention: RMSE is computed per forecast field, then the
    *RMSE values* (not MSEs) are averaged over init times.
    """
    per_field = np.sqrt(weighted_mean((pred - truth) ** 2, lat_w))
    return per_field.mean(axis=0) if per_field.ndim > 0 else per_field


def acc(
    pred: np.ndarray, truth: np.ndarray, clim: np.ndarray, lat_w: np.ndarray
) -> np.ndarray:
    """Anomaly correlation coefficient per field, averaged over the first axis.

    pred/truth/clim: (N, H, W) or (H, W).
    """
    ap = pred - clim
    at = truth - clim
    num = weighted_mean(ap * at, lat_w)
    den = np.sqrt(weighted_mean(ap**2, lat_w) * weighted_mean(at**2, lat_w))
    val = num / np.maximum(den, 1e-12)
    return val.mean(axis=0) if val.ndim > 0 else val


def crps_ensemble(ens: np.ndarray, truth: np.ndarray, lat_w: np.ndarray) -> float:
    """Fair (unbiased) ensemble CRPS, latitude-weighted, averaged over fields.

    ens: (M, N, H, W) ensemble of M members over N fields; truth: (N, H, W).
    CRPS = E|X - y| - 1/(2 M (M-1)) * sum_{i,j} |X_i - X_j|  (fair estimator)
    """
    M = ens.shape[0]
    skill = np.abs(ens - truth[None]).mean(axis=0)
    spread = np.zeros_like(truth, dtype=np.float64)
    for i in range(M):
        for j in range(i + 1, M):
            spread += np.abs(ens[i] - ens[j])
    spread *= 2.0 / (M * (M - 1)) if M > 1 else 0.0
    pointwise = skill - 0.5 * spread
    return float(weighted_mean(pointwise, lat_w).mean())


def wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(u**2 + v**2)


# --------------------------------------------------------------------------
# Sharpness diagnostics.
#
# An RMSE-optimal forecast is the conditional MEAN, so every model scored here
# -- ours and the frontier ones alike -- is deliberately blurred: hedging
# towards the mean is what minimises squared error. RMSE cannot see that, and
# for wind it is not cosmetic, because power goes as v^3. A forecast that is
# 12% better on RMSE can still under-predict the energy in a wind field and
# miss the high-wind events that decide grid operations.
#
# These four measure the blur directly. None of them replaces RMSE; they are
# reported next to it, because sharpening a forecast always costs RMSE and
# hiding that trade would be the whole trick.


def variance_ratio(pred: np.ndarray, truth: np.ndarray,
                   lat_w: np.ndarray) -> float:
    """Forecast spatial variance / truth spatial variance. 1.0 = calibrated.

    The most direct read on under-dispersion. Below 1 means the forecast is
    smoother than reality, and it should fall as lead time grows because the
    conditional mean flattens towards climatology.
    """
    def var(x):
        m = weighted_mean(x, lat_w)
        extra = (slice(None),) * m.ndim + (None, None)
        return weighted_mean((x - m[extra]) ** 2, lat_w)

    vp, vt = var(pred), var(truth)
    return float(np.mean(vp) / max(float(np.mean(vt)), 1e-12))


def zonal_power_spectrum(field: np.ndarray, lat_w: np.ndarray) -> np.ndarray:
    """Mean power per zonal wavenumber, latitude-weighted.

    Longitude is periodic, so the FFT along it needs no windowing -- this is
    one of the few places the 64x32 grid is exactly what the transform wants.
    Returns (W//2 + 1,): index k is the power at k waves around a latitude
    circle. Blur shows up as a deficit at high k.
    """
    f = np.fft.rfft(field, axis=-1)
    power = (np.abs(f) ** 2) / field.shape[-1]
    weighted = power * lat_w[:, None]
    return weighted.mean(axis=tuple(range(power.ndim - 2)) + (-2,))


def spectral_ratio(pred: np.ndarray, truth: np.ndarray, lat_w: np.ndarray,
                   high_k_from: float = 0.5) -> float:
    """Forecast/truth power over the top half of the wavenumber range.

    One number for "does it have the small scales". `high_k_from` is a
    fraction of the available wavenumbers, so it does not silently change
    meaning if the grid does.
    """
    sp, st = zonal_power_spectrum(pred, lat_w), zonal_power_spectrum(truth, lat_w)
    k0 = max(int(len(st) * high_k_from), 1)
    return float(sp[k0:].sum() / max(st[k0:].sum(), 1e-12))


def extreme_scores(pred: np.ndarray, truth: np.ndarray, lat_w: np.ndarray,
                   percentile: float = 95.0) -> dict[str, float]:
    """Error and bias where the TRUTH is extreme.

    Conditioning on truth, not on the forecast: the question is whether the
    model captures events that happened, which is the one a grid operator
    asks. Latitude weights still apply, but as weights on a masked mean rather
    than through weighted_mean, since the mask breaks the field structure.
    """
    thr = np.percentile(truth, percentile)
    mask = truth >= thr
    if not mask.any():
        return {"threshold": float(thr), "rmse": float("nan"),
                "bias": float("nan"), "n": 0}
    w = np.broadcast_to(lat_w[:, None], truth.shape)[mask]
    d = (pred - truth)[mask]
    return {
        "threshold": float(thr),
        "rmse": float(np.sqrt(np.average(d ** 2, weights=w))),
        # negative bias here is the signature of a blurred forecast: it
        # systematically under-shoots the events it did predict
        "bias": float(np.average(d, weights=w)),
        "n": int(mask.sum()),
    }


# IEC class-II-ish reference turbine, the shape rather than any real machine:
# nothing below cut-in, cubic through the ramp, flat at rated, zero past
# cut-out. Absolute capacity factors depend on the curve, so only differences
# between forecasts scored with the SAME curve are meaningful.
CUT_IN, RATED, CUT_OUT = 3.0, 12.0, 25.0


def capacity_factor(speed: np.ndarray) -> np.ndarray:
    """Turbine power curve applied to a 10 m wind speed field, in [0, 1]."""
    cf = np.zeros_like(speed, dtype=np.float64)
    ramp = (speed >= CUT_IN) & (speed < RATED)
    cf[ramp] = ((speed[ramp] ** 3 - CUT_IN ** 3)
                / (RATED ** 3 - CUT_IN ** 3))
    cf[(speed >= RATED) & (speed < CUT_OUT)] = 1.0
    return cf


def capacity_factor_error(pred_speed: np.ndarray, truth_speed: np.ndarray,
                          lat_w: np.ndarray) -> dict[str, float]:
    """Capacity-factor bias and RMSE -- the wind-power view of the same forecast.

    The cubic ramp is what makes this differ from wind-speed RMSE: an error at
    8 m/s costs far more energy than the same error at 4 m/s, and a blurred
    forecast concentrates near the mean where the curve is steep.
    """
    cp, ct = capacity_factor(pred_speed), capacity_factor(truth_speed)
    return {
        "bias": float(np.mean(weighted_mean(cp - ct, lat_w))),
        "rmse": float(np.mean(np.sqrt(weighted_mean((cp - ct) ** 2, lat_w)))),
        "pred_mean": float(np.mean(weighted_mean(cp, lat_w))),
        "truth_mean": float(np.mean(weighted_mean(ct, lat_w))),
    }
