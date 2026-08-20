"""Analytic tests for the sharpness diagnostics.

These metrics are about to carry a claim -- that the frontier models are
blurred -- so each one is pinned against a case whose answer is known by
construction rather than by running it once and blessing the output.
"""
from __future__ import annotations

import numpy as np
import pytest

from windml.eval.metrics import (
    capacity_factor,
    capacity_factor_error,
    extreme_scores,
    spectral_ratio,
    variance_ratio,
    zonal_power_spectrum,
)
from windml.utils.grid import latitude_weights

H, W = 32, 64
LAT = np.linspace(-87.1875, 87.1875, H)


@pytest.fixture
def lat_w() -> np.ndarray:
    return latitude_weights(LAT)


def test_variance_ratio_is_one_for_a_perfect_forecast(lat_w):
    rng = np.random.default_rng(0)
    truth = rng.normal(0, 3, (5, H, W))
    assert variance_ratio(truth, truth, lat_w) == pytest.approx(1.0, rel=1e-9)


def test_variance_ratio_detects_a_known_amount_of_blur(lat_w):
    """Scaling anomalies by a is exactly a^2 in variance -- the metric must say so."""
    rng = np.random.default_rng(1)
    truth = rng.normal(10, 3, (8, H, W))
    for a in (0.5, 0.8, 1.0):
        blurred = truth.mean() + a * (truth - truth.mean())
        assert variance_ratio(blurred, truth, lat_w) == pytest.approx(a**2, rel=0.02)


def test_variance_ratio_of_a_constant_field_is_zero(lat_w):
    truth = np.random.default_rng(2).normal(0, 1, (3, H, W))
    flat = np.full_like(truth, truth.mean())
    assert variance_ratio(flat, truth, lat_w) == pytest.approx(0.0, abs=1e-9)


def test_spectrum_puts_a_pure_wave_at_its_own_wavenumber(lat_w):
    """A field with k waves around the circle must have all its power at k."""
    lon = np.arange(W) * 2 * np.pi / W
    for k in (1, 3, 8):
        field = np.broadcast_to(np.cos(k * lon), (H, W)).copy()
        sp = zonal_power_spectrum(field, lat_w)
        assert int(np.argmax(sp)) == k
        others = np.delete(sp, k)
        assert others.max() < sp[k] * 1e-9


def test_spectral_ratio_is_one_against_itself_and_falls_when_smoothed(lat_w):
    rng = np.random.default_rng(3)
    truth = rng.normal(0, 1, (4, H, W))
    assert spectral_ratio(truth, truth, lat_w) == pytest.approx(1.0, rel=1e-9)

    # a 3-point zonal box filter is a low-pass, so high-k power must drop
    smooth = (np.roll(truth, 1, -1) + truth + np.roll(truth, -1, -1)) / 3.0
    assert spectral_ratio(smooth, truth, lat_w) < 0.5


def test_extreme_scores_condition_on_truth_not_forecast(lat_w):
    """A forecast that misses only the extremes must be caught here.

    Built so overall error is dominated by the extreme region: flat elsewhere,
    badly under-predicting the top tail. Plain RMSE dilutes that across the
    whole field; this metric should not.
    """
    # 12.5% of the field is extreme, so the 95th percentile lands inside it.
    # (An earlier version put 6.25% at 20 and asked for the 90th percentile,
    # which is 0.0 for that distribution -- the mask then selected the whole
    # field and the metric correctly returned the diluted 5.0.)
    truth = np.zeros((1, H, W))
    truth[0, :, :8] = 20.0
    pred = np.zeros_like(truth)                  # misses it entirely
    s = extreme_scores(pred, truth, lat_w, percentile=95.0)
    assert s["threshold"] == pytest.approx(20.0)
    assert s["n"] == H * 8
    assert s["rmse"] == pytest.approx(20.0, rel=1e-6)
    assert s["bias"] == pytest.approx(-20.0, rel=1e-6)   # under-shoot is negative


def test_extreme_scores_are_zero_for_a_perfect_forecast(lat_w):
    truth = np.random.default_rng(4).gamma(2.0, 3.0, (2, H, W))
    s = extreme_scores(truth, truth, lat_w, percentile=95.0)
    assert s["rmse"] == pytest.approx(0.0, abs=1e-12)
    assert s["bias"] == pytest.approx(0.0, abs=1e-12)


def test_power_curve_hits_its_defined_points():
    speeds = np.array([0.0, 2.9, 3.0, 12.0, 20.0, 24.9, 25.0, 40.0])
    cf = capacity_factor(speeds)
    assert cf[0] == 0.0 and cf[1] == 0.0          # below cut-in
    assert cf[2] == pytest.approx(0.0, abs=1e-12)  # exactly cut-in
    assert cf[3] == 1.0 and cf[4] == 1.0 and cf[5] == 1.0   # rated plateau
    assert cf[6] == 0.0 and cf[7] == 0.0          # cut-out and beyond
    assert np.all((cf >= 0) & (cf <= 1))


def test_power_curve_is_monotone_through_the_ramp():
    s = np.linspace(3.0, 12.0, 50)
    cf = capacity_factor(s)
    assert np.all(np.diff(cf) > 0)


def test_capacity_factor_penalises_blur_asymmetrically(lat_w):
    """The cubic ramp is why this is not just wind-speed RMSE.

    A forecast blurred toward the mean has the SAME speed bias as one shifted
    down by a constant, but the two differ in energy because the curve is
    convex. The blurred one must under-produce.
    """
    rng = np.random.default_rng(5)
    truth = np.clip(rng.normal(8.0, 3.0, (6, H, W)), 0, None)
    blurred = truth.mean() + 0.5 * (truth - truth.mean())

    e = capacity_factor_error(blurred, truth, lat_w)
    assert e["rmse"] > 0
    # shrinking spread around a mean on the steep part of a convex-then-capped
    # curve moves energy off the tails; the net here is a loss
    assert e["pred_mean"] < e["truth_mean"]
    assert e["bias"] < 0

    perfect = capacity_factor_error(truth, truth, lat_w)
    assert perfect["rmse"] == pytest.approx(0.0, abs=1e-12)
    assert perfect["bias"] == pytest.approx(0.0, abs=1e-12)
