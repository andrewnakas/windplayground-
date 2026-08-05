"""The blend's weighted least squares must recover a known affine relationship."""
import numpy as np

from windml.eval.forecasters import Forecaster


class _Fixed(Forecaster):
    def __init__(self, arr, name):
        self.arr = arr  # (n_init, K, C, H, W)
        self.name = name

    def forecast(self, init_idx, K):
        return self.arr[init_idx, :K]


def _fit_one_lead(members, target, lat_w):
    """Mirror of scripts.blend.fit_weights for a single (lead, channel)."""
    sqrt_w = np.sqrt(lat_w)[:, None]
    M = len(members)
    X = np.stack([m * sqrt_w for m in members]).reshape(M, -1)
    y = (target * sqrt_w).ravel()
    intercept = np.broadcast_to(np.broadcast_to(sqrt_w, target.shape[-2:]),
                                target.shape).ravel()
    A = np.concatenate([X, intercept[None]]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def test_recovers_known_affine_combination():
    rng = np.random.default_rng(0)
    H, W, N = 8, 16, 20
    lat_w = np.linspace(0.2, 1.8, H)
    m1 = rng.normal(size=(N, H, W))
    m2 = rng.normal(size=(N, H, W))
    truth = 0.7 * m1 + 0.2 * m2 + 1.5  # exact affine target

    coef = _fit_one_lead([m1, m2], truth, lat_w)
    np.testing.assert_allclose(coef, [0.7, 0.2, 1.5], atol=1e-6)


def test_intercept_is_a_constant_offset():
    """A pure constant bias must be recovered as the intercept, not smeared
    into the member weights (the fit and BlendForecaster must agree)."""
    rng = np.random.default_rng(1)
    H, W, N = 8, 16, 30
    lat_w = np.linspace(0.3, 1.7, H)
    m1 = rng.normal(size=(N, H, W))
    truth = m1 + 3.25

    coef = _fit_one_lead([m1], truth, lat_w)
    assert abs(coef[0] - 1.0) < 1e-6
    assert abs(coef[1] - 3.25) < 1e-6
    # applying the fitted weights the way BlendForecaster does reproduces truth
    applied = coef[0] * m1 + coef[1]
    np.testing.assert_allclose(applied, truth, atol=1e-6)
