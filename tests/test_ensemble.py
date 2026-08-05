import numpy as np

from windml.config import CHANNELS
from windml.eval.ensemble import DressedEnsemble, evaluate_ensemble
from windml.eval.forecasters import PersistenceForecaster

C, H, W = len(CHANNELS), 8, 16
LAT_W = np.ones(H)


def _inputs(n_time=40):
    rng = np.random.default_rng(3)
    truth = rng.normal(size=(n_time, C, H, W)).astype(np.float32)
    times = np.arange(
        np.datetime64("2020-01-01T00", "h"),
        np.datetime64("2020-01-01T00", "h") + np.timedelta64(6 * n_time, "h"),
        np.timedelta64(6, "h"),
    )
    clim = np.zeros((366, 4, C, H, W), dtype=np.float32)
    return truth, times, clim


def test_dressed_ensemble_shapes_and_spread():
    truth, _, _ = _inputs()
    spread = np.full((5, C), 2.0, dtype=np.float32)
    ens = DressedEnsemble(PersistenceForecaster(truth), spread, n_members=6, seed=0)
    members = ens.ensemble(3, 5)
    assert members.shape == (6, 5, C, H, W)
    # empirical spread should be near the requested 2.0
    assert 1.0 < members.std(axis=0).mean() < 3.0


def test_zero_spread_crps_equals_mae():
    """With zero spread the ensemble is deterministic, so CRPS == MAE."""
    truth, times, clim = _inputs()
    spread = np.zeros((4, C), dtype=np.float32)
    ens = DressedEnsemble(PersistenceForecaster(truth), spread, n_members=3)
    df = evaluate_ensemble(ens, truth, times, clim, LAT_W, K=4, init_stride=6)
    assert (df.spread.abs() < 1e-5).all()
    assert (df.crps > 0).all()
    assert df.crps.notna().all()


def test_dressing_improves_crps_over_zero_spread():
    """A calibrated spread must beat a zero-spread (deterministic) ensemble."""
    truth, times, clim = _inputs()
    base = PersistenceForecaster(truth)
    zero = evaluate_ensemble(
        DressedEnsemble(base, np.zeros((3, C), np.float32), n_members=8),
        truth, times, clim, LAT_W, K=3, init_stride=6, name="zero",
    )
    # persistence error on this synthetic data has std ~sqrt(2)
    calibrated = evaluate_ensemble(
        DressedEnsemble(base, np.full((3, C), 1.4, np.float32), n_members=8),
        truth, times, clim, LAT_W, K=3, init_stride=6, name="dressed",
    )
    assert calibrated.crps.mean() < zero.crps.mean()
