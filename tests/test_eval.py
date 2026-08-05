import numpy as np
import pandas as pd
import torch

from windml.config import CHANNELS
from windml.eval import metrics
from windml.eval.forecasters import ModelForecaster, PersistenceForecaster
from windml.eval.rollout import SCORED_VARS, evaluate_forecaster

C, H, W = len(CHANNELS), 8, 16
LAT_W = np.ones(H)


def test_rmse_analytic():
    truth = np.zeros((3, H, W))
    pred = truth + 2.0
    assert np.isclose(metrics.rmse(pred, truth, LAT_W), 2.0)


def test_acc_perfect_and_anti():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(4, H, W))
    clim = np.zeros_like(truth)
    assert np.isclose(metrics.acc(truth, truth, clim, LAT_W), 1.0)
    assert np.isclose(metrics.acc(-truth, truth, clim, LAT_W), -1.0)


def test_crps_deterministic_reduces_to_mae():
    rng = np.random.default_rng(1)
    truth = rng.normal(size=(4, H, W))
    ens = truth[None] + 1.0  # single member, constant error 1
    assert np.isclose(metrics.crps_ensemble(ens, truth, LAT_W), 1.0)


def _synthetic_eval_inputs(n_time=60):
    rng = np.random.default_rng(2)
    truth = rng.normal(size=(n_time, C, H, W)).astype(np.float32)
    times = np.arange(
        np.datetime64("2020-01-01T00", "h"),
        np.datetime64("2020-01-01T00", "h") + np.timedelta64(6 * n_time, "h"),
        np.timedelta64(6, "h"),
    )
    clim = np.zeros((366, 4, C, H, W), dtype=np.float32)
    return truth, times, clim


def test_identity_model_equals_persistence():
    truth, times, clim = _synthetic_eval_inputs()

    class _DS:
        array = truth
        two_frame = False
        static = np.zeros((5, H, W), dtype=np.float32)
        time_feats = np.zeros((len(times), 4), dtype=np.float32)

    class _Norm:
        def norm_state(self, x):
            return x

        def denorm_residual(self, dx):
            return dx

    identity = torch.nn.Conv2d(C + 5 + 4, C, 1, bias=False)
    torch.nn.init.zeros_(identity.weight)

    df_model = evaluate_forecaster(
        ModelForecaster(identity, _DS(), _Norm(), "identity"),
        truth, times, clim, LAT_W, K=4, init_stride=3,
    )
    df_pers = evaluate_forecaster(
        PersistenceForecaster(truth), truth, times, clim, LAT_W, K=4, init_stride=3
    )
    pd.testing.assert_series_equal(
        df_model["rmse"], df_pers["rmse"], check_names=False, atol=1e-5, rtol=1e-5
    )


def test_evaluator_output_schema():
    truth, times, clim = _synthetic_eval_inputs()
    df = evaluate_forecaster(
        PersistenceForecaster(truth), truth, times, clim, LAT_W, K=3, init_stride=5
    )
    assert set(df.variable) == set(SCORED_VARS)
    assert sorted(df.lead_h.unique()) == [6, 12, 18]
    assert (df.n_inits > 0).all()
    assert df.rmse.notna().all()
