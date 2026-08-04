import numpy as np
import pytest

from windml.config import CHANNELS, DataConfig
from windml.data.climatology import climatology_at, compute_climatology, doy_hour_index
from windml.data.dataset import Era5Dataset, year_range_times
from windml.data.normalization import Normalizer, compute_stats

H, W, C = 32, 64, len(CHANNELS)


@pytest.fixture()
def fake_cache(tmp_path):
    """Two fake cached years (2018, 2019) + statics, in a tmp cache dir."""
    rng = np.random.default_rng(0)
    cfg = DataConfig(cache_dir=str(tmp_path))
    root = tmp_path / "era5_64x32"
    root.mkdir()
    for year in (2018, 2019):
        n = len(year_range_times((year, year)))
        np.save(root / f"{year}.npy", rng.normal(size=(n, C, H, W)).astype(np.float32))
    np.savez(
        root / "statics.npz",
        land_sea_mask=rng.random((H, W)).astype(np.float32),
        geopotential_at_surface=rng.normal(size=(H, W)).astype(np.float32),
        latitude=np.linspace(-87.1875, 87.1875, H).astype(np.float32),
        longitude=np.linspace(0, 354.375, W).astype(np.float32),
    )
    return cfg


def test_stats_and_normalizer_roundtrip():
    rng = np.random.default_rng(1)
    x = rng.normal(loc=5.0, scale=3.0, size=(50, C, H, W)).astype(np.float32)
    stats = compute_stats(x)
    norm = Normalizer(stats)
    z = norm.norm_state(x)
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1.0) < 0.05
    np.testing.assert_allclose(norm.denorm_state(z), x, atol=1e-3)
    d = np.diff(x, axis=0)
    np.testing.assert_allclose(norm.denorm_residual(norm.norm_residual(d)), d, atol=1e-3)


def test_doy_hour_index():
    times = np.array(["2020-01-01T00", "2020-01-01T18", "2020-12-31T06"], dtype="datetime64[h]")
    doy, slot = doy_hour_index(times)
    assert list(doy) == [0, 0, 365]  # 2020 is a leap year
    assert list(slot) == [0, 3, 1]


def test_climatology_constant_field():
    times = year_range_times((2018, 2018))
    arr = np.full((len(times), 1, 2, 2), 7.0, dtype=np.float32)
    clim = compute_climatology(arr, times, window_days=3)
    looked_up = climatology_at(clim, times[:10])
    np.testing.assert_allclose(looked_up, 7.0, atol=1e-5)


def test_dataset_shapes_and_no_boundary_crossing(fake_cache):
    arr = np.load(fake_cache.cache_dir + "/era5_64x32/2018.npy")
    stats = compute_stats(arr[:200])
    ds = Era5Dataset(fake_cache, (2018, 2019), Normalizer(stats), rollout_steps=2, two_frame=True)
    n_total = len(year_range_times((2018, 2019)))
    # margin of 1 at the start (two-frame) and K=2 at the end
    assert len(ds) == n_total - 3
    x, y = ds[0]
    assert x.shape == (ds.n_input_channels, H, W) == (2 * C + 5 + 4, H, W)
    assert y.shape == (2, C, H, W)
    # last sample's targets stay inside the loaded span
    x_last, y_last = ds[len(ds) - 1]
    assert np.isfinite(x_last.numpy()).all() and np.isfinite(y_last.numpy()).all()


def test_dataset_residual_target_correctness(fake_cache):
    arr = np.load(fake_cache.cache_dir + "/era5_64x32/2018.npy")
    stats = compute_stats(arr[:200])
    norm = Normalizer(stats)
    ds = Era5Dataset(fake_cache, (2018, 2018), norm, rollout_steps=1, two_frame=False)
    _, y = ds[10]
    expected = norm.norm_residual((arr[11:12] - arr[10:11]).astype(np.float32))[0]
    np.testing.assert_allclose(y.numpy()[0], expected, atol=1e-5)


def test_direct_lead_target_and_horizon(fake_cache):
    """A direct-lead dataset targets one jump of L steps, not L 6-hourly ones."""
    arr = np.load(fake_cache.cache_dir + "/era5_64x32/2018.npy")
    norm = Normalizer(compute_stats(arr[:200]))
    L = 12  # 72 h at 6 h steps
    ds = Era5Dataset(fake_cache, (2018, 2018), norm, two_frame=False, direct_steps=L)
    assert ds.horizon == L
    assert len(ds) == arr.shape[0] - L
    x, y = ds[5]
    assert y.shape == (1, C, H, W)          # a single target, not L residuals
    expected = (arr[5 + L] - arr[5]).astype(np.float32) / ds.direct_std[0]
    np.testing.assert_allclose(y.numpy()[0], expected, rtol=1e-4, atol=1e-4)


def test_direct_forecaster_scores_only_its_lead(fake_cache):
    """DirectForecaster fills its own lead and leaves the rest NaN to skip."""
    import torch

    from windml.eval.forecasters import DirectForecaster

    arr = np.load(fake_cache.cache_dir + "/era5_64x32/2018.npy")
    norm = Normalizer(compute_stats(arr[:200]))
    ds = Era5Dataset(fake_cache, (2018, 2018), norm, two_frame=False, direct_steps=4)
    model = torch.nn.Conv2d(ds.n_input_channels, C, 1)
    torch.nn.init.zeros_(model.weight); torch.nn.init.zeros_(model.bias)
    out = DirectForecaster(model, ds, "direct").forecast(10, 8)
    assert out.shape == (8, C, H, W)
    assert np.isfinite(out[3]).all()                       # lead 4 (index 3)
    assert np.isnan(out[[0, 1, 2, 4, 5, 6, 7]]).all()      # all other leads
    # zero-weight model predicts no change, so the field equals the init state
    np.testing.assert_allclose(out[3], arr[10], rtol=1e-4, atol=1e-4)
