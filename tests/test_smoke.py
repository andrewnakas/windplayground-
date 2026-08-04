import numpy as np

from windml.config import CHANNELS, Config
from windml.utils.grid import latitude_weights, spatial_encodings, time_encodings


def test_config_roundtrip():
    cfg = Config.from_dict({"run_name": "x", "train": {"batch_size": 8}})
    assert cfg.run_name == "x"
    assert cfg.train.batch_size == 8
    assert len(cfg.hash()) == 10
    assert len(CHANNELS) == 8


def test_latitude_weights_mean_one():
    lat = np.linspace(-87.1875, 87.1875, 32)
    w = latitude_weights(lat)
    assert w.shape == (32,)
    np.testing.assert_allclose(w.mean(), 1.0, rtol=1e-6)
    assert w[16] > w[0]  # equator outweighs pole


def test_spatial_encodings_shape_and_range():
    enc = spatial_encodings(np.linspace(-87, 87, 32), np.linspace(0, 354.375, 64))
    assert enc.shape == (3, 32, 64)
    assert np.abs(enc).max() <= 1.0 + 1e-6


def test_time_encodings():
    enc = time_encodings(np.array([0.0, 6.0, 12.0, 24.0 * 100]))
    assert enc.shape == (4, 4)
    np.testing.assert_allclose(np.linalg.norm(enc[:, :2], axis=1), 1.0, rtol=1e-5)
