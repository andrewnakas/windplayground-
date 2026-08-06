"""Fidelity checks on the Rasp & Thuerey (2021) Resnet copy and its TISR input."""
from __future__ import annotations

import numpy as np
import torch

from windml.data.solar import SOLAR_CONSTANT, toa_incident_solar_radiation
from windml.models import build_model
from windml.models.rt_resnet import WeatherResNetRT
from windml.utils.grid import latitude_weights


def test_param_count_matches_paper():
    """~6.36M against the paper's stated ~6.3M.

    This is the cheapest check that the copy is structurally right: a wrong
    block count, width, or kernel size all move this number.
    """
    n = sum(p.numel() for p in WeatherResNetRT().parameters())
    assert 6.2e6 < n < 6.5e6, f"got {n/1e6:.2f}M params, paper reports ~6.3M"


def test_architecture_shape_is_the_papers():
    m = WeatherResNetRT()
    assert len(m.blocks) == 19
    assert m.stem.conv.conv.kernel_size == (7, 7)
    assert m.stem.conv.conv.out_channels == 128
    for block in m.blocks:
        for cb in (block.a, block.b):
            assert cb.conv.conv.kernel_size == (3, 3)
            assert cb.conv.conv.out_channels == 128


def test_block_order_is_conv_act_norm_drop():
    """The paper puts LeakyReLU BEFORE BatchNorm. Guard against a 'fix'."""
    cb = WeatherResNetRT().blocks[0].a
    assert isinstance(cb.act, torch.nn.LeakyReLU)
    assert cb.act.negative_slope == 0.3
    assert isinstance(cb.norm, torch.nn.BatchNorm2d)
    # eval mode so dropout is deterministic and BatchNorm stops updating its
    # running stats between the two forward passes we are comparing
    cb.eval()
    x = torch.randn(2, 128, 8, 16)
    expected = cb.drop(cb.norm(cb.act(cb.conv(x))))
    torch.testing.assert_close(cb(x), expected)


def test_forward_shape_and_registry():
    model = build_model("rt_resnet", in_channels=117, out_channels=3)
    model.eval()
    out = model(torch.randn(2, 117, 32, 64))
    assert out.shape == (2, 3, 32, 64)


def test_longitude_is_periodic_but_latitude_is_not():
    """Rolling the globe in longitude must roll the prediction identically."""
    model = WeatherResNetRT(in_channels=4, out_channels=2, width=8, n_blocks=1,
                            dropout=0.0)
    model.eval()
    x = torch.randn(1, 4, 32, 64)
    with torch.no_grad():
        rolled_after = torch.roll(model(x), shifts=7, dims=-1)
        rolled_before = model(torch.roll(x, shifts=7, dims=-1))
    torch.testing.assert_close(rolled_after, rolled_before, atol=1e-5, rtol=1e-4)


def test_conv_params_excludes_norm_and_bias():
    """L2 1e-5 applies to conv kernels only, as Keras kernel_regularizer does."""
    m = WeatherResNetRT(in_channels=4, out_channels=2, width=8, n_blocks=2)
    convs = m.conv_params()
    assert len(convs) == 1 + 2 * 2 + 1  # stem + 2 blocks x 2 convs + head
    ids = {id(p) for p in convs}
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            assert id(mod.weight) not in ids and id(mod.bias) not in ids
    assert all(p.dim() == 4 for p in convs)


def test_can_overfit_one_batch():
    """Sanity that the stack trains at all before spending Kaggle quota."""
    torch.manual_seed(0)
    model = WeatherResNetRT(in_channels=6, out_channels=2, width=16, n_blocks=2,
                            dropout=0.0)
    x = torch.randn(2, 6, 16, 32)
    y = torch.randn(2, 2, 16, 32)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        loss = torch.nn.functional.mse_loss(model(x), y)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.25 * first, f"{first:.3f} -> {loss.item():.3f}"


# --- TOA incident solar radiation -------------------------------------------


def _grid():
    lat = np.linspace(-87.1875, 87.1875, 32)
    lon = np.linspace(0.0, 354.375, 64)
    return lat, lon


def test_tisr_night_side_is_exactly_zero_and_never_negative():
    lat, lon = _grid()
    t = np.array(["2020-03-20T12:00:00"], dtype="datetime64[s]")
    r = toa_incident_solar_radiation(t, lat, lon)
    assert r.min() == 0.0
    assert (r >= 0).all()
    # At any instant roughly half the globe is dark.
    assert 0.3 < (r == 0).mean() < 0.7


def test_tisr_global_mean_is_solar_constant_over_four():
    """Area-weighted TOA insolation averages S0/4 -- the textbook identity."""
    lat, lon = _grid()
    times = np.arange("2020-01-01T00", "2021-01-01T00", 6,
                      dtype="datetime64[h]").astype("datetime64[s]")
    r = toa_incident_solar_radiation(times, lat, lon)
    w = latitude_weights(lat)[None, :, None]
    mean = (r * w).mean()
    assert abs(mean - SOLAR_CONSTANT / 4) < 6.0, mean


def test_tisr_seasonal_cycle_has_the_right_sign():
    """Northern midsummer lights the Arctic; midwinter leaves it dark."""
    lat, lon = _grid()
    times = np.array(["2020-06-21T12:00:00", "2020-12-21T12:00:00"],
                     dtype="datetime64[s]")
    r = toa_incident_solar_radiation(times, lat, lon)
    arctic = lat > 80
    assert r[0, arctic].mean() > 300      # polar day
    assert r[1, arctic].max() == 0.0      # polar night


def test_tisr_peaks_near_the_subsolar_longitude():
    """At 12:00 UTC the sun is overhead near longitude 0, not 180."""
    lat, lon = _grid()
    t = np.array(["2020-03-20T12:00:00"], dtype="datetime64[s]")
    r = toa_incident_solar_radiation(t, lat, lon)[0]
    equator = np.argmin(np.abs(lat))
    peak_lon = lon[np.argmax(r[equator])]
    assert min(peak_lon, 360 - peak_lon) < 15.0, peak_lon


# --- RT2021 channel bookkeeping ---------------------------------------------


def test_rt2021_input_channels_reproduce_the_papers_117():
    from windml.config import active_variables, rt_input_channels

    # 37 stored + 1 computed (TISR) = 38 fields/frame; x3 frames = 114 dynamic,
    # +3 statics = 117. The paper states the 114.
    assert len(active_variables("rt2021")) == 37
    assert rt_input_channels("rt2021") == 117
    # CMIP6 has no t2m or precip, so pretraining sees a narrower stack.
    assert rt_input_channels("rt2021_cmip") == 111


def test_rt_targets_are_the_first_three_channels():
    """Loss, metrics and the model head all index channels positionally."""
    from windml.config import RT_TARGETS, active_variables

    assert [v["short"] for v in active_variables("rt2021")][:3] == RT_TARGETS


def test_cmip_subset_is_pressure_levels_only():
    """Verified against the WeatherBench data repo: no surface fields in CMIP."""
    from windml.config import active_variables

    era = {v["short"] for v in active_variables("rt2021")}
    cmip = {v["short"] for v in active_variables("rt2021_cmip")}
    assert era - cmip == {"t2m", "tp"}
    assert len(cmip) == 35  # 5 variables x 7 levels
