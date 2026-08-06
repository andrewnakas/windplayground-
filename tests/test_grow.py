"""Pretrain -> fine-tune channel growth must preserve the learned function."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from windml.config import RT_INPUT_FRAMES, RT_N_STATIC, active_variables
from windml.data.normalization import Normalizer, inverse_log_transform, log_transform
from windml.models.grow import channel_index_map, grow_input_channels
from windml.models.rt_resnet import WeatherResNetRT


def test_grown_model_is_bit_identical_at_step_zero():
    """The whole point: zero-filled columns contribute nothing."""
    torch.manual_seed(0)
    small = WeatherResNetRT(in_channels=6, out_channels=2, width=8, n_blocks=2,
                            dropout=0.0)
    big = WeatherResNetRT(in_channels=10, out_channels=2, width=8, n_blocks=2,
                          dropout=0.0)
    small.eval()
    grow_input_channels(big, small.state_dict(), keep=list(range(6)))
    big.eval()

    x_old = torch.randn(2, 6, 16, 32)
    # New channels carry arbitrary values; they must not affect the output.
    x_new = torch.cat([x_old, torch.randn(2, 4, 16, 32)], dim=1)
    with torch.no_grad():
        torch.testing.assert_close(small(x_old), big(x_new))


def test_new_columns_are_exactly_zero_and_old_ones_preserved():
    small = WeatherResNetRT(in_channels=4, out_channels=1, width=8, n_blocks=1)
    big = WeatherResNetRT(in_channels=7, out_channels=1, width=8, n_blocks=1)
    old_w = small.state_dict()["stem.conv.conv.weight"].clone()
    grow_input_channels(big, small.state_dict(), keep=[0, 2, 4, 6])
    new_w = big.stem.conv.conv.weight.detach()
    torch.testing.assert_close(new_w[:, [0, 2, 4, 6]], old_w)
    assert new_w[:, [1, 3, 5]].abs().sum() == 0.0


def test_shrinking_is_refused():
    big = WeatherResNetRT(in_channels=9, out_channels=1, width=8, n_blocks=1)
    small = WeatherResNetRT(in_channels=4, out_channels=1, width=8, n_blocks=1)
    with pytest.raises(ValueError, match="more than the target"):
        grow_input_channels(small, big.state_dict())


def test_mismatched_architecture_is_refused():
    """A silent partial load would produce a randomly-initialized 'pretrained' model."""
    src = WeatherResNetRT(in_channels=4, out_channels=1, width=8, n_blocks=1)
    dst = WeatherResNetRT(in_channels=6, out_channels=1, width=8, n_blocks=3)
    with pytest.raises(ValueError, match="does not match"):
        grow_input_channels(dst, src.state_dict())


def test_channel_index_map_places_cmip_channels_in_the_rt2021_stack():
    pre = [v["short"] for v in active_variables("rt2021_cmip")]
    fine = [v["short"] for v in active_variables("rt2021")]
    keep = channel_index_map(pre, fine, RT_INPUT_FRAMES, RT_N_STATIC)

    assert len(keep) == RT_INPUT_FRAMES * len(pre) + RT_N_STATIC == 111
    assert len(set(keep)) == len(keep), "no channel may be mapped twice"
    # t2m and tp are the two the pretrained model never saw, in every frame.
    absent = {frame * len(fine) + fine.index(c)
              for frame in range(RT_INPUT_FRAMES) for c in ("t2m", "tp")}
    assert absent.isdisjoint(keep)


def test_channel_index_map_rejects_unknown_channel():
    with pytest.raises(ValueError, match="absent from fine-tune"):
        channel_index_map(["z500", "nope"], ["z500"], 1, 0)


# --- precipitation transform -------------------------------------------------


def test_log_transform_keeps_zero_at_zero():
    """The reason for subtracting log(eps): 'no rain' must stay exactly zero."""
    x = np.zeros((2, 2, 4, 4), dtype=np.float32)
    out = log_transform(x, ["z500", "tp"])
    assert out[:, 1].max() == 0.0


def test_log_transform_round_trips_and_is_monotone():
    rng = np.random.default_rng(0)
    x = np.zeros((5, 2, 4, 4), dtype=np.float32)
    x[:, 1] = rng.gamma(0.5, 2.0, size=(5, 4, 4)).astype(np.float32)
    chans = ["z500", "tp"]
    back = inverse_log_transform(log_transform(x, chans), chans)
    np.testing.assert_allclose(back, x, rtol=1e-4, atol=1e-6)
    flat = np.sort(x[:, 1].ravel()).reshape(1, 1, -1, 1)
    tf = log_transform(flat, ["tp"])[0, 0, :, 0]
    assert np.all(np.diff(tf) >= 0), "transform must preserve ordering"


def test_log_transform_leaves_other_channels_untouched():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 2, 4, 4)).astype(np.float32)
    out = log_transform(x, ["z500", "tp"])
    np.testing.assert_array_equal(out[:, 0], x[:, 0])


def test_normalizer_does_not_subtract_the_mean_from_precip():
    stats = {"channels": ["z500", "tp"], "mean": [100.0, 7.0],
             "std": [10.0, 2.0], "diff_std": [1.0, 1.0]}
    norm = Normalizer(stats)
    assert norm.mean[0, 0, 0, 0] == 100.0
    assert norm.mean[0, 1, 0, 0] == 0.0
    # so a zero-precip cell normalizes to zero, not to -mean/std
    x = np.zeros((1, 2, 2, 2), dtype=np.float32)
    assert norm.norm_state(x)[0, 1].max() == 0.0


# --- head growth (2 pretrain outputs -> 3 fine-tune outputs) -----------------


def test_grown_head_preserves_pretrained_outputs_and_zeros_the_new_one():
    """CMIP cannot supervise t2m, so it is added at fine-tune time as zeros."""
    from windml.models.grow import grow_output_channels

    torch.manual_seed(0)
    src = WeatherResNetRT(in_channels=6, out_channels=2, width=8, n_blocks=1,
                          dropout=0.0)
    dst = WeatherResNetRT(in_channels=6, out_channels=3, width=8, n_blocks=1,
                          dropout=0.0)
    patched = grow_output_channels(dst, src.state_dict())
    dst.load_state_dict(patched)
    src.eval()
    dst.eval()

    x = torch.randn(2, 6, 16, 32)
    with torch.no_grad():
        a, b = src(x), dst(x)
    # the two pretrained outputs are untouched...
    torch.testing.assert_close(a, b[:, :2])
    # ...and the new one predicts exactly zero residual, i.e. "no change"
    assert b[:, 2].abs().max() == 0.0


def test_head_growth_refuses_to_shrink():
    from windml.models.grow import grow_output_channels

    wide = WeatherResNetRT(in_channels=4, out_channels=5, width=8, n_blocks=1)
    narrow = WeatherResNetRT(in_channels=4, out_channels=2, width=8, n_blocks=1)
    with pytest.raises(ValueError, match="more than the target"):
        grow_output_channels(narrow, wide.state_dict())


def test_head_growth_is_a_noop_when_widths_already_match():
    from windml.models.grow import grow_output_channels

    a = WeatherResNetRT(in_channels=4, out_channels=3, width=8, n_blocks=1)
    b = WeatherResNetRT(in_channels=4, out_channels=3, width=8, n_blocks=1)
    src = a.state_dict()
    out = grow_output_channels(b, src)
    assert out is src, "same width should return the dict untouched"


def test_cmip_pretrain_transfers_into_the_era5_model_exactly():
    """The real pretrain -> fine-tune step, both ends grown at once.

    CMIP gives 111 inputs / 2 outputs; ERA5 fine-tuning wants 117 / 3. This is
    the combination scripts/train.py --init-ckpt performs, and the whole point
    is that it is exact: the two pretrained outputs must be untouched.
    """
    from windml.config import RT_INPUT_FRAMES, DataConfig, active_variables, rt_input_channels
    from windml.models.grow import grow_output_channels

    pre_cfg, fine_cfg = (DataConfig(variable_set="rt2021_cmip"),
                         DataConfig(variable_set="rt2021"))
    n_static = RT_N_STATIC
    assert rt_input_channels("rt2021_cmip") == 111
    assert rt_input_channels("rt2021") == 117

    torch.manual_seed(0)
    pre = WeatherResNetRT(in_channels=111, out_channels=len(pre_cfg.target_channels),
                          width=8, n_blocks=1, dropout=0.0)
    fine = WeatherResNetRT(in_channels=117, out_channels=len(fine_cfg.target_channels),
                           width=8, n_blocks=1, dropout=0.0)
    assert (pre.head.conv.out_channels, fine.head.conv.out_channels) == (2, 3)

    keep = channel_index_map(
        [v["short"] for v in active_variables("rt2021_cmip")],
        [v["short"] for v in active_variables("rt2021")],
        frames=RT_INPUT_FRAMES, n_static=n_static,
    )
    state = grow_output_channels(fine, pre.state_dict())
    grow_input_channels(fine, state, keep=keep)

    pre.eval()
    fine.eval()
    x_pre = torch.randn(2, 111, 8, 16)
    x_fine = torch.zeros(2, 117, 8, 16)
    x_fine[:, keep] = x_pre
    # the channels CMIP never had carry junk; they must not change anything yet
    absent = [i for i in range(117) if i not in set(keep)]
    x_fine[:, absent] = torch.randn(2, len(absent), 8, 16)

    with torch.no_grad():
        a, b = pre(x_pre), fine(x_fine)
    torch.testing.assert_close(a, b[:, :2])      # z500, t850 preserved exactly
    assert b[:, 2].abs().max() == 0.0            # t2m starts at "no change"
