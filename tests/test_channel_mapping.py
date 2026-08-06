"""Scoring must match model outputs to truth variables BY NAME, not position.

The failure this guards against is silent: rt2021 emits [z500, t850, t2m]
against a truth array ordered [u10, v10, t2m, msl, u850, v850, t850, z500], so
a positional slice compares z500 against u10 and yields plausible nonsense.
"""
from __future__ import annotations

import numpy as np

from windml.config import CHANNELS, DataConfig
from windml.eval.forecasters import N_SCORED, scored_channel_map, to_scored


def test_core_and_levels_keep_the_fast_path():
    """None means 'already aligned' -- every result computed so far is safe."""
    assert scored_channel_map(DataConfig(variable_set="core").target_channels) is None
    assert scored_channel_map(DataConfig(variable_set="levels").target_channels) is None


def test_rt2021_maps_by_name_not_position():
    cmap = scored_channel_map(DataConfig(variable_set="rt2021").target_channels)
    assert cmap is not None, "rt2021 must NOT take the positional fast path"
    # z500 is last in the truth order, t850 second-last, t2m third
    assert cmap == [CHANNELS.index("z500"), CHANNELS.index("t850"),
                    CHANNELS.index("t2m")]
    assert cmap == [7, 6, 2]


def test_to_scored_places_values_in_the_right_slots():
    cmap = scored_channel_map(DataConfig(variable_set="rt2021").target_channels)
    out = np.zeros((1, 1, 3, 2, 2), dtype=np.float32)
    out[0, 0, 0] = 500.0   # z500
    out[0, 0, 1] = 850.0   # t850
    out[0, 0, 2] = 2.0     # t2m
    scored = to_scored(out, cmap)
    assert scored.shape == (1, 1, N_SCORED, 2, 2)
    assert scored[0, 0, CHANNELS.index("z500")].mean() == 500.0
    assert scored[0, 0, CHANNELS.index("t850")].mean() == 850.0
    assert scored[0, 0, CHANNELS.index("t2m")].mean() == 2.0
    # everything the model does not forecast stays NaN, so it is skipped
    for name in ("u10", "v10", "msl", "u850", "v850"):
        assert np.isnan(scored[0, 0, CHANNELS.index(name)]).all()


def test_positional_slice_would_have_been_wrong():
    """Demonstrates the bug concretely, so the guard cannot be argued away."""
    out = np.zeros((1, 1, 3, 2, 2), dtype=np.float32)
    out[0, 0, 0] = 500.0
    naive = out[:, :, :N_SCORED]  # what the code used to do
    # the naive path puts z500's value where u10 is scored
    assert naive.shape[2] == 3 != N_SCORED  # and is the wrong width besides
    assert naive[0, 0, 0].mean() == 500.0
    assert CHANNELS[0] == "u10"


def test_target_channels_are_identity_for_non_rt_sets():
    for vs in ("core", "levels"):
        cfg = DataConfig(variable_set=vs)
        assert cfg.target_channels == cfg.channels
        assert not cfg.predicts_subset
        assert cfg.target_indices == list(range(len(cfg.channels)))


def test_cmip_pretrain_drops_t2m_from_the_targets():
    """CMIP has no 2m temperature, so it cannot be supervised during pretraining."""
    cfg = DataConfig(variable_set="rt2021_cmip")
    assert cfg.target_channels == ["z500", "t850"]
    assert cfg.predicts_subset
