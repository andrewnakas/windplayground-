"""The RT2021 training recipe: 3-frame inputs, plain-Adam L2, plateau schedule."""
from __future__ import annotations

import torch

from windml.config import Config
from windml.models.rt_resnet import WeatherResNetRT
from windml.train.trainer import Trainer


class _Cfg:
    """Minimal stand-in for the parts of Config that _build_optimizer reads."""

    def __init__(self, optimizer="adamw", wd=1e-5, lr=5e-5):
        self.train = type("T", (), {"optimizer": optimizer, "weight_decay": wd,
                                    "lr": lr})()


def test_rt_uses_plain_adam_not_adamw():
    """AdamW's decoupled decay is a different algorithm from Keras l2()."""
    m = WeatherResNetRT(in_channels=4, out_channels=2, width=8, n_blocks=1)
    opt = Trainer._build_optimizer(m, _Cfg(optimizer="adam"))
    assert type(opt) is torch.optim.Adam
    assert type(Trainer._build_optimizer(m, _Cfg())) is torch.optim.AdamW


def test_l2_applies_to_conv_kernels_only():
    m = WeatherResNetRT(in_channels=4, out_channels=2, width=8, n_blocks=2)
    opt = Trainer._build_optimizer(m, _Cfg(optimizer="adam", wd=1e-5))
    decayed, free = opt.param_groups[0], opt.param_groups[1]
    assert decayed["weight_decay"] == 1e-5
    assert free["weight_decay"] == 0.0
    # every decayed tensor is a 4-D conv kernel; no biases, no BatchNorm scales
    assert all(p.dim() == 4 for p in decayed["params"])
    assert len(decayed["params"]) == len(m.conv_params())
    # and nothing is lost between the two groups
    assert len(decayed["params"]) + len(free["params"]) == len(list(m.parameters()))


def test_models_without_conv_params_still_get_an_optimizer():
    """The registry has models that predate conv_params(); they must not break."""
    m = torch.nn.Sequential(torch.nn.Linear(4, 4))
    opt = Trainer._build_optimizer(m, _Cfg(optimizer="adam"))
    assert type(opt) is torch.optim.Adam


def test_rt_configs_carry_the_papers_hyperparameters():
    cfg = Config.from_yaml("configs/rt2021_72h.yaml")
    assert cfg.train.lr == 5e-5
    assert cfg.train.batch_size == 32
    assert cfg.train.weight_decay == 1e-5
    assert cfg.train.optimizer == "adam"
    assert cfg.train.lr_schedule == "plateau"
    assert cfg.train.plateau_factor == 0.2
    assert cfg.train.max_lr_drops == 2
    assert cfg.train.early_stop_patience == 5
    assert cfg.model.n_frames == 3
    # out_channels is DERIVED from the target set, never set in the config --
    # train.py passes it positionally, so a config that also set it would
    # raise "got multiple values for keyword argument 'out_channels'".
    assert "out_channels" not in cfg.model.params
    assert cfg.data.target_channels == ["z500", "t850", "t2m"]
    # their split, not our usual 2020 test year
    assert cfg.data.train_years == (1979, 2015)
    assert cfg.data.val_years == (2016, 2016)
    assert cfg.data.test_years == (2017, 2018)


def test_pretrain_config_is_narrower_and_dropout_free():
    """CMIP lacks t2m/tp, and the paper disables dropout when pretraining."""
    cfg = Config.from_yaml("configs/rt2021_pretrain.yaml")
    assert cfg.data.variable_set == "rt2021_cmip"
    assert cfg.model.params["dropout"] == 0.0


def test_every_lead_has_a_config():
    for lead in (6, 24, 72, 120):
        cfg = Config.from_yaml(f"configs/rt2021_{lead}h.yaml")
        assert cfg.train.direct_lead_h == lead
        assert cfg.run_name == f"rt2021_{lead}h"


def test_rollout_handles_three_frames():
    """The frame stack must shift correctly for n_frames=3, not just 2."""
    torch.manual_seed(0)
    C, nf, static = 3, 3, 2
    model = WeatherResNetRT(in_channels=C * nf + static, out_channels=C,
                            width=8, n_blocks=1, dropout=0.0)

    class _DS:
        n_frames = nf
        two_frame = True

    tr = Trainer.__new__(Trainer)
    tr.device = torch.device("cpu")
    tr.model = model
    tr.train_ds = _DS()
    tr.K = 3
    tr.loss_fn = lambda p, t: torch.nn.functional.mse_loss(p, t)
    ones = torch.ones(1, C, 1, 1)
    tr.diff_std_t, tr.std_t, tr.mean_t = ones, ones, torch.zeros(1, C, 1, 1)

    x = torch.randn(2, C * nf + static, 8, 16)
    y = torch.randn(2, 3, C, 8, 16)
    loss = tr._rollout_loss(x, y)
    assert torch.isfinite(loss) and loss.item() > 0
