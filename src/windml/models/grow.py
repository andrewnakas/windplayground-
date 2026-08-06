"""Grow a pretrained model's input stack when fine-tuning adds channels.

CMIP6 cannot pretrain the whole RT2021 input set. The WeatherBench CMIP archive
carries only the five pressure-level variables -- no 2m temperature and no
precipitation (see `CMIP_AVAILABLE_VARIABLES` in `windml.config`) -- so
pretraining runs on 111 input channels and ERA5 fine-tuning wants 117.

The safe way to bridge that is **zero-initializing the new stem columns**. A
convolution is a sum over input channels, so a column of zeros contributes
exactly nothing: the grown model computes a bit-identical function to the
pretrained one at step 0, and gradient descent grows the new inputs in from
there. The alternative -- random-initializing them -- injects noise into every
feature map at step 0 and throws away part of what pretraining bought.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _stem_conv(model: nn.Module) -> nn.Conv2d:
    """The first Conv2d, i.e. the only layer whose in_channels can change."""
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            return module
    raise ValueError("model has no Conv2d layer to grow")


def grow_input_channels(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    keep: list[int] | None = None,
) -> nn.Module:
    """Load `state_dict` into `model`, zero-filling stem columns it lacks.

    `keep` maps each pretrained input channel to its index in the new model's
    input layout; if omitted the pretrained channels are assumed to occupy the
    first positions. Everything except the stem convolution must match exactly,
    which is the point -- only the input width is allowed to differ.
    """
    stem = _stem_conv(model)
    stem_key = next(
        (k for k, v in state_dict.items()
         if v.shape[1:] == stem.weight.shape[1:] and v.dim() == 4
         and v.shape[0] == stem.weight.shape[0]),
        None,
    )
    # Locate the pretrained stem by name rather than shape when widths differ.
    if stem_key is None:
        stem_key = next(k for k, v in state_dict.items()
                        if v.dim() == 4 and v.shape[0] == stem.weight.shape[0]
                        and v.shape[2:] == stem.weight.shape[2:])

    old = state_dict[stem_key]
    n_old, n_new = old.shape[1], stem.weight.shape[1]
    if n_old > n_new:
        raise ValueError(
            f"pretrained stem has {n_old} input channels, more than the "
            f"target's {n_new}; growing can only add channels"
        )
    if keep is None:
        keep = list(range(n_old))
    if len(keep) != n_old:
        raise ValueError(f"keep has {len(keep)} entries, expected {n_old}")

    grown = torch.zeros_like(stem.weight)
    grown[:, keep] = old.to(grown.dtype)

    patched = dict(state_dict)
    patched[stem_key] = grown
    missing, unexpected = model.load_state_dict(patched, strict=False)
    # Only the stem may differ in shape; anything else missing means the two
    # architectures are not the same model and the transfer is meaningless.
    if missing or unexpected:
        raise ValueError(
            f"state dict does not match the model: missing={missing}, "
            f"unexpected={unexpected}"
        )
    return model


def channel_index_map(pretrain_channels: list[str], finetune_channels: list[str],
                      frames: int, n_static: int) -> list[int]:
    """Positions in the fine-tune input stack for each pretrained channel.

    Both stacks are laid out frame-major -- all fields at t, then t-6h, then
    t-12h -- with the statics appended last, so a pretrained channel's new home
    depends on which frame it belongs to as well as which variable it is.
    """
    lookup = {c: i for i, c in enumerate(finetune_channels)}
    missing = [c for c in pretrain_channels if c not in lookup]
    if missing:
        raise ValueError(f"pretrain channels absent from fine-tune set: {missing}")

    n_pre, n_fine = len(pretrain_channels), len(finetune_channels)
    keep = [
        frame * n_fine + lookup[c]
        for frame in range(frames)
        for c in pretrain_channels
    ]
    # Statics sit after every dynamic frame in both stacks and always carry over.
    keep += [frames * n_fine + i for i in range(n_static)]
    assert len(keep) == frames * n_pre + n_static
    return keep
