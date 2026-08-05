import numpy as np
import pytest
import torch

from windml.models import build_model

IN_CH, OUT_CH, H, W = 25, 8, 32, 64


@pytest.mark.parametrize(
    "name,params,max_params_m",
    [
        ("unet", {"width": 48}, 3.0),
        ("vit", {"patch": 2, "dim": 128, "depth": 6, "heads": 4}, 4.0),
        ("afno", {"patch": 2, "dim": 128, "depth": 6, "num_blocks": 8}, 4.0),
        ("graph", {"refinements": 2, "hidden": 96, "rounds": 4}, 4.0),
    ],
)
def test_forward_shape_params_and_gradients(name, params, max_params_m):
    model = build_model(name, in_channels=IN_CH, out_channels=OUT_CH, **params)
    n_params = sum(p.numel() for p in model.parameters())
    assert 0.2e6 < n_params < max_params_m * 1e6, f"{name}: {n_params/1e6:.2f}M params"

    x = torch.randn(2, IN_CH, H, W)
    y = model(x)
    assert y.shape == (2, OUT_CH, H, W)
    # (y - 1)^2 keeps dL/dy nonzero even for the zero-initialized heads
    (y - 1).square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    total = sum(float(g.abs().sum()) for g in grads)
    assert np.isfinite(total) and total > 0


def test_unet_longitude_periodicity():
    """Rolling the input along longitude must roll the output identically."""
    model = build_model("unet", in_channels=IN_CH, out_channels=OUT_CH, width=16).eval()
    x = torch.randn(1, IN_CH, H, W)
    with torch.no_grad():
        base = model(x)
        rolled = model(torch.roll(x, shifts=8, dims=-1))
    torch.testing.assert_close(rolled, torch.roll(base, shifts=8, dims=-1), atol=1e-4, rtol=1e-4)


def test_zero_init_head_predicts_zero_residual():
    """ViT/AFNO heads start at zero, so the initial forecast is persistence."""
    for name, params in [
        ("vit", {"patch": 2, "dim": 64, "depth": 2, "heads": 4}),
        ("afno", {"patch": 2, "dim": 64, "depth": 2, "num_blocks": 4}),
    ]:
        model = build_model(name, in_channels=IN_CH, out_channels=OUT_CH, **params).eval()
        with torch.no_grad():
            y = model(torch.randn(1, IN_CH, H, W))
        assert float(y.abs().max()) == 0.0, name
