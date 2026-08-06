"""AFNO: adaptive Fourier neural operator (FourCastNet lineage).

Token mixing happens in the 2D Fourier domain: rfft2 over the token grid, a
block-diagonal complex MLP per frequency mode with soft-shrinkage sparsity,
then irfft2. Channel mixing is a standard MLP. See Guibas et al. 2022 /
Pathak et al. 2022 (FourCastNet).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class AFNOMixer(nn.Module):
    def __init__(self, dim: int, num_blocks: int = 8, sparsity: float = 0.01):
        super().__init__()
        assert dim % num_blocks == 0
        self.nb = num_blocks
        self.bs = dim // num_blocks
        scale = 0.02
        self.w1 = nn.Parameter(scale * torch.randn(2, self.nb, self.bs, self.bs))
        self.b1 = nn.Parameter(scale * torch.randn(2, self.nb, self.bs))
        self.w2 = nn.Parameter(scale * torch.randn(2, self.nb, self.bs, self.bs))
        self.b2 = nn.Parameter(scale * torch.randn(2, self.nb, self.bs))
        self.sparsity = sparsity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, D)
        B, H, W, D = x.shape
        fx = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")  # (B, H, W//2+1, D)
        fx = fx.reshape(B, H, W // 2 + 1, self.nb, self.bs)

        def cmul(t, w, b):
            re = torch.einsum("...bi,bio->...bo", t.real, w[0]) - torch.einsum(
                "...bi,bio->...bo", t.imag, w[1]
            )
            im = torch.einsum("...bi,bio->...bo", t.real, w[1]) + torch.einsum(
                "...bi,bio->...bo", t.imag, w[0]
            )
            return torch.complex(re + b[0], im + b[1])

        h = cmul(fx, self.w1, self.b1)
        h = torch.complex(F.relu(h.real), F.relu(h.imag))
        h = cmul(h, self.w2, self.b2)
        h = torch.stack([h.real, h.imag], dim=-1)
        h = F.softshrink(h, lambd=self.sparsity)
        h = torch.complex(h[..., 0], h[..., 1])
        h = h.reshape(B, H, W // 2 + 1, D)
        return torch.fft.irfft2(h, s=(H, W), dim=(1, 2), norm="ortho")


class AFNOBlock(nn.Module):
    def __init__(self, dim: int, num_blocks: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = AFNOMixer(dim, num_blocks)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class AFNONet(nn.Module):
    def __init__(
        self,
        in_channels: int = 25,
        out_channels: int = 8,
        img_size: tuple[int, int] = (32, 64),
        patch: int = 2,
        dim: int = 128,
        depth: int = 6,
        num_blocks: int = 8,
    ):
        super().__init__()
        H, W = img_size
        self.h, self.w = H // patch, W // patch
        self.patch = patch
        self.out_channels = out_channels
        self.embed = nn.Conv2d(in_channels, dim, patch, stride=patch)
        self.pos = nn.Parameter(0.02 * torch.randn(1, self.h, self.w, dim))
        self.blocks = nn.ModuleList([AFNOBlock(dim, num_blocks) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_channels * patch * patch)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        t = self.embed(x).permute(0, 2, 3, 1)  # (B, h, w, dim)
        t = t + self.pos
        for blk in self.blocks:
            t = blk(t)
        t = self.head(self.norm(t))  # (B, h, w, C*p*p)
        t = t.reshape(B, self.h, self.w, self.out_channels, self.patch, self.patch)
        t = t.permute(0, 3, 1, 4, 2, 5)
        return t.reshape(B, self.out_channels, self.h * self.patch, self.w * self.patch)
