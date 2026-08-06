"""Stormer-style vision transformer.

Follows Stormer's finding that a plain pre-norm ViT with small patches and the
right training recipe (residual targets, lat-weighted loss, rollout fine-tune)
matches far fancier architectures. 2x2 patches on the 32x64 grid -> 512 tokens.
"""
from __future__ import annotations

import torch
from torch import nn


def sincos_pos_embed(dim: int, h: int, w: int) -> torch.Tensor:
    """Standard 2D sin-cos positional embedding, shape (h*w, dim)."""
    assert dim % 4 == 0
    quarter = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(quarter) / quarter))
    ys = torch.arange(h).float()[:, None] * omega[None]
    xs = torch.arange(w).float()[:, None] * omega[None]
    emb_y = torch.cat([ys.sin(), ys.cos()], dim=1)  # (h, dim/2)
    emb_x = torch.cat([xs.sin(), xs.cos()], dim=1)  # (w, dim/2)
    grid = torch.cat(
        [
            emb_y[:, None].expand(h, w, dim // 2),
            emb_x[None, :].expand(h, w, dim // 2),
        ],
        dim=-1,
    )
    return grid.reshape(h * w, dim)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class WeatherViT(nn.Module):
    def __init__(
        self,
        in_channels: int = 25,
        out_channels: int = 8,
        img_size: tuple[int, int] = (32, 64),
        patch: int = 2,
        dim: int = 128,
        depth: int = 6,
        heads: int = 4,
    ):
        super().__init__()
        H, W = img_size
        self.h, self.w = H // patch, W // patch
        self.patch = patch
        self.out_channels = out_channels
        self.embed = nn.Conv2d(in_channels, dim, patch, stride=patch)
        self.register_buffer("pos", sincos_pos_embed(dim, self.h, self.w)[None])
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_channels * patch * patch)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        t = self.embed(x).flatten(2).transpose(1, 2)  # (B, N, dim)
        t = t + self.pos
        for blk in self.blocks:
            t = blk(t)
        t = self.head(self.norm(t))  # (B, N, C*p*p)
        t = t.reshape(B, self.h, self.w, self.out_channels, self.patch, self.patch)
        t = t.permute(0, 3, 1, 4, 2, 5)  # (B, C, h, p, w, p)
        return t.reshape(B, self.out_channels, self.h * self.patch, self.w * self.patch)
