"""GraphCast-mini: icosahedral-mesh encode-process-decode GNN.

Hand-rolled (no torch-geometric): message passing uses precomputed edge index
tensors and index_add_ scatter sums. Grid pixels are encoded to their nearest
mesh node, processed with several rounds of mesh message passing, and decoded
back to each pixel from its 3 nearest mesh nodes with distance-aware MLPs.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def icosphere(refinements: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices (V, 3) unit sphere, edges (E, 2) undirected unique)."""
    phi = (1 + 5**0.5) / 2
    verts = np.array(
        [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
        ],
        dtype=np.float64,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ]
    )
    for _ in range(refinements):
        verts_list = list(verts)
        midpoint: dict[tuple[int, int], int] = {}

        def mid(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in midpoint:
                m = verts_list[a] + verts_list[b]
                m /= np.linalg.norm(m)
                midpoint[key] = len(verts_list)
                verts_list.append(m)
            return midpoint[key]

        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = np.array(new_faces)
        verts = np.array(verts_list)

    edges = set()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add((min(u, v), max(u, v)))
    return verts.astype(np.float32), np.array(sorted(edges), dtype=np.int64)


def latlon_to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    ).astype(np.float32)


def mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, out_dim)
    )


class MeshGNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 25,
        out_channels: int = 8,
        img_size: tuple[int, int] = (32, 64),
        refinements: int = 3,
        hidden: int = 128,
        rounds: int = 6,
    ):
        super().__init__()
        H, W = img_size
        self.out_channels = out_channels
        lat = np.linspace(-87.1875, 87.1875, H)
        lon = np.arange(W) * (360.0 / W)
        grid_xyz = latlon_to_xyz(
            np.repeat(lat, W), np.tile(lon, H)
        )  # (H*W, 3)
        mesh_xyz, mesh_edges = icosphere(refinements)
        V = len(mesh_xyz)

        # grid -> nearest mesh node; decode from 3 nearest mesh nodes
        d = grid_xyz @ mesh_xyz.T  # cosine similarity
        g2m = d.argmax(axis=1)
        m2g_idx = np.argsort(-d, axis=1)[:, :3]  # (H*W, 3)
        rel = grid_xyz[:, None, :] - mesh_xyz[m2g_idx]  # (H*W, 3, 3)

        self.register_buffer("g2m", torch.from_numpy(g2m))
        self.register_buffer("m2g_idx", torch.from_numpy(m2g_idx))
        self.register_buffer("m2g_rel", torch.from_numpy(rel.astype(np.float32)))
        self.register_buffer("edge_src", torch.from_numpy(
            np.concatenate([mesh_edges[:, 0], mesh_edges[:, 1]])))
        self.register_buffer("edge_dst", torch.from_numpy(
            np.concatenate([mesh_edges[:, 1], mesh_edges[:, 0]])))
        self.register_buffer("mesh_pos", torch.from_numpy(mesh_xyz))
        edge_vec = mesh_xyz[mesh_edges[:, 1]] - mesh_xyz[mesh_edges[:, 0]]
        edge_feat = np.concatenate([edge_vec, -edge_vec])
        self.register_buffer("edge_vec", torch.from_numpy(edge_feat))
        self.V, self.HW = V, H * W

        self.grid_embed = mlp(in_channels, hidden, hidden)
        self.node_init = mlp(hidden + 3, hidden, hidden)
        self.edge_mlps = nn.ModuleList(
            [mlp(2 * hidden + 3, hidden, hidden) for _ in range(rounds)]
        )
        self.node_mlps = nn.ModuleList(
            [mlp(2 * hidden, hidden, hidden) for _ in range(rounds)]
        )
        self.decode_gather = mlp(hidden + 3, hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, out_channels)
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        g = x.permute(0, 2, 3, 1).reshape(B, self.HW, C)
        g = self.grid_embed(g)  # (B, HW, h)

        # encode: mean of grid features per mesh node
        nodes = g.new_zeros(B, self.V, g.shape[-1])
        counts = g.new_zeros(self.V)
        counts.index_add_(0, self.g2m, torch.ones_like(self.g2m, dtype=g.dtype))
        nodes.index_add_(1, self.g2m, g)
        nodes = nodes / counts.clamp(min=1)[None, :, None]
        pos = self.mesh_pos[None].expand(B, -1, -1)
        nodes = self.node_init(torch.cat([nodes, pos], dim=-1))

        # process: rounds of edge->node message passing with residuals
        for edge_mlp, node_mlp in zip(self.edge_mlps, self.node_mlps):
            src = nodes[:, self.edge_src]
            dst = nodes[:, self.edge_dst]
            ev = self.edge_vec[None].expand(B, -1, -1)
            msg = edge_mlp(torch.cat([src, dst, ev], dim=-1))
            agg = nodes.new_zeros(nodes.shape)
            agg.index_add_(1, self.edge_dst, msg)
            nodes = nodes + node_mlp(torch.cat([nodes, agg], dim=-1))

        # decode: gather 3 nearest nodes with relative-position MLP
        gathered = nodes[:, self.m2g_idx]  # (B, HW, 3, h)
        rel = self.m2g_rel[None].expand(B, -1, -1, -1)
        dec = self.decode_gather(torch.cat([gathered, rel], dim=-1)).mean(dim=2)
        out = self.head(torch.cat([dec, g], dim=-1))  # (B, HW, C_out)
        return out.permute(0, 2, 1).reshape(B, self.out_channels, H, W)
