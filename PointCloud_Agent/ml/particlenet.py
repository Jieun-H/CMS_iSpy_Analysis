"""
ParticleNet with a track encoder in front (HANDOFF.md section 2: two-level model).

  1. Track encoder: each track (33 Bezier points, inner -> outer, + per-track scalars) -> one
     token. "conv": 1D convolution along the points (order-aware). "mlp": end points and
     end directions only (the 12 numbers the Bezier curve is built from) - the ablation that
     shows whether the sequence model adds anything.
  2. Every other node (calorimeter hit, CSC/DT segment, RPC hit) -> token by a small MLP.
     Each node gets a learned node-type embedding (track, ecal_eb, ..., rpc_hits).
  3. EdgeConv blocks (Qu & Gouskos, ParticleNet). The first block finds the k nearest
     neighbours in (eta, phi) with phi wrapped, so a track's outer end point sits next to the
     calorimeter hits in front of it; later blocks use the learned features (dynamic graph).
     Padding nodes are never neighbours and are averaged out.
  4. Head. task "track": one logit per track (track token after EdgeConv + event mean).
     task "event": one logit per event (mean over nodes).

All shapes: B events, T tracks, M other nodes, N = T + M nodes, k neighbours.
"""
import math

import torch
from torch import nn


def _mlp(sizes, last_act=True):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers += [nn.LayerNorm(sizes[i + 1]), nn.ReLU()]
    return nn.Sequential(*layers)


class ConvTrackEncoder(nn.Module):
    """(B, T, 33, 3) points + (B, T, F) scalars -> (B, T, C)."""

    def __init__(self, n_scalars, dim, channels=(32, 64, 64)):
        super().__init__()
        layers, c_in = [], 3
        for i, c in enumerate(channels):
            layers += [nn.Conv1d(c_in, c, 3, stride=1 if i == 0 else 2, padding=1),
                       nn.BatchNorm1d(c), nn.ReLU()]
            c_in = c
        self.conv = nn.Sequential(*layers)
        self.out = _mlp([2 * c_in + n_scalars, dim, dim])

    def forward(self, points, scalars):
        B, T, P, D = points.shape
        x = self.conv(points.reshape(B * T, P, D).transpose(1, 2))      # (BT, C, P')
        x = torch.cat([x.mean(-1), x.amax(-1)], dim=-1).reshape(B, T, -1)
        return self.out(torch.cat([x, scalars], dim=-1))


class MLPTrackEncoder(nn.Module):
    """End points + end directions (12 numbers) + scalars -> (B, T, C)."""

    def __init__(self, n_scalars, dim):
        super().__init__()
        self.out = _mlp([12 + n_scalars, dim, dim])

    def forward(self, points, scalars):
        d_in = points[:, :, 1] - points[:, :, 0]
        d_out = points[:, :, -1] - points[:, :, -2]
        d_in = d_in / d_in.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        d_out = d_out / d_out.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return self.out(torch.cat([points[:, :, 0], points[:, :, -1], d_in, d_out, scalars], -1))


def knn(coord, mask, k, wrap_phi):
    """coord (B, N, D), mask (B, N) -> neighbour index (B, N, k), neighbour valid (B, N, k).
    wrap_phi: coord is (eta, phi) and the phi difference is taken modulo 2 pi."""
    coord = coord.float()
    diff = coord.unsqueeze(2) - coord.unsqueeze(1)                       # (B, N, N, D)
    if wrap_phi:
        dphi = torch.remainder(diff[..., 1] + math.pi, 2 * math.pi) - math.pi
        dist = diff[..., 0] ** 2 + dphi ** 2
    else:
        dist = (diff ** 2).sum(-1)
    N = coord.shape[1]
    big = torch.finfo(dist.dtype).max
    dist = dist.masked_fill(~mask.unsqueeze(1), big)                     # padding never a neighbour
    dist = dist.masked_fill(torch.eye(N, dtype=torch.bool, device=coord.device), big)  # no self loop
    kk = min(k, max(N - 1, 1))
    d, idx = dist.topk(kk, dim=-1, largest=False)
    return idx, d < big


def knn_features(x, mask, k):
    """Same as knn on features (B, N, C), memory-light: |a-b|^2 = |a|^2 + |b|^2 - 2ab."""
    x = x.float()
    sq = (x * x).sum(-1)
    dist = sq.unsqueeze(2) + sq.unsqueeze(1) - 2 * x @ x.transpose(1, 2)
    N = x.shape[1]
    big = torch.finfo(dist.dtype).max
    dist = dist.masked_fill(~mask.unsqueeze(1), big)
    dist = dist.masked_fill(torch.eye(N, dtype=torch.bool, device=x.device), big)
    kk = min(k, max(N - 1, 1))
    d, idx = dist.topk(kk, dim=-1, largest=False)
    return idx, d < big


class EdgeConv(nn.Module):
    """x (B, N, C) -> (B, N, C_out): mean over neighbours j of MLP([x_i, x_j - x_i]) + shortcut."""

    def __init__(self, c_in, channels):
        super().__init__()
        layers, c = [], 2 * c_in
        for c_out in channels:
            layers += [nn.Conv2d(c, c_out, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU()]
            c = c_out
        self.mlp = nn.Sequential(*layers)
        self.shortcut = nn.Sequential(nn.Conv1d(c_in, c, 1, bias=False), nn.BatchNorm1d(c))
        self.act = nn.ReLU()

    def forward(self, x, idx, valid, mask):
        B, N, C = x.shape
        k = idx.shape[-1]
        nb = torch.gather(x, 1, idx.reshape(B, N * k, 1).expand(-1, -1, C)).reshape(B, N, k, C)
        xi = x.unsqueeze(2).expand(-1, -1, k, -1)
        e = torch.cat([xi, nb - xi], dim=-1).permute(0, 3, 1, 2)          # (B, 2C, N, k)
        e = self.mlp(e)                                                   # (B, C', N, k)
        w = valid.unsqueeze(1).to(e.dtype)
        agg = (e * w).sum(-1) / w.sum(-1).clamp_min(1)                    # (B, C', N)
        out = self.act(agg + self.shortcut(x.transpose(1, 2)))
        return out.transpose(1, 2) * mask.unsqueeze(-1).to(out.dtype)


class TrackParticleNet(nn.Module):
    def __init__(self, n_scalars, n_types, other_width, task="track", track_encoder="conv",
                 dim=64, k=16, edge_channels=((64, 64, 64), (128, 128, 128), (256, 256, 256)),
                 fusion=128, head=(128,), dropout=0.1):
        super().__init__()
        if task not in ("track", "event"):
            raise ValueError(f"task must be track or event, not {task}")
        self.task = task
        self.k = k
        enc = {"conv": ConvTrackEncoder, "mlp": MLPTrackEncoder}[track_encoder]
        self.track_enc = enc(n_scalars, dim)
        self.other_enc = _mlp([other_width, dim, dim])
        self.type_emb = nn.Embedding(n_types, dim)
        self.blocks = nn.ModuleList()
        c = dim
        for chans in edge_channels:
            self.blocks.append(EdgeConv(c, chans))
            c = chans[-1]
        total = sum(ch[-1] for ch in edge_channels)
        self.fusion = nn.Sequential(nn.Linear(total, fusion), nn.LayerNorm(fusion), nn.ReLU())
        head_in = 2 * fusion if task == "track" else fusion
        layers, h = [], head_in
        for size in head:
            layers += [nn.Linear(h, size), nn.ReLU(), nn.Dropout(dropout)]
            h = size
        layers.append(nn.Linear(h, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, b):
        T = b["track_points"].shape[1]
        trk = self.track_enc(b["track_points"], b["track_scalars"])
        oth = self.other_enc(b["other_feats"])
        x = torch.cat([trk, oth], dim=1)                                   # (B, N, dim)
        types = torch.cat([torch.zeros_like(b["track_mask"], dtype=torch.long), b["other_type"]], 1)
        mask = torch.cat([b["track_mask"], b["other_mask"]], dim=1)
        coord = torch.cat([b["track_coord"], b["other_coord"]], dim=1)
        x = (x + self.type_emb(types)) * mask.unsqueeze(-1).to(x.dtype)

        outs = []
        for i, block in enumerate(self.blocks):
            if i == 0:
                idx, valid = knn(coord, mask, self.k, wrap_phi=True)
            else:
                idx, valid = knn_features(x, mask, self.k)
            x = block(x, idx, valid, mask)
            outs.append(x)
        x = self.fusion(torch.cat(outs, dim=-1))                           # (B, N, fusion)
        m = mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * m).sum(1) / m.sum(1).clamp_min(1)                    # (B, fusion)
        if self.task == "event":
            return self.head(pooled).squeeze(-1)                           # (B,)
        trk = x[:, :T]
        h = torch.cat([trk, pooled.unsqueeze(1).expand(-1, T, -1)], dim=-1)
        return self.head(h).squeeze(-1)                                    # (B, T)
