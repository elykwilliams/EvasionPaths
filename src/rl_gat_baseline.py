# ******************************************************************************
#  Copyright (c) 2026, Marco Campos - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************

from __future__ import annotations

from typing import Dict

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch.nn import functional as F


class EdgeBiasedGATLayer(nn.Module):
    """Node-centric dense GAT layer with edge-feature attention bias."""

    def __init__(self, hidden_dim: int, edge_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads),
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self,
        node_h: torch.Tensor,
        edge_x: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, _ = node_h.shape

        q = self.q_proj(node_h).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(node_h).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(node_h).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.einsum("bhid,bhjd->bhij", q, k) / (self.head_dim**0.5)
        scores = self.leaky_relu(scores)
        edge_bias = self.edge_bias(edge_x).permute(0, 3, 1, 2)
        scores = scores + edge_bias

        valid = edge_mask.bool()
        eye = torch.eye(num_nodes, device=edge_mask.device, dtype=torch.bool).unsqueeze(0)
        valid = torch.logical_or(valid, eye)
        scores = scores.masked_fill(~valid.unsqueeze(1), -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.hidden_dim)
        out = self.out_proj(out)
        node_h = self.norm(node_h + out)
        node_h = F.relu(node_h)
        return node_h, attn


class BaselineGraphAttentionExtractor(BaseFeaturesExtractor):
    """
    Baseline node-centric GAT extractor over the alpha-complex graph.

    The learned state lives on nodes; edge features only bias message passing.
    Output is a single flat feature vector for the standard shared PPO actor/critic heads.
    """

    def __init__(
        self,
        observation_space,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 2,
        dropout: float = 0.0,
    ) -> None:
        node_shape = observation_space.spaces["node_x"].shape
        edge_shape = observation_space.spaces["edge_x"].shape
        global_shape = observation_space.spaces["global_x"].shape

        self.num_nodes = int(node_shape[0])
        self.node_dim = int(node_shape[1])
        self.edge_dim = int(edge_shape[-1])
        self.global_dim = int(global_shape[0])
        self.hidden_dim = int(hidden_dim)

        features_dim = self.num_nodes * self.hidden_dim + 2 * self.hidden_dim + self.global_dim
        super().__init__(observation_space, features_dim=features_dim)

        self.node_proj = nn.Linear(self.node_dim, self.hidden_dim)
        self.layers = nn.ModuleList(
            [
                EdgeBiasedGATLayer(
                    hidden_dim=self.hidden_dim,
                    edge_dim=self.edge_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.last_attention: Dict[str, torch.Tensor] = {}

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_x = obs["node_x"].float()
        edge_x = obs["edge_x"].float()
        edge_mask = obs["edge_mask"].float()
        global_x = obs["global_x"].float()

        if node_x.dim() == 2:
            node_x = node_x.unsqueeze(0)
            edge_x = edge_x.unsqueeze(0)
            edge_mask = edge_mask.unsqueeze(0)
            global_x = global_x.unsqueeze(0)

        node_h = self.node_proj(node_x)
        self.last_attention = {}

        for layer_idx, layer in enumerate(self.layers):
            node_h, attn = layer(node_h, edge_x, edge_mask)
            self.last_attention[f"layer_{layer_idx}"] = attn

        pooled_nodes = node_h.mean(dim=1)
        masked_edge_features = edge_x * edge_mask.unsqueeze(-1)
        pooled_edges = masked_edge_features.sum(dim=(1, 2)) / torch.clamp(edge_mask.sum(dim=(1, 2)).unsqueeze(-1), min=1.0)
        pooled_edges = F.pad(
            pooled_edges,
            (0, max(0, self.hidden_dim - pooled_edges.shape[-1])),
            mode="constant",
            value=0.0,
        )[:, : self.hidden_dim]
        return torch.cat([node_h.reshape(node_h.shape[0], -1), pooled_nodes, pooled_edges, global_x], dim=-1)
