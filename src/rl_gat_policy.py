# ******************************************************************************
#  Copyright (c) 2026, Marco Campos - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************

from __future__ import annotations

from typing import Dict

import torch
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch.nn import functional as F


class DartAttentionAggregator(nn.Module):
    """Attention from node queries onto directed-dart embeddings."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self,
        node_h: torch.Tensor,
        dart_h: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # node_h: [B, N, D], dart_h: [B, N, N, D], mask: [B, N, N]
        batch_size, num_nodes, _ = node_h.shape

        q = self.q_proj(node_h).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(dart_h).view(batch_size, num_nodes, num_nodes, self.num_heads, self.head_dim)
        k = k.permute(0, 3, 1, 2, 4)
        v = self.v_proj(dart_h).view(batch_size, num_nodes, num_nodes, self.num_heads, self.head_dim)
        v = v.permute(0, 3, 1, 2, 4)

        scores = torch.einsum("bhid,bhijd->bhij", q, k) / (self.head_dim**0.5)
        scores = self.leaky_relu(scores)
        valid = mask.bool()
        eye = torch.eye(num_nodes, device=mask.device, dtype=torch.bool).unsqueeze(0)
        valid = torch.logical_or(valid, eye)
        scores = scores.masked_fill(~valid.unsqueeze(1), -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("bhij,bhijd->bhid", attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.hidden_dim)
        out = self.out_proj(out)
        return out, attn


class DartUpdate(nn.Module):
    """Update directed-dart embeddings from endpoint node states and raw dart features."""

    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, node_h: torch.Tensor, dart_h: torch.Tensor, edge_x: torch.Tensor) -> torch.Tensor:
        src = node_h.unsqueeze(2).expand(-1, -1, node_h.shape[1], -1)
        dst = node_h.unsqueeze(1).expand(-1, node_h.shape[1], -1, -1)
        return self.mlp(torch.cat([src, dst, dart_h, edge_x], dim=-1))


class DartAwareBlock(nn.Module):
    """Alternating node-edge block with separate incoming/outgoing dart attention."""

    def __init__(self, hidden_dim: int, num_heads: int, edge_dim: int, global_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dart_update = DartUpdate(hidden_dim=hidden_dim, edge_dim=edge_dim)
        self.dart_norm = nn.LayerNorm(hidden_dim)

        self.in_agg = DartAttentionAggregator(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.out_agg = DartAttentionAggregator(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.node_update = nn.Sequential(
            nn.Linear(3 * hidden_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_h: torch.Tensor,
        dart_h: torch.Tensor,
        edge_x: torch.Tensor,
        edge_mask: torch.Tensor,
        global_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dart_delta = self.dart_update(node_h, dart_h, edge_x)
        dart_h = self.dart_norm(dart_h + dart_delta)
        dart_h = F.relu(dart_h)

        out_msg, out_attn = self.out_agg(node_h, dart_h, edge_mask > 0.5)
        in_msg_raw, in_attn_raw = self.in_agg(node_h, dart_h.transpose(1, 2), edge_mask.transpose(1, 2) > 0.5)

        global_rep = global_x.unsqueeze(1).expand(-1, node_h.shape[1], -1)
        node_delta = self.node_update(torch.cat([node_h, in_msg_raw, out_msg, global_rep], dim=-1))
        node_h = self.node_norm(node_h + node_delta)
        node_h = F.relu(node_h)
        return node_h, dart_h, in_attn_raw, out_attn


class GraphAttentionExtractor(BaseFeaturesExtractor):
    """
    Dart-aware node-edge attention encoder for structured evasion state.

    Expects observation dict keys:
      - node_x: [B, N, F_n]
      - edge_x: [B, N, N, F_e]
      - edge_mask: [B, N, N]
      - global_x: [B, F_g]
    """

    def __init__(
        self,
        observation_space,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(observation_space, features_dim=1)

        node_shape = observation_space.spaces["node_x"].shape
        edge_shape = observation_space.spaces["edge_x"].shape
        global_shape = observation_space.spaces["global_x"].shape

        self.num_nodes = int(node_shape[0])
        self.node_dim = int(node_shape[1])
        self.edge_dim = int(edge_shape[-1])
        self.global_dim = int(global_shape[0])
        self.hidden_dim = int(hidden_dim)

        self.node_proj = nn.Linear(self.node_dim, self.hidden_dim)
        self.edge_proj = nn.Linear(self.edge_dim, self.hidden_dim)
        self.layers = nn.ModuleList(
            [
                DartAwareBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=num_heads,
                    edge_dim=self.edge_dim,
                    global_dim=self.global_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self._features_dim = 1
        self.last_attention: Dict[str, torch.Tensor] = {}

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
        dart_h = self.edge_proj(edge_x)
        self.last_attention = {}

        for layer_idx, layer in enumerate(self.layers):
            node_h, dart_h, in_attn, out_attn = layer(node_h, dart_h, edge_x, edge_mask, global_x)
            self.last_attention[f"layer_{layer_idx}_in"] = in_attn
            self.last_attention[f"layer_{layer_idx}_out"] = out_attn

        return {
            "node_h": node_h,
            "dart_h": dart_h,
            "global_x": global_x,
            "node_x": node_x,
            "edge_mask": edge_mask,
        }


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation_fn: type[nn.Module]) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation_fn())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class DartAwareActorCriticPolicy(ActorCriticPolicy):
    """Actor-critic split with local mobile-node actor and global pooled critic."""

    def _build_mlp_extractor(self) -> None:
        # PPO internals expect this attribute to exist even though we bypass it below.
        self.mlp_extractor = nn.Identity()

    def _build(self, lr_schedule) -> None:
        self._build_mlp_extractor()

        if not isinstance(self.features_extractor, GraphAttentionExtractor):
            raise TypeError("DartAwareActorCriticPolicy requires GraphAttentionExtractor.")

        if not isinstance(self.net_arch, dict):
            actor_hidden = list(self.net_arch)
            critic_hidden = list(self.net_arch)
        else:
            actor_hidden = list(self.net_arch.get("pi", []))
            critic_hidden = list(self.net_arch.get("vf", []))

        self.num_mobile = int(self.action_space.shape[0])
        self.action_dim_per_mobile = int(self.action_space.shape[1])
        self.total_action_dim = int(np.prod(self.action_space.shape))

        hidden_dim = int(self.features_extractor.hidden_dim)
        global_dim = int(self.features_extractor.global_dim)
        critic_in_dim = hidden_dim + hidden_dim + global_dim
        actor_in_dim = hidden_dim + critic_in_dim

        self.actor_head = _build_mlp(
            input_dim=actor_in_dim,
            hidden_dims=actor_hidden,
            output_dim=self.action_dim_per_mobile,
            activation_fn=self.activation_fn,
        )
        self.critic_head = _build_mlp(
            input_dim=critic_in_dim,
            hidden_dims=critic_hidden,
            output_dim=1,
            activation_fn=self.activation_fn,
        )
        self.log_std = nn.Parameter(torch.ones(self.total_action_dim) * self.log_std_init, requires_grad=True)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.actor_head: 0.01,
                self.critic_head: 1.0,
            }
            for module, gain in module_gains.items():
                module.apply(lambda m: self.init_weights(m, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _mobile_node_embeddings(self, structured: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_h = structured["node_h"]
        node_x = structured["node_x"]
        mobile_mask = node_x[:, :, 1] > 0.5
        batch_size = node_h.shape[0]
        mobile_counts = mobile_mask.sum(dim=1)
        if not torch.all(mobile_counts == self.num_mobile):
            raise ValueError(f"Expected {self.num_mobile} mobile nodes per batch item, got {mobile_counts.tolist()}.")
        return node_h[mobile_mask].view(batch_size, self.num_mobile, node_h.shape[-1])

    def _global_summary(self, structured: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_h = structured["node_h"]
        dart_h = structured["dart_h"]
        global_x = structured["global_x"]
        edge_mask = structured["edge_mask"]

        pooled_nodes = node_h.mean(dim=1)
        dart_mask = edge_mask.unsqueeze(-1)
        pooled_darts = (dart_h * dart_mask).sum(dim=(1, 2)) / torch.clamp(dart_mask.sum(dim=(1, 2)), min=1.0)
        return torch.cat([pooled_nodes, pooled_darts, global_x], dim=-1)

    def _actor_mean_actions(self, structured: Dict[str, torch.Tensor]) -> torch.Tensor:
        mobile_h = self._mobile_node_embeddings(structured)
        global_summary = self._global_summary(structured).unsqueeze(1).expand(-1, self.num_mobile, -1)
        actor_in = torch.cat([mobile_h, global_summary], dim=-1)
        return self.actor_head(actor_in).reshape(mobile_h.shape[0], -1)

    def _critic_values(self, structured: Dict[str, torch.Tensor]) -> torch.Tensor:
        critic_input = self._global_summary(structured)
        return self.critic_head(critic_input)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        return self.action_dist.proba_distribution(latent_pi, self.log_std)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        structured = self.extract_features(obs)
        mean_actions = self._actor_mean_actions(structured)
        distribution = self._get_action_dist_from_latent(mean_actions)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self._critic_values(structured)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        structured = self.extract_features(obs)
        mean_actions = self._actor_mean_actions(structured)
        distribution = self._get_action_dist_from_latent(mean_actions)
        flat_actions = actions.reshape(actions.shape[0], -1)
        log_prob = distribution.log_prob(flat_actions)
        entropy = distribution.entropy()
        values = self._critic_values(structured)
        return values, log_prob, entropy

    def get_distribution(self, obs: torch.Tensor):
        structured = self.extract_features(obs)
        mean_actions = self._actor_mean_actions(structured)
        return self._get_action_dist_from_latent(mean_actions)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        structured = self.extract_features(obs)
        return self._critic_values(structured)
