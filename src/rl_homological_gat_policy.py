# ******************************************************************************
#  Copyright (c) 2026, Marco Campos - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from rl_gat_baseline import EdgeBiasedGATLayer


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation_fn: type[nn.Module]) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation_fn())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class HomologicalTokenStructuredExtractor(BaseFeaturesExtractor):
    """Pass through cycle-token graph observations for the homological GAT."""

    def __init__(self, observation_space) -> None:
        super().__init__(observation_space, features_dim=1)
        self.node_dim = int(observation_space.spaces["node_x"].shape[-1])
        self.edge_dim = int(observation_space.spaces["edge_x"].shape[-1])
        self.global_dim = int(observation_space.spaces["global_x"].shape[-1])
        self.token_dim = int(observation_space.spaces["node_cycle_token_x"].shape[-1])
        self.num_nodes = int(observation_space.spaces["node_x"].shape[0])

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        node_x = obs["node_x"].float()
        edge_x = obs["edge_x"].float()
        edge_mask = obs["edge_mask"].float()
        global_x = obs["global_x"].float()
        token_x = obs["node_cycle_token_x"].float()
        token_mask = obs["node_cycle_token_mask"].float()

        if node_x.dim() == 2:
            node_x = node_x.unsqueeze(0)
            edge_x = edge_x.unsqueeze(0)
            edge_mask = edge_mask.unsqueeze(0)
            global_x = global_x.unsqueeze(0)
            token_x = token_x.unsqueeze(0)
            token_mask = token_mask.unsqueeze(0)

        return {
            "node_x": node_x,
            "edge_x": edge_x,
            "edge_mask": edge_mask,
            "global_x": global_x,
            "node_cycle_token_x": token_x,
            "node_cycle_token_mask": token_mask,
        }


class NodeCycleTokenEncoder(nn.Module):
    """Encode incident cycle tokens at each node via tiny self-attention + true/false query pooling."""

    def __init__(self, token_dim: int, hidden_dim: int = 16, num_heads: int = 2) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.true_query = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.false_query = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = scores.masked_fill(~mask, -1e9)
        probs = torch.softmax(scores, dim=-1)
        probs = torch.where(mask, probs, torch.zeros_like(probs))
        denom = torch.clamp(probs.sum(dim=-1, keepdim=True), min=1e-12)
        return probs / denom

    def forward(self, token_x: torch.Tensor, token_mask: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # token_x: [B, N, T, F], token_mask: [B, N, T]
        batch_size, num_nodes, max_tokens, _ = token_x.shape
        flat_tokens = token_x.reshape(batch_size * num_nodes, max_tokens, token_x.shape[-1])
        flat_mask = token_mask.reshape(batch_size * num_nodes, max_tokens) > 0.5

        embedded = self.token_mlp(flat_tokens)
        token_repr = torch.zeros_like(embedded)
        true_summary = torch.zeros((batch_size * num_nodes, self.hidden_dim), device=token_x.device, dtype=token_x.dtype)
        false_summary = torch.zeros_like(true_summary)
        true_weights = torch.zeros((batch_size * num_nodes, max_tokens), device=token_x.device, dtype=token_x.dtype)
        false_weights = torch.zeros_like(true_weights)
        any_valid = flat_mask.any(dim=1)

        if torch.any(any_valid):
            valid_tokens = embedded[any_valid]
            valid_mask = flat_mask[any_valid]
            attn_out, _ = self.self_attn(
                valid_tokens,
                valid_tokens,
                valid_tokens,
                key_padding_mask=~valid_mask,
                need_weights=False,
            )
            valid_tokens = self.norm1(valid_tokens + attn_out)
            ff_out = self.ff(valid_tokens)
            valid_tokens = self.norm2(valid_tokens + ff_out)
            token_repr[any_valid] = valid_tokens

            valid_raw = flat_tokens[any_valid]
            valid_true_mask = valid_mask & (valid_raw[..., 5] > 0.5)
            valid_false_mask = valid_mask & (valid_raw[..., 6] > 0.5)

            true_scores = torch.einsum("btd,d->bt", valid_tokens, self.true_query)
            false_scores = torch.einsum("btd,d->bt", valid_tokens, self.false_query)
            valid_true_weights = self._masked_softmax(true_scores, valid_true_mask)
            valid_false_weights = self._masked_softmax(false_scores, valid_false_mask)

            true_summary[any_valid] = torch.einsum("bt,btd->bd", valid_true_weights, valid_tokens)
            false_summary[any_valid] = torch.einsum("bt,btd->bd", valid_false_weights, valid_tokens)
            true_weights[any_valid] = valid_true_weights
            false_weights[any_valid] = valid_false_weights

        pooled = torch.cat([true_summary, false_summary], dim=-1).reshape(batch_size, num_nodes, 2 * self.hidden_dim)
        extras = {
            "token_pool_true": true_weights.reshape(batch_size, num_nodes, max_tokens),
            "token_pool_false": false_weights.reshape(batch_size, num_nodes, max_tokens),
        }
        return pooled, extras


class HomologicalGraphAttentionExtractor(BaseFeaturesExtractor):
    """Local homological token encoder followed by a compact node-centric GAT."""

    def __init__(
        self,
        observation_space,
        token_hidden_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(observation_space, features_dim=1)

        node_shape = observation_space.spaces["node_x"].shape
        edge_shape = observation_space.spaces["edge_x"].shape
        global_shape = observation_space.spaces["global_x"].shape
        token_shape = observation_space.spaces["node_cycle_token_x"].shape

        self.num_nodes = int(node_shape[0])
        self.node_dim = int(node_shape[1])
        self.edge_dim = int(edge_shape[-1])
        self.global_dim = int(global_shape[0])
        self.hidden_dim = int(hidden_dim)
        self.token_hidden_dim = int(token_hidden_dim)
        self.token_encoder = NodeCycleTokenEncoder(token_dim=int(token_shape[-1]), hidden_dim=token_hidden_dim, num_heads=2)
        self.node_proj = nn.Linear(self.node_dim + 2 * token_hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                EdgeBiasedGATLayer(
                    hidden_dim=hidden_dim,
                    edge_dim=self.edge_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.last_attention: Dict[str, torch.Tensor] = {}
        self.last_token_attention: Dict[str, torch.Tensor] = {}

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        node_x = obs["node_x"].float()
        edge_x = obs["edge_x"].float()
        edge_mask = obs["edge_mask"].float()
        global_x = obs["global_x"].float()
        token_x = obs["node_cycle_token_x"].float()
        token_mask = obs["node_cycle_token_mask"].float()

        if node_x.dim() == 2:
            node_x = node_x.unsqueeze(0)
            edge_x = edge_x.unsqueeze(0)
            edge_mask = edge_mask.unsqueeze(0)
            global_x = global_x.unsqueeze(0)
            token_x = token_x.unsqueeze(0)
            token_mask = token_mask.unsqueeze(0)

        cycle_feat, token_attn = self.token_encoder(token_x, token_mask)
        gat_input = torch.cat([node_x, cycle_feat], dim=-1)
        node_h = self.node_proj(gat_input)
        self.last_attention = {}
        self.last_token_attention = token_attn

        for layer_idx, layer in enumerate(self.layers):
            node_h, attn = layer(node_h, edge_x, edge_mask)
            self.last_attention[f"layer_{layer_idx}"] = attn

        return {
            "node_h": node_h,
            "node_x": node_x,
            "edge_x": edge_x,
            "edge_mask": edge_mask,
            "global_x": global_x,
        }


class HomologicalGATActorCriticPolicy(ActorCriticPolicy):
    """Joint actor on top of a compact homological token + GAT encoder."""

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = nn.Identity()

    def _build(self, lr_schedule) -> None:
        self._build_mlp_extractor()

        if not isinstance(self.features_extractor, HomologicalGraphAttentionExtractor):
            raise TypeError("HomologicalGATActorCriticPolicy requires HomologicalGraphAttentionExtractor.")

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
        self.actor_summary_dim = hidden_dim + self.features_extractor.edge_dim + global_dim
        self.critic_summary_dim = 2 * hidden_dim + 3 + global_dim
        critic_in_dim = self.critic_summary_dim
        actor_in_dim = self.features_extractor.num_nodes * hidden_dim + self.actor_summary_dim

        self.actor_head = _build_mlp(
            input_dim=actor_in_dim,
            hidden_dims=actor_hidden,
            output_dim=self.total_action_dim,
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

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float()
        while mask.dim() < values.dim():
            mask = mask.unsqueeze(-1)
        weighted = values * mask
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        return weighted.sum(dim=1) / denom

    def _actor_summary(self, structured: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_h = structured["node_h"]
        edge_x = structured["edge_x"]
        edge_mask = structured["edge_mask"]
        global_x = structured["global_x"]

        pooled_nodes = node_h.mean(dim=1)
        masked_edge_features = edge_x * edge_mask.unsqueeze(-1)
        pooled_edges = masked_edge_features.sum(dim=(1, 2)) / torch.clamp(edge_mask.sum(dim=(1, 2)).unsqueeze(-1), min=1.0)
        return torch.cat([pooled_nodes, pooled_edges, global_x], dim=-1)

    def _critic_summary(self, structured: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_h = structured["node_h"]
        node_x = structured["node_x"]
        global_x = structured["global_x"]

        mobile_mask = node_x[:, :, 1] > 0.5
        true_mask = node_x[:, :, 4] > 0.5
        incident_true = node_x[:, :, 6:7]

        mobile_pool = self._masked_mean(node_h, mobile_mask)
        true_pool = self._masked_mean(node_h, true_mask)
        true_inc_mean = self._masked_mean(incident_true, true_mask)
        true_inc_sum = (incident_true * true_mask.unsqueeze(-1).float()).sum(dim=1)
        true_node_fraction = true_mask.float().mean(dim=1, keepdim=True)

        return torch.cat(
            [mobile_pool, true_pool, true_inc_mean, true_inc_sum, true_node_fraction, global_x],
            dim=-1,
        )

    def _actor_mean_actions(self, structured: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_h = structured["node_h"]
        global_summary = self._actor_summary(structured)
        actor_in = torch.cat([node_h.reshape(node_h.shape[0], -1), global_summary], dim=-1)
        return self.actor_head(actor_in)

    def _critic_values(self, structured: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.critic_head(self._critic_summary(structured))

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


class HomologicalGATLegacyActorCriticPolicy(HomologicalGATActorCriticPolicy):
    """Compatibility policy for homological GAT checkpoints saved before critic changes."""

    def _build(self, lr_schedule) -> None:
        self._build_mlp_extractor()

        if not isinstance(self.features_extractor, HomologicalGraphAttentionExtractor):
            raise TypeError("HomologicalGATLegacyActorCriticPolicy requires HomologicalGraphAttentionExtractor.")

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
        self.actor_summary_dim = hidden_dim + self.features_extractor.edge_dim + global_dim
        self.critic_summary_dim = self.actor_summary_dim
        actor_in_dim = self.features_extractor.num_nodes * hidden_dim + self.actor_summary_dim
        critic_in_dim = self.critic_summary_dim

        self.actor_head = _build_mlp(
            input_dim=actor_in_dim,
            hidden_dims=actor_hidden,
            output_dim=self.total_action_dim,
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

    def _critic_summary(self, structured: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._actor_summary(structured)


def load_homological_gat_ppo(checkpoint_path, **kwargs):
    """Load PPO with compatibility fallback for legacy homological GAT checkpoints."""
    from stable_baselines3 import PPO

    try:
        return PPO.load(str(checkpoint_path), **kwargs)
    except RuntimeError as exc:
        message = str(exc)
        if (
            "HomologicalGATActorCriticPolicy" not in message
            or "critic_head.0.weight" not in message
            or "size mismatch" not in message
        ):
            raise

    custom_objects = dict(kwargs.pop("custom_objects", {}))
    custom_objects["policy_class"] = HomologicalGATLegacyActorCriticPolicy
    return PPO.load(str(checkpoint_path), custom_objects=custom_objects, **kwargs)
