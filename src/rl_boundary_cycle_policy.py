# ******************************************************************************
#  Copyright (c) 2026, Marco Campos - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


def _build_mlp(input_dim: int, hidden_dims: List[int], output_dim: int, activation_fn: type[nn.Module]) -> nn.Sequential:
    layers: List[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation_fn())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class BoundaryCycleStructuredExtractor(BaseFeaturesExtractor):
    """Pass through the structured cycle-graph observation for the custom actor."""

    def __init__(self, observation_space) -> None:
        super().__init__(observation_space, features_dim=1)
        self.node_dim = int(observation_space.spaces["node_x"].shape[-1])
        self.edge_dim = int(observation_space.spaces["edge_x"].shape[-1])
        self.global_dim = int(observation_space.spaces["global_x"].shape[-1])
        self.num_nodes = int(observation_space.spaces["node_x"].shape[0])

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        node_x = obs["node_x"].float()
        edge_x = obs["edge_x"].float()
        edge_mask = obs["edge_mask"].float()
        global_x = obs["global_x"].float()
        cycle_index = obs["cycle_node_index"].long()
        cycle_node_mask = obs["cycle_node_mask"].float()
        cycle_mask = obs["cycle_mask"].float()
        cycle_is_true = obs["cycle_is_true"].float()

        if node_x.dim() == 2:
            node_x = node_x.unsqueeze(0)
            edge_x = edge_x.unsqueeze(0)
            edge_mask = edge_mask.unsqueeze(0)
            global_x = global_x.unsqueeze(0)
            cycle_index = cycle_index.unsqueeze(0)
            cycle_node_mask = cycle_node_mask.unsqueeze(0)
            cycle_mask = cycle_mask.unsqueeze(0)
            cycle_is_true = cycle_is_true.unsqueeze(0)

        return {
            "node_x": node_x,
            "edge_x": edge_x,
            "edge_mask": edge_mask,
            "global_x": global_x,
            "cycle_node_index": cycle_index,
            "cycle_node_mask": cycle_node_mask,
            "cycle_mask": cycle_mask,
            "cycle_is_true": cycle_is_true,
        }


class CircularBoundaryBlock(nn.Module):
    """Local circular residual block over boundary-node tokens."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prev_x = torch.roll(x, shifts=1, dims=1)
        next_x = torch.roll(x, shifts=-1, dims=1)
        delta = self.update(torch.cat([prev_x, x, next_x], dim=-1))
        return torch.relu(self.norm(x + delta))


class BoundaryCycleEncoder(nn.Module):
    """Encode one cycle in both orientations and average the aligned embeddings."""

    def __init__(self, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.token_proj = nn.Linear(8, hidden_dim)
        self.blocks = nn.ModuleList([CircularBoundaryBlock(hidden_dim) for _ in range(num_layers)])
        self.pool_score = nn.Linear(hidden_dim, 1)

    def _encode_direction(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.token_proj(tokens)
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_fwd = self._encode_direction(tokens)
        h_rev = self._encode_direction(torch.flip(tokens, dims=[1]))
        h = 0.5 * (h_fwd + torch.flip(h_rev, dims=[1]))
        weights = torch.softmax(self.pool_score(h).squeeze(-1), dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * h, dim=1)
        return h, pooled


class SetContextBlock(nn.Module):
    """Small set encoder for cycle summaries plus the time token."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)


class MobilePropagationBlock(nn.Module):
    """Local message passing over the mobile-only alpha-complex graph."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, mobile_h: torch.Tensor, edge_feat: torch.Tensor, edge_mask: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        num_mobile = mobile_h.shape[0]
        context_rep = context.unsqueeze(0).expand(num_mobile, -1)
        src = mobile_h.unsqueeze(1).expand(-1, num_mobile, -1)
        dst = mobile_h.unsqueeze(0).expand(num_mobile, -1, -1)
        ctx = context_rep.unsqueeze(1).expand(-1, num_mobile, -1)
        messages = self.message_mlp(torch.cat([src, dst, edge_feat, ctx], dim=-1))
        mask = edge_mask.unsqueeze(-1)
        masked_messages = messages * mask
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        agg = masked_messages.sum(dim=1) / denom
        delta = self.update_mlp(torch.cat([mobile_h, agg, context_rep], dim=-1))
        return torch.relu(self.norm(mobile_h + delta))


class BoundaryCycleActorCriticPolicy(ActorCriticPolicy):
    """Boundary-cycle actor with local mobile propagation and a shared critic."""

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = nn.Identity()

    def _build(self, lr_schedule) -> None:
        self._build_mlp_extractor()

        if not isinstance(self.features_extractor, BoundaryCycleStructuredExtractor):
            raise TypeError("BoundaryCycleActorCriticPolicy requires BoundaryCycleStructuredExtractor.")

        if not isinstance(self.net_arch, dict):
            actor_hidden = list(self.net_arch)
            critic_hidden = list(self.net_arch)
        else:
            actor_hidden = list(self.net_arch.get("pi", []))
            critic_hidden = list(self.net_arch.get("vf", []))

        hidden_dim = 128
        self.hidden_dim = hidden_dim
        self.time_proj = nn.Linear(self.features_extractor.global_dim, hidden_dim)
        self.cycle_encoder = BoundaryCycleEncoder(hidden_dim=hidden_dim, num_layers=3)
        self.set_context = SetContextBlock(hidden_dim=hidden_dim, num_heads=2)
        self.mobile_proj = nn.Linear(4, hidden_dim)
        self.seed_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.mobile_blocks = nn.ModuleList([MobilePropagationBlock(hidden_dim=hidden_dim) for _ in range(3)])

        self.num_mobile = int(self.action_space.shape[0])
        self.action_dim_per_mobile = int(self.action_space.shape[1])
        self.total_action_dim = int(np.prod(self.action_space.shape))

        actor_in_dim = 2 * hidden_dim
        critic_in_dim = 3 * hidden_dim
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
        self.last_attention: Dict[str, torch.Tensor] = {}

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.cycle_encoder: np.sqrt(2),
                self.time_proj: np.sqrt(2),
                self.mobile_proj: np.sqrt(2),
                self.seed_proj: np.sqrt(2),
                self.actor_head: 0.01,
                self.critic_head: 1.0,
            }
            for module, gain in module_gains.items():
                module.apply(lambda m: self.init_weights(m, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    @staticmethod
    def _edge_relative_vector(edge_x: torch.Tensor) -> torch.Tensor:
        return edge_x[..., 0:1] * edge_x[..., 1:3]

    def _mobile_global_indices(self, structured: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        node_x = structured["node_x"][batch_idx]
        mobile_mask = node_x[:, 1] > 0.5
        indices = torch.nonzero(mobile_mask, as_tuple=False).squeeze(-1)
        if indices.numel() != self.num_mobile:
            raise ValueError(f"Expected {self.num_mobile} mobile nodes, got {int(indices.numel())}.")
        return indices

    def _build_boundary_tokens(
        self,
        structured: Dict[str, torch.Tensor],
        batch_idx: int,
        cycle_nodes: torch.Tensor,
        cycle_is_true: torch.Tensor,
    ) -> torch.Tensor:
        node_x = structured["node_x"][batch_idx]
        edge_x = structured["edge_x"][batch_idx]
        prev_nodes = torch.roll(cycle_nodes, shifts=1, dims=1)
        next_nodes = torch.roll(cycle_nodes, shifts=-1, dims=1)

        in_edge = edge_x[prev_nodes, cycle_nodes]
        out_edge = edge_x[cycle_nodes, next_nodes]
        x_in = self._edge_relative_vector(in_edge)
        x_out = self._edge_relative_vector(out_edge)

        prev_mobile = node_x[prev_nodes, 1:2]
        center_mobile = node_x[cycle_nodes, 1:2]
        next_mobile = node_x[next_nodes, 1:2]
        cycle_label = cycle_is_true.view(-1, 1, 1).expand(-1, cycle_nodes.shape[1], 1)
        return torch.cat([x_in, x_out, center_mobile, prev_mobile, next_mobile, cycle_label], dim=-1)

    def _encode_batch_item(
        self,
        structured: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cycle_index = structured["cycle_node_index"][batch_idx]
        cycle_node_mask = structured["cycle_node_mask"][batch_idx]
        cycle_mask = structured["cycle_mask"][batch_idx]
        cycle_is_true = structured["cycle_is_true"][batch_idx]
        node_x = structured["node_x"][batch_idx]
        edge_x = structured["edge_x"][batch_idx]
        edge_mask = structured["edge_mask"][batch_idx]
        global_x = structured["global_x"][batch_idx]

        time_token = self.time_proj(global_x.unsqueeze(0)).squeeze(0)
        mobile_indices = self._mobile_global_indices(structured, batch_idx)
        mobile_node_x = node_x[mobile_indices]
        mobile_h = self.mobile_proj(torch.cat([mobile_node_x[:, 2:4], mobile_node_x[:, 6:8]], dim=-1))

        seed_accum_true = torch.zeros_like(mobile_h)
        seed_accum_false = torch.zeros_like(mobile_h)
        seed_count_true = torch.zeros((self.num_mobile, 1), device=mobile_h.device, dtype=mobile_h.dtype)
        seed_count_false = torch.zeros((self.num_mobile, 1), device=mobile_h.device, dtype=mobile_h.dtype)
        cycle_embeddings: List[torch.Tensor] = []
        valid_counts = cycle_node_mask.sum(dim=1).long()
        valid_rows = torch.nonzero((cycle_mask > 0.5) & (valid_counts >= 2), as_tuple=False).squeeze(-1)

        if valid_rows.numel() > 0:
            mobile_lookup = torch.full(
                (node_x.shape[0],),
                -1,
                dtype=torch.long,
                device=node_x.device,
            )
            mobile_lookup[mobile_indices] = torch.arange(self.num_mobile, device=node_x.device, dtype=torch.long)

            cycle_nodes_all = cycle_index[valid_rows]
            cycle_lengths = valid_counts[valid_rows]
            unique_lengths = torch.unique(cycle_lengths, sorted=True)

            for length in unique_lengths.tolist():
                group_mask = cycle_lengths == int(length)
                group_rows = valid_rows[group_mask]
                cycle_nodes = cycle_index[group_rows, : int(length)]
                group_cycle_is_true = cycle_is_true[group_rows]
                tokens = self._build_boundary_tokens(structured, batch_idx, cycle_nodes, group_cycle_is_true)
                node_h, cycle_h = self.cycle_encoder(tokens)
                cycle_embeddings.extend(cycle_h.unbind(dim=0))

                mobile_targets = mobile_lookup[cycle_nodes]
                valid_mobile = mobile_targets >= 0
                if torch.any(valid_mobile):
                    flat_targets = mobile_targets[valid_mobile]
                    flat_h = node_h[valid_mobile]
                    flat_cycle_is_true = group_cycle_is_true.unsqueeze(1).expand(-1, int(length))[valid_mobile] > 0.5
                    if torch.any(flat_cycle_is_true):
                        true_targets = flat_targets[flat_cycle_is_true]
                        true_h = flat_h[flat_cycle_is_true]
                        seed_accum_true.index_add_(0, true_targets, true_h)
                        seed_count_true.index_add_(
                            0,
                            true_targets,
                            torch.ones((true_targets.shape[0], 1), device=seed_count_true.device, dtype=seed_count_true.dtype),
                        )
                    false_mobile = ~flat_cycle_is_true
                    if torch.any(false_mobile):
                        false_targets = flat_targets[false_mobile]
                        false_h = flat_h[false_mobile]
                        seed_accum_false.index_add_(0, false_targets, false_h)
                        seed_count_false.index_add_(
                            0,
                            false_targets,
                            torch.ones((false_targets.shape[0], 1), device=seed_count_false.device, dtype=seed_count_false.dtype),
                        )

        seeded = (seed_count_true.squeeze(-1) > 0) | (seed_count_false.squeeze(-1) > 0)
        if torch.any(seeded):
            true_seed = seed_accum_true / torch.clamp(seed_count_true, min=1.0)
            false_seed = seed_accum_false / torch.clamp(seed_count_false, min=1.0)
            seed_features = torch.cat([true_seed, false_seed], dim=-1)
            mobile_h[seeded] = mobile_h[seeded] + self.seed_proj(seed_features[seeded])

        if cycle_embeddings:
            cycle_tensor = torch.stack(cycle_embeddings, dim=0)
            set_tokens = torch.cat([time_token.unsqueeze(0), cycle_tensor], dim=0).unsqueeze(0)
            set_encoded = self.set_context(set_tokens).squeeze(0)
            context = set_encoded[0]
            cycle_summary = set_encoded[1:].mean(dim=0)
        else:
            context = time_token
            cycle_summary = torch.zeros_like(time_token)

        mobile_edge_x = edge_x[mobile_indices][:, mobile_indices]
        mobile_edge_mask = edge_mask[mobile_indices][:, mobile_indices]
        relative_vec = self._edge_relative_vector(mobile_edge_x)
        mobile_edge_feat = torch.cat([relative_vec, mobile_edge_x[..., 0:1]], dim=-1)

        eye = torch.eye(self.num_mobile, device=mobile_h.device, dtype=mobile_edge_mask.dtype)
        mobile_edge_mask = torch.clamp(mobile_edge_mask - eye, min=0.0)

        for block in self.mobile_blocks:
            mobile_h = block(mobile_h, mobile_edge_feat, mobile_edge_mask, context)

        current_velocities = mobile_node_x[:, 2:4]
        return mobile_h, context, cycle_summary, current_velocities

    def _structured_forward(self, structured: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = structured["node_x"].shape[0]
        policy_outputs: List[torch.Tensor] = []
        value_outputs: List[torch.Tensor] = []
        self.last_attention = {}

        true_cycle_count = structured["global_x"][:, 1]
        for batch_idx in range(batch_size):
            mobile_h, context, cycle_summary, current_velocities = self._encode_batch_item(structured, batch_idx)
            actor_in = torch.cat(
                [mobile_h, context.unsqueeze(0).expand(self.num_mobile, -1)],
                dim=-1,
            )
            mean_actions = self.actor_head(actor_in)
            if float(true_cycle_count[batch_idx].item()) <= 0.0:
                mean_actions = current_velocities
            policy_outputs.append(mean_actions.reshape(-1))

            critic_in = torch.cat([mobile_h.mean(dim=0), cycle_summary, context], dim=-1)
            value_outputs.append(self.critic_head(critic_in).reshape(1))

        return torch.stack(policy_outputs, dim=0), torch.stack(value_outputs, dim=0)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        return self.action_dist.proba_distribution(latent_pi, self.log_std)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        structured = self.extract_features(obs)
        mean_actions, values = self._structured_forward(structured)
        distribution = self._get_action_dist_from_latent(mean_actions)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        structured = self.extract_features(obs)
        mean_actions, values = self._structured_forward(structured)
        distribution = self._get_action_dist_from_latent(mean_actions)
        flat_actions = actions.reshape(actions.shape[0], -1)
        log_prob = distribution.log_prob(flat_actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: torch.Tensor):
        structured = self.extract_features(obs)
        mean_actions, _ = self._structured_forward(structured)
        return self._get_action_dist_from_latent(mean_actions)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        structured = self.extract_features(obs)
        _, values = self._structured_forward(structured)
        return values
