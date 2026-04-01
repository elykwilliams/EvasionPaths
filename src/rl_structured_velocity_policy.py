# ******************************************************************************
#  Copyright (c) 2026, Marco Campos - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from rl_gat_baseline import EdgeBiasedGATLayer


def _safe_tensor(tensor: torch.Tensor, *, clamp: float = 1e4) -> torch.Tensor:
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=clamp, neginf=-clamp)
    return torch.clamp(tensor, min=-clamp, max=clamp)


def _safe_softmax(logits: torch.Tensor, dim: int) -> torch.Tensor:
    weights = torch.softmax(_safe_tensor(logits), dim=dim)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    denom = torch.clamp(weights.sum(dim=dim, keepdim=True), min=1e-6)
    return weights / denom


def _build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation_fn: type[nn.Module],
) -> nn.Sequential:
    layers: List[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(activation_fn())
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)


def _cycle_local_attn_mask(length: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((length, length), fill_value=-1e9, device=device)
    idx = torch.arange(length, device=device)
    mask[idx, idx] = 0.0
    mask[idx, (idx - 1) % length] = 0.0
    mask[idx, (idx + 1) % length] = 0.0
    return mask


class StructuredVelocityCycleExtractor(BaseFeaturesExtractor):
    """Pass through cycle graph observations for the structured velocity policy."""

    def __init__(self, observation_space) -> None:
        super().__init__(observation_space, features_dim=1)
        self._expected_dims = {
            key: len(space.shape)
            for key, space in observation_space.spaces.items()
        }

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        payload: Dict[str, torch.Tensor] = {}
        for key, value in obs.items():
            tensor = value
            if key == "cycle_node_index":
                tensor = tensor.long()
            else:
                tensor = tensor.float()
            if tensor.dim() == self._expected_dims[key]:
                tensor = tensor.unsqueeze(0)
            payload[key] = tensor
        return payload


class _CycleDartTokenizer(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.width = int(width)
        self.dart_embed = nn.Sequential(
            nn.Linear(3, self.width),
            nn.ReLU(),
        )
        self.cycle_attn = nn.MultiheadAttention(self.width, num_heads=2, batch_first=True)
        self.cycle_norm = nn.LayerNorm(self.width)
        self.cycle_score = nn.Linear(self.width, 1)
        self.cycle_pool_proj = nn.Sequential(
            nn.LayerNorm(self.width),
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width),
        )
        self.inter_cycle_attn = nn.MultiheadAttention(self.width, num_heads=2, batch_first=True)
        self.inter_cycle_norm = nn.LayerNorm(self.width)
        self.inter_cycle_score = nn.Linear(self.width, 1)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(self.width),
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width),
        )

    @staticmethod
    def _normalize_cycle_orientation(darts: torch.Tensor) -> torch.Tensor:
        if darts.shape[0] < 3:
            return darts
        verts = [torch.zeros(2, device=darts.device, dtype=darts.dtype)]
        for idx in range(darts.shape[0] - 1):
            verts.append(verts[-1] + darts[idx])
        poly = torch.stack(verts, dim=0)
        x = poly[:, 0]
        y = poly[:, 1]
        area = 0.5 * torch.sum(x * torch.roll(y, shifts=-1, dims=0) - y * torch.roll(x, shifts=-1, dims=0))
        if float(area) >= 0.0:
            return darts
        return torch.flip(-darts, dims=[0])

    def encode_cycle(self, dart_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dart_xy = dart_features[:, :2]
        dart_xy = self._normalize_cycle_orientation(dart_xy)
        dart_norm = torch.linalg.norm(dart_xy, dim=-1, keepdim=True)
        normalized_features = _safe_tensor(torch.cat([dart_xy, dart_norm], dim=-1))

        dart_embed = _safe_tensor(self.dart_embed(normalized_features)).unsqueeze(0)
        attn_mask = _cycle_local_attn_mask(dart_embed.shape[1], dart_embed.device)
        attn_out, _ = self.cycle_attn(dart_embed, dart_embed, dart_embed, attn_mask=attn_mask, need_weights=False)
        dart_embed = _safe_tensor(self.cycle_norm(_safe_tensor(dart_embed + attn_out)).squeeze(0))
        dart_logits = _safe_tensor(self.cycle_score(dart_embed).squeeze(-1))
        dart_weights = _safe_softmax(dart_logits, dim=0)
        cycle_embed = _safe_tensor(self.cycle_pool_proj(torch.sum(dart_embed * dart_weights.unsqueeze(-1), dim=0)))
        return cycle_embed, dart_weights, _safe_tensor(dart_xy)

    def tokenize_collection(self, cycle_embeddings: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not cycle_embeddings:
            zero = self.token_proj[1].weight.new_zeros((self.width,))
            return zero, zero.new_zeros((0,))
        collection = _safe_tensor(torch.stack(cycle_embeddings, dim=0)).unsqueeze(0)
        attn_out, _ = self.inter_cycle_attn(collection, collection, collection, need_weights=False)
        collection = _safe_tensor(self.inter_cycle_norm(_safe_tensor(collection + attn_out)).squeeze(0))
        logits = _safe_tensor(self.inter_cycle_score(collection).squeeze(-1))
        weights = _safe_softmax(logits, dim=0)
        token = _safe_tensor(self.token_proj(torch.sum(collection * weights.unsqueeze(-1), dim=0)))
        return token, weights


class StructuredVelocityGraphExtractor(BaseFeaturesExtractor):
    """Encode cycle collections and the alpha-complex graph for the structured actor."""

    def __init__(
        self,
        observation_space,
        cycle_width: int = 8,
        graph_hidden_dim: int = 32,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(observation_space, features_dim=1)
        self.node_dim = int(observation_space.spaces["node_x"].shape[-1])
        self.edge_dim = 1
        self.global_dim = 16
        self.num_nodes = int(observation_space.spaces["node_x"].shape[0])
        self.cycle_width = int(cycle_width)
        self.graph_hidden_dim = int(graph_hidden_dim)

        self.tokenizer = _CycleDartTokenizer(width=self.cycle_width)
        self.node_proj = nn.Linear(4, self.graph_hidden_dim)
        self.layers = nn.ModuleList(
            [
                EdgeBiasedGATLayer(
                    hidden_dim=self.graph_hidden_dim,
                    edge_dim=self.edge_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.rho = nn.Sequential(
            nn.LayerNorm(self.graph_hidden_dim),
            nn.Linear(self.graph_hidden_dim, 16),
            nn.ReLU(),
        )
        self.psi = nn.Sequential(
            nn.LayerNorm(2 * self.cycle_width + 16),
            nn.Linear(2 * self.cycle_width + 16, 16),
            nn.ReLU(),
        )

        self.last_attention: Dict[str, torch.Tensor] = {}
        self.last_cycle_attention: Dict[str, List[torch.Tensor]] = {}
        self.last_global: torch.Tensor | None = None
        self.last_inward: torch.Tensor | None = None

    @staticmethod
    def _rotate_minus_clockwise(vec: torch.Tensor) -> torch.Tensor:
        return torch.stack([-vec[..., 1], vec[..., 0]], dim=-1)

    def _encode_batch_item(
        self,
        edge_x: torch.Tensor,
        cycle_node_index: torch.Tensor,
        cycle_node_mask: torch.Tensor,
        cycle_mask: torch.Tensor,
        cycle_is_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        device = edge_x.device
        dtype = edge_x.dtype
        true_embeds: List[torch.Tensor] = []
        false_embeds: List[torch.Tensor] = []
        true_nodes: List[torch.Tensor] = []
        true_darts: List[torch.Tensor] = []
        true_dart_weights: List[torch.Tensor] = []
        false_dart_weights: List[torch.Tensor] = []

        for row in range(cycle_node_index.shape[0]):
            if float(cycle_mask[row]) <= 0.5:
                continue
            active = cycle_node_mask[row] > 0.5
            nodes = cycle_node_index[row, active]
            if nodes.numel() < 3:
                continue

            darts: List[torch.Tensor] = []
            for idx in range(nodes.numel()):
                src = int(nodes[idx].item())
                dst = int(nodes[(idx + 1) % nodes.numel()].item())
                dist = edge_x[src, dst, 0]
                direction = edge_x[src, dst, 1:3]
                darts.append(torch.cat([dist * direction, dist.unsqueeze(0)], dim=0))
            dart_features = torch.stack(darts, dim=0)
            cycle_embed, dart_weights, normalized_darts = self.tokenizer.encode_cycle(dart_features)
            if float(cycle_is_true[row]) > 0.5:
                true_embeds.append(cycle_embed)
                true_nodes.append(nodes.clone())
                true_darts.append(normalized_darts)
                true_dart_weights.append(dart_weights)
            else:
                false_embeds.append(cycle_embed)
                false_dart_weights.append(dart_weights)

        true_token, true_cycle_weights = self.tokenizer.tokenize_collection(true_embeds)
        false_token, _ = self.tokenizer.tokenize_collection(false_embeds)

        n_accum = edge_x.new_zeros((edge_x.shape[0], 2))
        for alpha, nodes, darts in zip(true_cycle_weights, true_nodes, true_darts):
            count = int(nodes.numel())
            for idx in range(count):
                d_in = darts[(idx - 1) % count]
                d_out = darts[idx]
                inward = self._rotate_minus_clockwise(d_in + d_out)
                inward = inward / torch.clamp(torch.linalg.norm(inward), min=1e-6)
                n_accum[nodes[idx]] = n_accum[nodes[idx]] + alpha * inward

        n_norm = torch.linalg.norm(n_accum, dim=-1, keepdim=True)
        n_hat = torch.where(n_norm > 1e-6, n_accum / n_norm, torch.zeros_like(n_accum))
        n_hat = _safe_tensor(n_hat)

        return true_token, false_token, n_hat, true_dart_weights, false_dart_weights

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        node_x = obs["node_x"].float()
        edge_x = obs["edge_x"].float()
        edge_mask = obs["edge_mask"].float()
        cycle_node_index = obs["cycle_node_index"].long()
        cycle_node_mask = obs["cycle_node_mask"].float()
        cycle_mask = obs["cycle_mask"].float()
        cycle_is_true = obs["cycle_is_true"].float()

        if node_x.dim() == 2:
            node_x = node_x.unsqueeze(0)
            edge_x = edge_x.unsqueeze(0)
            edge_mask = edge_mask.unsqueeze(0)
            cycle_node_index = cycle_node_index.unsqueeze(0)
            cycle_node_mask = cycle_node_mask.unsqueeze(0)
            cycle_mask = cycle_mask.unsqueeze(0)
            cycle_is_true = cycle_is_true.unsqueeze(0)

        batch_size = node_x.shape[0]
        device = node_x.device
        dtype = node_x.dtype

        true_tokens = []
        false_tokens = []
        n_hats = []
        all_true_dart_weights: List[torch.Tensor] = []
        all_false_dart_weights: List[torch.Tensor] = []

        for batch_idx in range(batch_size):
            true_token, false_token, n_hat, true_dart_weights, false_dart_weights = self._encode_batch_item(
                edge_x[batch_idx],
                cycle_node_index[batch_idx],
                cycle_node_mask[batch_idx],
                cycle_mask[batch_idx],
                cycle_is_true[batch_idx],
            )
            true_tokens.append(true_token)
            false_tokens.append(false_token)
            n_hats.append(n_hat)
            all_true_dart_weights.extend(true_dart_weights)
            all_false_dart_weights.extend(false_dart_weights)

        true_token = torch.stack(true_tokens, dim=0) if true_tokens else torch.zeros((batch_size, self.cycle_width), device=device, dtype=dtype)
        false_token = torch.stack(false_tokens, dim=0) if false_tokens else torch.zeros((batch_size, self.cycle_width), device=device, dtype=dtype)
        n_hat = torch.stack(n_hats, dim=0) if n_hats else torch.zeros((batch_size, self.num_nodes, 2), device=device, dtype=dtype)

        node_input = _safe_tensor(torch.cat([node_x[:, :, 2:4], n_hat], dim=-1))
        node_h = _safe_tensor(self.node_proj(node_input))
        dist_edge_x = edge_x[:, :, :, 0:1]

        self.last_attention = {}
        for layer_idx, layer in enumerate(self.layers):
            node_h, attn = layer(node_h, dist_edge_x, edge_mask)
            node_h = _safe_tensor(node_h)
            self.last_attention[f"layer_{layer_idx}"] = attn

        pooled_nodes = _safe_tensor(self.rho(_safe_tensor(node_h.sum(dim=1))))
        g_raw = _safe_tensor(torch.cat([true_token, false_token, pooled_nodes], dim=-1))
        g = _safe_tensor(self.psi(g_raw), clamp=100.0)

        self.last_cycle_attention = {
            "true_dart_weights": all_true_dart_weights,
            "false_dart_weights": all_false_dart_weights,
        }
        self.last_global = g.detach()
        self.last_inward = n_hat.detach()

        return {
            "node_h": node_h,
            "node_x": node_x,
            "n_hat": n_hat,
            "g": g,
        }


class StructuredVelocityActorCriticPolicy(ActorCriticPolicy):
    """Structured nodewise actor with inward-direction modulation."""

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = nn.Identity()

    def _build(self, lr_schedule) -> None:
        self._build_mlp_extractor()

        if not isinstance(self.features_extractor, StructuredVelocityGraphExtractor):
            raise TypeError("StructuredVelocityActorCriticPolicy requires StructuredVelocityGraphExtractor.")

        self.num_mobile = int(self.action_space.shape[0])
        self.action_dim_per_mobile = int(self.action_space.shape[1])
        self.total_action_dim = int(np.prod(self.action_space.shape))

        hidden_dim = int(self.features_extractor.graph_hidden_dim)
        global_dim = int(self.features_extractor.global_dim)
        self.local_proj = nn.Sequential(nn.Linear(hidden_dim, 24), self.activation_fn())
        self.global_proj = nn.Sequential(nn.Linear(global_dim, 8), self.activation_fn())
        self.actor_trunk = nn.Sequential(nn.Linear(32, 16), self.activation_fn())
        self.tau_head = nn.Linear(16, self.action_dim_per_mobile)
        self.beta_head = nn.Linear(16, 1)
        self.critic_head = _build_mlp(
            input_dim=global_dim,
            hidden_dims=[16],
            output_dim=1,
            activation_fn=self.activation_fn,
        )
        self.log_std = nn.Parameter(torch.ones(self.total_action_dim) * self.log_std_init, requires_grad=False)
        self.last_actor_terms: Dict[str, torch.Tensor] = {}

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.local_proj: np.sqrt(2),
                self.global_proj: np.sqrt(2),
                self.actor_trunk: np.sqrt(2),
                self.tau_head: 0.01,
                self.beta_head: 0.01,
                self.critic_head: 1.0,
            }
            for module, gain in module_gains.items():
                module.apply(lambda m: self.init_weights(m, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _structured_actor_mean(self, structured: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_h = structured["node_h"]
        node_x = structured["node_x"]
        n_hat = structured["n_hat"]
        g = structured["g"]

        mobile_mask = node_x[:, :, 1] > 0.5
        mobile_h = node_h[mobile_mask].reshape(node_h.shape[0], self.num_mobile, node_h.shape[-1])
        mobile_n = n_hat[mobile_mask].reshape(node_h.shape[0], self.num_mobile, 2)

        local = _safe_tensor(self.local_proj(mobile_h), clamp=100.0)
        global_branch = _safe_tensor(self.global_proj(g), clamp=100.0).unsqueeze(1).expand(-1, self.num_mobile, -1)
        trunk = _safe_tensor(self.actor_trunk(torch.cat([local, global_branch], dim=-1)), clamp=100.0)
        tau = torch.tanh(_safe_tensor(self.tau_head(trunk), clamp=20.0))
        beta = torch.sigmoid(_safe_tensor(self.beta_head(trunk), clamp=20.0))
        velocity = _safe_tensor(tau + beta * mobile_n, clamp=10.0)

        self.last_actor_terms = {
            "tau": tau.detach(),
            "beta": beta.detach(),
            "n_hat_mobile": mobile_n.detach(),
            "velocity": velocity.detach(),
        }
        return velocity.reshape(node_h.shape[0], self.total_action_dim)

    def _critic_values(self, structured: Dict[str, torch.Tensor]) -> torch.Tensor:
        return _safe_tensor(self.critic_head(structured["g"].detach()), clamp=100.0)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        safe_log_std = _safe_tensor(self.log_std, clamp=5.0).detach()
        return self.action_dist.proba_distribution(_safe_tensor(latent_pi, clamp=10.0), safe_log_std)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        structured = self.extract_features(obs)
        mean_actions = self._structured_actor_mean(structured)
        distribution = self._get_action_dist_from_latent(mean_actions)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self._critic_values(structured)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        structured = self.extract_features(obs)
        mean_actions = self._structured_actor_mean(structured)
        distribution = self._get_action_dist_from_latent(mean_actions)
        flat_actions = actions.reshape(actions.shape[0], -1)
        log_prob = distribution.log_prob(flat_actions)
        entropy = distribution.entropy()
        values = self._critic_values(structured)
        return values, log_prob, entropy

    def get_distribution(self, obs: torch.Tensor):
        structured = self.extract_features(obs)
        mean_actions = self._structured_actor_mean(structured)
        return self._get_action_dist_from_latent(mean_actions)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        structured = self.extract_features(obs)
        return self._critic_values(structured)
