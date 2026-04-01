# ******************************************************************************
#  Copyright (c) 2026, Marco Campos - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


@dataclass(frozen=True)
class AttentionLogConfig:
    enabled: bool = True
    topk: int = 40
    every_n_steps_train: int = 10
    every_n_steps_eval: int = 1
    log_full_attention: bool = False
    path: str = ""


class JsonlAttentionLogger:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def _ensure_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _first_batch(array: np.ndarray) -> np.ndarray:
    return array[0] if array.ndim > 0 and array.shape[0] == 1 else array


def _entropy(prob: np.ndarray) -> float:
    p = prob[prob > 1e-12]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def summarize_attention(last_attention: Dict[str, Any], obs: Dict[str, Any], *, topk: int, log_full_attention: bool) -> Dict[str, Any]:
    if not last_attention:
        return {"attention_layers": {}}

    node_x = _first_batch(_ensure_numpy(obs["node_x"]))
    edge_x = _first_batch(_ensure_numpy(obs["edge_x"]))
    edge_mask = _first_batch(_ensure_numpy(obs["edge_mask"])) > 0.5

    true_edge_mask = edge_x[:, :, 3] > 0.5
    is_fence = node_x[:, 0] > 0.5
    is_mobile = node_x[:, 1] > 0.5

    ff_mask = np.logical_and(is_fence[:, None], is_fence[None, :])
    mm_mask = np.logical_and(is_mobile[:, None], is_mobile[None, :])
    fm_mask = np.logical_or(
        np.logical_and(is_fence[:, None], is_mobile[None, :]),
        np.logical_and(is_mobile[:, None], is_fence[None, :]),
    )

    valid = edge_mask.astype(bool)
    records: Dict[str, Any] = {}

    for layer_name, layer_tensor in last_attention.items():
        attn = _ensure_numpy(layer_tensor)
        # [B, H, N, N]
        if attn.ndim != 4:
            continue
        attn = _first_batch(attn)

        layer_heads: List[Dict[str, Any]] = []
        head_entropies = []
        head_maxima = []
        head_means = []

        for head_idx in range(attn.shape[0]):
            scores = attn[head_idx]
            valid_scores = np.where(valid, scores, 0.0)
            total_mass = float(np.sum(valid_scores))
            denom = max(total_mass, 1e-12)

            flat_idx = np.argwhere(valid)
            flat_vals = valid_scores[valid]
            k = min(int(topk), int(flat_vals.shape[0]))
            if k > 0:
                order = np.argpartition(flat_vals, -k)[-k:]
                order = order[np.argsort(flat_vals[order])[::-1]]
                top_pairs = flat_idx[order]
                top_vals = flat_vals[order]
                top_edges = [[int(i), int(j)] for i, j in top_pairs]
                top_scores = [float(v) for v in top_vals]
            else:
                top_edges, top_scores = [], []

            prob = valid_scores / denom
            entropy = _entropy(prob)
            max_score = float(np.max(flat_vals)) if flat_vals.size else 0.0
            mean_score = float(np.mean(flat_vals)) if flat_vals.size else 0.0

            true_mass = float(np.sum(np.where(np.logical_and(valid, true_edge_mask), scores, 0.0)))
            ff_mass = float(np.sum(np.where(np.logical_and(valid, ff_mask), scores, 0.0)))
            fm_mass = float(np.sum(np.where(np.logical_and(valid, fm_mask), scores, 0.0)))
            mm_mass = float(np.sum(np.where(np.logical_and(valid, mm_mask), scores, 0.0)))

            head_entropies.append(entropy)
            head_maxima.append(max_score)
            head_means.append(mean_score)

            head_record: Dict[str, Any] = {
                "head_index": int(head_idx),
                "topk_edges": top_edges,
                "topk_scores": top_scores,
                "mass_on_true_cycle_edges": float(true_mass / denom),
                "mass_on_fence_fence": float(ff_mass / denom),
                "mass_on_fence_mobile": float(fm_mass / denom),
                "mass_on_mobile_mobile": float(mm_mass / denom),
                "entropy": entropy,
                "max_score": max_score,
                "mean_score": mean_score,
            }
            if log_full_attention:
                head_record["attention_matrix"] = scores.tolist()
            layer_heads.append(head_record)

        records[layer_name] = {
            "heads": layer_heads,
            "mean_entropy": float(np.mean(head_entropies)) if head_entropies else 0.0,
            "mean_max_score": float(np.mean(head_maxima)) if head_maxima else 0.0,
            "mean_score": float(np.mean(head_means)) if head_means else 0.0,
        }

    return {"attention_layers": records}


class AttentionLoggingCallback:  # Loaded only when SB3 is available in training script.
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Import create_attention_logging_callback() instead of AttentionLoggingCallback directly.")


def create_attention_logging_callback(config: AttentionLogConfig):
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("stable-baselines3 is required for attention callback.") from exc

    class _AttentionLoggingCallback(BaseCallback):
        def __init__(self, cfg: AttentionLogConfig):
            super().__init__()
            self.cfg = cfg
            self._writer = JsonlAttentionLogger(cfg.path) if cfg.enabled and cfg.path else None

        def _on_step(self) -> bool:
            if not self.cfg.enabled or self._writer is None:
                return True
            if self.n_calls % max(1, int(self.cfg.every_n_steps_train)) != 0:
                return True

            policy = self.model.policy
            extractor = getattr(policy, "features_extractor", None)
            last_attention = getattr(extractor, "last_attention", None)
            if not last_attention:
                return True

            obs = self.locals.get("new_obs")
            infos = self.locals.get("infos", [{}])
            info = infos[0] if infos else {}

            if not isinstance(obs, dict):
                return True

            summary = summarize_attention(
                last_attention,
                obs,
                topk=self.cfg.topk,
                log_full_attention=self.cfg.log_full_attention,
            )
            event = {
                "phase": "train",
                "global_step": int(self.num_timesteps),
                "callback_step": int(self.n_calls),
                "episode_index": int(info.get("event_record", {}).get("episode_index", -1)),
                "step_index": int(info.get("event_record", {}).get("step_index", -1)),
                "sim_time": float(info.get("time", -1.0)),
            }
            event.update(summary)
            self._writer.append(event)
            return True

    return _AttentionLoggingCallback(config)
