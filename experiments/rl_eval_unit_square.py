#!/usr/bin/env python3
"""Evaluate a trained PPO model on fixed unit-square RL environment seeds."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PPO on fixed unit-square evasion RL task.")
    parser.add_argument("--model", type=str, required=True, help="Path to .zip PPO model")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--outdir", type=str, default="experiments/output/rl_unit_square_eval")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--attention-topk", type=int, default=40)
    parser.add_argument("--attention-log-full", action="store_true")
    parser.add_argument("--attention-log-every", type=int, default=1)
    parser.add_argument("--attention-log-path", type=str, default="")
    return parser


def _safe_mean(values):
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "stable-baselines3 is required for evaluation. Install with: pip install stable-baselines3"
        ) from exc

    from rl_attention_logging import AttentionLogConfig, JsonlAttentionLogger, summarize_attention
    # Ensure custom extractor/policy classes are importable during model deserialization.
    from rl_boundary_cycle_policy import BoundaryCycleActorCriticPolicy, BoundaryCycleStructuredExtractor  # noqa: F401
    from rl_gat_baseline import BaselineGraphAttentionExtractor  # noqa: F401
    from rl_gat_policy import DartAwareActorCriticPolicy, GraphAttentionExtractor  # noqa: F401
    from rl_homological_gat_policy import (  # noqa: F401
        HomologicalGATActorCriticPolicy,
        HomologicalGATLegacyActorCriticPolicy,
        HomologicalGraphAttentionExtractor,
        load_homological_gat_ppo,
    )
    from rl_structured_velocity_policy import (  # noqa: F401
        StructuredVelocityActorCriticPolicy,
        StructuredVelocityCycleExtractor,
        StructuredVelocityGraphExtractor,
    )
    from rl_unit_square_experiment import UnitSquareRLConfig, make_eval_env

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"eval_unitsquare_{timestamp}"
    run_dir = Path(args.outdir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = UnitSquareRLConfig()
    attention_config = AttentionLogConfig(
        enabled=True,
        topk=max(1, args.attention_topk),
        every_n_steps_train=config.attention_log_config.every_n_steps_train,
        every_n_steps_eval=max(1, args.attention_log_every),
        log_full_attention=bool(args.attention_log_full),
        path=args.attention_log_path or str(run_dir / "attention_eval.jsonl"),
    )
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        payload = config.as_serializable()
        payload["attention_log_config"] = {
            "enabled": attention_config.enabled,
            "topk": attention_config.topk,
            "every_n_steps_train": attention_config.every_n_steps_train,
            "every_n_steps_eval": attention_config.every_n_steps_eval,
            "log_full_attention": attention_config.log_full_attention,
            "path": attention_config.path,
        }
        json.dump(payload, handle, indent=2, sort_keys=True)

    env = make_eval_env(
        config,
        event_log_path=str(run_dir / "eval_events.jsonl"),
        enable_event_logging=True,
    )
    model = load_homological_gat_ppo(args.model)
    attention_writer = JsonlAttentionLogger(attention_config.path) if attention_config.enabled else None

    episode_returns = []
    episode_steps = []
    episode_final_times = []
    cleared_flags = []
    timeout_flags = []

    for _ in range(max(1, args.episodes)):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_return = 0.0
        ep_steps = 0

        while not (done or truncated):
            obs_for_attention = obs
            action, _ = model.predict(obs_for_attention, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_steps += 1

            if attention_writer and ep_steps % attention_config.every_n_steps_eval == 0:
                extractor = getattr(model.policy, "features_extractor", None)
                last_attention = getattr(extractor, "last_attention", None)
                if last_attention:
                    summary = summarize_attention(
                        last_attention,
                        obs_for_attention,
                        topk=attention_config.topk,
                        log_full_attention=attention_config.log_full_attention,
                    )
                    attention_writer.append(
                        {
                            "phase": "eval",
                            "episode_index": int(len(episode_returns) + 1),
                            "step_index": int(ep_steps),
                            "sim_time": float(info.get("time", -1.0)),
                            "event_time": float(info.get("elapsed", -1.0)),
                            "true_cycles_closed": int(info.get("true_cycles_closed", 0)),
                            "true_cycles_added": int(info.get("true_cycles_added", 0)),
                            **summary,
                        }
                    )

        episode_returns.append(ep_return)
        episode_steps.append(ep_steps)
        episode_final_times.append(float(info["time"]))
        cleared_flags.append(bool(done))
        timeout_flags.append(bool(truncated and not done))

    records = env.event_log
    summary = {
        "episodes": max(1, args.episodes),
        "clear_rate": float(sum(cleared_flags) / len(cleared_flags)) if cleared_flags else 0.0,
        "timeout_rate": float(sum(timeout_flags) / len(timeout_flags)) if timeout_flags else 0.0,
        "mean_return": _safe_mean(episode_returns),
        "mean_steps": _safe_mean(episode_steps),
        "mean_final_time": _safe_mean(episode_final_times),
        "mean_true_cycles_closed_per_step": _safe_mean([r.true_cycles_closed for r in records]),
        "mean_true_cycles_added_per_step": _safe_mean([r.true_cycles_added for r in records]),
        "mean_trace_eval_count": _safe_mean([r.trace_evaluation_count for r in records]),
        "mean_trace_split_count": _safe_mean([r.trace_split_count for r in records]),
        "mean_trace_max_depth": _safe_mean([r.trace_max_recursion_depth for r in records]),
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Evaluation artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
