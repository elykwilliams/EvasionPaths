#!/usr/bin/env python3
"""Post-training rollout capture for RL checkpoints with Reeb artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from UI import _build_reeb_graph_plot_data
from plotting_tools import show_state
from reeb_graph import ReebEventGraphBuilder
from rl_attention_logging import AttentionLogConfig
from rl_env import RewardConfig
from rl_unit_square_experiment import UnitSquareRLConfig, make_eval_env


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture RL+Reeb rollout artifacts for saved checkpoints.")
    parser.add_argument("--run-dir", type=str, required=True, help="Training run dir (contains models/ and config.json).")
    parser.add_argument("--outdir", type=str, default="", help="Output root (default: <run-dir>/rl_reeb_checkpoints).")
    parser.add_argument("--episodes", type=int, default=1, help="Rollout episodes per checkpoint.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional per-episode step cap (0 = use environment termination only).",
    )
    parser.add_argument("--frame-interval-ms", type=int, default=130)
    parser.add_argument("--every-n-steps", type=int, default=0, help="Filter step checkpoints by interval (0 = all).")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy sampling instead of deterministic actions.")
    parser.add_argument("--no-include-init", action="store_true")
    parser.add_argument("--no-include-final", action="store_true")
    parser.add_argument("--no-include-best", action="store_true")
    parser.add_argument("--no-include-step-checkpoints", action="store_true")
    return parser


def _render_2d_frame(simulation, frame_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.clear()
    ax.axis("off")
    ax.axis("equal")
    ax.set_title(f"T = {simulation.time:6.3f}", loc="left")
    show_state(simulation, ax=ax)
    fig.tight_layout()
    fig.savefig(frame_path, dpi=120)
    plt.close(fig)


def _load_config(run_dir: Path) -> UnitSquareRLConfig:
    default = UnitSquareRLConfig()
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return default

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    reward_payload = payload.get("reward_config", {})
    attention_payload = payload.get("attention_log_config", {})

    reward_config = RewardConfig(**reward_payload) if isinstance(reward_payload, dict) else default.reward_config
    attention_config = (
        AttentionLogConfig(**attention_payload)
        if isinstance(attention_payload, dict)
        else default.attention_log_config
    )

    return UnitSquareRLConfig(
        num_mobile_sensors=int(payload.get("num_mobile_sensors", default.num_mobile_sensors)),
        sensing_radius=float(payload.get("sensing_radius", default.sensing_radius)),
        fence_sensing_radius=(
            default.fence_sensing_radius
            if payload.get("fence_sensing_radius", default.fence_sensing_radius) is None
            else float(payload.get("fence_sensing_radius", default.fence_sensing_radius))
        ),
        fence_offset_ratio=(
            default.fence_offset_ratio
            if payload.get("fence_offset_ratio", default.fence_offset_ratio) is None
            else float(payload.get("fence_offset_ratio", default.fence_offset_ratio))
        ),
        use_weighted_alpha=bool(payload.get("use_weighted_alpha", default.use_weighted_alpha)),
        dt=float(payload.get("dt", default.dt)),
        tmax=float(payload.get("tmax", default.tmax)),
        initial_sensor_speed=float(payload.get("initial_sensor_speed", default.initial_sensor_speed)),
        max_speed_scale=float(payload.get("max_speed_scale", default.max_speed_scale)),
        coordinate_free=bool(payload.get("coordinate_free", default.coordinate_free)),
        state_mode=str(payload.get("state_mode", default.state_mode)),
        reward_config=reward_config,
        attention_log_config=attention_config,
        train_seeds=tuple(int(x) for x in payload.get("train_seeds", list(default.train_seeds))),
        eval_seeds=tuple(int(x) for x in payload.get("eval_seeds", list(default.eval_seeds))),
    )


def _step_from_checkpoint_name(path: Path) -> int | None:
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    if not match:
        return None
    return int(match.group(1))


def _discover_checkpoints(models_dir: Path, *, include_init: bool, include_final: bool, include_best: bool, include_step: bool, every_n_steps: int) -> List[Tuple[str, Path]]:
    discovered: List[Tuple[str, Path]] = []

    init_path = models_dir / "ppo_unitsquare_init.zip"
    if include_init and init_path.exists():
        discovered.append(("init", init_path))

    if include_step:
        step_paths = sorted(models_dir.glob("ppo_unitsquare_*_steps.zip"), key=lambda p: _step_from_checkpoint_name(p) or -1)
        for step_path in step_paths:
            step = _step_from_checkpoint_name(step_path)
            if step is None:
                continue
            if every_n_steps > 0 and step % every_n_steps != 0:
                continue
            discovered.append((f"step_{step:07d}", step_path))

    final_path = models_dir / "ppo_unitsquare_final.zip"
    if include_final and final_path.exists():
        discovered.append(("final", final_path))

    best_path = models_dir / "best" / "best_model.zip"
    if include_best and best_path.exists():
        discovered.append(("best", best_path))

    return discovered


def _nontrivial_history_entries(entries: Iterable[Tuple[dict, tuple, tuple, float]]) -> List[Tuple[dict, tuple, tuple, float]]:
    return [
        (labels, alpha_change, boundary_change, event_time)
        for labels, alpha_change, boundary_change, event_time in entries
        if any(alpha_change) or tuple(boundary_change) != (0, 0)
    ]


def _summary_payload(step_summary) -> Dict[str, int]:
    if step_summary is None:
        return {
            "n_cycles": 0,
            "n_true": 0,
            "n_false": 0,
            "n_birth": 0,
            "n_death": 0,
            "n_continue": 0,
            "n_split_edges": 0,
            "n_merge_edges": 0,
            "n_transform_edges": 0,
            "n_label_flips": 0,
        }

    return {
        "n_cycles": int(step_summary.n_cycles),
        "n_true": int(step_summary.n_true),
        "n_false": int(step_summary.n_false),
        "n_birth": int(step_summary.n_birth),
        "n_death": int(step_summary.n_death),
        "n_continue": int(step_summary.n_continue),
        "n_split_edges": int(step_summary.n_split_edges),
        "n_merge_edges": int(step_summary.n_merge_edges),
        "n_transform_edges": int(step_summary.n_transform_edges),
        "n_label_flips": int(step_summary.n_label_flips),
    }


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _capture_checkpoint(
    *,
    checkpoint_path: Path,
    checkpoint_tag: str,
    checkpoint_dir: Path,
    config: UnitSquareRLConfig,
    episodes: int,
    max_steps: int,
    deterministic: bool,
    frame_interval_ms: int,
) -> Dict:
    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "stable-baselines3 is required for checkpoint rollout capture. Install with: pip install stable-baselines3"
        ) from exc

    # Ensure custom extractor/policy classes are available for model load.
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

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    env = make_eval_env(
        config,
        event_log_path=str(checkpoint_dir / "env_event_log.jsonl"),
        enable_event_logging=True,
    )
    model = load_homological_gat_ppo(checkpoint_path)

    episode_summaries: List[Dict] = []

    for episode_idx in range(1, max(1, episodes) + 1):
        episode_dir = checkpoint_dir / f"episode_{episode_idx:03d}"
        frames_dir = episode_dir / "frames"
        episode_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)

        obs, _ = env.reset()
        sim = env.simulation

        builder = ReebEventGraphBuilder()
        builder.add_snapshot(
            step=0,
            time=float(sim.time),
            labels=sim.cycle_label.label,
            alpha_cycle=sim.topology.outer_cycle,
        )

        frame_records: List[Dict] = []
        step_records: List[Dict] = []
        history_cursor = len(sim.cycle_label.history)
        done = False
        truncated = False
        step = 0
        ep_return = 0.0

        frame_name = "frame_00000.png"
        _render_2d_frame(sim, frames_dir / frame_name)
        frame_records.append(
            {
                "step": 0,
                "time": float(sim.time),
                "image": str(Path("frames") / frame_name),
                "summary": _summary_payload(builder.summaries[-1] if builder.summaries else None),
                "atomic_events": [],
                "snapshot_debug": builder.get_snapshot_debug(0),
            }
        )

        while not (done or truncated):
            if max_steps > 0 and step >= max_steps:
                break

            action, _ = model.predict(obs, deterministic=deterministic)
            action_arr = np.asarray(action, dtype=float)
            action_norms = np.linalg.norm(action_arr, axis=1) if action_arr.size else np.asarray([], dtype=float)

            obs, reward, done, truncated, info = env.step(action)
            ep_return += float(reward)
            step += 1
            sim = env.simulation

            new_history_entries = sim.cycle_label.history[history_cursor:]
            history_cursor = len(sim.cycle_label.history)
            nontrivial_entries = _nontrivial_history_entries(new_history_entries)
            atomic_events = []

            if nontrivial_entries:
                for labels, alpha_change, boundary_change, event_time in nontrivial_entries:
                    builder.add_snapshot(
                        step=step,
                        time=float(event_time),
                        labels=labels,
                        alpha_cycle=sim.topology.outer_cycle,
                    )
                    atomic_events.append(
                        {
                            "time": float(event_time),
                            "alpha_change": list(alpha_change),
                            "boundary_change": list(boundary_change),
                            "uncovered_cycles": max(0, sum(1 for v in labels.values() if bool(v)) - 1),
                        }
                    )
            else:
                builder.add_snapshot(
                    step=step,
                    time=float(sim.time),
                    labels=sim.cycle_label.label,
                    alpha_cycle=sim.topology.outer_cycle,
                )

            frame_name = f"frame_{step:05d}.png"
            _render_2d_frame(sim, frames_dir / frame_name)
            frame_records.append(
                {
                    "step": step,
                    "time": float(sim.time),
                    "image": str(Path("frames") / frame_name),
                    "summary": _summary_payload(builder.summaries[-1] if builder.summaries else None),
                    "atomic_events": atomic_events,
                    "snapshot_debug": builder.get_snapshot_debug(step),
                }
            )

            step_records.append(
                {
                    "step": int(step),
                    "time": float(info.get("time", sim.time)),
                    "event_time": float(info.get("elapsed", 0.0)),
                    "reward": float(reward),
                    "done": bool(done),
                    "truncated": bool(truncated),
                    "true_cycles_closed": int(info.get("true_cycles_closed", 0)),
                    "true_cycles_added": int(info.get("true_cycles_added", 0)),
                    "trace_eval_count": int(info.get("trace_evaluation_count", 0)),
                    "trace_split_count": int(info.get("trace_split_count", 0)),
                    "trace_max_depth": int(info.get("trace_max_recursion_depth", 0)),
                    "trace_recursion_limit_hit": bool(info.get("trace_recursion_limit_hit", False)),
                    "action_l2_mean": float(np.mean(action_norms)) if action_norms.size else 0.0,
                    "action_l2_max": float(np.max(action_norms)) if action_norms.size else 0.0,
                }
            )

        builder.close(step=step + 1, time=float(sim.time))

        timeline = {
            "interval_ms": int(max(1, frame_interval_ms)),
            "highlight_half_width": float(max(0.001, 0.48 * config.dt)),
            "reeb_graph": _build_reeb_graph_plot_data(builder),
            "frames": frame_records,
        }
        (episode_dir / "timeline.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")

        graph_payload = nx.node_link_data(builder.graph, edges="links")
        (episode_dir / "reeb_graph_node_link.json").write_text(json.dumps(graph_payload), encoding="utf-8")

        with (episode_dir / "rollout.jsonl").open("w", encoding="utf-8") as handle:
            for record in step_records:
                handle.write(json.dumps(record, sort_keys=True))
                handle.write("\n")

        episode_summary = {
            "episode_index": episode_idx,
            "return": float(ep_return),
            "steps": int(step),
            "final_time": float(sim.time),
            "cleared": bool(done),
            "timed_out": bool(truncated and not done),
            "mean_trace_depth": _mean([float(r["trace_max_depth"]) for r in step_records]),
            "mean_trace_splits": _mean([float(r["trace_split_count"]) for r in step_records]),
            "recursion_limit_hit_rate": _mean(
                [1.0 if bool(r["trace_recursion_limit_hit"]) else 0.0 for r in step_records]
            ),
        }
        (episode_dir / "summary.json").write_text(json.dumps(episode_summary, indent=2, sort_keys=True), encoding="utf-8")
        episode_summaries.append(episode_summary)

    env.close()

    checkpoint_summary = {
        "checkpoint_tag": checkpoint_tag,
        "checkpoint_path": str(checkpoint_path),
        "episodes": int(max(1, episodes)),
        "clear_rate": _mean([1.0 if e["cleared"] else 0.0 for e in episode_summaries]),
        "timeout_rate": _mean([1.0 if e["timed_out"] else 0.0 for e in episode_summaries]),
        "mean_return": _mean([float(e["return"]) for e in episode_summaries]),
        "mean_steps": _mean([float(e["steps"]) for e in episode_summaries]),
        "mean_final_time": _mean([float(e["final_time"]) for e in episode_summaries]),
        "mean_trace_depth": _mean([float(e["mean_trace_depth"]) for e in episode_summaries]),
        "mean_trace_splits": _mean([float(e["mean_trace_splits"]) for e in episode_summaries]),
        "mean_recursion_limit_hit_rate": _mean([float(e["recursion_limit_hit_rate"]) for e in episode_summaries]),
        "episode_summaries": episode_summaries,
    }
    (checkpoint_dir / "checkpoint_summary.json").write_text(
        json.dumps(checkpoint_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return checkpoint_summary


def main() -> None:
    args = _make_parser().parse_args()

    run_dir = Path(args.run_dir).resolve()
    models_dir = run_dir / "models"
    if not models_dir.exists():
        raise SystemExit(f"Missing models directory: {models_dir}")

    output_root = Path(args.outdir).resolve() if args.outdir else (run_dir / "rl_reeb_checkpoints")
    output_root.mkdir(parents=True, exist_ok=True)

    config = _load_config(run_dir)
    checkpoints = _discover_checkpoints(
        models_dir,
        include_init=not bool(args.no_include_init),
        include_final=not bool(args.no_include_final),
        include_best=not bool(args.no_include_best),
        include_step=not bool(args.no_include_step_checkpoints),
        every_n_steps=max(0, int(args.every_n_steps)),
    )
    if not checkpoints:
        raise SystemExit("No checkpoints found with the selected filters.")

    all_summaries = []
    for tag, ckpt_path in checkpoints:
        checkpoint_dir = output_root / tag
        summary = _capture_checkpoint(
            checkpoint_path=ckpt_path,
            checkpoint_tag=tag,
            checkpoint_dir=checkpoint_dir,
            config=config,
            episodes=max(1, int(args.episodes)),
            max_steps=max(0, int(args.max_steps)),
            deterministic=not bool(args.stochastic),
            frame_interval_ms=max(1, int(args.frame_interval_ms)),
        )
        all_summaries.append(summary)
        print(
            f"[{tag}] clear_rate={summary['clear_rate']:.3f} "
            f"mean_return={summary['mean_return']:.3f} "
            f"mean_final_time={summary['mean_final_time']:.3f}"
        )

    manifest = {
        "run_dir": str(run_dir),
        "output_root": str(output_root),
        "config": asdict(config),
        "checkpoint_summaries": all_summaries,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote checkpoint artifacts to: {output_root}")


if __name__ == "__main__":
    main()
