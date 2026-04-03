#!/usr/bin/env python3
"""Run post-training RL simulations from one checkpoint and save only selected Reeb visualizations."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from UI import _build_reeb_graph_plot_data, _open_via_local_viewer_server, _write_timeline_html
from plotting_tools import show_domain_boundary, show_state
from reeb_graph import ReebEventGraphBuilder
from rl_attention_logging import AttentionLogConfig
from rl_env import RewardConfig
from rl_unit_square_experiment import UnitSquareRLConfig, make_eval_env


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one saved RL checkpoint over many sims and save Reeb visualizations for a small subset."
    )
    parser.add_argument("--run-dir", type=str, required=True, help="Training run directory.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        help="Checkpoint selector: best|final|init|step_2000 (or explicit .zip path).",
    )
    parser.add_argument("--num-sims", type=int, default=100, help="Total simulations to run.")
    parser.add_argument("--num-save", type=int, default=5, help="How many simulations to save visualizations for.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional per-sim step cap (0 = natural termination).")
    parser.add_argument("--frame-interval-ms", type=int, default=130)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions instead of deterministic policy.")
    parser.add_argument("--outdir", type=str, default="", help="Output dir (default: <run-dir>/rl_reeb_from_model_<checkpoint>).")
    parser.add_argument("--open", action="store_true", help="Open the gallery page after completion.")
    return parser


def _load_config(run_dir: Path) -> UnitSquareRLConfig:
    default = UnitSquareRLConfig()
    path = run_dir / "config.json"
    if not path.exists():
        return default
    payload = json.loads(path.read_text(encoding="utf-8"))
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


def _resolve_checkpoint(run_dir: Path, selector: str) -> Tuple[str, Path]:
    candidate = Path(selector)
    if candidate.exists():
        return candidate.stem, candidate.resolve()

    models = run_dir / "models"
    if selector == "best":
        path = models / "best" / "best_model.zip"
    elif selector == "final":
        path = models / "ppo_unitsquare_final.zip"
    elif selector == "init":
        path = models / "ppo_unitsquare_init.zip"
    elif selector.startswith("step_"):
        step = selector.split("_", 1)[1]
        path = models / f"ppo_unitsquare_{step}_steps.zip"
    else:
        raise ValueError(f"Unknown checkpoint selector: {selector}")

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return selector, path.resolve()


def _render_2d_frame(simulation, frame_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.clear()
    ax.axis("off")
    ax.axis("equal")
    ax.set_title(f"T = {simulation.time:6.3f}", loc="left")
    show_state(simulation, ax=ax)
    show_domain_boundary(
        simulation.sensor_network.domain,
        ax=ax,
        color="#111111",
        linewidth=2.0,
        linestyle="--",
        zorder=10,
    )
    fig.tight_layout()
    fig.savefig(frame_path, dpi=120)
    plt.close(fig)


def _choose_saved_indices(num_sims: int, num_save: int) -> List[int]:
    if num_sims <= 0 or num_save <= 0:
        return []
    num_save = min(num_sims, num_save)
    if num_save == num_sims:
        return list(range(num_sims))
    # Evenly spaced sample indices across all sims.
    return sorted(set(int(round(i * (num_sims - 1) / max(1, num_save - 1))) for i in range(num_save)))


def _policy_term_maps(
    *,
    reward_config: RewardConfig,
    true_cycles_closed: float,
    true_cycles_added: float,
    merge_hazard_count: float,
    elapsed: float,
    effort: float,
    neighbor_close_violation: float,
    neighbor_far_violation: float,
    fence_close_violation: float,
    fence_far_violation: float,
    area_progress: float,
    perimeter_progress: float,
    largest_area_progress: float,
    largest_perimeter_progress: float,
    area_regress: float,
    perimeter_regress: float,
    largest_area_regress: float,
    largest_perimeter_regress: float,
    area_residual: float,
    perimeter_residual: float,
    largest_area_residual: float,
    largest_perimeter_residual: float,
    one_hole_linger_penalty: float,
    one_hole_area_scale: float,
    one_hole_perimeter_scale: float,
    clear_indicator: float,
    success_time_bonus: float,
    timeout_indicator: float,
) -> Dict[str, Dict[str, float]]:
    raw = {
        "true_cycles_closed": float(true_cycles_closed),
        "true_cycles_added": float(true_cycles_added),
        "merge_hazard_count": float(merge_hazard_count),
        "elapsed": float(elapsed),
        "effort": float(effort),
        "neighbor_close_violation": float(neighbor_close_violation),
        "neighbor_far_violation": float(neighbor_far_violation),
        "fence_close_violation": float(fence_close_violation),
        "fence_far_violation": float(fence_far_violation),
        "area_progress_norm": float(area_progress),
        "perimeter_progress_norm": float(perimeter_progress),
        "largest_area_progress_norm": float(largest_area_progress),
        "largest_perimeter_progress_norm": float(largest_perimeter_progress),
        "area_regress_norm": float(area_regress),
        "perimeter_regress_norm": float(perimeter_regress),
        "largest_area_regress_norm": float(largest_area_regress),
        "largest_perimeter_regress_norm": float(largest_perimeter_regress),
        "area_residual_norm": float(area_residual),
        "perimeter_residual_norm": float(perimeter_residual),
        "largest_area_residual_norm": float(largest_area_residual),
        "largest_perimeter_residual_norm": float(largest_perimeter_residual),
        "one_hole_linger_penalty": float(one_hole_linger_penalty),
        "one_hole_area_scale": float(one_hole_area_scale),
        "one_hole_perimeter_scale": float(one_hole_perimeter_scale),
        "clear_indicator": float(clear_indicator),
        "success_time_bonus": float(success_time_bonus),
        "timeout_indicator": float(timeout_indicator),
    }
    weighted = {
        "true_cycles_closed": reward_config.true_cycle_closed_reward * raw["true_cycles_closed"],
        "true_cycles_added": -reward_config.true_cycle_added_penalty * raw["true_cycles_added"],
        "merge_hazard_count": -reward_config.merge_hazard_penalty_weight * raw["merge_hazard_count"],
        "elapsed": -reward_config.time_penalty * raw["elapsed"],
        "effort": -reward_config.control_effort_penalty * raw["effort"],
        "neighbor_close_violation": -reward_config.neighbor_close_penalty_weight * raw["neighbor_close_violation"],
        "neighbor_far_violation": -reward_config.neighbor_far_penalty_weight * raw["neighbor_far_violation"],
        "fence_close_violation": -reward_config.fence_close_penalty_weight * raw["fence_close_violation"],
        "fence_far_violation": -reward_config.fence_far_penalty_weight * raw["fence_far_violation"],
        "area_progress_norm": reward_config.area_progress_reward_weight * raw["area_progress_norm"],
        "perimeter_progress_norm": reward_config.perimeter_progress_reward_weight * raw["perimeter_progress_norm"],
        "largest_area_progress_norm": reward_config.largest_area_progress_reward_weight * raw["largest_area_progress_norm"],
        "largest_perimeter_progress_norm": reward_config.largest_perimeter_progress_reward_weight * raw["largest_perimeter_progress_norm"],
        "area_regress_norm": -reward_config.area_regress_penalty_weight * raw["area_regress_norm"],
        "perimeter_regress_norm": -reward_config.perimeter_regress_penalty_weight * raw["perimeter_regress_norm"],
        "largest_area_regress_norm": -reward_config.largest_area_regress_penalty_weight * raw["largest_area_regress_norm"],
        "largest_perimeter_regress_norm": -reward_config.largest_perimeter_regress_penalty_weight * raw["largest_perimeter_regress_norm"],
        "area_residual_norm": -reward_config.area_residual_penalty_weight * raw["area_residual_norm"],
        "perimeter_residual_norm": -reward_config.perimeter_residual_penalty_weight * raw["perimeter_residual_norm"],
        "largest_area_residual_norm": -reward_config.largest_area_residual_penalty_weight * raw["largest_area_residual_norm"],
        "largest_perimeter_residual_norm": -reward_config.largest_perimeter_residual_penalty_weight * raw["largest_perimeter_residual_norm"],
        "one_hole_linger_penalty": raw["one_hole_linger_penalty"],
        "one_hole_area_scale": raw["one_hole_area_scale"],
        "one_hole_perimeter_scale": raw["one_hole_perimeter_scale"],
        "clear_indicator": reward_config.clear_bonus * raw["clear_indicator"],
        "success_time_bonus": reward_config.success_time_bonus_weight * raw["success_time_bonus"],
        "timeout_indicator": -reward_config.timeout_penalty * raw["timeout_indicator"],
    }
    return {"raw": raw, "weighted": weighted}


def _active_policy_term_names(reward_config: RewardConfig) -> List[str]:
    always_include = [
        "true_cycles_closed",
        "true_cycles_added",
        "elapsed",
        "effort",
        "clear_indicator",
        "timeout_indicator",
        "success_time_bonus",
    ]
    weighted_terms = [
        ("merge_hazard_count", reward_config.merge_hazard_penalty_weight),
        ("neighbor_close_violation", reward_config.neighbor_close_penalty_weight),
        ("neighbor_far_violation", reward_config.neighbor_far_penalty_weight),
        ("fence_close_violation", reward_config.fence_close_penalty_weight),
        ("fence_far_violation", reward_config.fence_far_penalty_weight),
        ("area_progress_norm", reward_config.area_progress_reward_weight),
        ("perimeter_progress_norm", reward_config.perimeter_progress_reward_weight),
        ("largest_area_progress_norm", reward_config.largest_area_progress_reward_weight),
        ("largest_perimeter_progress_norm", reward_config.largest_perimeter_progress_reward_weight),
        ("area_regress_norm", reward_config.area_regress_penalty_weight),
        ("perimeter_regress_norm", reward_config.perimeter_regress_penalty_weight),
        ("largest_area_regress_norm", reward_config.largest_area_regress_penalty_weight),
        ("largest_perimeter_regress_norm", reward_config.largest_perimeter_regress_penalty_weight),
        ("area_residual_norm", reward_config.area_residual_penalty_weight),
        ("perimeter_residual_norm", reward_config.perimeter_residual_penalty_weight),
        ("largest_area_residual_norm", reward_config.largest_area_residual_penalty_weight),
        ("largest_perimeter_residual_norm", reward_config.largest_perimeter_residual_penalty_weight),
        ("one_hole_linger_penalty", reward_config.one_hole_linger_penalty_weight),
    ]
    if reward_config.area_progress_reward_weight != 0.0 or reward_config.area_regress_penalty_weight != 0.0:
        weighted_terms.append(("one_hole_area_scale", reward_config.one_hole_area_scale_alpha))
    if reward_config.perimeter_progress_reward_weight != 0.0 or reward_config.perimeter_regress_penalty_weight != 0.0:
        weighted_terms.append(("one_hole_perimeter_scale", reward_config.one_hole_perimeter_scale_alpha))

    names = list(always_include)
    names.extend(name for name, weight in weighted_terms if float(weight) != 0.0)
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(names))


def main() -> None:
    args = _make_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    config = _load_config(run_dir)
    ckpt_tag, ckpt_path = _resolve_checkpoint(run_dir, args.checkpoint)

    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise SystemExit("stable-baselines3 is required. Install with: pip install stable-baselines3") from exc

    # Ensure custom extractor/policy classes are importable for checkpoint load.
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

    outdir = Path(args.outdir).resolve() if args.outdir else (run_dir / f"rl_reeb_from_model_{ckpt_tag}")
    outdir.mkdir(parents=True, exist_ok=True)

    model = load_homological_gat_ppo(ckpt_path)
    env = make_eval_env(config, enable_event_logging=False, event_log_path=None)

    num_sims = max(1, int(args.num_sims))
    save_indices = set(_choose_saved_indices(num_sims, max(0, int(args.num_save))))
    deterministic = not bool(args.stochastic)

    sim_summaries: List[Dict] = []
    gallery_rows: List[Tuple[str, str]] = []

    for sim_idx in range(num_sims):
        save_this = sim_idx in save_indices
        sim_dir = outdir / f"sim_{sim_idx:03d}"
        frames_dir = sim_dir / "frames"
        if save_this:
            sim_dir.mkdir(parents=True, exist_ok=True)
            frames_dir.mkdir(parents=True, exist_ok=True)

        obs, reset_info = env.reset(seed=int(config.eval_seeds[sim_idx % len(config.eval_seeds)]))
        sim = env.simulation
        done = False
        truncated = False
        ep_return = 0.0
        step = 0

        builder = ReebEventGraphBuilder()
        builder.add_snapshot(
            step=0,
            time=float(sim.time),
            labels=sim.cycle_label.label,
            alpha_cycle=sim.topology.outer_cycle,
        )
        history_cursor = len(sim.cycle_label.history)
        frame_records: List[Dict] = []

        if save_this:
            frame_name = "frame_00000.png"
            _render_2d_frame(sim, frames_dir / frame_name)
            frame_records.append(
                {
                    "step": 0,
                    "time": float(sim.time),
                    "phase": str(reset_info.get("phase", "simplify")),
                    "image": str(Path("frames") / frame_name),
                    "atomic_events": [],
                    "snapshot_debug": builder.get_snapshot_debug(0),
                    "policy_terms": _policy_term_maps(
                        reward_config=config.reward_config,
                        true_cycles_closed=0.0,
                        true_cycles_added=0.0,
                        merge_hazard_count=0.0,
                        elapsed=0.0,
                        effort=0.0,
                        neighbor_close_violation=0.0,
                        neighbor_far_violation=0.0,
                        fence_close_violation=0.0,
                        fence_far_violation=0.0,
                        area_progress=0.0,
                        perimeter_progress=0.0,
                        largest_area_progress=0.0,
                        largest_perimeter_progress=0.0,
                        area_regress=0.0,
                        perimeter_regress=0.0,
                        largest_area_regress=0.0,
                        largest_perimeter_regress=0.0,
                        area_residual=0.0,
                        perimeter_residual=0.0,
                        largest_area_residual=0.0,
                        largest_perimeter_residual=0.0,
                        one_hole_linger_penalty=0.0,
                        one_hole_area_scale=1.0,
                        one_hole_perimeter_scale=1.0,
                        clear_indicator=0.0,
                        success_time_bonus=0.0,
                        timeout_indicator=0.0,
                    ),
                }
            )

        while not (done or truncated):
            if args.max_steps > 0 and step >= int(args.max_steps):
                break

            action, _ = model.predict(obs, deterministic=deterministic)
            action_arr = np.asarray(action, dtype=float)
            effort = float(np.mean(np.sum(action_arr * action_arr, axis=1))) if action_arr.size else 0.0
            obs, reward, done, truncated, info = env.step(action)
            ep_return += float(reward)
            step += 1
            sim = env.simulation

            new_history = sim.cycle_label.history[history_cursor:]
            history_cursor = len(sim.cycle_label.history)
            nontrivial = [
                (labels, alpha_change, boundary_change, event_time)
                for labels, alpha_change, boundary_change, event_time in new_history
                if any(alpha_change) or tuple(boundary_change) != (0, 0)
            ]

            atomic_events = []
            if nontrivial:
                for labels, alpha_change, boundary_change, event_time in nontrivial:
                    builder.add_snapshot(
                        step=step,
                        time=float(event_time),
                        labels=labels,
                        alpha_cycle=sim.topology.outer_cycle,
                    )
                    if save_this:
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

            if save_this:
                policy_terms = _policy_term_maps(
                    reward_config=config.reward_config,
                    true_cycles_closed=float(info.get("true_cycles_closed", 0.0)),
                    true_cycles_added=float(info.get("true_cycles_added", 0.0)),
                    merge_hazard_count=float(info.get("merge_hazard_count", 0.0)),
                    elapsed=float(info.get("elapsed", 0.0)),
                    effort=effort,
                    neighbor_close_violation=float(info.get("neighbor_close_violation", 0.0)),
                    neighbor_far_violation=float(info.get("neighbor_far_violation", 0.0)),
                    fence_close_violation=float(info.get("fence_close_violation", 0.0)),
                    fence_far_violation=float(info.get("fence_far_violation", 0.0)),
                    area_progress=float(info.get("true_cycle_area_delta_norm", 0.0)),
                    perimeter_progress=float(info.get("true_cycle_perimeter_delta_norm", 0.0)),
                    largest_area_progress=float(info.get("largest_true_cycle_area_delta_norm", 0.0)),
                    largest_perimeter_progress=float(info.get("largest_true_cycle_perimeter_delta_norm", 0.0)),
                    area_regress=float(max(0.0, -float(info.get("true_cycle_area_delta_norm", 0.0)))),
                    perimeter_regress=float(max(0.0, -float(info.get("true_cycle_perimeter_delta_norm", 0.0)))),
                    largest_area_regress=float(max(0.0, -float(info.get("largest_true_cycle_area_delta_norm", 0.0)))),
                    largest_perimeter_regress=float(max(0.0, -float(info.get("largest_true_cycle_perimeter_delta_norm", 0.0)))),
                    area_residual=float(info.get("true_cycle_area_norm", 0.0)),
                    perimeter_residual=float(info.get("true_cycle_perimeter_norm", 0.0)),
                    largest_area_residual=float(info.get("largest_true_cycle_area_norm", 0.0)),
                    largest_perimeter_residual=float(info.get("largest_true_cycle_perimeter_norm", 0.0)),
                    one_hole_linger_penalty=float(info.get("reward_terms", {}).get("one_hole_linger_penalty", 0.0)),
                    one_hole_area_scale=float(info.get("reward_terms", {}).get("one_hole_area_scale", 1.0)),
                    one_hole_perimeter_scale=float(info.get("reward_terms", {}).get("one_hole_perimeter_scale", 1.0)),
                    clear_indicator=float(1.0 if done else 0.0),
                    success_time_bonus=(
                        1.0 / max(float(info.get("time", 0.0)), 1e-8)
                        if done
                        else 0.0
                    ),
                    timeout_indicator=float(1.0 if (truncated and not done) else 0.0),
                )
                frame_name = f"frame_{step:05d}.png"
                _render_2d_frame(sim, frames_dir / frame_name)
                frame_records.append(
                    {
                        "step": step,
                        "time": float(sim.time),
                        "phase": str(info.get("phase", "simplify")),
                        "image": str(Path("frames") / frame_name),
                        "atomic_events": atomic_events,
                        "snapshot_debug": builder.get_snapshot_debug(step),
                        "policy_terms": policy_terms,
                    }
                )

        builder.close(step=step + 1, time=float(sim.time))
        cleared = bool(done)
        timed_out = bool(truncated and not done)

        sim_summary = {
            "sim_index": sim_idx,
            "saved": bool(save_this),
            "return": float(ep_return),
            "steps": int(step),
            "final_time": float(sim.time),
            "cleared": cleared,
            "timed_out": timed_out,
        }
        sim_summaries.append(sim_summary)

        if save_this:
            timeline = {
                "interval_ms": int(max(1, args.frame_interval_ms)),
                "highlight_half_width": float(max(0.001, 0.48 * config.dt)),
                "reeb_graph": _build_reeb_graph_plot_data(builder),
                "active_term_names": _active_policy_term_names(config.reward_config),
                "frames": frame_records,
                "export_config": {
                    "source_dir": str(sim_dir.resolve()),
                    "category": "rl",
                    "motion_model": f"RL ({getattr(config, 'model_kind', 'policy')})",
                    "display_name": run_dir.name,
                    "source_run": run_dir.name,
                    "checkpoint": ckpt_tag,
                },
            }
            (sim_dir / "timeline.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")
            _write_timeline_html(sim_dir / "index.html", timeline)
            (sim_dir / "summary.json").write_text(json.dumps(sim_summary, indent=2, sort_keys=True), encoding="utf-8")
            gallery_rows.append((f"sim_{sim_idx:03d}", str((sim_dir / "index.html").relative_to(outdir))))

    env.close()

    clear_rate = float(np.mean([1.0 if s["cleared"] else 0.0 for s in sim_summaries])) if sim_summaries else 0.0
    timeout_rate = float(np.mean([1.0 if s["timed_out"] else 0.0 for s in sim_summaries])) if sim_summaries else 0.0
    mean_return = float(np.mean([s["return"] for s in sim_summaries])) if sim_summaries else 0.0
    mean_final_time = float(np.mean([s["final_time"] for s in sim_summaries])) if sim_summaries else 0.0

    manifest = {
        "run_dir": str(run_dir),
        "checkpoint_tag": ckpt_tag,
        "checkpoint_path": str(ckpt_path),
        "config": asdict(config),
        "num_sims": num_sims,
        "num_saved": len(save_indices),
        "saved_indices": sorted(int(i) for i in save_indices),
        "summary": {
            "clear_rate": clear_rate,
            "timeout_rate": timeout_rate,
            "mean_return": mean_return,
            "mean_final_time": mean_final_time,
        },
        "sim_summaries": sim_summaries,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    gallery_html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>RL Reeb Gallery</title></head><body>",
        f"<h2>Checkpoint: {ckpt_tag}</h2>",
        f"<p>clear_rate={clear_rate:.3f} timeout_rate={timeout_rate:.3f} mean_return={mean_return:.3f} mean_final_time={mean_final_time:.3f}</p>",
        "<ul>",
    ]
    for name, rel in gallery_rows:
        gallery_html.append(f"<li><a href='{rel}'>{name}</a></li>")
    gallery_html.extend(["</ul>", "</body></html>"])
    (outdir / "index.html").write_text("\n".join(gallery_html), encoding="utf-8")

    print(
        f"Finished {num_sims} simulations from checkpoint '{ckpt_tag}'. "
        f"Saved {len(save_indices)} visualization runs to: {outdir}"
    )

    if args.open:
        import webbrowser

        webbrowser.open(_open_via_local_viewer_server(outdir))


if __name__ == "__main__":
    main()
