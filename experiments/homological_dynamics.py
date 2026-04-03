#!/usr/bin/env python3
"""Run 2D homological dynamics and visualize coupled sensor + Reeb evolution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from UI import (
    _build_reeb_graph_plot_data,
    _collect_atomic_events_by_step,
    _render_2d_frame,
    _write_timeline_html,
)
from boundary_geometry import RectangularDomain
from motion_model import HomologicalDynamicsMotion
from plotting_tools import show_state
from reeb_graph import (
    _compact_view_graph,
    _draw_compact_graph_panel,
    _optimize_compact_layout,
    all_interior_cycles_false,
    default_gif_path,
    print_atomic_change_report,
    run_online_reeb_simulation,
    save_animation_gif,
)
from sensor_network import SensorNetwork, generate_fence_sensors, generate_mobile_sensors
from time_stepping import EvasionPathSimulation


def build_homological_simulation(
    *,
    num_sensors: int,
    sensing_radius: float,
    timestep_size: float,
    sensor_velocity: float,
    seed: int,
    max_speed: float,
    lambda_shrink: float,
    mu_curvature: float,
    eta_cohesion: float,
    repulsion_strength: float,
    repulsion_power: float,
    auto_d_safe: bool,
    d_safe_manual: float,
) -> EvasionPathSimulation:
    np.random.seed(seed)
    domain = RectangularDomain()
    motion_model = HomologicalDynamicsMotion(
        sensing_radius=sensing_radius,
        max_speed=max_speed,
        lambda_shrink=lambda_shrink,
        mu_curvature=mu_curvature,
        eta_cohesion=eta_cohesion,
        repulsion_strength=repulsion_strength,
        repulsion_power=repulsion_power,
        d_safe_manual=d_safe_manual,
        auto_d_safe=auto_d_safe,
    )
    fence = generate_fence_sensors(domain, sensing_radius)
    mobile_sensors = generate_mobile_sensors(domain, num_sensors, sensing_radius, sensor_velocity)
    sensor_network = SensorNetwork(mobile_sensors, motion_model, fence, sensing_radius, domain)
    return EvasionPathSimulation(sensor_network, timestep_size)


def animate_homological_sensor_and_reeb(
    builder,
    *,
    run_steps: int,
    num_sensors: int,
    sensing_radius: float,
    timestep_size: float,
    sensor_velocity: float,
    seed: int,
    max_speed: float,
    lambda_shrink: float,
    mu_curvature: float,
    eta_cohesion: float,
    repulsion_strength: float,
    repulsion_power: float,
    auto_d_safe: bool,
    d_safe_manual: float,
    interval_ms: int = 130,
):
    simulation = build_homological_simulation(
        num_sensors=num_sensors,
        sensing_radius=sensing_radius,
        timestep_size=timestep_size,
        sensor_velocity=sensor_velocity,
        seed=seed,
        max_speed=max_speed,
        lambda_shrink=lambda_shrink,
        mu_curvature=mu_curvature,
        eta_cohesion=eta_cohesion,
        repulsion_strength=repulsion_strength,
        repulsion_power=repulsion_power,
        auto_d_safe=auto_d_safe,
        d_safe_manual=d_safe_manual,
    )
    if simulation.topology.dim != 2:
        raise ValueError("Coupled animation is currently only supported for 2D simulations.")

    compact, event_nodes = _compact_view_graph(builder.graph)
    layout_pos = _optimize_compact_layout(compact, event_nodes)
    nx.set_node_attributes(compact, layout_pos, "pos")
    xs = [p[0] for p in layout_pos.values()]
    ys = [p[1] for p in layout_pos.values()]
    x_pad = max(0.01, 0.02 * (max(xs) - min(xs) if xs else 1.0))
    y_pad = 1.0
    highlight_half_width = max(0.001, 0.48 * timestep_size)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(12, 10),
        gridspec_kw={"height_ratios": [3, 2]},
    )
    state = {"step": 0, "clear_streak": 0}

    def draw_graph_panel():
        ax_bottom.clear()
        ax_bottom.set_xlabel("time")
        ax_bottom.set_ylabel("cycle lane")
        _draw_compact_graph_panel(
            ax_bottom,
            compact,
            event_nodes,
            highlight_time=simulation.time,
            highlight_half_width=highlight_half_width,
        )
        if xs and ys:
            ax_bottom.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
            ax_bottom.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

    def draw_sensor_panel():
        ax_top.clear()
        ax_top.axis("off")
        ax_top.axis("equal")
        ax_top.set_title(f"T = {simulation.time:6.3f}", loc="left")
        show_state(simulation, ax=ax_top)

    def update(frame_idx: int):
        if frame_idx > 0 and state["step"] < run_steps:
            simulation.do_timestep()
            state["step"] += 1
            if all_interior_cycles_false(simulation.cycle_label.label, simulation.topology.outer_cycle):
                state["clear_streak"] += 1
            else:
                state["clear_streak"] = 0

        draw_sensor_panel()
        draw_graph_panel()

    draw_sensor_panel()
    draw_graph_panel()
    return FuncAnimation(fig, update, interval=interval_ms, frames=range(run_steps + 1), repeat=False)


def _write_homological_timeline_bundle(
    *,
    output_dir: Path,
    simulation,
    builder,
    run_steps: int,
    num_sensors: int,
    sensing_radius: float,
    timestep_size: float,
    sensor_velocity: float,
    seed: int,
    max_speed: float,
    lambda_shrink: float,
    mu_curvature: float,
    eta_cohesion: float,
    repulsion_strength: float,
    repulsion_power: float,
    auto_d_safe: bool,
    d_safe_manual: float,
    interval_ms: int,
    cleared: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_dir = output_dir / "sim_000"
    frames_dir = sim_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    replay = build_homological_simulation(
        num_sensors=num_sensors,
        sensing_radius=sensing_radius,
        timestep_size=timestep_size,
        sensor_velocity=sensor_velocity,
        seed=seed,
        max_speed=max_speed,
        lambda_shrink=lambda_shrink,
        mu_curvature=mu_curvature,
        eta_cohesion=eta_cohesion,
        repulsion_strength=repulsion_strength,
        repulsion_power=repulsion_power,
        auto_d_safe=auto_d_safe,
        d_safe_manual=d_safe_manual,
    )

    frame_records = []
    step_to_summary = {s.step: s for s in builder.summaries}
    atomic_events_by_step = _collect_atomic_events_by_step(
        history=simulation.cycle_label.history,
        summaries=builder.summaries,
    )

    for step in range(run_steps + 1):
        if step > 0:
            replay.do_timestep()
        frame_name = f"frame_{step:05d}.png"
        _render_2d_frame(replay, frames_dir / frame_name)
        summary = step_to_summary.get(step)
        frame_records.append(
            {
                "step": step,
                "time": float(replay.time),
                "phase": "homological",
                "image": str(Path("frames") / frame_name),
                "summary": {
                    "n_cycles": int(summary.n_cycles) if summary else 0,
                    "n_true": int(summary.n_true) if summary else 0,
                    "n_false": int(summary.n_false) if summary else 0,
                    "n_birth": int(summary.n_birth) if summary else 0,
                    "n_death": int(summary.n_death) if summary else 0,
                    "n_continue": int(summary.n_continue) if summary else 0,
                    "n_split_edges": int(summary.n_split_edges) if summary else 0,
                    "n_merge_edges": int(summary.n_merge_edges) if summary else 0,
                    "n_transform_edges": int(summary.n_transform_edges) if summary else 0,
                    "n_label_flips": int(summary.n_label_flips) if summary else 0,
                },
                "atomic_events": atomic_events_by_step.get(step, []),
                "snapshot_debug": builder.get_snapshot_debug(step),
                "policy_terms": {"raw": {}, "weighted": {}},
            }
        )

    timeline = {
        "interval_ms": int(interval_ms),
        "highlight_half_width": float(max(0.001, 0.48 * timestep_size)),
        "reeb_graph": _build_reeb_graph_plot_data(builder),
        "active_term_names": [],
        "frames": frame_records,
        "export_config": {
            "source_dir": str(sim_dir.resolve()),
            "category": "homological",
            "motion_model": "Homological Motion",
            "display_name": output_dir.name,
            "source_run": output_dir.name,
            "checkpoint": "",
        },
    }
    (sim_dir / "timeline.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    _write_timeline_html(sim_dir / "index.html", timeline)

    summary = {
        "sim_index": 0,
        "saved": True,
        "return": 0.0,
        "steps": int(run_steps),
        "final_time": float(simulation.time),
        "cleared": bool(cleared),
        "timed_out": bool(not cleared),
    }
    (sim_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "source": "homological_dynamics",
        "num_sims": 1,
        "num_saved": 1,
        "summary": {
            "clear_rate": 1.0 if cleared else 0.0,
            "timeout_rate": 0.0 if cleared else 1.0,
            "mean_return": 0.0,
            "mean_final_time": float(simulation.time),
        },
        "sim_summaries": [summary],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "index.html").write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset='utf-8'><title>Homological Timeline Export</title></head><body>",
                "<h2>Homological Timeline Export</h2>",
                "<ul><li><a href='sim_000/index.html'>sim_000</a></li></ul>",
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2D simulation with HomologicalDynamicsMotion.")
    parser.add_argument("--num-sensors", type=int, default=12)
    parser.add_argument("--radius", type=float, default=0.3)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--velocity", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--clear-streak", type=int, default=8)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--interval-ms", type=int, default=130)

    parser.add_argument("--max-speed", type=float, default=1.0)
    parser.add_argument("--lambda-shrink", type=float, default=1.0)
    parser.add_argument("--mu-curvature", type=float, default=0.5)
    parser.add_argument("--eta-cohesion", type=float, default=0.2)
    parser.add_argument("--repulsion-strength", type=float, default=0.1)
    parser.add_argument("--repulsion-power", type=float, default=2.0)
    parser.add_argument("--auto-d-safe", action="store_true", default=True)
    parser.add_argument("--manual-d-safe", type=float, default=None)

    parser.add_argument("--save-gif", action="store_true", default=False)
    parser.add_argument("--gif-path", type=str, default="")
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument(
        "--save-timeline-dir",
        type=str,
        default="",
        help="Optional output dir for a publishable timeline bundle (manifest + sim_000/timeline.json).",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    auto_d_safe = bool(args.auto_d_safe)
    d_safe_manual = float(2.0 * args.radius + 0.02)
    if args.manual_d_safe is not None:
        auto_d_safe = False
        d_safe_manual = float(args.manual_d_safe)

    simulation = build_homological_simulation(
        num_sensors=args.num_sensors,
        sensing_radius=args.radius,
        timestep_size=args.dt,
        sensor_velocity=args.velocity,
        seed=args.seed,
        max_speed=args.max_speed,
        lambda_shrink=args.lambda_shrink,
        mu_curvature=args.mu_curvature,
        eta_cohesion=args.eta_cohesion,
        repulsion_strength=args.repulsion_strength,
        repulsion_power=args.repulsion_power,
        auto_d_safe=auto_d_safe,
        d_safe_manual=d_safe_manual,
    )

    simulation, builder, steps, _ = run_online_reeb_simulation(
        simulation,
        max_steps=args.max_steps,
        clear_streak_needed=args.clear_streak,
    )
    print_atomic_change_report(simulation, builder.summaries, dt=args.dt)

    cleared = all_interior_cycles_false(simulation.cycle_label.label, simulation.topology.excluded_cycles)

    if args.save_timeline_dir:
        outdir = _write_homological_timeline_bundle(
            output_dir=Path(args.save_timeline_dir).resolve(),
            simulation=simulation,
            builder=builder,
            run_steps=steps,
            num_sensors=args.num_sensors,
            sensing_radius=args.radius,
            timestep_size=args.dt,
            sensor_velocity=args.velocity,
            seed=args.seed,
            max_speed=args.max_speed,
            lambda_shrink=args.lambda_shrink,
            mu_curvature=args.mu_curvature,
            eta_cohesion=args.eta_cohesion,
            repulsion_strength=args.repulsion_strength,
            repulsion_power=args.repulsion_power,
            auto_d_safe=auto_d_safe,
            d_safe_manual=d_safe_manual,
            interval_ms=args.interval_ms,
            cleared=cleared,
        )
        print(f"Saved homological timeline bundle to: {outdir}")

    if args.no_plot:
        return

    ani = animate_homological_sensor_and_reeb(
        builder,
        run_steps=steps,
        num_sensors=args.num_sensors,
        sensing_radius=args.radius,
        timestep_size=args.dt,
        sensor_velocity=args.velocity,
        seed=args.seed,
        max_speed=args.max_speed,
        lambda_shrink=args.lambda_shrink,
        mu_curvature=args.mu_curvature,
        eta_cohesion=args.eta_cohesion,
        repulsion_strength=args.repulsion_strength,
        repulsion_power=args.repulsion_power,
        auto_d_safe=auto_d_safe,
        d_safe_manual=d_safe_manual,
        interval_ms=args.interval_ms,
    )

    if args.save_gif:
        gif_path = Path(args.gif_path) if args.gif_path else default_gif_path("homological_dynamics")
        save_animation_gif(ani, gif_path, fps=args.gif_fps)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
