# ******************************************************************************
#  Copyright (c) 2023, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************
"""Sample script for 2D rectangle simulation with optional Reeb-graph visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from UI import build_and_open_2d_timeline_ui
from plotting_tools import show_state
from reeb_graph import (
    animate_sensor_and_reeb,
    default_gif_path,
    print_atomic_change_report,
    run_2d_reeb_simulation,
    save_animation_gif,
)


def animate_sensor_only(simulation, *, max_steps: int, interval_ms: int = 130):
    fig = plt.figure(figsize=(9, 7))
    ax = plt.gca()
    state = {"step": 0}

    def update(_):
        if state["step"] < max_steps:
            simulation.do_timestep()
            state["step"] += 1

        ax.clear()
        ax.axis("off")
        ax.axis("equal")
        ax.set_title(f"T = {simulation.time:6.3f}", loc="left")
        show_state(simulation, ax=ax)

    return FuncAnimation(fig, update, interval=interval_ms, frames=range(max_steps + 1), repeat=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample 2D run with optional Reeb graph panel.")
    parser.add_argument("--num-sensors", type=int, default=12)
    parser.add_argument("--radius", type=float, default=0.3)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--velocity", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--clear-streak", type=int, default=8)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--with-reeb", action="store_true", default=True)
    parser.add_argument("--save-gif", action="store_true", default=True)
    parser.add_argument("--gif-path", type=str, default="")
    parser.add_argument("--gif-fps", type=int, default=8)
    args = parser.parse_args()

    simulation, builder, steps, _clear_streak = run_2d_reeb_simulation(
        num_sensors=args.num_sensors,
        sensing_radius=args.radius,
        timestep_size=args.dt,
        sensor_velocity=args.velocity,
        max_steps=args.max_steps,
        clear_streak_needed=args.clear_streak,
        seed=args.seed,
    )

    print_atomic_change_report(simulation, builder.summaries, dt=args.dt)

    try:
        launch_ui = input("Simulation finished. Open 2D timeline UI? [y/N]: ").strip().lower() == "y"
    except EOFError:
        launch_ui = False

    if launch_ui:
        html_path = build_and_open_2d_timeline_ui(
            simulation=simulation,
            builder=builder,
            run_steps=steps,
            num_sensors=args.num_sensors,
            sensing_radius=args.radius,
            timestep_size=args.dt,
            sensor_velocity=args.velocity,
            seed=args.seed,
            interval_ms=130,
        )
        print(f"Opened timeline UI: {html_path}")

    if args.with_reeb:
        ani = animate_sensor_and_reeb(
            builder,
            run_steps=steps,
            num_sensors=args.num_sensors,
            sensing_radius=args.radius,
            timestep_size=args.dt,
            sensor_velocity=args.velocity,
            seed=args.seed,
        )
        gif_prefix = "sample_reeb_coupled"
    else:
        ani = animate_sensor_only(
            simulation,
            max_steps=steps,
            interval_ms=130,
        )
        gif_prefix = "sample_sensor_only"

    if args.save_gif and not launch_ui:
        gif_path = Path(args.gif_path) if args.gif_path else default_gif_path(gif_prefix)
        save_animation_gif(ani, gif_path, fps=args.gif_fps)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
