"""Run the 2D rectangular Reeb-event visualization experiment using shared src logic."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from reeb_graph import (
    animate_reeb_graph,
    animate_sensor_and_reeb,
    default_gif_path,
    draw_reeb_graph,
    make_frames_dir,
    print_atomic_change_report,
    run_2d_reeb_simulation,
    save_animation_gif,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2D simulation and build a Reeb event graph.")
    parser.add_argument("--num-sensors", type=int, default=12)
    parser.add_argument("--radius", type=float, default=0.3)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--velocity", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--clear-streak", type=int, default=8)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--animate", action="store_true", default=True)
    parser.add_argument("--coupled-animate", action="store_true", default=True)
    parser.add_argument("--save-frames", action="store_true", default=False)
    parser.add_argument("--frames-dir", type=str, default="")
    parser.add_argument("--save-gif", action="store_true", default=False)
    parser.add_argument("--gif-path", type=str, default="")
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    frames_dir = make_frames_dir(args.frames_dir if args.frames_dir else None) if args.save_frames else None

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

    if args.no_plot:
        return

    if args.coupled_animate:
        ani = animate_sensor_and_reeb(
            builder,
            run_steps=steps,
            num_sensors=args.num_sensors,
            sensing_radius=args.radius,
            timestep_size=args.dt,
            sensor_velocity=args.velocity,
            seed=args.seed,
            save_frames_dir=frames_dir,
        )
        if args.save_gif:
            gif_path = Path(args.gif_path) if args.gif_path else default_gif_path("reeb_coupled")
            save_animation_gif(ani, gif_path, fps=args.gif_fps)
        plt.tight_layout()
        plt.show()
    elif args.animate:
        ani = animate_reeb_graph(builder.graph, save_frames_dir=frames_dir)
        if args.save_gif:
            gif_path = Path(args.gif_path) if args.gif_path else default_gif_path("reeb_graph")
            save_animation_gif(ani, gif_path, fps=args.gif_fps)
        plt.show()
    else:
        draw_reeb_graph(builder.graph, title="")


if __name__ == "__main__":
    main()
