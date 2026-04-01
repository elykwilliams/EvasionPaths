#!/usr/bin/env python3
"""Animate billiard motion in a Bunimovich stadium without running the evasion-path simulation."""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boundary_geometry import BunimovichStadium
from motion_model import BilliardMotion
from plotting_tools import show_domain_boundary, show_fence_sensors
from sensor_network import SensorNetwork, generate_fence_sensors, generate_mobile_sensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize billiard motion in a Bunimovich stadium.")
    parser.add_argument("--width", type=float, default=1.0, help="Half-length of the stadium's flat section.")
    parser.add_argument("--radius", type=float, default=1.0, help="Radius of the stadium endcaps.")
    parser.add_argument("--n-sensors", type=int, default=12, help="Number of mobile sensors.")
    parser.add_argument("--speed", type=float, default=0.75, help="Initial speed magnitude for each mobile sensor.")
    parser.add_argument("--sensing-radius", type=float, default=0.18, help="Radius used for fence spacing and optional disks.")
    parser.add_argument("--dt", type=float, default=0.02, help="Simulation timestep.")
    parser.add_argument("--frames", type=int, default=600, help="Number of animation frames.")
    parser.add_argument("--interval-ms", type=float, default=20.0, help="Delay between frames in milliseconds.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible initial conditions.")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Advance the system before displaying the first frame.")
    parser.add_argument("--trail-length", type=int, default=35, help="Number of previous positions to keep per sensor.")
    parser.add_argument("--show-fence", action="store_true", help="Plot the fence sensor locations.")
    parser.add_argument("--show-radius", action="store_true", help="Draw sensing disks for mobile sensors.")
    parser.add_argument("--save", type=str, default="", help="Optional path to save the animation, e.g. out.mp4 or out.gif.")
    return parser.parse_args()


def mobile_points(sensor_network: SensorNetwork) -> np.ndarray:
    return np.array([sensor.pos for sensor in sensor_network.mobile_sensors], dtype=float)


def mobile_velocities(sensor_network: SensorNetwork) -> np.ndarray:
    return np.array([sensor.vel for sensor in sensor_network.mobile_sensors], dtype=float)


def draw_mobile_sensing_disks(ax: plt.Axes, sensor_network: SensorNetwork) -> None:
    for sensor in sensor_network.mobile_sensors:
        disk = plt.Circle(sensor.pos, sensor.radius, color="tab:blue", alpha=0.10, linewidth=0)
        ax.add_artist(disk)


def draw_frame(
    ax: plt.Axes,
    sensor_network: SensorNetwork,
    domain: BunimovichStadium,
    histories: list[deque[np.ndarray]],
    sim_time: float,
    *,
    show_fence: bool,
    show_radius: bool,
) -> None:
    ax.cla()
    ax.set_aspect("equal", adjustable="box")
    pad = 0.2 * domain.r + 0.1
    ax.set_xlim(-(domain.w + domain.r) - pad, domain.w + domain.r + pad)
    ax.set_ylim(-domain.r - pad, domain.r + pad)
    ax.set_title(f"Bunimovich billiard motion, t = {sim_time:0.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    show_domain_boundary(domain, ax=ax)
    if show_fence:
        show_fence_sensors(sensor_network, ax=ax)

    if show_radius:
        draw_mobile_sensing_disks(ax, sensor_network)

    for history in histories:
        if len(history) < 2:
            continue
        pts = np.asarray(history, dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], color="tab:orange", alpha=0.45, linewidth=1.0)

    points = mobile_points(sensor_network)
    velocities = mobile_velocities(sensor_network)
    if len(points):
        ax.scatter(points[:, 0], points[:, 1], s=30, color="tab:red", zorder=3, label="mobile sensors")
        ax.quiver(
            points[:, 0],
            points[:, 1],
            velocities[:, 0],
            velocities[:, 1],
            angles="xy",
            scale_units="xy",
            scale=2.5,
            width=0.004,
            color="tab:green",
            alpha=0.85,
            zorder=4,
        )

    if show_fence or len(points):
        ax.legend(loc="upper right")
    ax.grid(alpha=0.20)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    domain = BunimovichStadium(w=float(args.width), r=float(args.radius), L=2.0 * (float(args.width) + float(args.radius)))
    motion_model = BilliardMotion()
    fence = generate_fence_sensors(domain, float(args.sensing_radius))
    mobile = generate_mobile_sensors(domain, int(args.n_sensors), float(args.sensing_radius), float(args.speed))
    sensor_network = SensorNetwork(mobile, motion_model, fence, float(args.sensing_radius), domain)

    for _ in range(max(0, int(args.warmup_steps))):
        sensor_network.move(float(args.dt))
        sensor_network.update()

    histories = [deque([sensor.pos.copy()], maxlen=max(1, int(args.trail_length))) for sensor in sensor_network.mobile_sensors]
    sim_time = 0.0

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)

    def update(_frame_idx: int):
        nonlocal sim_time
        sensor_network.move(float(args.dt))
        sensor_network.update()
        sim_time += float(args.dt)

        for history, sensor in zip(histories, sensor_network.mobile_sensors):
            history.append(sensor.pos.copy())

        draw_frame(
            ax,
            sensor_network,
            domain,
            histories,
            sim_time,
            show_fence=bool(args.show_fence),
            show_radius=bool(args.show_radius),
        )
        return ()

    draw_frame(
        ax,
        sensor_network,
        domain,
        histories,
        sim_time,
        show_fence=bool(args.show_fence),
        show_radius=bool(args.show_radius),
    )

    ani = FuncAnimation(
        fig,
        update,
        frames=int(args.frames),
        interval=float(args.interval_ms),
        blit=False,
        cache_frame_data=False,
    )

    if args.save:
        ani.save(args.save)
    plt.show()


if __name__ == "__main__":
    main()
