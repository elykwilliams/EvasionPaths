#!/usr/bin/env python3
"""Animate Vicsek-style motion in a Bunimovich stadium without running the evasion-path simulation."""

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
from motion_model import Viscek
from plotting_tools import show_domain_boundary, show_fence_sensors
from sensor_network import SensorNetwork, generate_fence_sensors, generate_mobile_sensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Vicsek motion in a Bunimovich stadium.")
    parser.add_argument("--width", type=float, default=1.0, help="Half-length of the stadium's flat section.")
    parser.add_argument("--radius", type=float, default=1.0, help="Radius of the stadium endcaps.")
    parser.add_argument("--n-sensors", type=int, default=20, help="Number of mobile sensors.")
    parser.add_argument("--speed", type=float, default=0.75, help="Initial speed magnitude for each mobile sensor.")
    parser.add_argument("--sensing-radius", type=float, default=0.18, help="Radius used for fence spacing and optional disks.")
    parser.add_argument("--interaction-radius", type=float, default=0.30, help="Vicsek alignment radius.")
    parser.add_argument("--noise-scale", type=float, default=float(np.pi / 12.0), help="Maximum angular noise magnitude in radians.")
    parser.add_argument("--dt", type=float, default=0.02, help="Simulation timestep.")
    parser.add_argument("--frames", type=int, default=600, help="Number of animation frames.")
    parser.add_argument("--interval-ms", type=float, default=20.0, help="Delay between frames in milliseconds.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible initial conditions.")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Advance the system before displaying the first frame.")
    parser.add_argument("--trail-length", type=int, default=35, help="Number of previous positions to keep per sensor.")
    parser.add_argument("--show-fence", action="store_true", help="Plot the fence sensor locations.")
    parser.add_argument("--show-radius", action="store_true", help="Draw sensing disks for mobile sensors.")
    parser.add_argument("--square-init", action="store_true", help="Initialize mobile sensors from the centered student-style square box.")
    parser.add_argument("--save", type=str, default="", help="Optional path to save the animation, e.g. out.mp4 or out.gif.")
    return parser.parse_args()


def mobile_points(sensor_network: SensorNetwork) -> np.ndarray:
    return np.array([sensor.pos for sensor in sensor_network.mobile_sensors], dtype=float)


def mobile_velocities(sensor_network: SensorNetwork) -> np.ndarray:
    return np.array([sensor.vel for sensor in sensor_network.mobile_sensors], dtype=float)


def polarization(sensor_network: SensorNetwork) -> float:
    velocities = mobile_velocities(sensor_network)
    if len(velocities) == 0:
        return 0.0
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    headings = velocities / np.maximum(speeds, 1e-12)
    return float(np.linalg.norm(np.sum(headings, axis=0)) / len(headings))


def draw_mobile_sensing_disks(ax: plt.Axes, sensor_network: SensorNetwork) -> None:
    for sensor in sensor_network.mobile_sensors:
        disk = plt.Circle(sensor.pos, sensor.radius, color="tab:blue", alpha=0.10, linewidth=0)
        ax.add_artist(disk)


def draw_frame(
    domain_ax: plt.Axes,
    polar_ax: plt.Axes,
    sensor_network: SensorNetwork,
    domain: BunimovichStadium,
    histories: list[deque[np.ndarray]],
    sim_time: float,
    polarization_times: list[float],
    polarization_values: list[float],
    *,
    show_fence: bool,
    show_radius: bool,
    interaction_radius: float,
    noise_scale: float,
) -> None:
    domain_ax.cla()
    polar_ax.cla()

    domain_ax.set_aspect("equal", adjustable="box")
    pad = 0.2 * domain.r + 0.1
    domain_ax.set_xlim(-(domain.w + domain.r) - pad, domain.w + domain.r + pad)
    domain_ax.set_ylim(-domain.r - pad, domain.r + pad)
    domain_ax.set_title(
        f"Bunimovich Vicsek motion, t = {sim_time:0.2f}\n"
        f"interaction radius = {interaction_radius:0.2f}, noise = {noise_scale:0.3f}"
    )
    domain_ax.set_xlabel("x")
    domain_ax.set_ylabel("y")

    show_domain_boundary(domain, ax=domain_ax)
    if show_fence:
        show_fence_sensors(sensor_network, ax=domain_ax)

    if show_radius:
        draw_mobile_sensing_disks(domain_ax, sensor_network)

    for history in histories:
        if len(history) < 2:
            continue
        pts = np.asarray(history, dtype=float)
        domain_ax.plot(pts[:, 0], pts[:, 1], color="tab:orange", alpha=0.45, linewidth=1.0)

    points = mobile_points(sensor_network)
    velocities = mobile_velocities(sensor_network)
    if len(points):
        domain_ax.scatter(points[:, 0], points[:, 1], s=30, color="tab:red", zorder=3, label="mobile sensors")
        domain_ax.quiver(
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
        domain_ax.legend(loc="upper right")
    domain_ax.grid(alpha=0.20)

    polar_ax.plot(polarization_times, polarization_values, color="#1f77b4", linewidth=1.8)
    polar_ax.scatter([polarization_times[-1]], [polarization_values[-1]], color="#1f77b4", s=28, zorder=3)
    polar_ax.set_xlim(0.0, max(polarization_times[-1], sim_time, 1e-6))
    polar_ax.set_ylim(0.0, 1.02)
    polar_ax.set_title("Polarization")
    polar_ax.set_xlabel("time")
    polar_ax.set_ylabel("P(t)")
    polar_ax.grid(alpha=0.25)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    domain = BunimovichStadium(
        w=float(args.width),
        r=float(args.radius),
        L=2.0 * (float(args.width) + float(args.radius)),
        square_init=bool(args.square_init),
        square_init_length=min(float(args.width), float(args.radius)),
    )
    motion_model = Viscek(
        large_dt=float(args.dt),
        radius=float(args.interaction_radius),
        noise_scale=float(args.noise_scale),
    )
    fence = generate_fence_sensors(domain, float(args.sensing_radius))
    mobile = generate_mobile_sensors(domain, int(args.n_sensors), float(args.sensing_radius), float(args.speed))
    sensor_network = SensorNetwork(mobile, motion_model, fence, float(args.sensing_radius), domain)

    for _ in range(max(0, int(args.warmup_steps))):
        sensor_network.move(float(args.dt))
        sensor_network.update()

    histories = [deque([sensor.pos.copy()], maxlen=max(1, int(args.trail_length))) for sensor in sensor_network.mobile_sensors]
    sim_time = 0.0
    polarization_times = [0.0]
    polarization_values = [polarization(sensor_network)]

    fig, (domain_ax, polar_ax) = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)

    def update(_frame_idx: int):
        nonlocal sim_time
        sensor_network.move(float(args.dt))
        sensor_network.update()
        sim_time += float(args.dt)

        for history, sensor in zip(histories, sensor_network.mobile_sensors):
            history.append(sensor.pos.copy())
        polarization_times.append(sim_time)
        polarization_values.append(polarization(sensor_network))

        draw_frame(
            domain_ax,
            polar_ax,
            sensor_network,
            domain,
            histories,
            sim_time,
            polarization_times,
            polarization_values,
            show_fence=bool(args.show_fence),
            show_radius=bool(args.show_radius),
            interaction_radius=float(args.interaction_radius),
            noise_scale=float(args.noise_scale),
        )
        return ()

    draw_frame(
        domain_ax,
        polar_ax,
        sensor_network,
        domain,
        histories,
        sim_time,
        polarization_times,
        polarization_values,
        show_fence=bool(args.show_fence),
        show_radius=bool(args.show_radius),
        interaction_radius=float(args.interaction_radius),
        noise_scale=float(args.noise_scale),
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
