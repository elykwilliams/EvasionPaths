# ******************************************************************************
#  Copyright (c) 2026, Marco Campos - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************
"""2D weighted-alpha animation sample with separate fence/mobile radii.

This mirrors examples/sample_animation.py, but the fence sensors and mobile
sensors may use different sensing radii. The weighted alpha complex is enabled
by passing per-sensor radii through SensorNetwork(use_weighted_alpha=True).
"""

from __future__ import annotations

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from boundary_geometry import RectangularDomain
from motion_model import BilliardMotion
from plotting_tools import show_domain_boundary, show_state
from sensor_network import SensorNetwork, generate_fence_sensors, generate_mobile_sensors
from time_stepping import EvasionPathSimulation


def build_simulation(
    *,
    num_sensors: int,
    mobile_radius: float,
    fence_radius: float,
    timestep_size: float,
    sensor_velocity: float,
    seed: int | None,
    end_time: float,
) -> EvasionPathSimulation:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    domain = RectangularDomain()
    motion_model = BilliardMotion()

    fence = generate_fence_sensors(domain, fence_radius)
    mobile_sensors = generate_mobile_sensors(domain, num_sensors, mobile_radius, sensor_velocity)

    sensor_network = SensorNetwork(
        mobile_sensors=mobile_sensors,
        motion_model=motion_model,
        fence=fence,
        sensing_radius=mobile_radius,
        domain=domain,
        use_weighted_alpha=True,
    )
    return EvasionPathSimulation(sensor_network, timestep_size, end_time=end_time)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animate a 2D weighted alpha-complex simulation with separate fence/mobile radii."
    )
    parser.add_argument("--num-sensors", type=int, default=20)
    parser.add_argument("--mobile-radius", type=float, default=0.2)
    parser.add_argument("--fence-radius", type=float, default=0.3)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--velocity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--n-steps", type=int, default=250)
    parser.add_argument("--end-time", type=float, default=0.0)
    parser.add_argument("--save-mp4", action="store_true")
    parser.add_argument("--filename-base", type=str, default="WeightedACAnimation")
    args = parser.parse_args()

    simulation = build_simulation(
        num_sensors=args.num_sensors,
        mobile_radius=args.mobile_radius,
        fence_radius=args.fence_radius,
        timestep_size=args.dt,
        sensor_velocity=args.velocity,
        seed=args.seed,
        end_time=args.end_time,
    )

    def draw_state() -> None:
        axis = plt.gca()
        axis.cla()
        axis.axis("off")
        axis.axis("equal")
        status = "active" if simulation.cycle_label.has_intruder() else "cleared"
        axis.set_title(
            (
                f"T = {simulation.time:5.2f} ({status})\n"
                f"weighted_ac=True, mobile_r={args.mobile_radius:.2f}, fence_r={args.fence_radius:.2f}"
            ),
            loc="left",
        )
        show_state(simulation, ax=axis)
        show_domain_boundary(
            simulation.sensor_network.domain,
            ax=axis,
            color="#111111",
            linewidth=1.5,
            linestyle="--",
            zorder=10,
        )

    def update(_):
        if simulation.cycle_label.has_intruder():
            simulation.do_timestep()
        draw_state()

    ms_per_frame = 5000 * args.dt
    fig = plt.figure(1)
    draw_state()
    print(
        "Initial weighted AC status:",
        {
            "has_intruder": bool(simulation.cycle_label.has_intruder()),
            "n_cycles": len(simulation.cycle_label.label),
            "mobile_radius": float(args.mobile_radius),
            "fence_radius": float(args.fence_radius),
        },
    )
    ani = FuncAnimation(fig, update, interval=ms_per_frame, frames=args.n_steps, repeat=False)

    plt.show()
    if args.save_mp4:
        ani.save(args.filename_base + ".mp4")


if __name__ == "__main__":
    main()
