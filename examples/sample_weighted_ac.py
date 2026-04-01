# ******************************************************************************
#  Copyright (c) 2023, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************
"""Single-run UnitCube sample with optional weighted alpha complex.

When --weighted-ac is enabled:
- mobile sensors keep --radius
- fence sensors use a computed radius from --fence-subdivisions over [0, 1]
"""

from __future__ import annotations

import argparse
import random

import numpy as np

from boundary_geometry import UnitCube, get_unitcube_fence
from motion_model import BilliardMotion
from sensor_network import SensorNetwork, generate_mobile_sensors
from time_stepping import EvasionPathSimulation


def fence_radius_from_subdivisions(subdivisions: int) -> float:
    if subdivisions <= 0:
        raise ValueError("fence_subdivisions must be a positive integer.")
    # If we want N subdivisions on [0,1], the lattice spacing is 1/N.
    return 1.0 / float(subdivisions)


def build_simulation(
    *,
    num_sensors: int,
    mobile_radius: float,
    fence_radius: float,
    timestep_size: float,
    sensor_velocity: float,
    seed: int | None,
    weighted_ac: bool,
    end_time: float,
) -> EvasionPathSimulation:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    domain = UnitCube()
    motion_model = BilliardMotion()

    fence = get_unitcube_fence(spacing=fence_radius)
    mobile_sensors = generate_mobile_sensors(domain, num_sensors, mobile_radius, sensor_velocity)

    sensor_network = SensorNetwork(
        mobile_sensors=mobile_sensors,
        motion_model=motion_model,
        fence=fence,
        sensing_radius=mobile_radius,
        domain=domain,
        use_weighted_alpha=weighted_ac,
    )
    return EvasionPathSimulation(sensor_network, timestep_size, end_time=end_time)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UnitCube sample with optional weighted alpha complex.")
    parser.add_argument("--num-sensors", type=int, default=20)
    parser.add_argument("--radius", type=float, default=0.2, help="Mobile sensor radius.")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--velocity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--weighted-ac", action="store_true", default=False)
    parser.add_argument(
        "--fence-subdivisions",
        type=int,
        default=5,
        help="Number of subdivisions over [0,1] per wall direction (used only with --weighted-ac).",
    )
    parser.add_argument(
        "--fence-radius",
        type=float,
        default=0.2,
        help="Used only when --weighted-ac is off (uniform AC mode).",
    )
    parser.add_argument("--end-time", type=float, default=0.0)
    args = parser.parse_args()

    if args.weighted_ac:
        fence_radius = fence_radius_from_subdivisions(args.fence_subdivisions)
    else:
        fence_radius = args.fence_radius

    simulation = build_simulation(
        num_sensors=args.num_sensors,
        mobile_radius=args.radius,
        fence_radius=fence_radius,
        timestep_size=args.dt,
        sensor_velocity=args.velocity,
        seed=args.seed,
        weighted_ac=args.weighted_ac,
        end_time=args.end_time,
    )

    result_time = simulation.run()
    print(
        f"UnitCube run complete: weighted_ac={args.weighted_ac}, t={result_time:.4f}, "
        f"mobile_radius={args.radius:.4f}, fence_radius={fence_radius:.4f}"
    )


if __name__ == "__main__":
    main()
