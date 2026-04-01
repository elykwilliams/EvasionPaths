# ******************************************************************************
#  Copyright (c) 2023, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************
"""Single-run 3D UnitCube simulation smoke test."""

from __future__ import annotations

import argparse
import random

import numpy as np

from boundary_geometry import UnitCube, get_unitcube_fence
from motion_model import BilliardMotion
from sensor_network import SensorNetwork, generate_mobile_sensors
from time_stepping import EvasionPathSimulation


def build_simulation(
    *,
    num_sensors: int,
    sensing_radius: float,
    timestep_size: float,
    sensor_velocity: float,
    seed: int | None,
    end_time: float,
) -> EvasionPathSimulation:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    domain = UnitCube()
    motion_model = BilliardMotion()
    fence = get_unitcube_fence(spacing=sensing_radius)
    mobile_sensors = generate_mobile_sensors(domain, num_sensors, sensing_radius, sensor_velocity)

    sensor_network = SensorNetwork(
        mobile_sensors=mobile_sensors,
        motion_model=motion_model,
        fence=fence,
        sensing_radius=sensing_radius,
        domain=domain,
    )
    return EvasionPathSimulation(sensor_network, timestep_size, end_time=end_time)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single 3D UnitCube simulation.")
    parser.add_argument("--num-sensors", type=int, default=20)
    parser.add_argument("--radius", type=float, default=0.3)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--velocity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--end-time",
        type=float,
        default=0.0,
        help="Set >0 to cap runtime. 0 means run-until-clear per EvasionPathSimulation.",
    )
    args = parser.parse_args()

    simulation = build_simulation(
        num_sensors=args.num_sensors,
        sensing_radius=args.radius,
        timestep_size=args.dt,
        sensor_velocity=args.velocity,
        seed=args.seed,
        end_time=args.end_time,
    )

    result_time = simulation.run()
    print(f"3D UnitCube simulation complete: t={result_time:.4f}")


if __name__ == "__main__":
    main()
