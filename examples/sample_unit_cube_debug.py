# ******************************************************************************
#  Copyright (c) 2023, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************
"""Single-run 3D UnitCube simulation with atomic-change debug logging."""

from __future__ import annotations

import argparse
import random

import numpy as np

from boundary_geometry import UnitCube, get_unitcube_fence
from motion_model import BilliardMotion
from sensor_network import SensorNetwork, generate_mobile_sensors
from time_stepping import EvasionPathSimulation
from utilities import MaxRecursionDepthError


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


def _label_counts(labels: dict) -> tuple[int, int]:
    n_true = sum(1 for value in labels.values() if bool(value))
    n_false = len(labels) - n_true
    return n_true, n_false


def _is_trivial_change(alpha_change: tuple, boundary_change: tuple) -> bool:
    return (not any(alpha_change)) and tuple(boundary_change) == (0, 0)


def _printable_boundary_change(boundary_change: tuple) -> bool:
    return tuple(boundary_change) not in {(0, 0), (1, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single 3D UnitCube simulation with debug output.")
    parser.add_argument("--num-sensors", type=int, default=20)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--velocity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means unlimited outer timesteps.")
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

    history_cursor = len(simulation.cycle_label.history)
    event_count = 0
    skipped_trivial = 0
    skipped_boundary_filter = 0
    outer_step = 0

    init_labels = simulation.cycle_label.history[0][0]
    init_true, init_false = _label_counts(init_labels)
    print(
        "INIT"
        f" t=0.000000"
        f" boundary_cycles_present={len(init_labels)}"
        f" labels_true={init_true}"
        f" labels_false={init_false}"
    )

    while simulation.cycle_label.has_intruder():
        if args.max_steps > 0 and outer_step >= args.max_steps:
            print(f"Reached --max-steps={args.max_steps}; stopping early.")
            break
        if 0 < args.end_time < simulation.time:
            print(f"Reached --end-time={args.end_time}; stopping early.")
            break

        try:
            simulation.do_timestep()
        except MaxRecursionDepthError as exc:
            print(f"[ERROR] MaxRecursionDepthError at t={simulation.time:.6f}: {exc}")
            raise

        outer_step += 1
        new_entries = simulation.cycle_label.history[history_cursor:]
        history_cursor = len(simulation.cycle_label.history)

        for labels, alpha_change, boundary_change, event_time in new_entries:
            if _is_trivial_change(alpha_change, boundary_change):
                skipped_trivial += 1
                continue
            event_count += 1
            if not _printable_boundary_change(boundary_change):
                skipped_boundary_filter += 1
                continue
            n_true, n_false = _label_counts(labels)
            print(
                "ATOMIC"
                f" event={event_count}"
                f" outer_step={outer_step}"
                f" t={float(event_time):.6f}"
                f" alpha_change={alpha_change}"
                f" boundary_change={boundary_change}"
                f" boundary_cycles_present={len(labels)}"
                f" labels_true={n_true}"
                f" labels_false={n_false}"
            )

    print("----- Simulation Summary -----")
    print(
        f"3D UnitCube debug run complete: t={simulation.time:.4f}, "
        f"outer_steps={outer_step}, atomic_events={event_count}, "
        f"skipped_trivial={skipped_trivial}, skipped_boundary_filter={skipped_boundary_filter}"
    )


if __name__ == "__main__":
    main()
