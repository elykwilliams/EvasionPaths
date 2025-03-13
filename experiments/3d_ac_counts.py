import itertools
import os
import csv
import logging
from boundary_geometry import UnitCube, UnitCubeFence, get_unitcube_fence
from motion_model import BilliardMotion
import numpy as np
from sensor_network import Sensor, generate_mobile_sensors, SensorNetwork, read_fence, generate_fence_sensors
from topology import generate_topology
from state_change import StateChange
from utilities import MaxRecursionDepthError
import argparse

import sys
from contextlib import contextmanager
import io

@contextmanager
def no_stdout():
    # Save original stdout object
    old_stdout = sys.stdout
    try:
        # Redirect stdout to an in-memory stream that does nothing
        sys.stdout = io.StringIO()
        yield
    finally:
        # Restore original stdout
        sys.stdout = old_stdout

# def generate_unit_cube_fence_with_faces_only(spacing=0.5, perturbation=0.01):
#     # Define the boundaries with a slight offset to cover the cube
#     margin = np.sqrt(3) / 2 * spacing
#     grid_min, grid_max = -margin, 1 + margin
#
#     # Create a regular grid that extends slightly beyond the unit cube boundaries
#     x = np.arange(grid_min, grid_max, spacing)
#     grid_points = np.array(list(itertools.product(x, x, x)))
#
#     # Apply a small random perturbation to each point for general position
#     perturbed_points = grid_points + np.random.uniform(-perturbation, perturbation, grid_points.shape)
#
#     # Keep only points that lie outside the unit cube to create the "fence"
#     fence_points = perturbed_points[
#         (perturbed_points[:, 0] < 0) | (perturbed_points[:, 0] > 1) |
#         (perturbed_points[:, 1] < 0) | (perturbed_points[:, 1] > 1) |
#         (perturbed_points[:, 2] < 0) | (perturbed_points[:, 2] > 1)
#     ]
#
#     return fence_points


def export_to_csv(counts, number, radius, f_path):
    # Create the filename with the timestamp, radius, and number
    filename = f"radius_{radius}_num_{number}.csv"
    file_path = os.path.join(f_path, filename)
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        writer.writerow(['Radius:', radius])
        writer.writerow(['Number:', number])
        writer.writerow(['Element', 'Count'])

        # Write each row of data
        for element, num in counts.items():
            if element != "((0, 0, 0, 0, 0, 0), (0, 0))":
                writer.writerow([element, num])


class AtomicChangeDetection:
    def __init__(self,
                 sensor_network: SensorNetwork,
                 dt: float,
                 end_time: int = 0,
                 f_path: str = ".",
                 max_ac: int = 0) -> None:
        self.dt = dt
        self.Tend = end_time
        self.time = 0
        self.f_path = f_path

        self.sensor_network = sensor_network
        self.topology = generate_topology(sensor_network.points, sensor_network.sensing_radius)
        self.topology_stack = []
        self.atomic_changes = {}
        self.max_ac = max_ac

    def run(self) -> float:
        print("max atomic changes", self.max_ac)
        while True:
            # Recalculate the total number of atomic changes after each timestep
            print(f"Total atomic changes so far: {self.total_atomic_changes()} / {self.max_ac}")

            # Check if we have reached the maximum number of atomic changes
            if self.total_atomic_changes() >= self.max_ac:
                print("Reached the maximum number of atomic changes, stopping simulation.")
                break

            try:
                self.do_timestep()
            except MaxRecursionDepthError:
                raise  # Handle recursion depth issues

            self.save_atomic_changes_csv()
            # Check if the time limit is reached (optional)
            if 0 < self.Tend < self.time:
                print("Out of time")
                break

        print("sim over")

        return self.time

    def do_timestep(self, level: int = 0) -> None:

        adaptive_dt = self.dt / (2 ** level)
        for loop_id in range(2):
            self.sensor_network.move(adaptive_dt)

            if loop_id == 0:
                new_topology = generate_topology(self.sensor_network.points, self.sensor_network.sensing_radius)
            else:
                # new_topology = self.topology_stack.pop()
                new_topology = generate_topology(self.sensor_network.points, self.sensor_network.sensing_radius)

            state_change = StateChange(new_topology, self.topology)
            case = str((state_change.alpha_complex_change(), state_change.boundary_cycle_change()))

            if case == "((0, 0, 0, 0, 0, 0), (0, 0))":
                self.update(state_change, adaptive_dt)
            else:
                if level == 25:
                    # raise MaxRecursionDepthError(state_change)
                    if case not in self.atomic_changes:
                        self.atomic_changes[case] = 1
                        print(f"Adding new case: {case}")
                    else:
                        self.atomic_changes[case] += 1

                    print("Unknown edge change occured. Might be due to disconnections.")
                    self.update(state_change, adaptive_dt)
                    if self.total_atomic_changes() >= self.max_ac:
                        self.save_atomic_changes_csv()
                        print("Reached max number of atomic changes")
                        return

                if not state_change.is_atomic_change():
                    # self.topology_stack.append(new_topology)
                    self.do_timestep(level + 1)
                else:
                    if case not in self.atomic_changes:
                        self.atomic_changes[case] = 1
                        print(f"Adding new case: {case}")
                    else:
                        self.atomic_changes[case] += 1

                    self.update(state_change, adaptive_dt)
                    if self.total_atomic_changes() >= self.max_ac:
                        self.save_atomic_changes_csv()
                        print("Reached max number of atomic changes")
                        return
                if level == 0:
                    break

    def total_atomic_changes(self):
        return sum(value for key, value in self.atomic_changes.items() if key != "((0, 0, 0, 0, 0, 0), (0, 0))")

    def save_atomic_changes_csv(self) -> None:
        export_to_csv(
            self.atomic_changes,
            len(self.sensor_network.mobile_sensors) + len(self.sensor_network.fence_sensors),
            self.sensor_network.sensing_radius,
            self.f_path)

    def update(self, state_change: StateChange, adaptive_dt: float) -> None:
        self.time += adaptive_dt
        self.topology = state_change.new_topology
        self.sensor_network.update()


def simulate(n_sensors, radii, velocities, dt, output_file, max_atomic_changes, debug) -> float:
    logging.info("Starting a simulation.")
    try:
        domain = UnitCube()
        motion_model = BilliardMotion()

        mobile_sensors = generate_mobile_sensors(domain, n_sensors, radii, velocities)

        sensor_network = SensorNetwork(
            mobile_sensors=mobile_sensors,
            motion_model=motion_model,
            fence=[],
            sensing_radius=radii,
            domain=domain
        )
        simulation = AtomicChangeDetection(
            sensor_network=sensor_network,
            dt=dt,
            f_path=output_file,
            max_ac=max_atomic_changes
        )

        logging.info("Simulation set up")
        print("Simulation set up")

        if not debug:
            with no_stdout():
                return simulation.run()
        else:
            return simulation.run()
    except Exception as e:
        logging.error(f"Simulation failed due to {e}. Retrying...")
        return -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the atomic change detection simulation with specified parameters.")

    # Define arguments for the main parameters
    parser.add_argument("--num_sensors", type=int, default=30, help="Number of mobile sensors")
    parser.add_argument("--lower_bound", type=float, default=0.05, help="Lower bound of sensing radius")
    parser.add_argument("--upper_bound", type=float, default=0.45, help="Upper bound of sensing radius")
    parser.add_argument("--delta", type=int, default=0.01, help="Subdivision increase for the sensing radius.")
    parser.add_argument("--timestep_size", type=float, default=0.05, help="Size of each timestep")
    parser.add_argument("--sensor_velocity", type=float, default=1, help="Velocity of the sensors")
    parser.add_argument("--max_changes", type=int, default=1000, help="Maximum number of atomic changes to look for")
    parser.add_argument("--output_dir", type=str, default="./output/atomic_change_counts/", help="Directory to save output CSV files")

    # Parse the arguments
    args = parser.parse_args()

    # Use parsed arguments
    num_sensors = args.num_sensors
    lower_bound = args.lower_bound
    upper_bound = args.upper_bound
    delta = args.delta
    timestep_size = args.timestep_size
    sensor_velocity = args.sensor_velocity
    max_changes = args.max_changes
    output_dir = args.output_dir

    print("Number of Mobile Sensors: ", num_sensors)

    sensing_radii = np.arange(lower_bound, upper_bound + delta, delta)
    sensing_radii = np.round(sensing_radii, 2)

    # For testing purposes
    # sensing_radii = [0.05]
    print(sensing_radii)

    log_name = f"AC_count_{num_sensors}"
    log_file = log_name + ".log"
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for sensing_radius in sensing_radii:
        print("Current Radius: ", sensing_radius)
        csv_fn = output_dir
        simulate(
            n_sensors=num_sensors,
            radii=sensing_radius,
            velocities=sensor_velocity,
            dt=timestep_size,
            output_file=csv_fn,
            max_atomic_changes=max_changes,
            debug=False)
