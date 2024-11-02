import os
import csv
import sys
import pickle
import logging
from boundary_geometry import UnitCube, UnitCubeFence, get_unitcube_fence
from motion_model import BilliardMotion
from sensor_network import Sensor, generate_mobile_sensors, SensorNetwork, read_fence, generate_fence_sensors
from topology import generate_topology
from time_stepping import EvasionPathSimulation
import numpy as np
import random
from tqdm import tqdm
from state_change import StateChange
from utilities import MaxRecursionDepthError
import argparse



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
        with tqdm(total=float('inf'), desc="Atomic Changes", unit=" ac") as pbar:
            while True:
                # Recalculate the total number of atomic changes after each timestep
                print(f"Total atomic changes so far: {self.total_atomic_changes()} / {self.max_ac}")

                # Check if we have reached the maximum number of atomic changes
                if self.total_atomic_changes() >= self.max_ac:
                    print("Reached the maximum number of atomic changes, stopping simulation.")
                    break
                # elif sum(self.atomic_changes.values()) > 100 * self.max_ac:
                #     print("Too many trivial changes, adjust the radius.")
                #     break

                try:
                    self.do_timestep()
                except MaxRecursionDepthError:
                    raise  # Handle recursion depth issues

                # Recalculate total_atomic_changes after a change is detected
                pbar.set_description(f"Num AC's: {self.total_atomic_changes()}")
                pbar.update(1)  # Increment progress

                self.save_atomic_changes_csv()

                if self.total_atomic_changes() >= self.max_ac:
                    print("breaking")
                    break

                # Check if the time limit is reached (optional)
                if 0 < self.Tend < self.time:
                    print("Out of time")
                    break

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

            if level == 25:
                raise MaxRecursionDepthError(state_change)

            if not state_change.is_atomic_change():
                # self.topology_stack.append(new_topology)
                self.do_timestep(level + 1)
            else:
                case = str((state_change.alpha_complex_change(),state_change.boundary_cycle_change()))
                if case not in self.atomic_changes:
                    self.atomic_changes[case] = 1
                    print(f"Adding new case: {case}")  # Debugging new case
                else:
                    self.atomic_changes[case] += 1
                sys.stdout.write(
                    f"\rTotal Atomic Changes: {self.total_atomic_changes()}")  # Use carriage return to overwrite
                sys.stdout.flush()  # Flush to ensure the output is updated immediately

                self.update(state_change, adaptive_dt)

            if level == 0:
                break

            if self.total_atomic_changes() >= self.max_ac:
                break

    def total_atomic_changes(self):
        return sum(value for key, value in self.atomic_changes.items() if key != "((0, 0, 0, 0, 0, 0), (0, 0))")

    def save_atomic_changes_csv(self, f_path=".") -> None:
        export_to_csv(
            self.atomic_changes,
            len(self.sensor_network.mobile_sensors) + len(self.sensor_network.fence_sensors),
            self.sensor_network.sensing_radius,
            self.f_path)

    def update(self, state_change: StateChange, adaptive_dt: float) -> None:
        self.time += adaptive_dt
        self.topology = state_change.new_topology
        self.sensor_network.update()


def simulate(n_sensors, radii, velocities, dt, output_file, max_atomic_changes) -> float:
    logging.info("Starting a simulation.")
    try:
        domain = UnitCube()
        motion_model = BilliardMotion()

        fence = get_unitcube_fence(spacing=radii)
        mobile_sensors = generate_mobile_sensors(domain, n_sensors, radii, velocities)

        sensor_network = SensorNetwork(
            mobile_sensors=mobile_sensors,
            motion_model=motion_model,
            fence=fence,
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

        return simulation.run()
    except Exception as e:
        logging.error(f"Simulation failed due to {e}. Retrying...")
        return -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the atomic change detection simulation with specified parameters.")

    # Define arguments for the main parameters
    parser.add_argument("--num_sensors", type=int, default=10, help="Number of mobile sensors")
    parser.add_argument("--lower_bound", type=float, default=0.02, help="Lower bound of sensing radius")
    parser.add_argument("--upper_bound", type=float, default=0.24, help="Upper bound of sensing radius")
    parser.add_argument("--subdivisions", type=int, default=20, help="Number of subdivisions for sensing radius range")
    parser.add_argument("--timestep_size", type=float, default=0.05, help="Size of each timestep")
    parser.add_argument("--sensor_velocity", type=float, default=1, help="Velocity of the sensors")
    parser.add_argument("--max_changes", type=int, default=5, help="Maximum number of atomic changes to look for")
    parser.add_argument("--output_dir", type=str, default="./output/atomic_change_counts/", help="Directory to save output CSV files")

    # Parse the arguments
    args = parser.parse_args()

    # Use parsed arguments
    num_sensors = args.num_sensors
    lower_bound = args.lower_bound
    upper_bound = args.upper_bound
    subdivisions = args.subdivisions
    timestep_size = args.timestep_size
    sensor_velocity = args.sensor_velocity
    max_changes = args.max_changes
    output_dir = args.output_dir

    print("Number of Mobile Sensors: ", num_sensors)

    sensing_radii = [round(lower_bound + i * (upper_bound - lower_bound) / (subdivisions - 1), 2) for i in
                     range(subdivisions)]
    sensing_radii = sensing_radii[::-1]

    # For testing purposes
    # sensing_radii = [0.24]
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
            max_atomic_changes=max_changes)
