# ************************************************************
# Copyright (c) 2021, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************
import os

from joblib import Parallel, delayed

from boundary_geometry import RectangularDomain
from motion_model import BilliardMotion
from sensor_network import generate_fence_sensors, generate_mobile_sensors, SensorNetwork
from time_stepping import EvasionPathSimulation

num_sensors: int = 20
sensing_radius: float = 0.2
timestep_size: float = 0.01
sensor_velocity = 1

output_dir: str = "./output"
filename_base: str = "data"

n_runs: int = 5


domain = RectangularDomain()
motion_model = BilliardMotion(domain)
fence = generate_fence_sensors(domain, sensing_radius)


def simulate() -> float:
    mobile_sensors = generate_mobile_sensors(domain, num_sensors, sensing_radius, sensor_velocity)
    sensor_network = SensorNetwork(mobile_sensors, motion_model, fence, sensing_radius)
    simulation = EvasionPathSimulation(sensor_network, timestep_size)

    return simulation.run()


def output_data(filename: str, data_points: list) -> None:
    with open(filename, 'a+') as file:
        file.writelines("%.2f\n" % d for d in data_points)


def run_experiment() -> None:
    times = Parallel(n_jobs=-1)(
        delayed(simulate)() for _ in range(n_runs)
    )
    output_data(output_dir + "/" + filename_base + ".txt", times)


if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    run_experiment()
