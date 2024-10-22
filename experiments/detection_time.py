# ******************************************************************************
#  Copyright (c) 2023, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************
import os
import logging
from boundary_geometry import UnitCube, UnitCubeFence, get_unitcube_fence
from motion_model import BilliardMotion
from sensor_network import Sensor, generate_mobile_sensors, SensorNetwork, read_fence, generate_fence_sensors
from time_stepping import EvasionPathSimulation
import numpy as np
import random
from tqdm import tqdm

# seed = 10
# np.random.seed(seed)
# random.seed(seed)

# 69,42 are working

num_sensors: int = 10
sensing_radius: float = 0.2
timestep_size: float = 0.05
sensor_velocity = 1
n_runs: int = 10

log_name = f"{num_sensors}sensors_{sensing_radius}r_{n_runs}nr"
log_file = log_name + ".log"
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


domain = UnitCube()
motion_model = BilliardMotion()
"""
Testing fence stuff
"""

####################### Fence stuff #####################3

# fence = UnitCubeFence(sensing_radius)
# fence_sensors = [Sensor(point, (0, 0, 0), sensing_radius, True) for point in fence]
# fence = read_fence("../examples/Fence/fence.csv", sensing_radius)
fence = get_unitcube_fence(spacing=sensing_radius)

points = domain.point_generator(num_sensors)

output_dir: str = "./output"
r_string = str(sensing_radius).replace(".", "")

filename_base: str = f"{n_runs}_runs_{num_sensors}s_{r_string}r_10_22"

def simulate() -> float:
    logging.info("Starting a simulation.")
    try:

        mobile_sensors = generate_mobile_sensors(domain, num_sensors, sensing_radius, sensor_velocity)
        sensor_network = SensorNetwork(
            mobile_sensors=mobile_sensors,
            motion_model=motion_model,
            fence=fence,
            sensing_radius=sensing_radius,
            domain=domain
        )
        simulation = EvasionPathSimulation(sensor_network, timestep_size)
        logging.info("Simulation set up")
        print("Simulation set up")
        return simulation.run()
    except Exception as e:
        logging.error(f"Simulation failed due to {e}. Retrying...")
        return None


def output_data(filename: str, data_points: list) -> None:
    with open(filename, 'a+') as file:
        for d in data_points:
            if type(d) != str:
                file.writelines("%.2f\n" % d)
            else:
                file.writelines(str(d) + "\n")


def run_experiment() -> None:
    times = []
    while len(times) < n_runs:
        result = simulate()
        if result is not None:
            times.append(result)
        logging.info(f"Simulation completed successfully: {result}. Total completed: {len(times)}/{n_runs}")
    logging.info(f"All {n_runs} simulations completed.")
    filename = output_dir + "/" + filename_base + ".txt"
    output_data(filename, times)


if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    run_experiment()
