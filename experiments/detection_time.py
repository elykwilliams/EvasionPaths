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
from itertools import product
import pickle
import argparse


# seed = 10
# np.random.seed(seed)
# random.seed(seed)

# 69,42 are working

def simulate(sim) -> float:
    try:
        return sim.run()
    except Exception as e:
        print(f"Simulation failed due to {e}. Retrying...")
        return None


def sim_runs(radius, n_sens, num_runs):
    times = []
    while len(times) < num_runs:
        try:
            sim = sim_init(n_sensors=n_sens, radii=radius)
            result = simulate(sim)
            if result is not None:
                times.append(result)
        except Exception as e:
            print(f"An error occurred: {e}. Retrying initialization...")
    return times


def sim_init(n_sensors, radii):
    timestep_size: float = 0.05
    sensor_velocity = 1
    domain = UnitCube()
    motion_model = BilliardMotion()

    ####################### Fence stuff #####################3

    # fence = UnitCubeFence(sensing_radius)
    # fence_sensors = [Sensor(point, (0, 0, 0), sensing_radius, True) for point in fence]
    # fence = read_fence("../examples/Fence/fence.csv", sensing_radius)
    fence = get_unitcube_fence(spacing=radii)

    mobile_sensors = generate_mobile_sensors(domain, n_sensors, radii, sensor_velocity)
    sensor_network = SensorNetwork(
        mobile_sensors=mobile_sensors,
        motion_model=motion_model,
        fence=fence,
        sensing_radius=radii,
        domain=domain
    )
    sim = EvasionPathSimulation(sensor_network, timestep_size)

    return sim


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the atomic change detection simulation with specified parameters.")

    # Define arguments for the main parameters
    parser.add_argument("--min_sensors", type=int, default=15, help="Min number of mobile sensors")
    parser.add_argument("--max_sensors", type=int, default=25, help="Max number of mobile sensors")
    parser.add_argument("--lower_r", type=float, default=0.35, help="Lower bound of sensing radius")
    parser.add_argument("--upper_r", type=float, default=0.45, help="Upper bound of sensing radius")
    parser.add_argument("--subdivisions", type=int, default=11, help="Number of subdivisions for sensing radius range")
    parser.add_argument("--output_dir", type=str, default="./output/Tmax/", help="Directory to save output CSV files")
    parser.add_argument("--fn", type=str, default="tmax.pkl", help="filename for tmax")
    parser.add_argument("--n_runs", type=int, default=100, help="Number of simulations")

    args = parser.parse_args()

    min_num_sensors = args.min_sensors
    max_num_sensors = args.max_sensors
    min_r = args.lower_r
    max_r = args.upper_r
    subdivisions = args.subdivisions
    output_dir = args.output_dir
    fn = args.fn
    n_runs = args.n_runs

    sensing_radii = [round(min_r + i * (max_r - min_r) / (subdivisions - 1), 2) for i in
                     range(subdivisions)]
    sensing_radii = sensing_radii[::-1]

    num_sensors = [int(min_num_sensors + i * (max_num_sensors - min_num_sensors) / (subdivisions - 1)) for i in
                   range(subdivisions)]
    num_sensors = num_sensors[::-1]

    results_dict = {}

    print(sensing_radii, num_sensors)
    for r, n in product(sensing_radii, num_sensors):
        print("Current parameters: ", r, n)

        T_max_arr = sim_runs(r, n, num_runs=n_runs)
        key = f"{r},{n}"
        results_dict[key] = T_max_arr
        print(results_dict)

    with open(fn, 'wb') as f:
        pickle.dump(output_dir, f)

