# ******************************************************************************
#  Copyright (c) 2023, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************
import os
from sensor_network import generate_fence_sensors, generate_mobile_sensors, SensorNetwork
from time_stepping import EvasionPathSimulation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from motion_model import BilliardMotion
from sensor_network import Sensor
from boundary_geometry import UnitCube, get_unitcube_fence
from topology import generate_topology
from alpha_complex import AlphaComplex

num_sensors: int = 20
sensing_radius: float = 0.3
timestep_size: float = 0.01
sensor_velocity = 1

domain = UnitCube()
motion_model = BilliardMotion()

fence = get_unitcube_fence(sensing_radius)
print(len(fence))
n_runs: int = 1

def simulate() -> float:
    mobile_sensors = generate_mobile_sensors(domain, num_sensors, sensing_radius, sensor_velocity)
    sensor_network = SensorNetwork(mobile_sensors, motion_model, fence, sensing_radius, domain)
    simulation = EvasionPathSimulation(sensor_network, timestep_size)

    return simulation.run()

def run_experiment() -> None:
    times = [simulate() for _ in range(n_runs)]
    print(times)

def confirm_fence(points):
    fence_coords = np.array([sensor.pos for sensor in points])
    ac = AlphaComplex(fence_coords, sensing_radius)
    print(ac)
    print(ac.simplex_tree)
    st = ac.simplex_tree
    st.compute_persistence()

    # Get Betti numbers (b0: components, b1: loops, b2: voids)
    betti_nums = st.betti_numbers()

    print(f"Betti numbers: β₀ = {betti_nums[0]}, β₁ = {betti_nums[1]}, β₂ = {betti_nums[2]}")
    if betti_nums[0] != 1 and betti_nums[2] != 1:
        print("Fence does not enclose the region.")
        return False
    else:
        return True

if __name__ == "__main__":
    # run_experiment()
    if confirm_fence(fence):
        times = [simulate() for _ in range(n_runs)]

    # x = fence_coords[:, 0]
    # y = fence_coords[:, 1]
    # z = fence_coords[:, 2]
    #
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.scatter(x, y, z, c='blue', marker='o', alpha=0.7)  # Scatter plot of points
    #
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('3D Point Cloud Visualization')
    #
    # plt.show()