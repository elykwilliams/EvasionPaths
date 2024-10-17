# ******************************************************************************
#  Copyright (c) 2023, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************
import os

from boundary_geometry import UnitCube, UnitCubeFence, get_unitcube_fence
from motion_model import BilliardMotion
from sensor_network import Sensor, generate_mobile_sensors, SensorNetwork, read_fence, generate_fence_sensors
from topology import generate_topology
from cycle_labelling import CycleLabelling
from time_stepping import EvasionPathSimulation
import numpy as np
import random
from tqdm import tqdm

# seed = 10
# np.random.seed(seed)
# random.seed(seed)

# 69,42 are working

num_sensors: int = 20
sensing_radius: float = 0.12
timestep_size: float = 0.05
sensor_velocity = 1
n_runs: int = 10


domain = UnitCube()
motion_model = BilliardMotion()

fence = get_unitcube_fence(spacing=sensing_radius)

points = domain.point_generator(num_sensors)
mobile_sensors = generate_mobile_sensors(domain, num_sensors, sensing_radius, sensor_velocity)
sensor_network = SensorNetwork(
    mobile_sensors=mobile_sensors,
    motion_model=motion_model,
    fence=fence,
    sensing_radius=sensing_radius,
    domain=domain
)
time = 0
topology = generate_topology(sensor_network.points, sensor_network.sensing_radius)
# if not self.topology.is_face_connected():
#     raise ValueError("The provided sensor network is not face connected")
cycle_label = CycleLabelling(topology)

network_coordinates = sensor_network.points
labels = cycle_label.label

print(network_coordinates)
print(len(network_coordinates))
print(labels)

