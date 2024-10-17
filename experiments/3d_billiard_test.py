import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from motion_model import BilliardMotion
from sensor_network import Sensor
from boundary_geometry import UnitCube

domain = UnitCube()
motion_model = BilliardMotion()

# Create a sensor with an initial position inside the cube and a velocity vector
initial_pos = np.array([0.3, 0.3, 0.3])
initial_vel = np.array([0.2, 0.35, 0.15])
sensor = Sensor(position=initial_pos, velocity=initial_vel, sensing_radius=0.1)

# Simulation settings
timestep_size = 0.05
n_steps = 1000

positions = [sensor.pos.copy()]

for step in range(n_steps):
    motion_model.local_update(sensor, timestep_size)

    if sensor.pos not in domain:
        motion_model.reflect(domain, sensor)

    positions.append(sensor.pos.copy())

    sensor.update()

positions = np.array(positions)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Billiard Motion of a Single Sensor in a Unit Cube')

sensor_point, = ax.plot([], [], [], 'bo', markersize=10)

def update(frame):
    sensor_point.set_data([positions[frame, 0]], [positions[frame, 1]])
    sensor_point.set_3d_properties([positions[frame, 2]])
    return sensor_point,

ani = FuncAnimation(fig, update, frames=n_steps, interval=50, blit=True)

plt.show()
