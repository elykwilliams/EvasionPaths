from itertools import product

import numpy as np


class SensorNetwork:
    def __init__(self, points, velocities, sensing_radius):

        self.sensing_radius = sensing_radius

        self.mobile_sensors = points
        self.fence_sensors = self.get_fence_sensors()
        self.velocities = velocities

        self.old_pos = self.mobile_sensors
        self.old_vel = self.velocities

    def get_fence_sensors(self):
        dx = np.sqrt(3) * self.sensing_radius / 2

        spacing = np.arange(-dx, 1.001 + dx, self.sensing_radius)
        grid = list(product(spacing, spacing))
        x0_face = [(-dx, y, z) for y, z in grid]
        y0_face = [(x, -dx, z) for x, z in grid]
        z0_face = [(x, y, -dx) for x, y in grid]
        x1_face = [(1 + dx, y, z) for y, z in grid]
        y1_face = [(x, 1 + dx, z) for x, z in grid]
        z1_face = [(x, y, 1 + dx) for x, y in grid]

        fence_sensors = np.concatenate((x0_face, y0_face, z0_face, x1_face, y1_face, z1_face))
        fence_sensors = np.unique(fence_sensors, axis=0)
        return fence_sensors

    @property
    def points(self):
        return np.concatenate((self.fence_sensors, self.mobile_sensors))

    def move(self, dt):
        self.mobile_sensors = self.old_pos + self.velocities * dt
        self.check_reflections()

    def update(self):
        self.old_pos = self.mobile_sensors
        self.old_vel = self.velocities

    def check_reflections(self):
        # check if each point is in the interior
        for coor in range(3):
            for sensor, vel in zip(self.mobile_sensors, self.velocities):
                if sensor[coor] >= 1:
                    sensor[coor] = -sensor[coor] + 2
                    vel[coor] = -vel[coor]
                elif sensor[coor] <= 0:
                    sensor[coor] = -sensor[coor]
                    vel[coor] = -vel[coor]

    def __repr__(self):
        fence_pos_str = '\n'.join(str(p) for p in enumerate(self.fence_sensors))
        new_pos_str = '\n'.join(str(p) for p in enumerate(self.mobile_sensors, start=len(self.fence_sensors)))
        old_pos_str = '\n'.join(str(p) for p in enumerate(self.old_pos, start=len(self.fence_sensors)))
        return (f"Num mobile sensors: {len(self.mobile_sensors)}\n"
                f"Num boundary sensors: {len(self.fence_sensors)}\n"
                f"Fence Sensor Positions:\n"
                f"{fence_pos_str}\n"
                f"Mobile Sensor Positions:\n"
                f"{new_pos_str}\n"
                f"Old Mobile Sensor Positions:\n"
                f"{old_pos_str}\n"
                )
