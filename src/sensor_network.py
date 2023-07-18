# ************************************************************
# Copyright (c) 2021, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************

import numpy as np
import pandas as pd


class Sensor:
    """
    This class contains all information local to a sensor. The sensor is able to compute its movement given a model
    of motion. old_pos and new_pos should not updated unless update is called, in case the timestep is subdivided and
    positions need to be interpolated.
    """

    def __init__(self, position: np.array, velocity: np.array, sensing_radius: float, boundary_sensor: bool = False):
        """
        Initialize Sensor.

        :param position: Initial position as an np.array [x, y].
        :param velocity: Initial velocity as an np.array [vx, vy].
        :param sensing_radius: Sensing radius of the sensor.
        :param boundary_sensor: Indicator if the sensor is part of the boundary/fence.
        """
        self.pos = self.old_pos = position
        self.vel = self.old_vel = velocity
        self.radius = sensing_radius
        self.boundary_flag = boundary_sensor

    def update(self):
        """
        Update sensor current state. This function updates the values on which the new position are computed.
        It should not be called until ready to move to next timestep.
        """
        if self.boundary_flag:
            return
        self.old_pos = self.pos
        self.old_vel = self.vel

    def dist(self, other: 'Sensor') -> float:
        """
        Compute distance between this sensor and another sensor.

        :param other: The other sensor.
        :return: The Euclidean distance between the two sensors.
        """
        return np.linalg.norm(self.old_pos - other.old_pos)


## This class represents a collection of sensors.
# A sensor network is a collection of sensors + motion model for how those
# sensors move.
# This class allows users to iterate over each sensor so that each sensor
# can be polled for information.
# We can iterate over an entire sensor network, or we can iterate over the
# mobile or fence sensors specifically.
# move() will compute any nonlocal computation from the motion model, and then
# move each sensor.
# Update() will update each sensor.
class SensorNetwork:

    ## Initialize sensor network.
    # Users can provide the positions and velocities of each sensor manually.
    # they should both be a list of tuples using cartesian coordinates. If the velocities
    # are not specified, then they will be initialized by the motion model using the vel_mag
    # parameter. If positions are not specified, they will be generated by the motion models'
    # domain using the n_sensors parameter.
    def __init__(self, mobile_sensors, motion_model, fence, sensing_radius):

        self.motion_model = motion_model
        self.sensing_radius = sensing_radius

        self.mobile_sensors = mobile_sensors
        self.fence_sensors = fence

    def __iter__(self):
        # WARNING: fence sensors must come first in order to compute alpha-cycle.
        return iter(self.fence_sensors + self.mobile_sensors)

    def move(self, dt):
        self.motion_model.nonlocal_update(self, dt)

        for sensor in self.mobile_sensors:
            self.motion_model.local_update(sensor, dt)

    ## Update each sensor's position.
    # This should be called when moving to the next timestep.
    def update(self):
        for s in self.mobile_sensors:
            s.update()

    @property
    def points(self):
        return np.array([s.pos for s in self])


def initial_vel(domain, vel_magnitude):
    random_vector = 2*np.random.rand(domain.dim)-1
    norm_v = np.linalg.norm(random_vector)
    unit_vec = random_vector / norm_v

    return vel_magnitude * unit_vec


def generate_mobile_sensors(domain, n_sensors, sensing_radius, vel_mag):
    # Initialize sensor positions
    points = domain.point_generator(n_sensors)

    sensors = []
    for sensor in range(n_sensors):
        velocity = initial_vel(domain, vel_mag)
        s = Sensor(np.array(points[sensor]), np.array(velocity), sensing_radius)
        sensors.append(s)
    return sensors


def read_mobile_sensors(filename, sensing_radius):
    sensor_dataFrame = pd.read_csv(filename, index_col=0)

    sensors = []
    for sensor_id in sensor_dataFrame:
        s = Sensor(np.array(sensor_dataFrame[sensor_id][0:3]), np.array(sensor_dataFrame[sensor_id][3:]),
                   sensing_radius, False)
        sensors.append(s)
    return sensors


def read_fence(filename, radius):
    sensor_dataFrame = pd.read_csv(filename, index_col=0, skiprows=1)

    fence = []
    for sensor_id in sensor_dataFrame:
        s = Sensor(np.array(sensor_dataFrame[sensor_id]), (0, 0, 0), radius, True)
        fence.append(s)
    return fence


def generate_fence_sensors(domain, radius):
    return [Sensor(point, (0, 0, 0), radius, True) for point in domain.fence(radius)]