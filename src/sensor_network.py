# ************************************************************
# Copyright (c) 2021, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************

from utilities import *


## This class contains all information local to a sensor.
# The sensor is able to compute its movement given a model
# of motion. Positions/velocities are not updated unless
# update is called, in case the timestep is subdivided and
# positions need to be recomputed. Update should be called
# when moving to the next timestep.
# Note that velocity is stored in polar form (rho, theta).
class Sensor:
    ## Initialize Sensor.
    # Position as (x, y), velocity as (rho, theta)
    # boundary_sensor indicates that it is part of the fence.
    def __init__(self, position, polar_vel, sensing_radius, boundary_sensor=False):
        self.position = position
        self.old_pos = position
        self.pvel = polar_vel
        self.old_pvel = polar_vel
        self.radius = sensing_radius
        self.boundary_flag = boundary_sensor

    ## Get new position.
    # This will compute the position using a given motion model
    # motion model will use the old_pos and old_vel for computation.
    # old_pos and old_vel should not be updated incase they are needed
    # to recompute with smaller timestep.
    def move(self, motion_model, dt):
        assert not self.boundary_flag, "Boundary sensors cannot be updated"
        motion_model.update_position(self, dt)

    ## Update sensor current state.
    # This function updates the values on which the new position are computed,
    # It should not be called until ready to move to next timestep.
    def update(self):
        assert not self.boundary_flag, "Boundary sensors cannot be updated"
        self.old_pos = self.position
        self.old_pvel = self.pvel


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
    def __init__(self, motion_model, domain, sensing_radius, n_sensors=0, vel_mag=None, points=(), velocities=()):
        if velocities:
            assert len(points) == len(velocities), \
                "len(points) != len(velocities)"
        if n_sensors and points:
            assert len(points) == n_sensors, \
                "The number points specified is different than the number of points provided"

        self.motion_model = motion_model
        self.sensing_radius = sensing_radius

        # Initialize sensor positions
        if velocities:
            velocities = [cart2pol(v) for v in velocities]
        elif points:
            velocities = (self.motion_model.initial_pvel(vel_mag) for _ in points)
        else:
            points = motion_model.domain.generate_interior_points(n_sensors)
            velocities = (self.motion_model.initial_pvel(vel_mag) for _ in points)

        self.mobile_sensors = [Sensor(pt, v, sensing_radius) for pt, v in zip(points, velocities)]
        self.fence_sensors = [Sensor(pt, (0, 0), sensing_radius, True) for pt in motion_model.domain.generate_fence()]

    ## Iterate through all sensors.
    # WARNING: fence sensors must come first in order to compute
    # alpha-cycle.
    def __iter__(self):
        return iter(self.fence_sensors + self.mobile_sensors)

    ## Update any nonlocal computations, then move each sensor.
    # Ideally each sensor can be given a model of motion and it will
    # compute how to move itself for a purely local computation. However,
    # often times a nonlocal computation needs to be done, or the values
    # on which the local computation is done need to be modified. This is
    # all done in compute_update(). After this, each sensor move() can be
    # called for and local computations.
    def move(self, dt):
        self.motion_model.compute_update(self, dt)
        for sensor in self.mobile_sensors:
            sensor.move(self.motion_model, dt)

    ## Update each sensor's position.
    # This should be called before moving to the next timestep.
    def update(self):
        for s in self.mobile_sensors:
            s.update()
