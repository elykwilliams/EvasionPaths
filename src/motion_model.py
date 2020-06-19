# ************************************************************
# Copyright (c) 2020, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************

from boundary_geometry import *
from numpy import sqrt, random, sin, cos, pi
from abc import ABC, abstractmethod


## This class provides the basic interface for a model of motion.
# it should do two things, update the point positions, and reflect
# points off of the boundary. Because the reflection depends on the
# boundary, motion models and boundaries must be compatible.
class MotionModel(ABC):

    ## Initialize motion model with boundary and time-scale
    def __init__(self, dt: float, boundary: Boundary) -> None:
        self.dt = dt
        self.boundary = boundary

    ## Update an individual point.
    # This function will be called on all points.
    # reflection should be separate. The index is the position
    # in the set of ALL points, and can be useful in looking up
    # sensor specific data. Will return new position or sensor.
    @abstractmethod
    def update_point(self, pt: tuple, index: int) -> tuple:
        return pt

    @abstractmethod
    def reflect(self, old_pt, new_pt, index):
        self.boundary.reflect_velocity(old_pt, new_pt)
        return self.boundary.reflect_point(old_pt, new_pt)

    ## Update all non-fence points.
    # If a point is not in the domain, reflect. It is sometimes
    # necessary to override this class method since this method is
    # called only once per time-step.
    def update_points(self, old_points: list, dt: float) -> list:
        self.dt = dt
        return self.boundary.points \
            + [self.update_point(pt, n) for n, pt in enumerate(old_points) if n >= len(self.boundary)]


## Provide random motion for rectangular domain.
# Will move a point randomly with an average step
# size of sigma*dt
class BrownianMotion(MotionModel):

    ## Initialize boundary with typical velocity.
    def __init__(self, dt: float, boundary: Boundary, sigma: float) -> None:
        super().__init__(dt, boundary)
        self.sigma = sigma

    ## Random function.
    def epsilon(self) -> float:
        return self.sigma * sqrt(self.dt) * random.normal(0, 1)

    ## Update each coordinate with brownian model.
    def update_point(self, old_pt: tuple, index) -> tuple:
        new_pt = old_pt[0] + self.epsilon(), old_pt[1] + self.epsilon()
        return new_pt if self.boundary.in_domain(new_pt) else self.reflect(old_pt, new_pt, index)

    def reflect(self, old_pt, new_pt, index):
        return self.boundary.reflect_point(old_pt, new_pt)


## Implement Billiard Motion for Rectangular Domain.
# All sensors will have same velocity bit will have random angles.
# Points will move a distance of vel*dt each update.
class BilliardMotion(MotionModel):

    ## Initialize Boundary with additional velocity and number of sensors.
    # The number of sensors is required to know how to initialize the velocity
    # angles.
    def __init__(self, dt: float, boundary: Boundary, vel: float, n_int_sensors: int) -> None:
        super().__init__(dt, boundary)
        self.vel = vel
        self.vel_angle = random.uniform(0, 2 * pi, n_int_sensors + len(boundary))

    ## Update point using x = x + v*dt.
    def update_point(self, pt: tuple, index: int) -> tuple:
        theta = self.vel_angle[index]
        new_pt = pt[0] + self.dt * self.vel * cos(theta), pt[1] + self.dt * self.vel * sin(theta)
        return new_pt if self.boundary.in_domain(new_pt) else self.reflect(pt, new_pt, index)

    def reflect(self, old_pt, new_pt, index):
        self.vel_angle[index] = self.boundary.reflect_velocity(old_pt, new_pt)
        return self.boundary.reflect_point(old_pt, new_pt)


## Implement randomized variant of Billiard motion.
# Each update, a sensor has a chance of randomly changing direction.
class RunAndTumble(BilliardMotion):

    ## Update angles before updating points.
    # Each update every point has a 1 : 5 chance of having its velocity
    # angle changed. Then update as normal.
    def update_point(self, pt: tuple, index: int) -> tuple:
        if random.randint(0, 5) == 4:
            self.vel_angle[index] = random.uniform(0, 2 * pi)
        return super().update_point(pt, index)
