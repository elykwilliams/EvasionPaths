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

    ## Move points back in domain.
    # If a point is moved outside of the domain, this function
    # provides a rule on how to move it back inside the domain.
    # The index is the position in the set of ALL points, and can
    # be useful in looking up sensor specific data. Reflect should
    # return position of point in the domain.
    @abstractmethod
    def reflect_point(self, pt: tuple, index: int) -> tuple:
        return pt

    @abstractmethod
    def reflect_velocity(self, pt, index):
        return

    ## Update all non-fence points.
    # If a point is not in the domain, reflect. It is sometimes
    # necessary to override this class method since this method is
    # called only once per time-step.
    def update_points(self, old_points: list) -> list:
        offset = len(self.boundary)
        interior_pts = old_points[offset:]
        for (n, pt) in enumerate(interior_pts):
            interior_pts[n] = self.update_point(pt, offset+n)
            if not self.boundary.in_domain(interior_pts[n]):
                self.reflect_velocity(pt, offset+n)
                interior_pts[n] = self.boundary.reflect_point(interior_pts[n], self.reflect_point(pt, offset + n))

        return self.boundary.points + interior_pts


## Provide random motion for rectangular domain.
# Will move a point randomly with an average step
# size of sigma*dt
class BrownianMotion(MotionModel):

    ## Initialize boundary with typical velocity.
    def __init__(self, dt: float, boundary: RectangularDomain, sigma: float) -> None:
        super().__init__(dt, boundary)
        self.sigma = sigma
        self.boundary = boundary

    ## Random function.
    def epsilon(self) -> float:
        return self.sigma*sqrt(self.dt)*random.normal(0, 1)

    ## Update each coordinate with brownian model.
    def update_point(self, pt: tuple, index=0) -> tuple:
        return pt[0] + self.epsilon(), pt[1] + self.epsilon()

    ## Move point inside domain.
    # If point moves outside vertical wall, fix y-coordinate, and update
    # x coordinate until in domain. Visa-versa for the horizontal walls.
    def reflect_point(self, pt: tuple, index) -> tuple:
        x, y = pt
        while not self.boundary.in_domain(pt):
            if x >= self.boundary.x_max:
                pt = (self.boundary.x_max - abs(self.epsilon()), y)
            elif x <= self.boundary.x_min:
                pt = (self.boundary.x_min + abs(self.epsilon()), y)
            if y >= self.boundary.y_max:
                pt = (x, self.boundary.y_max - abs(self.epsilon()))
            elif y <= self.boundary.y_min:
                pt = (x, self.boundary.y_min + abs(self.epsilon()))
        return pt

    def reflect_velocity(self, pt, index):
        return


## Implement Billiard Motion for Rectangular Domain.
# All sensors will have same velocity bit will have random angles.
# Points will move a distance of vel*dt each update.
class BilliardMotion(MotionModel):

    ## Initialize Boundary with additional velocity and number of sensors.
    # The number of sensors is required to know how to initialize the velocity
    # angles.
    def __init__(self, dt: float, boundary: RectangularDomain, vel: float, n_int_sensors: int) -> None:
        super().__init__(dt, boundary)
        self.vel = vel
        self.vel_angle = random.uniform(0, 2*pi, n_int_sensors+len(boundary))
        self.boundary = boundary  # not actually needed, just for type hinting.

    ## Update point using x = x + v*dt.
    def update_point(self, pt: tuple, index: int) -> tuple:
        theta = self.vel_angle[index]
        return pt[0] + self.dt*self.vel*cos(theta), pt[1] + self.dt*self.vel*sin(theta)


    def reflect_velocity(self, pt, index):
        if pt[0] <= self.boundary.x_min or pt[0] >= self.boundary.x_max:
            self.vel_angle[index] = pi - self.vel_angle[index]
        if pt[1] <= self.boundary.y_min or pt[1] >= self.boundary.y_max:
            self.vel_angle[index] = - self.vel_angle[index]
        self.vel_angle[index] %= 2 * pi

    ## Reflect using angle in = angle out.
    def reflect_point(self, pt: tuple, index: int) -> tuple:
        return self.update_point(pt, index)


## Implement randomized variant of Billiard motion.
# Each update, a sensor has a chance of randomly changing direction.
class RunAndTumble(BilliardMotion):

    ## Update angles before updating points.
    # Each update every point has a 1 : 5 chance of having its velocity
    # angle changed. Then update as normal.
    def update_points(self, old_points: list) -> list:

        for n in range(len(self.vel_angle)):
            if random.randint(0, 5) == 4:
                self.vel_angle[n] = random.uniform(0, 2 * pi)

        return super().update_points(old_points)
