# ************************************************************
# Copyright (c) 2021, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************

from utilities import pol2cart
import numpy as np
from numpy import arange, pi, array, arctan2
from numpy.linalg import norm
from abc import ABC, abstractmethod


## Specify domain geometry.
# The domain class handles all issues relating to the
# domain geometry. It should be unaware of sensors and
# operate with strictly geometric objects. A user defined
# domain should
# - Initialize fence sensors
# - Determine if a point is in the domain
# - generate sensor location inside the domain
# - provide boundary points for plotting
# - reflect position and velocity off boundary in elastic collision.
class Domain(ABC):

    ## Initialize Domain.
    # stores number of fence sensors.
    def __init__(self, spacing: float) -> None:
        self.spacing = spacing

    ## Determine if given point it in domain or not.
    @abstractmethod
    def __contains__(self, item) -> bool:
        return True

    ## Generate boundary points in counterclockwise order.
    # WARNING: Points must be generated in counterclockwise order so that the
    # alpha_cycle can be easily computed.
    @abstractmethod
    def generate_fence(self):
        return []

    ## Generate n_int sensors randomly inside the domain.
    @abstractmethod
    def generate_interior_points(self, n_int_sensors: int) -> list:
        return []

    ## Return points along the physical domain.
    # to be used for displaying the domain boundary as opposed to only
    # the fence.
    @abstractmethod
    def domain_boundary_points(self):
        x_pts, y_pts = [], []
        return x_pts, y_pts

    ## Reflect a point off the boundary.
    # Should be elastic collision
    @abstractmethod
    def reflect_point(self, old_pt, new_pt):
        return new_pt

    ## Reflect a velocity off the boundary.
    # Should be elastic collision. Leave velocity magnitude as is,
    # and compute reflected angle.
    @abstractmethod
    def reflect_velocity(self, old_pt, new_pt):
        vel_angle = np.arctan2(new_pt[1] - old_pt[1], new_pt[0] - old_pt[0])
        return vel_angle


## A rectangular domain.
# This domain implements a physical boundary separate from the fence so that
# sensors don't get too close and mess up the associated boundary cycle.
# The input parameters specify the dimension of the desired virtual boundary
# and the physical locations of the sensors are places slightly outside of
# this boundary in a way that still allows interior sensors to form simplices
# with fence sensors.
class RectangularDomain(Domain):

    ## Initialize with dimension of desired boundary.
    # Sensor positions will be reflected so that interior sensors stay in the
    # specified domain. Default to the unit square with spacing of 0.2. Spacing
    # should be less that 2*sensing_radius. Defaults to unit square
    def __init__(self, spacing: float,
                 x_min: float = 0, x_max: float = 1,
                 y_min: float = 0, y_max: float = 1) -> None:
        self.x_max, self.y_max = x_max, y_max
        self.x_min, self.y_min = x_min, y_min

        # Initialize fence position
        self.dx = spacing * np.sin(np.pi / 6)  # virtual boundary width
        self.vx_min, self.vx_max = self.x_min - self.dx, self.x_max + self.dx
        self.vy_min, self.vy_max = self.y_min - self.dx, self.y_max + self.dx

        super().__init__(spacing)

    ## Check if point is in domain.
    def __contains__(self, point: tuple) -> bool:
        return self.x_min <= point[0] <= self.x_max \
               and self.y_min <= point[1] <= self.y_max

    ## Generate fence in counter-clockwise order.
    def generate_fence(self) -> list:
        points = []
        points.extend([(x, self.vy_min) for x in np.arange(self.vx_min, 0.999*self.vx_max, self.spacing)])  # bottom
        points.extend([(self.vx_max, y) for y in np.arange(self.vy_min, 0.999*self.vx_max, self.spacing)])  # right
        points.extend([(self.x_max - x, self.vy_max)
                       for x in np.arange(self.vx_min, 0.999*self.vx_max, self.spacing)])  # top
        points.extend([(self.vx_min, self.y_max - y)
                       for y in np.arange(self.vy_min, 0.999*self.vy_max, self.spacing)])  # left
        return points

    ## Generate points distributed (uniformly) randomly in the interior.
    def generate_interior_points(self, n_int_sensors: int) -> list:
        rand_x = np.random.uniform(self.x_min, self.x_max, size=n_int_sensors)
        rand_y = np.random.uniform(self.y_min, self.y_max, size=n_int_sensors)
        return list(zip(rand_x, rand_y))

    ## Generate points to plot domain boundary.
    def domain_boundary_points(self):
        x_pts = [self.x_min, self.x_min, self.x_max, self.x_max, self.x_min]
        y_pts = [self.y_min, self.y_max, self.y_max, self.y_min, self.y_min]
        return x_pts, y_pts

    ## Reflect position of point outside of domain.
    def reflect_point(self, old_pt, new_pt):
        pt = new_pt
        if new_pt[0] <= self.x_min:
            pt = (self.x_min + abs(self.x_min - new_pt[0]), new_pt[1])
        elif new_pt[0] >= self.x_max:
            pt = (self.x_max - abs(self.x_max - new_pt[0]), new_pt[1])

        new_pt = pt
        if new_pt[1] <= self.y_min:
            pt = (new_pt[0], self.y_min + abs(self.y_min - new_pt[1]))
        elif new_pt[1] >= self.y_max:
            pt = (new_pt[0], self.y_max - abs(self.y_max - new_pt[1]))

        return pt

    ## Reflect velocity angle to keep velocity consistent.
    def reflect_velocity(self, old_pt, new_pt):
        vel_angle = np.arctan2(new_pt[1] - old_pt[1], new_pt[0] - old_pt[0])
        if new_pt[0] <= self.x_min or new_pt[0] >= self.x_max:
            vel_angle = np.pi - vel_angle
        if new_pt[1] <= self.y_min or new_pt[1] >= self.y_max:
            vel_angle = - vel_angle
        return float(vel_angle) % (2 * np.pi)


## A circular domain.
# This domain implements a physical boundary separate from the fence so that
# sensors don't get too close and mess up the associated boundary cycle.
# The input parameters specify the dimension of the desired virtual boundary
# and the physical locations of the sensors are places slightly outside of
# this boundary in a way that still allows interior sensors to form simplices
# with fence sensors.
class CircularDomain(Domain):
    def __init__(self, spacing, radius) -> None:

        self.radius = radius

        # Initialize fence
        self.dx = spacing
        self.v_rad = self.radius + self.dx

        super().__init__(spacing)

    ## Check if point is in domain.
    def __contains__(self, point: tuple) -> bool:
        return norm(point) < self.radius

    ## Generate points in counter-clockwise order.
    def generate_fence(self) -> list:
        return [pol2cart([self.v_rad, t]) for t in arange(0, 2 * pi, self.spacing)]

    ## Generate points distributed randomly (uniformly) in the interior.
    def generate_interior_points(self, n_int_sensors):
        theta = np.random.uniform(0, 2 * pi, size=n_int_sensors)
        radius = np.random.uniform(0, self.radius, size=n_int_sensors)
        return [pol2cart(p) for p in zip(radius, theta)]

    ## Generate Points to plot domain boundary.
    def domain_boundary_points(self):
        x_pts = [self.radius*np.cos(t) for t in arange(0, 2*pi, 0.01)]
        y_pts = [self.radius*np.sin(t) for t in arange(0, 2*pi, 0.01)]
        return x_pts, y_pts

    ## Compute trajectory intersection with boundary.
    # For internal use.
    def _get_intersection(self, old_pt, new_pt):
        d = new_pt - old_pt
        x0 = old_pt
        t_vals = np.roots([norm(d)**2, 2*np.dot(d, x0), norm(x0)**2 - self.radius**2])
        t = t_vals[0] if 0 <= t_vals[0] <= 1 else t_vals[1]
        return (1-t)*old_pt + t*new_pt

    ## reflect position if outside of domain.
    def reflect_point(self, old_pt, new_pt):
        old_pt, new_pt = array(old_pt), array(new_pt)
        boundary_pt = self._get_intersection(old_pt, new_pt)
        disp = new_pt - boundary_pt
        normal = boundary_pt/norm(boundary_pt)
        reflected_disp = disp - 2*np.dot(disp, normal)*normal
        reflected_pt = boundary_pt + reflected_disp
        return reflected_pt[0], reflected_pt[1]

    ## Reflect velocity angle to keep velocity consistent.
    def reflect_velocity(self, old_pt, new_pt):
        old_pt, new_pt = array(old_pt), array(new_pt)
        boundary_pt = self._get_intersection(old_pt, new_pt)
        reflected_pt = self.reflect_point(old_pt, new_pt)
        disp_from_wall = array(reflected_pt) - boundary_pt
        return arctan2(disp_from_wall[1], disp_from_wall[0]) % (2 * pi)
