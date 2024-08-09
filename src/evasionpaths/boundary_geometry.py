# ************************************************************
# Copyright (c) 2020, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************
import math

import numpy as np
from numpy import sin, cos, arange, pi, array, arctan2
from numpy.linalg import norm
from abc import ABC, abstractmethod


## Provide abstract base of features that a boundary must satisfy.
# The boundary class generates the positions of the boundary sensors,
# a way of specifying if a given point is in or out of the boundary.
# It will also know the boundary cycle associated with the outside of
# the fence (the "alpha_cycle". And finally, it will be the class to
# generate initial data since it can determine how best go generate
# random points inside the domain.
class Boundary(ABC):

    ## Must initialize the boundary points and "alpha_cycle".
    # The points stored are the points on the boundary only.
    def __init__(self) -> None:
        self.points = self.generate_boundary_points()
        self.alpha_cycle = self.get_alpha_cycle()

    ## length of boundary is number of boundary sensors.
    def __len__(self) -> int:
        return len(self.points)

    ## Determine if given point it in domain or not.
    @abstractmethod
    def in_domain(self, point: tuple) -> bool:
        return True

    ## Generate boundary points in counterclockwise order.
    # Points must be generated in counterclockwise order so that the
    # alpha_cycle can be easily computed.
    @abstractmethod
    def generate_boundary_points(self):
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

    ## Return list of all points.
    # Boundary points must come first and be non-empty.
    def generate_points(self, n_int_sensors: int) -> list:
        return self.points + self.generate_interior_points(n_int_sensors)

    ## construct boundary cycle.
    # the alpha_cycle is the boundary cycle going counter-closckwise around the outside
    # of the domain.
    def get_alpha_cycle(self) -> tuple:
        a = [str(n + 1) + "," + str(n) for n in range(len(self.points) - 1)] + ["0," + str(len(self.points) - 1)]
        return tuple(sorted(a))

    ## Reflect a point off the boundary.
    # If a sensor leaves the domain, we need to move the sensors back in
    @abstractmethod
    def reflect_point(self, old_pt, new_pt):
        return new_pt

    @abstractmethod
    def reflect_velocity(self, old_pt, new_pt):
        vel_angle = np.arctan2(new_pt[1] - old_pt[1], new_pt[0] - old_pt[0])
        return float(vel_angle)


## a rectangular domain using virtual boundary.
# This domain implements a physical boundary separate from the fence so that
# sensors don't get too close and mess up the associated boundary cycle.
# The input parameters specify the dimension of the desired virtual boundary
# and the physical locations of the sensors are places slightly outside of
# this boundary in a way that still allows interior sensors to form simplices
# with fence sensors.
class RectangularDomain(Boundary):

    ## Initialize with dimension of desired boundary.
    # Sensor positions will be reflected so that interior sensors stay in the
    # specified domain. Default to the unit square with spacing of 0.2. Spacing
    # should be less that 2*sensing_radius.
    def __init__(self, spacing: float = 0.2,  # default of 0.2, with unit square
                 x_min: float = 0, x_max: float = 1,
                 y_min: float = 0, y_max: float = 1) -> None:
        self.x_max, self.y_max = x_max, y_max
        self.x_min, self.y_min = x_min, y_min
        self.spacing = spacing

        # Initialize fence boundary
        self.dx = self.spacing * math.sin(np.pi / 6)  # virtual boundary width
        self.vx_min, self.vx_max = self.x_min - self.dx, self.x_max + self.dx
        self.vy_min, self.vy_max = self.y_min - self.dx, self.y_max + self.dx

        super().__init__()

    ## Check if point is in domain.
    def in_domain(self, point: tuple) -> bool:
        return self.x_min <= point[0] <= self.x_max \
               and self.y_min <= point[1] <= self.y_max

    ## Generate points in counter-clockwise order.
    def generate_boundary_points(self) -> list:
        points = []
        points.extend([(float(x), self.vy_min) for x in np.arange(self.vx_min, 0.999*self.vx_max, self.spacing)])  # bottom
        points.extend([(self.vx_max, float(y)) for y in np.arange(self.vy_min, 0.999*self.vx_max, self.spacing)])  # right
        points.extend([(self.x_max - float(x), self.vy_max)
                       for x in np.arange(self.vx_min, 0.999*self.vx_max, self.spacing)])  # top
        points.extend([(self.vx_min, self.y_max - float(y))
                       for y in np.arange(self.vy_min, 0.999*self.vy_max, self.spacing)])  # left
        return points

    ## Generate points distributed randomly (uniformly) in the interior.
    def generate_interior_points(self, n_int_sensors: int) -> list:
        rand_x = np.random.uniform(self.x_min, self.x_max, size=n_int_sensors)
        rand_y = np.random.uniform(self.y_min, self.y_max, size=n_int_sensors)
        return list(zip(rand_x, rand_y))

    ## Generate Points to plot domain boundary.
    def domain_boundary_points(self):
        x_pts = [self.x_min, self.x_min, self.x_max, self.x_max, self.x_min]
        y_pts = [self.y_min, self.y_max, self.y_max, self.y_min, self.y_min]
        return x_pts, y_pts

    ## reflect position if outside of domain.
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
        return float(vel_angle) % (2 * math.pi)


## a circular domain using virtual boundary.
# This domain implements a physical boundary separate from the fence so that
# sensors don't get too close and mess up the associated boundary cycle.
# The input parameters specify the dimension of the desired virtual boundary
# and the physical locations of the sensors are places slightly outside of
# this boundary in a way that still allows interior sensors to form simplices
# with fence sensors.
class CircularDomain(Boundary):
    def __init__(self, spacing, radius) -> None:

        self.spacing = spacing
        self.radius = radius

        # Initialize fence boundary
        self.dx = self.spacing
        self.v_rad = self.radius + self.dx

        super().__init__()

    ## Check if point is in domain.
    def in_domain(self, point: tuple) -> bool:
        return norm(point) < self.radius

    ## Generate points in counter-clockwise order.
    def generate_boundary_points(self) -> list:
        return [(self.v_rad*math.cos(t), self.v_rad*math.sin(t)) for t in arange(0, 2 * pi, self.spacing)]

    ## Generate points distributed randomly (uniformly) in the interior.
    def generate_interior_points(self, n_int_sensors):
        theta = np.random.uniform(0, 2 * pi, size=n_int_sensors)
        radius = np.random.uniform(0, self.radius, size=n_int_sensors)
        return [(r*cos(t), r*sin(t)) for r, t in zip(radius, theta)]

    ## Generate Points to plot domain boundary.
    def domain_boundary_points(self):
        x_pts = [self.radius*cos(t) for t in arange(0, 2*pi, 0.01)]
        y_pts = [self.radius*sin(t) for t in arange(0, 2*pi, 0.01)]
        return x_pts, y_pts

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
