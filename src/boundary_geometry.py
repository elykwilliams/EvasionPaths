# ************************************************************
# Copyright (c) 2021, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************
import random
from abc import ABC, abstractmethod
from itertools import product

import numpy as np
from dataclasses import dataclass
from numpy import arange, pi
from numpy.linalg import norm

from utilities import pol2cart


class BoundaryReflector(ABC):

    @abstractmethod
    def reflect_point(self, sensor):
        ...

    @abstractmethod
    def reflect_velocity(self, sensor):
        ...


@dataclass
class SquareReflector(BoundaryReflector):
    dim: int
    min: tuple
    max: tuple

    def reflect_point(self, sensor):
        for i in range(self.dim):
            if sensor.pos[i] <= self.min[i]:
                sensor.pos[i] = (self.min[i] - sensor.pos[i]) + self.min[i]
            elif sensor.pos[i] >= self.max[i]:
                sensor.pos[i] = 2 * self.max[i] - sensor.pos[i] + self.min[i]

    def reflect_velocity(self, sensor):
        for i in range(self.dim):
            if sensor.pos[i] <= self.min[i] or sensor.pos[i] >= self.max[i]:
                sensor.vel[i] = -sensor.vel[i]


@dataclass
class RadialReflector(BoundaryReflector):
    radius: float

    ## Compute trajectory intersection with boundary.
    # For internal use.
    def _get_intersection(self, old_pt, new_pt):
        d = new_pt - old_pt
        x0 = old_pt
        t_vals = np.roots([norm(d) ** 2, 2 * np.dot(d, x0), norm(x0) ** 2 - self.radius ** 2])
        t = t_vals[0] if 0 <= t_vals[0] <= 1 else t_vals[1]
        return (1 - t) * old_pt + t * new_pt

    ## reflect position if outside of domain.
    def reflect_point(self, sensor):
        boundary_pt = self._get_intersection(sensor.old_pos, sensor.pos)
        disp = sensor.pos - boundary_pt
        normal = boundary_pt / norm(boundary_pt)
        reflected_disp = disp - 2 * np.dot(disp, normal) * normal
        sensor.pos = boundary_pt + reflected_disp

    ## Reflect velocity angle to keep velocity consistent.
    def reflect_velocity(self, sensor):
        boundary_pt = self._get_intersection(sensor.old_pos, sensor.pos)
        disp = sensor.pos - boundary_pt
        normal = boundary_pt / norm(boundary_pt)
        reflected_disp = disp - 2 * np.dot(disp, normal) * normal
        unit_vec = reflected_disp / norm(reflected_disp)

        sensor.vel = norm(sensor.vel) * unit_vec


## Specify domain geometry.
# The domain class handles all issues relating to the
# domain geometry. It should be unaware of sensors and
# operate with strictly geometric objects. A user defined
# domain should
# - Initialize fence sensor positions
# - Determine if a point is in the domain
# - generate sensor locations inside the domain
# - provide boundary points for plotting
@dataclass
class Domain(ABC):
    spacing: float = 0

    @property
    @abstractmethod
    def reflector(self) -> BoundaryReflector:
        pass

    def reflect(self, sensor):
        self.reflector.reflect_velocity(sensor)  # Order important, uses unreflected positions
        self.reflector.reflect_point(sensor)

    @abstractmethod
    def __contains__(self, point: tuple) -> bool:
        ...

    @abstractmethod
    def point_generator(self, n_sensors: int) -> list:
        ...

    ## Return points along the boundary for plotting (not the fence).
    # Move to plotting tools
    @abstractmethod
    def domain_boundary_points(self):
        ...



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
        self.max = (x_max, y_max)
        self.min = (x_min, y_min)
        self.dim = 2
        self._reflector = SquareReflector(self.dim, self.min, self.max)
        super().__init__(spacing)

    ## Check if point is in domain.
    def __contains__(self, point: tuple) -> bool:
        return all(self.min[i] <= point[i] <= self.max[i] for i in range(self.dim))

    @property
    def reflector(self):
        return self._reflector

    ## Generate fence in counter-clockwise order.
    @property
    def fence(self) -> list:
        # Initialize fence position
        dx = self.spacing * np.sin(np.pi / 6)  # virtual boundary width
        vx_min, vx_max = self.min[0] - dx, self.max[0] + dx
        vy_min, vy_max = self.min[1] - dx, self.max[1] + dx

        points = []
        points.extend([(x, vy_min) for x in np.arange(vx_min, 0.999 * vx_max, self.spacing)])  # bottom
        points.extend([(vx_max, y) for y in np.arange(vy_min, 0.999 * vx_max, self.spacing)])  # right
        points.extend([(self.max[0] - x, vy_max)
                       for x in np.arange(vx_min, 0.999 * vx_max, self.spacing)])  # top
        points.extend([(vx_min, self.max[1] - y)
                       for y in np.arange(vy_min, 0.999 * vy_max, self.spacing)])  # left
        return points

    ## Generate points distributed (uniformly) randomly in the interior.
    def point_generator(self, n_sensors: int) -> list:
        rand_x = np.random.uniform(self.min[0], self.max[0], size=n_sensors)
        rand_y = np.random.uniform(self.min[1], self.max[1], size=n_sensors)
        return list(zip(rand_x, rand_y))

    ## Generate points to plot domain boundary.
    def domain_boundary_points(self):
        x_pts = [self.min[0], self.min[0], self.max[0], self.max[0], self.min[0]]
        y_pts = [self.min[1], self.max[1], self.max[1], self.min[1], self.min[1]]
        return x_pts, y_pts


class CircularDomain(Domain):
    def __init__(self, spacing, radius) -> None:
        self.radius = radius
        self._reflector = RadialReflector(self.radius)
        super().__init__(spacing)

    ## Check if point is in domain.
    def __contains__(self, point: tuple) -> bool:
        return norm(point) < self.radius

    @property
    def reflector(self):
        return self._reflector

    ## Generate points in counter-clockwise order.
    @property
    def fence(self) -> list:
        # Initialize fence
        dx = self.spacing
        v_rad = self.radius + dx
        return [pol2cart([v_rad, t]) for t in arange(0, 2 * pi, self.spacing)]

    ## Generate points distributed randomly (uniformly) in the interior.
    def point_generator(self, n_sensors):
        theta = np.random.uniform(0, 2 * pi, size=n_sensors)
        radius = np.random.uniform(0, self.radius, size=n_sensors)
        return [pol2cart(p) for p in zip(radius, theta)]

    ## Generate Points to plot domain boundary.
    def domain_boundary_points(self):
        x_pts = [self.radius * np.cos(t) for t in arange(0, 2 * pi, 0.01)]
        y_pts = [self.radius * np.sin(t) for t in arange(0, 2 * pi, 0.01)]
        return x_pts, y_pts


class UnitCube(Domain):

    def __init__(self) -> None:
        self.min = (0, 0, 0)
        self.max = (1, 1, 1)
        self.dim = 3
        self._reflector = SquareReflector(self.dim, self.min, self.max)

    ## Determine if given point it in domain or not.
    def __contains__(self, point) -> bool:
        return all(0 <= px <= 1 for px in point)

    @property
    def reflector(self):
        return self._reflector

    ## Generate n_int sensors randomly inside the domain.
    def point_generator(self, n_sensors: int):
        return np.random.rand(n_sensors, 3)

    def domain_boundary_points(self):
        return []


## Generate boundary points in counterclockwise order.
# WARNING: Points must be generated in counterclockwise order so that the
# alpha_cycle can be easily computed.
# Note: this function will not produce a fence suitable for computation and needs to be preprocessed, see warning
def UnitCubeFence(spacing):
    # TODO check that domain is contained in the fence covered region
    dx = np.sqrt(3) * spacing / 2

    points = np.arange(-dx, 1.001 + dx, spacing)
    grid = list(product(points, points))
    epsilon = pow(10, -5)
    x0_face = [(-dx + random.uniform(-epsilon, epsilon),
                y + random.uniform(-epsilon, epsilon),
                z + random.uniform(-epsilon, epsilon)) for y, z in grid]
    y0_face = [(x + random.uniform(-epsilon, epsilon),
                -dx + random.uniform(-epsilon, epsilon),
                z + random.uniform(-epsilon, epsilon)) for x, z in grid]
    z0_face = [(x + random.uniform(-epsilon, epsilon),
                y + random.uniform(-epsilon, epsilon),
                -dx + random.uniform(-epsilon, epsilon)) for x, y in grid]
    x1_face = [(1 + dx + random.uniform(-epsilon, epsilon),
                y + random.uniform(-epsilon, epsilon),
                z + random.uniform(-epsilon, epsilon)) for y, z in grid]
    y1_face = [(x + random.uniform(-epsilon, epsilon),
                1 + dx + random.uniform(-epsilon, epsilon),
                z + random.uniform(-epsilon, epsilon)) for x, z in grid]
    z1_face = [(x + random.uniform(-epsilon, epsilon),
                y + random.uniform(-epsilon, epsilon),
                1 + dx + random.uniform(-epsilon, epsilon)) for x, y in grid]
    fence_sensors = np.concatenate((x0_face, y0_face, z0_face, x1_face, y1_face, z1_face))
    return np.unique(fence_sensors, axis=0)
