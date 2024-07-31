# ******************************************************************************
#  Copyright (c) 2024, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy import arange, pi
from numpy.linalg import norm

from utilities import pol2cart


## Specify domain geometry.
# The domain class handles all issues relating to the
# domain geometry. It should be unaware of sensors and
# operate with strictly geometric objects. A user defined
# domain should
# - Determine if a point is in the domain
# - provide the outward UNIT normal for any point on the boundary
# - given a point inside and a point outside, determine the intersection
#   point of the segment connecting these two points with the domain.
# - generate random points inside the domain
# - optionally provide fence sensor positions
# - optionally provide boundary points for plotting
@dataclass
class Domain(ABC):

    @abstractmethod
    def __contains__(self, point: tuple) -> bool:
        ...

    @abstractmethod
    def intersects_boundary(self, old_pos: tuple, new_pos: tuple):
        ...

    @abstractmethod
    def point_generator(self, n_sensors: int) -> list:
        ...

    ## Return points along the boundary for plotting (not the fence).
    # Move to plotting tools
    @abstractmethod
    def domain_boundary_points(self):
        ...

    @abstractmethod
    def normal(self, boundary_point):
        pass

    @abstractmethod
    def get_intersection_point(self, old, new):
        pass


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
    # specified domain. Defaults to unit square
    def __init__(self,
                 x_min: float = 0, x_max: float = 1,
                 y_min: float = 0, y_max: float = 1) -> None:
        self.max = (x_max, y_max)
        self.min = (x_min, y_min)
        self.dim = 2

    ## Check if point is in domain.
    def __contains__(self, point: tuple) -> bool:
        return all(self.min[i] <= point[i] <= self.max[i] for i in range(self.dim))

    def intersects_boundary(self, old_pos: tuple, new_pos: tuple):
        return new_pos not in self

    @staticmethod
    def eps():
        epsilon = pow(10, -4)
        return random.uniform(-epsilon, epsilon)*0

    ## Generate fence in counter-clockwise order.
    # spacing should be less than 2*sensing_radius.
    def fence(self, spacing) -> list:
        # Initialize fence position
        dx = spacing * np.sin(np.pi / 3)  # virtual boundary width
        vx_min, vx_max = self.min[0] - dx, self.max[0] + dx
        vy_min, vy_max = self.min[1] - dx, self.max[1] + dx

        points = []
        points.extend([(x+self.eps(), vy_min+self.eps()) for x in np.arange(vx_min, 0.999 * vx_max, spacing)])  # bottom
        points.extend([(vx_max+self.eps(), y+self.eps()) for y in np.arange(vy_min, 0.999 * vx_max, spacing)])  # right
        points.extend([(self.max[0] - x+self.eps(), vy_max+self.eps())
                       for x in np.arange(vx_min, 0.999 * vx_max, spacing)])  # top
        points.extend([(vx_min+self.eps(), self.max[1] - y+self.eps())
                       for y in np.arange(vy_min, 0.999 * vy_max, spacing)])  # left
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

    def normal(self, boundary_point):
        center = 0.5*np.array([self.max[0] + self.min[0], self.max[1] + self.min[1]])
        offset_point = boundary_point - center      # center domain at (0, 0)
        if self.min[0] - center[0] < offset_point[0] < self.max[0] - center[0]:
            normal = np.array([0, offset_point[1]])
        else:
            normal = np.array([offset_point[0], 0])
        return normal / np.linalg.norm(normal)

    def get_intersection_point(self, old, new):
        assert (old in self and new not in self)
        tvals = []
        for dim in range(self.dim):
            tvals.append((self.max[dim] - old[dim]) / (new[dim] - old[dim]))
            tvals.append((self.min[dim] - old[dim]) / (new[dim] - old[dim]))
        t = min(filter(lambda x: 0 < x < 1, tvals))
        return old + t*(new - old)


class CircularDomain(Domain):
    def __init__(self, radius) -> None:
        self.radius = radius
        self.dim = 2

    ## Check if point is in domain.
    def __contains__(self, point: tuple) -> bool:
        return norm(point) < self.radius

    def intersects_boundary(self, old_pos: tuple, new_pos: tuple):
        return new_pos not in self

    ## Generate points in counter-clockwise order.
    def fence(self, spacing) -> list:
        # Initialize fence
        dx = spacing
        v_rad = self.radius + dx
        return [pol2cart((v_rad, t)) for t in arange(0, 2 * pi, spacing)]

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

    def normal(self, boundary_point):
        return boundary_point / np.linalg.norm(boundary_point)

    def get_intersection_point(self, old, new):
        disp = new - old
        tvals = np.roots([norm(disp) ** 2, 2 * np.dot(disp, old), norm(old) ** 2 - self.radius ** 2])
        t = min(filter(lambda x: 0 < x < 1, tvals))
        return old + t * disp


class UnitCube(Domain):

    def __init__(self) -> None:
        self.min = (0, 0, 0)
        self.max = (1, 1, 1)
        self.dim = 3

    ## Determine if given point it in domain or not.
    def __contains__(self, point) -> bool:
        return all(0 <= px <= 1 for px in point)

    def intersects_boundary(self, old_pos: tuple, new_pos: tuple):
        return new_pos not in self

    ## Generate n_int sensors randomly inside the domain.
    def point_generator(self, n_sensors: int):
        return np.random.rand(n_sensors, 3)

    def domain_boundary_points(self):
        return []

    def normal(self, boundary_point):
        center = 0.5*np.array([self.max[d] + self.min[d] for d in range(self.dim)])
        offset_point = boundary_point - center      # center domain at (0, 0)
        if self.min[0] < boundary_point[0] < self.max[0]:
            normal = np.array([0, offset_point[1]])
        else:
            normal = np.array([offset_point[0], 0])
        return normal / np.linalg.norm(normal)

    def get_intersection_point(self, old, new):
        assert (old in self and new not in self)
        tvals = []
        for dim in range(len(old)):
            tvals.append((self.max[dim] - old[dim]) / (new[dim] - old[dim]))
            tvals.append((self.min[dim] - old[dim]) / (new[dim] - old[dim]))
        t = min(filter(lambda x: 0 < x < 1, tvals))
        return old + t*(new - old)


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


class BunimovichStadium(Domain):

    def __init__(self, w, r, L: float):
        self.w = w
        self.r = r
        self.length = L
        self.dim = 2

    def __contains__(self, point):
        x, y = point
        if -self.w <= x <= self.w and -self.r <= y <= self.r:
            return True
        if (x + self.w) ** 2 + y ** 2 <= self.r ** 2:
            return True
        if (x - self.w) ** 2 + y ** 2 <= self.r ** 2:
            return True
        return False

    def intersects_boundary(self, old_pos: tuple, new_pos: tuple):
        return new_pos not in self

    def normal(self, pt):
        if pt[0] < -self.w:
            center_cir = np.array([-self.w, 0])
            normal = pt - center_cir
        elif pt[0] > self.w:
            center_cir = np.array([self.w, 0])
            normal = pt - center_cir
        else:
            normal = np.array([0, pt[1]])
        return normal / np.linalg.norm(normal)

    def get_circle_center(self, old_pos, new_pos):
        intersect = self.intersect_flat(old_pos, new_pos)
        w = self.w if intersect[0] > self.w else -self.w
        return np.array([w, 0])

    def intersect_circ(self, old_pos, new_pos):
        center = self.get_circle_center(old_pos, new_pos)
        a = np.linalg.norm(new_pos - old_pos)**2
        b = 2 * np.dot(old_pos - center, new_pos - old_pos)
        c = np.linalg.norm(old_pos - center) ** 2 - self.r ** 2
        t = min(filter(lambda x: 0 < x < 1, np.roots([a, b, c])))
        return old_pos + t * (new_pos - old_pos)

    def intersect_flat(self, old_pos, new_pos):
        r = self.r if new_pos[1] > old_pos[1] else -self.r
        t = (r - old_pos[1]) / (new_pos[1] - old_pos[1])
        return old_pos + t * (new_pos - old_pos)

    def get_intersection_point(self, old_pos, new_pos):
        inter_flat = self.intersect_flat(old_pos, new_pos)
        if -self.w <= inter_flat[0] <= self.w:
            return inter_flat
        return self.intersect_circ(old_pos, new_pos)

    def point_generator(self, N):
        if self.length > 2 * self.w or self.length > 2 * self.r:
            raise ValueError("Box doesn't fit in the stadium.")

        points = []
        while len(points) < N:
            p = np.random.uniform(-self.length / 2, self.length / 2, 2)
            if p in self:
                points.append(p)

        return points

    def domain_boundary_points(self):
        theta1 = np.linspace(np.pi / 2, 3 * np.pi / 2, 1000)
        x1 = -self.w + self.r * np.cos(theta1)
        y1 = self.r * np.sin(theta1)

        x3 = np.array([-self.w, self.w])  # bottom length
        y3 = np.array([-self.r, -self.r])

        theta2 = np.linspace(-1 * np.pi / 2, np.pi / 2, 1000)
        x2 = self.w + self.r * np.cos(theta2)
        y2 = self.r * np.sin(theta2)

        x4 = np.array([-self.w, self.w])  # top length
        y4 = np.array([self.r, self.r])

        x = np.concatenate((x1, x3, x2,  x4))
        y = np.concatenate((y1, y3, y2, y4))
        return x, y

    def fence(self, spacing):

        fence = []  # List to store the fence points

        theta = np.linspace(np.pi / 2, 0.999*3 * np.pi / 2,
                            int(np.ceil(2 * np.pi * self.r / spacing)))  # Angles spaced apart by 'spacing'
        x_pts = (self.r + (np.sqrt(3) / 2) * spacing) * np.cos(theta) + -self.w  # x-coordinates of the fence points
        y_pts = (self.r + (np.sqrt(3) / 2) * spacing) * np.sin(theta)  # y-coordinates of the fence points

        for x, y in zip(x_pts, y_pts):
            fence.append((x, y))

        for i in range(int(2*self.w // spacing)+1):
            x1 = -self.w + i * spacing
            y1 = -self.r - (np.sqrt(3) / 2) * spacing

            if x1 > self.w:
                break

            fence.append((x1, y1))

        theta2 = np.linspace(-1 * np.pi / 2, 0.999 * np.pi / 2, int(np.ceil(2 * np.pi * self.r / spacing)))
        x2 = (self.r + (np.sqrt(3) / 2) * spacing) * np.cos(theta2) + self.w  # x-coordinates of the fence points
        y2 = (self.r + (np.sqrt(3) / 2) * spacing) * np.sin(theta2)  # y-coordinates of the fence points

        for x, y in zip(x2, y2):
            fence.append((x, y))

        for i in range(int(2*self.w // spacing)+1):
            x3 = self.w - i * spacing
            y3 = self.r + (np.sqrt(3) / 2) * spacing

            if x3 < -self.w:
                break

            fence.append((x3, y3))

        return fence