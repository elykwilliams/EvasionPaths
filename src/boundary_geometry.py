# ************************************************************
# Copyright (c) 2020, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************

import numpy as np
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

    @abstractmethod
    def reflect_point(self, old_pt, new_pt):
        return new_pt


## a rectangular domain using virtual boundary.
# This domain implements a virtual boundary so that sensors don't get
# too close and mess up the boundary cycle associated with the fence.
# The input parameters specify the dimension of the desired virtual boundary
# and the physical locations of the sensors are places slightly outside of
# this boundary in a way that still allows sensors to form simplices with
# boundary sensors.
class RectangularDomain(Boundary):

    ## Initialize with dimension of virtual boundary.
    # sensor positions will be adapted to so that interior sensors may roam
    # specified domain. Default to the unit square with spacing of 0.2. Spacing
    # should be less that 2*sensing_radius.
    def __init__(self, spacing: float = 0.2,  # default of 0.2, with unit square
                 x_min: float = 0, x_max: float = 1,
                 y_min: float = 0, y_max: float = 1) -> None:
        self.x_max, self.y_max = x_max, y_max
        self.x_min, self.y_min = x_min, y_min
        self.spacing = spacing

        # Initialize virtual boundary
        self.dx = self.spacing * np.sin(np.pi / 6)  # virtual boundary width
        self.vx_min, self.vx_max = self.x_min - self.dx, self.x_max + self.dx
        self.vy_min, self.vy_max = self.y_min - self.dx, self.y_max + self.dx

        super().__init__()

    ## Check if point is in virtual domain.
    def in_domain(self, point: tuple) -> bool:
        return self.x_min < point[0] < self.x_max \
               and self.y_min < point[1] < self.y_max

    ## Generate points in counter-clockwise order.
    def generate_boundary_points(self) -> list:
        points = []
        points.extend([(x, self.vy_min) for x in np.arange(self.vx_min, 0.999*self.vx_max, self.spacing)])  # bottom
        points.extend([(self.vx_max, y) for y in np.arange(self.vy_min, 0.999*self.vx_max, self.spacing)])  # right
        points.extend([(self.x_max - x, self.vy_max) for x in np.arange(self.vx_min, 0.999*self.vx_max, self.spacing)])  # top
        points.extend([(self.vx_min, self.y_max - y) for y in np.arange(self.vy_min, 0.999*self.vy_max, self.spacing)])  # left
        return points

    ## Generate points distributed randomly (uniformly) in the interior.
    def generate_interior_points(self, n_int_sensors: int)-> list:
        rand_x = np.random.uniform(self.x_min, self.x_max, size=n_int_sensors)
        rand_y = np.random.uniform(self.y_min, self.y_max, size=n_int_sensors)
        return list(zip(rand_x, rand_y))

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
