# ************************************************************
# Copyright (c) 2020, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************

import numpy as np


class RectangularDomain:
    """This class defines the boundary and stores data related to the boundary geometry.
        Does not store interior points but can generate them"""

    def __init__(self, spacing: float = 0.2,  # default of 0.2, with unit square
                 x_min: float = 0, x_max: float = 1,
                 y_min: float = 0, y_max: float = 1) -> None:
        self.x_max, self.y_max = x_max, y_max
        self.x_min, self.y_min = x_min, y_min
        self.points = []

        # Initialize virtual boundary
        self.dx = spacing * np.sin(np.pi / 6)  # virtual boundary width
        self.vx_min, self.vx_max = self.x_min - self.dx, self.x_max + self.dx
        self.vy_min, self.vy_max = self.y_min - self.dx, self.y_max + self.dx

        self.generate_boundary_points(spacing)
        self.alpha_cycle = self.get_alpha_cycle()

    def __len__(self):
        return len(self.points)

    def at_boundary(self, point: tuple) -> bool:
        """This function defines the boundary by determining if a point is at the boundary or not. """
        return point[0] == self.x_min \
            or point[1] == self.y_min \
            or point[0] == self.x_max \
            or point[1] == self.y_max

    def in_domain(self, point: tuple) -> bool:
        """This function determines if a point is in the interior of the domain or not """
        return self.x_min < point[0] < self.x_max and self.y_min < point[1] < self.y_max

    def generate_boundary_points(self, spacing: float) -> list:
        """ Generate a list of points (represent by tuples)
            Spacing should be *at most* 2*sensing_radius when used.
            Points should be added in cyclic order.
        """

        self.points.extend([(x, self.vy_min) for x in np.arange(self.vx_min, 0.999*self.vx_max, spacing)])  # bottom
        self.points.extend([(self.vx_max, y) for y in np.arange(self.vy_min, 0.999*self.vx_max, spacing)])  # right
        self.points.extend([(self.x_max - x, self.vy_max) for x in np.arange(self.vx_min, 0.999*self.vx_max, spacing)])  # top
        self.points.extend([(self.vx_min, self.y_max - y) for y in np.arange(self.vy_min, 0.999*self.vy_max, spacing)])  # left

        return self.points

    def get_alpha_cycle(self):
        a = [str(n+1)+","+str(n) for n in range(len(self.points)-1)] + ["0,"+str(len(self.points)-1)]
        return tuple(sorted(a))

    def generate_interior(self, n_int_sensors):
        rand_x = np.random.uniform(self.x_min, self.x_max, size=n_int_sensors)
        rand_y = np.random.uniform(self.y_min, self.y_max, size=n_int_sensors)
        return list(zip(rand_x, rand_y))

    def generate_points(self, n_int_sensors):
        return self.points + self.generate_interior(n_int_sensors)
