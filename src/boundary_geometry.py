# Kyle Williams 6/3/20

import numpy as np


class RectangularDomain:
    """This class defines the boundary and stores data related to the boundary geometry."""

    def __init__(self, spacing: float = 0.2,  # default of 0.2, with unit square
                 x_min: float = 0, x_max: float = 1,
                 y_min: float = 0, y_max: float = 1) -> None:
        self.x_max, self.y_max = x_max, y_max
        self.x_min, self.y_min = x_min, y_min
        self.points = []

        self.dx = spacing * np.sin(np.pi / 6)  # virtual boundary width
        self.vx_min, self.vx_max = self.x_min - self.dx, self.x_max + self.dx
        self.vy_min, self.vy_max = self.y_min - self.dx, self.y_max + self.dx

        self.generate(spacing)

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

    def generate(self, spacing: float) -> list:
        """ Generate a list of points (represent by tuples)
            Spacing should be *at most* 2*sensing_radius when used.
            Points should be added in cyclic order.
        """

        self.points.extend([(x, self.vy_min) for x in np.arange(self.vx_min, self.vx_max, spacing)])  # bottom
        self.points.extend([(self.vx_max, y) for y in np.arange(self.vy_min, self.vx_max, spacing)])  # right
        self.points.extend([(self.x_max - x, self.vy_max) for x in np.arange(self.vx_min, self.vx_max, spacing)])  # top
        self.points.extend([(self.vx_min, self.y_max - y) for y in np.arange(self.vy_min, self.vy_max, spacing)])  # left

        return self.points


class CircularDomain:
    """This class defines the boundary and stores data related to the boundary geometry. It does not store the boundary
        points, simply generates and returns them """

    def __init__(self, spacing: float = 0.2, radius: float = 1):
        self.radius = radius
        self.points = []
        self.generate(spacing)

    def __len__(self):
        return len(self.points)

    def at_boundary(self, point: tuple) -> bool:
        """This function defines the boundary by determining if a point is at the boundary or not. """
        return point[0] ** 2 + point[1] ** 2 == self.radius ** 2

    def in_domain(self, point: tuple) -> bool:
        """This function determines if a point is in the interior of the domain or not """
        return point[0] ** 2 + point[1] ** 2 < self.radius ** 2

    def generate(self, spacing: float) -> list:
        """ Generate a list of points (represent by tuples)
            Add corners explicitly in case boundary width is not divisible by radius
            Spacing should be *at most* 2*sensing_radius when used.
        """
        dtheta = spacing / self.radius

        self.points = [(self.radius * np.cos(theta), self.radius * np.sin(theta))
                       for theta in np.arange(0, 2 * np.pi, dtheta)]

        return self.points


if __name__=="__main__":
    import matplotlib.pyplot as plt

    sensing_radius = 0.15

    # Draw vi
    unit_square = RectangularDomain(sensing_radius)
    x_pts = [x for x, _ in unit_square.points]
    y_pts = [y for _, y in unit_square.points]
    plt.plot(x_pts, y_pts, "*")

    # Draw unit square
    x_us = [0, 0, 1, 1, 0]
    y_us = [0, 1, 1, 0, 0]
    plt.plot(x_us, y_us)

    ax = plt.gca()

    for point in unit_square.points:
        ax.add_artist(plt.Circle(point, sensing_radius, color='b', alpha=0.1))

    ax.axis('equal')
    ax.set(xlim=(unit_square.vx_min - 1.1*sensing_radius, unit_square.vx_max + 1.1*sensing_radius),
           ylim=(unit_square.vy_min - 1.1*sensing_radius, unit_square.vy_max + 1.1*sensing_radius))
    ax.set_aspect('equal', 'box')
    plt.show()
