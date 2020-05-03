# Kyle Williams 2/25/20

import numpy as np


class RectangularDomain:
    """This class defines the boundary and stores data related to the boundary geometry. It does not store the boundary
        points, simply generates and returns them """

    def __init__(self, spacing: float = 0.2,  # default of 0.2, with unit square
                 x_min: float = 0, x_max: float = 1,
                 y_min: float = 0, y_max: float = 1) -> None:
        self.x_max, self.y_max = x_max, y_max
        self.x_min, self.y_min = x_min, y_min
        self.points = []

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
            Add corners explicitly in case boundary width is not divisible by radius
            Spacing should be *at most* 2*sensing_radius when used.
        """
        dx = spacing * np.sin(np.pi / 6)  # virtual boundary width
        corners = [(self.x_min-dx, self.y_min-dx), (self.x_min-dx, self.y_max+dx),
                   (self.x_max+dx, -dx), (self.x_max+dx, self.y_max+dx)]

        self.points = corners

        self.points.extend([(x, self.y_min-dx) for x in np.arange(self.x_min - dx, self.x_max+dx, spacing)])  # bottom
        self.points.extend([(self.x_min-dx, y) for y in np.arange(self.y_min - dx, self.y_max+dx, spacing)])  # left
        self.points.extend([(self.x_max - x, self.y_max+dx) for x in np.arange(self.x_min - dx, self.x_max+dx, spacing)])  # top
        self.points.extend([(self.x_max+dx, self.y_max - y) for y in np.arange(self.y_min - dx, self.y_max+dx, spacing)])  # right

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

    unit_square = RectangularDomain(0.15)
    x_pts = [x for x, _ in unit_square.points]
    y_pts = [y for _, y in unit_square.points]
    plt.plot(x_pts, y_pts, "*")

    # Draw unit square
    x_us = [0, 0, 1, 1, 0]
    y_us = [0, 1, 1, 0, 0]
    plt.plot(x_us, y_us)

    ax = plt.gca()
    ax.axis("equal")

    plt.show()