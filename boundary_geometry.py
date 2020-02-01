# Kyle Williams 1/31/19
# TODO: it may make more sense to store boundary points internally, but this will not have a significant effect on the \
#       rest of the code. If this to be done, generate() would be called in the __init__ function.

import numpy as np


class Boundary:
    """This class defines the boundary and stores data related to the boundary geometry. It does not store the boundary
        points, simply generates and returns them """

    def __init__(self):
        self.x_max, self.y_max = 1, 1
        self.x_min, self.y_min = 0, 0
        self.n_points = 0

    def at_boundary(self, point: tuple) -> bool:
        """This function defines the boundary by determining if a point is at the boundary or not. """
        return point[0] <= self.x_min or point[1] <= self.y_min or point[0] >= self.x_max or point[1] >= self.y_max

    def generate(self, spacing: float) -> list:
        """ Generate a list of points (represent by tuples)
            Add corners explicitly in case boundary width is not divisible by radius
            Spacing should be *at most* 2*sensing_radius when used.
        """
        corners = [(0, 0), (0, self.y_max), (self.x_max, 0), (self.x_max, self.y_max)]
        pts = corners
        for x in np.arange(spacing, self.x_max, spacing):
            pts.extend([(x, 0), (x, self.y_max)])
        for y in np.arange(spacing, self.y_max, spacing):
            pts.extend([(0, y), (self.x_max, y)])
        self.n_points = len(pts)
        return pts
