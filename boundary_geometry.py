from gudhi import AlphaComplex
import numpy as np
from itertools import filterfalse
from numpy import random, sqrt
from combinatorial_map import *


class Boundary:
    def __init__(self):
        self.x_max, self.y_max = 1, 1
        self.n_points = 0

    def at_boundary(self, point):
        return point[0] <= 0 or point[1] <= 0 or point[0] >= self.x_max or point[1] >= self.y_max

    def generate(self, radius):

        corners = [[0, 0], [0, self.y_max], [self.x_max, 0], [self.x_max, self.y_max]]
        pts = corners
        for x in np.arange(radius, self.x_max, radius):
            pts.extend([[x, 0], [x, self.y_max]])
        for y in np.arange(radius, self.y_max, radius):
            pts.extend([[0, y], [self.x_max, y]])
        self.n_points = len(pts)
        return pts
