# Kyle Williams 12/16/19

from boundary_geometry import *
from numpy import sqrt, random


class BrownianMotion:
    def __init__(self, dt, sigma):
        self.boundary = Boundary()
        self.sigma = sigma
        self.dt = dt

    def epsilon(self):  # Model selected by Deepjoyti
        return self.sigma*sqrt(self.dt)*random.normal(0, 1)

    def generate_points(self, n_interior_pts, radius):
        pts = self.boundary.generate(radius)
        for _ in range(n_interior_pts):
            rand_x = np.random.uniform(radius, self.boundary.x_max - radius)
            rand_y = np.random.uniform(radius, self.boundary.y_max - radius)
            pts.append([rand_x, rand_y])
        return pts

    def update_points(self, old_points):
        boundary_pts = old_points[:self.boundary.n_points]
        interior_pts = old_points[self.boundary.n_points:]

        interior_pts = [(x + self.epsilon(), y + self.epsilon()) for (x, y) in interior_pts]

        # Boundary cases as implemented by Deepjoyti
        for n, (x, y) in enumerate(interior_pts):
            if x >= 1:
                interior_pts[n] = (2 - x - self.epsilon(), y)
            if x <= 0:
                interior_pts[n] = (-x + self.epsilon(), y)
            if y >= 1:
                interior_pts[n] = (x, 2 - y - self.epsilon())
            if y <= 0:
                interior_pts[n] = (x, -y + self.epsilon())

        return boundary_pts + interior_pts
