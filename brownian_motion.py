# Kyle Williams 2/4/20

from boundary_geometry import *
from numpy import sqrt, random


class BrownianMotion:
    def __init__(self, dt, sigma, sensing_radius, boundary):
        self.radius = sensing_radius
        self.boundary = boundary
        self.sigma = sigma
        self.dt = dt

    def epsilon(self):  # Model selected by Deepjoyti
        return self.sigma*sqrt(self.dt)*random.normal(0, 1)

    def generate_points(self, n_interior_pts):
        interior_pts = []
        for _ in range(n_interior_pts):
            rand_x = np.random.uniform(self.boundary.x_min + self.radius/2, self.boundary.x_max - self.radius/2)
            rand_y = np.random.uniform(self.boundary.y_min + self.radius/2, self.boundary.y_max - self.radius/2)
            interior_pts.append((rand_x, rand_y))
        return self.boundary.points + interior_pts

    def update_points(self, old_points):
        interior_pts = old_points[len(self.boundary):]

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

        return self.boundary.points + interior_pts
