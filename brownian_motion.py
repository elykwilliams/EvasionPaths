# Kyle Williams 2/4/20

from boundary_geometry import *
from numpy import sqrt, random, sin, pi


def generate_points(boundary, n_sensors, radius):
    interior_pts = []
    for _ in range(n_sensors):
        rand_x = np.random.uniform(boundary.x_min + radius, boundary.x_max - radius)
        rand_y = np.random.uniform(boundary.y_min + radius, boundary.y_max - radius)
        interior_pts.append((rand_x, rand_y))
    return boundary.points + interior_pts


class BrownianMotion:
    def __init__(self, dt, sigma, sensing_radius, boundary):
        self.radius = sensing_radius
        self.boundary = boundary
        self.sigma = sigma
        self.dt = dt

    def epsilon(self):  # Model selected by Deepjoyti
        return self.sigma*sqrt(self.dt)*random.normal(0, 1)

    def update_points(self, old_points):
        dx = self.radius*sin(pi/3)

        interior_pts = old_points[len(self.boundary):]

        interior_pts = [(x + self.epsilon(), y + self.epsilon()) for (x, y) in interior_pts]

        for n, (x, y) in enumerate(interior_pts):
            if x >= self.boundary.x_max - dx:
                interior_pts[n] = (self.boundary.x_max - dx - 2*abs(self.epsilon()), y)
            if x <= self.boundary.x_min + dx:
                interior_pts[n] = (self.boundary.x_min + dx + 2*abs(self.epsilon()), y)
            if y >= self.boundary.y_max - dx:
                interior_pts[n] = (x, self.boundary.y_max - dx - 2*abs(self.epsilon()))
            if y <= self.boundary.y_min + dx:
                interior_pts[n] = (x, self.boundary.y_min + dx + 2*abs(self.epsilon()))

        return self.boundary.points + interior_pts
