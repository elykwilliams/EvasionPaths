# Kyle Williams 2/4/20

from boundary_geometry import *
from numpy import sqrt, random, sin, pi


def generate_points(boundary, n_sensors, radius):
    interior_pts = []
    dx = radius * sin(pi / 6)  # virtual boundary width
    for _ in range(n_sensors):
        rand_x = np.random.uniform(boundary.x_min + dx, boundary.x_max - dx)
        rand_y = np.random.uniform(boundary.y_min + dx, boundary.y_max - dx)
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
        dx = self.radius*sin(pi/6) # virtual boundary width

        interior_pts = old_points[len(self.boundary):]

        interior_pts = [(x + self.epsilon(), y + self.epsilon()) for (x, y) in interior_pts]

        for n, (x, y) in enumerate(interior_pts):
            if x >= self.boundary.x_max - dx:
                interior_pts[n] = (self.boundary.x_max - dx - abs(self.epsilon()), y)
            if x <= self.boundary.x_min + dx:
                interior_pts[n] = (self.boundary.x_min + dx + abs(self.epsilon()), y)
            if y >= self.boundary.y_max - dx:
                interior_pts[n] = (x, self.boundary.y_max - dx - abs(self.epsilon()))
            if y <= self.boundary.y_min + dx:
                interior_pts[n] = (x, self.boundary.y_min + dx + abs(self.epsilon()))

        return self.boundary.points + interior_pts


class BilliardMotion:
    def __init__(self, dt, sigma, sensing_radius, boundary, points):
        pass

    def epsilon(self):
        return self.sigma*sqrt(self.dt)*random.normal(0, 1)

    def update_points(self, old_points):
        pass
