# Kyle Williams 2/25/20

from boundary_geometry import *
from numpy import sqrt, random, sin, pi


def generate_points(boundary, n_sensors):
    interior_pts = []
    for _ in range(n_sensors):
        rand_x = np.random.uniform(boundary.x_min, boundary.x_max)
        rand_y = np.random.uniform(boundary.y_min, boundary.y_max)
        interior_pts.append((rand_x, rand_y))
    return boundary.points + interior_pts


class BrownianMotion:
    def __init__(self, dt, sigma, boundary):
        self.boundary = boundary
        self.sigma = sigma
        self.dt = dt

    def epsilon(self):  # Model selected by Deepjoyti
        return self.sigma*sqrt(self.dt)*random.normal(0, 1)

    def update_points(self, old_points):

        interior_pts = old_points[len(self.boundary):]

        interior_pts = [(x + self.epsilon(), y + self.epsilon()) for (x, y) in interior_pts]

        for n, (x, y) in enumerate(interior_pts):
            if x >= self.boundary.x_max:
                interior_pts[n] = (self.boundary.x_max - abs(self.epsilon()), y)
            if x <= self.boundary.x_min:
                interior_pts[n] = (self.boundary.x_min + abs(self.epsilon()), y)
            if y >= self.boundary.y_max:
                interior_pts[n] = (x, self.boundary.y_max - abs(self.epsilon()))
            if y <= self.boundary.y_min:
                interior_pts[n] = (x, self.boundary.y_min + abs(self.epsilon()))

        return self.boundary.points + interior_pts


from numpy import multiply, cos


class BilliardMotion:

    def __init__(self, dt, sensing_radius, boundary):

        theta = random.uniform(0, 360, 15)

        vel = random.uniform(0, 0.1, 15)
        self.radius = sensing_radius
        self.boundary = boundary
        self.dt = dt
        self.velx = multiply(vel, cos(theta)) * dt
        self.vely = multiply(vel, sin(theta)) * dt

    #    def epsilon(self):  # Model selected by Deepjoyti

    #        return self.vel*sqrt(self.dt)*random.normal(0, 1)

    #    def epsilon(self):

    #        theta = random.uniform(0,360)

    #        x_position = self.vel*cos(theta)

    #        y_position = self.vel*sin(theta)

    #        position = (x_position*self.dt, y_position*self.dt)

    #        return self.position

    def update_points(self, old_points):

        dx = self.radius * sin(pi / 6)  # virtual boundary width

        interior_pts = old_points[len(self.boundary):]

        interior_pts = [(x + self.velx, y + self.vely) for (x, y) in interior_pts]

        for n, (x, y) in enumerate(interior_pts):

            if x >= self.boundary.x_max - dx:
                interior_pts[n] = (self.boundary.x_max - dx - abs(self.velx), y)

                theta[n] = 180 - theta

            if x <= self.boundary.x_min + dx:
                interior_pts[n] = (self.boundary.x_min + dx + abs(self.velx), y)

                theta[n] = 180 - theta

            if y >= self.boundary.y_max - dx:
                interior_pts[n] = (x, self.boundary.y_max - dx - abs(self.vely))

                theta[n] = 360 - theta

            if y <= self.boundary.y_min + dx:
                interior_pts[n] = (x, self.boundary.y_min + dx + abs(self.vely))

                theta[n] = 360 - theta

        self.velx = multiply(vel, cos(theta)) * dt

        self.vely = multiply(vel, sin(theta)) * dt

        return self.boundary.points + interior_pts
