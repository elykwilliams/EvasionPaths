# Kyle Williams 2/25/20

from boundary_geometry import *
from numpy import sqrt, random, sin, cos, pi
from abc import ABC, abstractmethod


class MotionModel(ABC):
    def __init__(self, dt, boundary):
        self.dt = dt
        self.boundary = boundary

    @abstractmethod
    def update_point(self, pt, index):
        return point

    @abstractmethod
    def reflect(self, pt, index):
        return point

    def update_points(self, old_points):

        interior_pts = old_points[len(self.boundary):]

        interior_pts = [self.update_point(pt, len(self.boundary)+n) for (n, pt) in enumerate(interior_pts)]

        for n, pt in enumerate(interior_pts):
            while not self.boundary.in_domain(interior_pts[n]):
                interior_pts[n] = self.reflect(pt, len(self.boundary)+n)

        return self.boundary.points + interior_pts


class BrownianMotion(MotionModel):
    def __init__(self, dt, sigma, boundary) -> None:
        super().__init__(dt, boundary)
        self.sigma = sigma

    def epsilon(self) -> float:  # Model selected by Deepjoyti
        return self.sigma*sqrt(self.dt)*random.normal(0, 1)

    def update_point(self, pt: tuple, index=0) -> tuple:
        return pt[0] + self.epsilon(), pt[1] + self.epsilon()

    def reflect(self, pt: tuple, index) -> tuple:
        x, y = pt
        if x >= self.boundary.x_max:
            pt = (self.boundary.x_max - abs(self.epsilon()), y)
        elif x <= self.boundary.x_min:
            pt = (self.boundary.x_min + abs(self.epsilon()), y)
        if y >= self.boundary.y_max:
            pt = (x, self.boundary.y_max - abs(self.epsilon()))
        elif y <= self.boundary.y_min:
            pt = (x, self.boundary.y_min + abs(self.epsilon()))
        return pt


class BilliardMotion(MotionModel):
    """ Defines motion with constant velocity and sensors reflected
        at the boundary with angle in = angle out"""
    def __init__(self, dt: float, vel: float, boundary: RectangularDomain, n_sensors: int):
        super().__init__(dt, boundary)
        self.vel = vel
        self.vel_angle = random.uniform(0, 2*pi, n_sensors)

    def update_point(self, pt, index):
        theta = self.vel_angle[index]
        return pt[0] + self.dt*self.vel*cos(theta), pt[1] + self.dt*self.vel*sin(theta)

    def reflect(self, pt, index):
        if pt[0] <= self.boundary.x_min or pt[0] >= self.boundary.x_max:
            self.vel_angle[index] = pi - self.vel_angle[index]
        if pt[1] <= self.boundary.y_min or pt[1] >= self.boundary.y_max:
            self.vel_angle[index] = - self.vel_angle[index]
        self.vel_angle[index] %= 2 * pi
        return self.update_point(pt, index)


class RunAndTumble(BilliardMotion):

    def update_point(self, pt, index):
        theta = self.vel_angle[index]
        return pt[0] + self.dt * self.vel * cos(theta), pt[1] + self.dt * self.vel * sin(theta)

    def update_points(self, old_points):

        for n in range(len(self.vel_angle)):
            if random.randint(0, 10) == 9:
                self.vel_angle[n] = random.uniform(0, 2 * pi)

        return super().update_points(old_points)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    sensing_radius = 0.15

    unit_square = RectangularDomain(sensing_radius)
    points = unit_square.generate_points(25)
    n_sensors = 25 + len(unit_square)
    mm = BilliardMotion(0.1, 0.2, unit_square, n_sensors)


    def plot_boundary(boundary):
        x_pts = [x for x, _ in boundary.points]
        y_pts = [y for _, y in boundary.points]
        plt.plot(x_pts, y_pts, "*")

        x_us = [0, 0, 1, 1, 0]
        y_us = [0, 1, 1, 0, 0]
        plt.plot(x_us, y_us)

        ax = plt.gca()

        for point in unit_square.points:
            ax.add_artist(plt.Circle(point, sensing_radius, color='b', alpha=0.1))

    def plot_points(points):
        x_pts = [x for x, _ in points]
        y_pts = [y for _, y in points]
        plt.plot(x_pts, y_pts, "k*")

    def update(index):
        global points
        fig = plt.gcf()
        ax = plt.gca()

        ax.cla()
        ax.axis('equal')
        ax.set(xlim=(unit_square.vx_min - 1.1 * sensing_radius, unit_square.vx_max + 1.1 * sensing_radius),
               ylim=(unit_square.vy_min - 1.1 * sensing_radius, unit_square.vy_max + 1.1 * sensing_radius))
        ax.set_aspect('equal', 'box')
        points = mm.update_points(points)

        plot_boundary(unit_square)
        plot_points(points)

    def animate():
        n_steps = 100
        dt = 0.1
        ms_per_frame = 1000 * dt

        fig = plt.figure(1)

        ani = FuncAnimation(fig, update, interval=ms_per_frame, frames=n_steps)
        plt.show()

    for _ in range(10):
        points = mm.update_points(points)
    animate()

