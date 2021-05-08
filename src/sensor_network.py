import math
from numpy.linalg import norm


def cart2pol(points):
    return [(norm(p), math.atan2(p[1], p[0])) for p in points]


class Sensor:
    def __init__(self, position, polar_vel, sensing_radius, boundary_sensor=False):
        self.position = position
        self.old_pos = position
        self.pvel = polar_vel
        self.old_pvel = self.pvel
        self.radius = sensing_radius
        self.boundary_flag = boundary_sensor

    def move(self, motion_model, dt):
        if self.boundary_flag:
            return
        motion_model.update_position(self, dt)

    def update(self):
        assert not self.boundary_flag, "Boundary sensors cannot be updated"
        self.old_pos = self.position
        self.old_pvel = self.pvel


class SensorNetwork:
    def __init__(self, motion_model, boundary, sensing_radius, n_sensors=0, vel_mag=None, points=(), velocities=()):
        if velocities:
            assert len(points) == len(velocities), \
                "len(points) != len(velocities)"
        if n_sensors and points:
            assert len(points) == n_sensors, \
                "The number points specified is different than the number of points provided"

        self.motion_model = motion_model
        self.sensing_radius = sensing_radius

        # Initialize sensor positions

        self.fence_sensors = [Sensor(pt, (0, 0), sensing_radius, True) for pt in boundary.generate_fence()]

        if velocities:
            self.mobile_sensors = [Sensor(pt, v, sensing_radius) for pt, v in zip(points, cart2pol(velocities))]
        elif points:
            self.mobile_sensors = []
            for pt in points:
                v = self.motion_model.initial_pvel(vel_mag)
                sensor = Sensor(pt, v, sensing_radius)
                self.mobile_sensors.append(sensor)
        else:
            self.mobile_sensors = []
            for pt in boundary.generate_interior_points(n_sensors):
                v = self.motion_model.initial_pvel(vel_mag)
                sensor = Sensor(pt, v, sensing_radius)
                self.mobile_sensors.append(sensor)

    def __iter__(self):
        return iter(self.fence_sensors + self.mobile_sensors)

    def move(self, dt):
        self.motion_model.compute_update(self, dt)
        for sensor in self.mobile_sensors:
            sensor.move(self.motion_model, dt)

    def update(self):
        for s in self.mobile_sensors:
            s.update()
