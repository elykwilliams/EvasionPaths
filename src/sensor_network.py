from numpy.linalg import norm
from numpy import array
from utilities import cart2pol


class Sensor:
    def __init__(self, position, polar_vel, sensing_radius, boundary_sensor=False):
        self.position = position
        self.old_pos = position
        self.pvel = polar_vel
        self.old_pvel = polar_vel
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
    def __init__(self, motion_model, domain, sensing_radius, n_sensors=0, vel_mag=None, points=(), velocities=()):
        if velocities:
            assert len(points) == len(velocities), \
                "len(points) != len(velocities)"
        if n_sensors and points:
            assert len(points) == n_sensors, \
                "The number points specified is different than the number of points provided"

        self.motion_model = motion_model
        self.sensing_radius = sensing_radius

        # Initialize sensor positions
        if velocities:
            velocities = [cart2pol(v) for v in velocities]
        elif points:
            velocities = (self.motion_model.initial_pvel(vel_mag) for _ in points)
        else:
            points = domain.generate_interior_points(n_sensors)
            velocities = (self.motion_model.initial_pvel(vel_mag) for _ in points)

        self.mobile_sensors = [Sensor(pt, v, sensing_radius) for pt, v in zip(points, velocities)]
        self.fence_sensors = [Sensor(pt, (0, 0), sensing_radius, True) for pt in domain.generate_fence()]

    def __iter__(self):
        return iter(self.fence_sensors + self.mobile_sensors)

    def move(self, dt):
        self.motion_model.compute_update(self, dt)
        for sensor in self.mobile_sensors:
            sensor.move(self.motion_model, dt)

    def update(self):
        for s in self.mobile_sensors:
            s.update()
