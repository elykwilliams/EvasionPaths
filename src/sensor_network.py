
class Sensor:
    def __init__(self, position, velocity, sensing_radius, boundary_sensor=False):
        self.position = position
        self.old_pos = position
        self.velocity = velocity
        self.radius = sensing_radius
        self.boundary_flag = boundary_sensor

    def move(self, motion_model, dt, pt=()):
        if self.boundary_flag:
            return

        if pt:
            self.position = pt
        else:
            pass

    def update(self):
        assert not self.boundary_flag, "Boundary sensors cannot be updated"
        self.old_pos = self.position


class SensorNetwork:
    def __init__(self, motion_model, boundary, sensing_radius, n_sensors=0, points=(), velocities=()):
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
            self.mobile_sensors = [Sensor(pt, v, sensing_radius) for pt, v in zip(points, velocities)]
        elif points:
            self.mobile_sensors = [Sensor(pt, None, sensing_radius) for pt in points]
        else:
            # vel = self.motion_model.initialize_velocity()
            vel = None
            self.mobile_sensors = [Sensor(pt, vel, sensing_radius) for pt in boundary.generate_interior_points(n_sensors)]

    def __iter__(self):
        return iter(self.fence_sensors + self.mobile_sensors)

    def move(self, dt):
        sensors = self.fence_sensors + self.mobile_sensors
        points = [s.old_pos for s in sensors]
        motion_model_points = self.motion_model.update_points(points, dt)

        for sensor, pt in zip(sensors, motion_model_points):
            sensor.move(self.motion_model, dt, pt)

    def update(self):
        for s in self.mobile_sensors:
            s.update()
