
class Sensor:
    def __init__(self, position, velocity, sensing_radius, boundary_sensor=False):
        self.position = position
        self.old_pos = position
        self.velocity = velocity
        self.radius = sensing_radius
        self.boundary_flag = boundary_sensor

    def update_position(self, motion_model, dt, pt=()):
        if pt:
            self.position = pt
        else:
            pass

    def update_old_position(self):
        self.old_pos = self.position


class SensorNetwork:
    def __init__(self, motion_model, boundary, sensing_radius, n_sensors, points):
        self.motion_model = motion_model

        # Initialize sensor positions
        if points and motion_model.n_sensors != len(points):
            assert False, \
                "motion_model.n_sensors != len(points) \n"\
                "Use the correct number of sensors when initializing the motion model."

        self.sensors = [Sensor(pt, (0, 0), sensing_radius, True) for pt in boundary.generate_boundary_points()]
        if points:
            self.sensors.extend([Sensor(pt, None, sensing_radius) for pt in points])
        else:
            self.sensors.extend([Sensor(pt, None, sensing_radius)
                                 for pt in boundary.generate_interior_points(n_sensors)])


