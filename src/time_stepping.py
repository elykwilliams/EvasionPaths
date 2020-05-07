# Kyle Williams 3/8/20
from cycle_labelling import *
from copy import deepcopy
from states import *


def interpolate_points(old_points, new_points, t):
    assert (len(old_points) == len(new_points))
    return [(old_points[n][0] * (1 - t) + new_points[n][0] * t,
             old_points[n][1] * (1 - t) + new_points[n][1] * t)
            for n in range(len(old_points))]


class EvasionPathSimulation:
    def __init__(self, boundary, motion_model, n_int_sensors, sensing_radius, dt, end_time=0):

        # Initialize Fields
        self.evasion_paths = ""
        self.cmap = None

        self.motion_model = motion_model
        self.boundary = boundary
        self.sensing_radius = sensing_radius

        # Parameters
        self.dt = dt
        self.Tend = end_time

        # Internal time keeping
        self.time = 0
        self.n_steps = 0

        # Point data
        self.points = boundary.generate_points(n_int_sensors)
        self.old_points = self.points

        self.state = State(self.points, self.sensing_radius, self.boundary)
        self.old_state = self.state
        self.state_change = StateChange(self.old_state, self.state)
        self.cycle_label = CycleLabelling(self.state)

    def update_old_data(self):
        self.old_points = self.points.copy()
        self.old_state = deepcopy(self.state)

    def run(self):
        while self.cycle_label.has_intruder():
            self.time += self.dt
            self.evasion_paths = ""
            self.evasion_paths += self.do_timestep()
            if 0 < self.Tend < self.time:
                break
        return self.time

    def do_timestep(self, new_points=(), level=0):

        if level == 25:
            raise MaxRecursionDepth(self.state_change)

        for t in [0.5, 1.0]:

            # Update Points
            if level == 0:
                self.points = self.motion_model.update_points(self.points)
            else:
                self.points = interpolate_points(self.old_points, new_points, t)

            self.state = State(self.points, self.sensing_radius, self.boundary)
            self.state_change = StateChange(self.old_state, self.state)

            try:
                self.cycle_label.update(self.state_change)

            except InvalidStateChange:
                self.do_timestep(self.points, level=level + 1)

            self.n_steps += 1
            self.update_old_data()

            if level == 0:
                return


if __name__ == "__main__":
    pass
