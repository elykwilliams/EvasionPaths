# ************************************************************
# Copyright (c) 2020, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************

from cycle_labelling import *
from copy import deepcopy
from topological_state import *


## Exception indicating that atomic transition not found.
# This can happen when two or more atomic transitions
# happen simultaniously. This is sometimes a problem for manufactured
# simulations. It can also indicate that a sensor has broken free of the
# virtual boundary and is interfering with the fence boundary cycle.
# There is a very rare change that a new atomic transition is discovered.
class MaxRecursionDepth(Exception):
    def __init__(self, state_change):
        self.state_change = state_change

    def __str__(self):
        return "Max Recursion depth exceeded! \n\n" \
               + str(self.state_change)


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

        self.state = TopologicalState(self.points, self.sensing_radius, self.boundary)
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
                self.points = self.interpolate_points(self.old_points, new_points, t)

            self.state = TopologicalState(self.points, self.sensing_radius, self.boundary)
            self.state_change = StateChange(self.old_state, self.state)

            try:
                self.cycle_label.update(self.state_change)

            except InvalidStateChange:
                self.do_timestep(self.points, level=level + 1)

            self.n_steps += 1
            self.update_old_data()

            if level == 0:
                return

    @staticmethod
    def interpolate_points(old_points, new_points, t):
        return [(old_points[n][0] * (1 - t) + new_points[n][0] * t,
                 old_points[n][1] * (1 - t) + new_points[n][1] * t)
                for n in range(len(old_points))]


if __name__ == "__main__":
    pass
