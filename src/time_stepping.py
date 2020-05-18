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
from motion_model import *


## Exception indicating that atomic transition not found.
# This can happen when two or more atomic transitions
# happen simultaneously. This is sometimes a problem for manufactured
# simulations. It can also indicate that a sensor has broken free of the
# virtual boundary and is interfering with the fence boundary cycle.
# There is a very rare change that a new atomic transition is discovered.
class MaxRecursionDepth(Exception):
    def __init__(self, state_change):
        self.state_change = state_change

    def __str__(self):
        return "Max Recursion depth exceeded! \n\n" \
               + str(self.state_change)


## This class provides the main interface for running a simulation.
# It provides the ability to preform a single timestep manually, run
# until there are no possible intruders, or until a max time is reached.
# evasion_paths provides the name of the transitions per timestep, is_connected
# is a flag that will be true as long as the graph remains connects.
class EvasionPathSimulation:

    ## Initialize
    # If end_time is set to a non-zero value, use sooner of max cutoff time or  cleared
    # domain. Set to 0 to disable.
    def __init__(self, boundary: Boundary, motion_model: MotionModel,
                 n_int_sensors: int, sensing_radius: float, dt: float, end_time: int =0) -> None:

        # Initialize Fields
        self.evasion_paths = ""
        self.is_connected = True
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

    ## Move to next time-step.
    def update_old_data(self) -> None:
        self.old_points = self.points.copy()
        self.old_state = deepcopy(self.state)

    ## Run until no more intruders.
    # exit if max time is set. Returns simulation time.
    def run(self) -> float:
        while self.cycle_label.has_intruder():
            self.time += self.dt
            self.evasion_paths = ""
            self.do_timestep()
            if 0 < self.Tend < self.time:
                break
        return self.time

    ## To single timestep.
    # Do recursive adaptive step if non-atomic transition is found.
    def do_timestep(self, new_points: list = (), level: int = 0) -> None:

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
            self.is_connected = self.is_connected and self.state.is_connected()
            self.update_old_data()

            if level == 0:
                return

    ## Linearly interpolate points.
    # Used when doing an adaptive step to interpolate between old an new points.
    @staticmethod
    def interpolate_points(old_points: list, new_points: list, t: float) -> list:
        return [(old_points[n][0] * (1 - t) + new_points[n][0] * t,
                 old_points[n][1] * (1 - t) + new_points[n][1] * t)
                for n in range(len(old_points))]