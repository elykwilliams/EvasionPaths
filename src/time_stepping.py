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

        self.state = TopologicalState(self.points, self.sensing_radius, self.boundary)
        self.old_state = self.state
        self.state_change = StateChange(self.old_state, self.state)
        self.cycle_label = CycleLabelling(self.state)

    ## Move to next time-step.
    def update_old_data(self) -> None:
        self.old_state = deepcopy(self.state)

    ## Run until no more intruders.
    # exit if max time is set. Returns simulation time.
    def run(self) -> float:
        while self.cycle_label.has_intruder():
            # self.time += self.dt
            self.evasion_paths = ""
            self.do_timestep()
            if 0 < self.Tend < self.time:
                break
        return self.time

    ## To single timestep.
    # Do recursive adaptive step if non-atomic transition is found.
    def do_timestep(self, level: int = 0) -> None:

        dt = self.dt * 2 ** -(level+1)

        if level == 25:
            raise MaxRecursionDepth(self.state_change)

        for _ in range(2):

            new_points = self.motion_model.update_points(self.points, dt)
            self.state = TopologicalState(new_points, self.sensing_radius, self.boundary)
            self.state_change = StateChange(self.old_state, self.state)

            if self.state_change.is_atomic():
                self.cycle_label.update(self.state_change)
                self.time += dt
                self.points = new_points
                self.is_connected &= self.state.is_connected()
                self.update_old_data()
            else:
                self.do_timestep(level=level + 1)

            if level == 0:
                return
