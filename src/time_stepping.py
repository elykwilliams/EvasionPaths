# ************************************************************
# Copyright (c) 2020, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************
import pickle
from cycle_labelling import *
from topological_state import *
from motion_model import *
from sensor_network import *


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
                 n_int_sensors: int, sensing_radius: float, dt: float, end_time: int = 0,
                 points=()) -> None:

        self.boundary = boundary

        # time settings
        self.dt = dt
        self.Tend = end_time
        self.time = 0

        self.sensor_network = SensorNetwork(motion_model, boundary, sensing_radius, n_int_sensors, points)
        self.state = TopologicalState(self.sensor_network.sensors, self.boundary)
        self.cycle_label = CycleLabelling(self.state)

    ## Run until no more intruders.
    # exit if max time is set. Returns simulation time.
    def run(self) -> float:
        while self.cycle_label.has_intruder() and self.Tend < self.time:
            self.do_timestep()
            self.time += self.dt
        return self.time

    ## To single timestep.
    # Do recursive adaptive step if non-atomic transition is found.
    def do_timestep(self, level: int = 0) -> None:

        dt = self.dt * 2 ** -level

        for _ in range(2):

            self.sensor_network.move(dt)
            new_state = TopologicalState(self.sensor_network.sensors, self.boundary)
            state_change = StateChange(self.state, new_state)

            if state_change.is_atomic():
                self.update(state_change)
            elif level + 1 == 25:
                raise MaxRecursionDepth(state_change)
            else:
                self.do_timestep(level=level + 1)

            if level == 0:
                return

    def update(self, state_change):
        self.cycle_label.update(state_change)
        self.sensor_network.update()
        self.state = state_change.new_state


## Takes output from save_state() to initialize a simulation.
# WARNING: Only use pickle files created by this software on a specific machine.
#          do not send pickle files over a network.
# Load a previously saved simulation.
def load_state(filename: str) -> EvasionPathSimulation:
    assert filename, "Error: Output filename not specified"
    with open(filename, "rb") as file:
        return pickle.load(file)


## Dumps current state to be resumed later.
# This function is used to save the current state of a simulation.
# This is useful for saving a random initial state for testing or
# for saving an incomplete simulation to restart later.
def save_state(simulation, filename: str) -> None:
    assert filename, "Error: Output filename not specified"
    with open(filename, "wb") as file:
        pickle.dump(simulation, file)
