# ************************************************************
# Copyright (c) 2021, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************
import pickle

from alpha_complex import AlphaComplex
from combinatorial_map import RotationInfo2D, CombinatorialMap2D
from cycle_labelling import CycleLabellingDict
from sensor_network import SensorNetwork
from state_change import StateChange2D
from topology import ConnectedTopology
from update_data import LabelUpdateFactory
from utilities import *


## This class provides the main interface for running a simulation.
# It provides the ability to preform a single timestep manually, as well as run
# until there are no possible intruders, and/or until a max time is reached.
# state_change provides the name of the transitions per timestep.
class EvasionPathSimulation:

    ## Initialize from a given sensor_network.
    # If end_time is set to a non-zero value, use minimum of max cutoff time and cleared
    # domain. Set Tend = 0 to run until no possible intruder.
    def __init__(self, sensor_network: SensorNetwork, dt: float, end_time: int = 0) -> None:

        # time settings
        self.dt = dt
        self.Tend = end_time
        self.time = 0

        self.sensor_network = sensor_network
        ac = AlphaComplex(sensor_network.points, sensor_network.sensing_radius)
        rot_info = RotationInfo2D(sensor_network.points, ac)
        cmap = CombinatorialMap2D(rot_info)
        self.state = ConnectedTopology(ac, cmap)
        self.cycle_label = CycleLabellingDict(self.state)

    ## Run until no more intruders.
    # exit if max time is set. Returns simulation time.
    def run(self) -> float:
        while self.cycle_label.has_intruder():
            self.do_timestep()
            self.time += self.dt
            if 0 < self.Tend < self.time:
                break
        return self.time

    ## Do single timestep.
    # Will attempt to move sensors forward and test if atomic topological change happens.
    # If change is non-atomic, split time step in half and solve recursively.
    # Once an atomic change is found, update sensor network position, and update labelling.
    def do_timestep(self, level: int = 0) -> None:

        dt = self.dt * 2 ** -level

        for _ in range(2):
            self.sensor_network.move(dt)
            ac = AlphaComplex(self.sensor_network.points, self.sensor_network.sensing_radius)
            rot_info = RotationInfo2D(self.sensor_network.points, ac)
            cmap = CombinatorialMap2D(rot_info)
            new_state = ConnectedTopology(ac, cmap)
            state_change = StateChange2D(new_state, self.state)

            label_update = LabelUpdateFactory().get_update(state_change, self.cycle_label)

            if label_update.is_atomic() and self.cycle_label.is_valid(label_update):
                self.cycle_label.update(label_update)
                self.sensor_network.update()
                self.state = state_change.new_topology
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
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise InitializationError(f'Unable to open file: {filename}\n{e}')


## Dumps current state to be resumed later.
# This function is used to save the current state of a simulation.
# This is useful for saving a random initial state for testing or
# for saving an incomplete simulation to restart later.
def save_state(simulation, filename: str) -> None:
    try:
        with open(filename, 'wb') as file:
            pickle.dump(simulation, file)
    except Exception as e:
        raise IOError(f'Unable to open file: {filename}\n{e}')
