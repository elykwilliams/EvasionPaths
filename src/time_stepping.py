# ************************************************************
# Copyright (c) 2021, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************
import pickle

from alpha_complex import Simplex
from cycle_labelling import CycleLabellingDict
from reeb_graph import ReebGraph
from sensor_network import SensorNetwork
from state_change import StateChange
from topology import generate_topology
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
        self.topology = generate_topology(sensor_network.points, sensor_network.sensing_radius)
        self.cycle_label = CycleLabellingDict(self.topology)

        self.stack = []
        self.history = [(self.make_hole_dict(), 'Initialization', 0)]

        self.reeb_graph = ReebGraph(self.history[0][0])

    ## Run until no more intruders.
    # exit if max time is set. Returns simulation time.
    def run(self) -> float:
        while self.cycle_label.has_intruder():
            try:
                self.do_timestep()
            except MaxRecursionDepthError:
                raise  # do self_dump
            self.time += self.dt
            if 0 < self.Tend < self.time:
                break
        self.reeb_graph.finalize(self.time, self.make_hole_dict())
        return self.time

    ## Do single timestep.
    # Will attempt to move sensors forward and test if atomic topological change happens.
    # If change is non-atomic, split time step in half and solve recursively.
    # Once an atomic change is found, update sensor network position, and update labelling.
    def do_timestep(self, level=0):
        if level == 25:
            s = StateChange(self.stack[-1], self.topology)
            raise MaxRecursionDepthError(s)

        adaptive_dt = self.dt * 2 **-level

        self.sensor_network.move(adaptive_dt)
        topology = generate_topology(self.sensor_network.points, self.sensor_network.sensing_radius)
        self.stack.append(topology)

        for _ in range(2):
            if not self.stack:
                return

            new_topology = self.stack.pop()
            label_update = LabelUpdateFactory().get_update(new_topology, self.topology, self.cycle_label)

            if label_update.is_atomic():
                self.update(label_update, new_topology, adaptive_dt)
            else:
                self.stack.append(new_topology)
                self.do_timestep(level+1)

    def update(self, label_update, new_topology, adaptive_dt):
        self.cycle_label.update(label_update)
        self.sensor_network.move(adaptive_dt)
        self.sensor_network.update()
        self.topology = new_topology
        self.time += adaptive_dt

        self.history.append((self.make_hole_dict(), type(label_update).__name__, self.time))
        self.reeb_graph.update(self.time, self.history[-1][0], self.history[-2][0])

    def make_hole_dict(self):
        d = {}
        for cycle, val in self.cycle_label.dict.items():
            if cycle == self.topology.alpha_cycle:
                continue
            elif len(cycle) > 4:
                d[cycle] = val
            elif Simplex(cycle.nodes) not in self.topology.simplices(self.topology.dim):
                d[cycle] = val
        return d

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
# This is useful for saving an incomplete simulation to restart later.
def save_state(simulation, filename: str) -> None:
    try:
        with open(filename, 'wb') as file:
            pickle.dump(simulation, file)
    except Exception as e:
        raise IOError(f'Unable to open file: {filename}\n{e}')
