# ************************************************************
# Copyright (c) 2021, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************
from cycle_labelling import CycleLabelling
from sensor_network import SensorNetwork
from state_change import StateChange
from topology import generate_topology
from utilities import MaxRecursionDepthError


## This class provides the main interface for running a simulation.
# It provides the ability to preform a single timestep manually, as well as run
# until there are no possible intruders, and/or until a max time is reached.
# state_change provides the name of the transitions per timestep.
class EvasionPathSimulation:

    ## Initialize from a given sensor_network.
    # If end_time is set to a non-zero value, use minimum of end_time time and cleared
    # domain. Set Tend = 0 to run until no possible intruder.
    def __init__(self, sensor_network: SensorNetwork, dt: float, end_time: int = 0) -> None:

        # time settings
        self.dt = dt
        self.Tend = end_time
        self.time = 0

        self.sensor_network = sensor_network
        self.topology = generate_topology(sensor_network.points, sensor_network.sensing_radius)
        self.cycle_label = CycleLabelling(self.topology)

        self.topology_stack = []

    ## Run until no more intruders.
    # exit if max time is set. Returns simulation time.
    def run(self) -> float:
        while self.cycle_label.has_intruder():
            try:
                self.do_timestep()
            except MaxRecursionDepthError:
                raise  # do self_dump
            # self.time += self.dt
            if 0 < self.Tend < self.time:
                break
        self.cycle_label.finalize(self.time)
        return self.time

    ## Do single timestep.
    # Will attempt to move sensors forward and test if atomic topological change happens.
    # If change is non-atomic, split time step in half and solve recursively.
    # Once an atomic change is found, update sensor network position, and update labelling.
    def do_timestep(self, level: int = 0):
        if level == 25:
            s = StateChange(self.topology_stack[-1], self.topology)
            raise MaxRecursionDepthError(s)

        adaptive_dt = self.dt / (2 ** level)

        # First half
        self.sensor_network.move(adaptive_dt)
        new_topology = generate_topology(self.sensor_network.points, self.sensor_network.sensing_radius)
        state_change = StateChange(new_topology, self.topology)

        if not state_change.is_atomic_change():
            self.topology_stack.append(new_topology)
            self.do_timestep(level + 1)
        else:
            self.update(state_change, adaptive_dt)

        if level == 0:
            return

        # Second half
        self.sensor_network.move(adaptive_dt)
        new_topology = self.topology_stack.pop()
        state_change = StateChange(new_topology, self.topology)

        if not state_change.is_atomic_change():
            self.topology_stack.append(new_topology)
            self.do_timestep(level + 1)
        else:
            self.update(state_change, adaptive_dt)

    def update(self, state_change, adaptive_dt: float):
        self.time += adaptive_dt
        self.cycle_label.update(state_change, self.time)
        self.topology = state_change.new_topology
        self.sensor_network.update()
