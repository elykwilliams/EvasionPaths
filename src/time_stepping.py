# ******************************************************************************
#  Copyright (c) 2023, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************

from cycle_labelling import CycleLabelling
from sensor_network import SensorNetwork
from state_change import StateChange
from topology import generate_topology
from utilities import MaxRecursionDepthError


class EvasionPathSimulation:
    """
    This class provides the main interface for running a simulation to build the Reeb graph
    of the uncovered region in a sensor network. It provides the ability to perform a single
    timestep manually, as well as run until there are no possible intruders, and/or until a
    max time is reached.
    """

    def __init__(self, sensor_network: SensorNetwork, dt: float, end_time: int = 0) -> None:
        """
        Initialize from a given sensor_network.
        If end_time is set to a non-zero value, use minimum of end_time time and cleared
        domain. Set end_time = 0 to run until no possible intruder.

        :param sensor_network: The sensor network to be simulated.
        :param dt: The timestep for the simulation.
        :param end_time: The end time for the simulation. Set to 0 to run until no possible intruder.
        """
        # time settings
        self.dt = dt
        self.Tend = end_time
        self.time = 0

        self.sensor_network = sensor_network
        self.topology = generate_topology(sensor_network.points, sensor_network.sensing_radius)
        self.cycle_label = CycleLabelling(self.topology)

        self.topology_stack = []

    def run(self) -> float:
        """
        Run simulation until no more intruders.
        Exit if max time is set. Returns simulation time.

        :return: The simulation time.
        """
        while self.cycle_label.has_intruder():
            try:
                self.do_timestep()
            except MaxRecursionDepthError:
                raise  # do self_dump
            if 0 < self.Tend < self.time:
                break
        # self.cycle_label.finalize(self.time)
        return self.time

    def do_timestep(self, level: int = 0) -> None:
        """
        Do single timestep.
        Will attempt to move sensors forward and test if atomic topological change happens.
        If change is non-atomic, split time step in half and continue recursively.
        Once an atomic change is found, update sensor network position, and update labelling.

        :param level: The recursion level of the adaptive time-stepping. Defaults to 0.
        """
        adaptive_dt = self.dt / (2 ** level)

        # Split interval in two
        for loop_id in range(2):
            self.sensor_network.move(adaptive_dt)

            if loop_id == 0:
                new_topology = generate_topology(self.sensor_network.points, self.sensor_network.sensing_radius)
            else:
                # new_topology = self.topology_stack.pop()
                new_topology = generate_topology(self.sensor_network.points, self.sensor_network.sensing_radius)

            state_change = StateChange(new_topology, self.topology)

            if level == 25:
                raise MaxRecursionDepthError(state_change)

            if not state_change.is_atomic_change():
                # self.topology_stack.append(new_topology)
                self.do_timestep(level + 1)
            else:
                self.update(state_change, adaptive_dt)

            if level == 0:
                break

    def update(self, state_change: StateChange, adaptive_dt: float) -> None:
        """
        Update the simulation state after a timestep.

        :param state_change: The state change object representing the difference in the topology.
        :param adaptive_dt: The timestep size for current recursive level.
        """
        self.time += adaptive_dt
        self.cycle_label.update(state_change, self.time)
        self.topology = state_change.new_topology
        self.sensor_network.update()
