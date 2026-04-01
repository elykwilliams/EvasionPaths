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
from tqdm import tqdm
import numpy as np


def _domain_interior_point(sensor_network: SensorNetwork):
    domain = sensor_network.domain
    if hasattr(domain, "min") and hasattr(domain, "max"):
        return 0.5 * (np.asarray(domain.min, dtype=float) + np.asarray(domain.max, dtype=float))

    point_generator = getattr(domain, "point_generator", None)
    if callable(point_generator):
        try:
            sample = point_generator(1)
            sample_array = np.asarray(sample, dtype=float)
            if sample_array.size:
                if sample_array.ndim == 1:
                    return sample_array
                return sample_array[0]
        except Exception:
            pass

    fence = np.asarray([s.pos for s in sensor_network.fence_sensors], dtype=float)
    if fence.size:
        return np.mean(fence, axis=0)
    return None

class EvasionPathSimulation:
    """
    This class provides the main interface for running a simulation to build the Reeb graph
    of the uncovered region in a sensor network. It provides the ability to perform a single
    timestep manually, as well as run until there are no possible intruders, and/or until a
    max time is reached.
    """

    def __init__(
        self,
        sensor_network: SensorNetwork,
        dt: float,
        end_time: int = 0,
        outer_winding_sign: int = -1,
    ) -> None:
        """
        Initialize from a given sensor_network.
        If end_time is set to a non-zero value, use minimum of end_time time and cleared
        domain. Set end_time = 0 to run until no possible intruder.

        :param sensor_network: The sensor network to be simulated.
        :param dt: The timestep for the simulation.
        :param end_time: The end time for the simulation. Set to 0 to run until no possible intruder.
        """
        print("initializing sim")
        # time settings
        self.dt = dt
        self.Tend = end_time
        self.time = 0

        self.sensor_network = sensor_network
        self._fence_node_count = len(sensor_network.fence_sensors)
        self._interior_point = _domain_interior_point(sensor_network)
        self._outer_winding_sign = -1 if outer_winding_sign < 0 else 1
        point_radii = sensor_network.point_radii if getattr(sensor_network, "use_weighted_alpha", False) else None
        self.topology = self._build_topology(point_radii=point_radii, topology_cache=None)
        # if not self.topology.is_face_connected():
        #     raise ValueError("The provided sensor network is not face connected")

        self.cycle_label = CycleLabelling(self.topology)
        self._push_motion_model_context()

        self.topology_stack = []

    def _push_motion_model_context(self):
        context_setter = getattr(self.sensor_network.motion_model, "set_homology_context", None)
        if callable(context_setter):
            context_setter(self.topology, self.cycle_label.label, self.time)

    def _build_topology(self, *, point_radii, topology_cache=None):
        points = np.asarray(self.sensor_network.points, dtype=float)
        radii_key = None
        if point_radii is not None:
            radii_key = np.asarray(point_radii, dtype=float).tobytes()

        cache_key = None
        if topology_cache is not None:
            cache_key = (points.tobytes(), radii_key)
            cached = topology_cache.get(cache_key)
            if cached is not None:
                return cached

        topology = generate_topology(
            points,
            self.sensor_network.sensing_radius,
            point_radii=point_radii,
            fence_node_count=self._fence_node_count,
            fence_node_groups=getattr(self.sensor_network, "fence_groups", ()),
            excluded_fence_groups=getattr(self.sensor_network, "excluded_fence_groups", ()),
            interior_point=self._interior_point,
            outer_winding_sign=self._outer_winding_sign,
        )
        if topology_cache is not None and cache_key is not None:
            topology_cache[cache_key] = topology
        return topology

    def run(self) -> float:
        """
        Run simulation until no more intruders.
        Exit if max time is set. Returns simulation time.

        :return: The simulation time.
        """
        print("Initial state of cycle_label:")
        print(f"Has intruder? {self.cycle_label.has_intruder()}")
        # Define the total time for the progress bar if Tend is set, otherwise use an arbitrary large value
        with tqdm(total=float('inf'), desc="Simulation Progress", unit=" ts") as pbar:
            while self.cycle_label.has_intruder():
                try:
                    self.do_timestep()
                except MaxRecursionDepthError:
                    raise  # do self_dump

                pbar.set_description(f"Current Time: {self.time:.2f}")
                pbar.update(0)  # Do not increment, just update display
                if 0 < self.Tend < self.time:
                    break
        # self.cycle_label.finalize(self.time)
        return self.time

    def do_timestep(self, level: int = 0, topology_cache=None) -> None:
        """
        Do single timestep.
        Will attempt to move sensors forward and test if atomic topological change happens.
        If change is non-atomic, split time step in half and continue recursively.
        Once an atomic change is found, update sensor network position, and update labelling.

        :param level: The recursion level of the adaptive time-stepping. Defaults to 0.
        """
        if topology_cache is None:
            topology_cache = {}

        adaptive_dt = self.dt / (2 ** level)
        # print("dotimestep, level:", level)
        # Split interval in two
        for _ in range(2):
            self._push_motion_model_context()
            self.sensor_network.move(adaptive_dt)

            # new_topology = self.topology_stack.pop()  # kept for potential future caching strategy
            point_radii = self.sensor_network.point_radii if getattr(self.sensor_network, "use_weighted_alpha", False) else None
            new_topology = self._build_topology(point_radii=point_radii, topology_cache=topology_cache)

            state_change = StateChange(new_topology, self.topology)
            is_atomic = state_change.is_atomic_change()
            if not is_atomic:
                if level == 25:
                    raise MaxRecursionDepthError(
                        state_change,
                        level=level,
                        adaptive_dt=adaptive_dt,
                        sim_time=self.time,
                    )
                # self.topology_stack.append(new_topology)
                self.do_timestep(level + 1, topology_cache=topology_cache)
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
        self._push_motion_model_context()
        self.sensor_network.update()
