# ******************************************************************************
#  Copyright (c) 2026, Marco Campos - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from cycle_labelling import CycleLabelling
from motion_model import BilliardMotion
from sensor_network import Sensor, SensorNetwork
from state_change import StateChange
from topology import Topology, generate_topology
from utilities import MaxRecursionDepthError


@dataclass(frozen=True)
class TraceEvaluation:
    """Evaluation payload at a candidate elapsed time."""

    elapsed: float
    topology: Topology
    state_change: StateChange
    mobile_positions: np.ndarray
    mobile_velocities: np.ndarray


@dataclass(frozen=True)
class TraceDiagnostics:
    """Diagnostics collected while searching for the earliest atomic event."""

    evaluation_count: int = 0
    split_count: int = 0
    max_recursion_depth: int = 0
    recursion_limit_hit: bool = False


@dataclass(frozen=True)
class RLStepResult:
    """Result returned after committing one RL-controlled simulator step."""

    elapsed: float
    event_found: bool
    topology: Topology
    state_change: StateChange
    mobile_positions: np.ndarray
    mobile_velocities: np.ndarray
    trace_diagnostics: TraceDiagnostics


class AtomicEventTracer:
    """
    Search for the earliest non-trivial atomic change in [0, interval] using
    recursive bisection with a left-first strategy.
    """

    def __init__(
        self,
        evaluate_at: Callable[[float], TraceEvaluation],
        *,
        max_depth: int = 25,
        time_tolerance: float = 1e-6,
    ) -> None:
        self.evaluate_at = evaluate_at
        self.max_depth = max_depth
        self.time_tolerance = time_tolerance
        self.last_diagnostics = TraceDiagnostics()

    @staticmethod
    def _is_trivial_atomic(state_change: StateChange) -> bool:
        if not state_change.is_atomic_change():
            return False
        alpha_change = state_change.alpha_complex_change()
        boundary_change = state_change.boundary_cycle_change()
        return all(value == 0 for value in alpha_change) and all(value == 0 for value in boundary_change)

    @classmethod
    def _is_nontrivial_atomic(cls, state_change: StateChange) -> bool:
        return state_change.is_atomic_change() and not cls._is_trivial_atomic(state_change)

    def find_first_nontrivial_atomic(self, interval: float) -> Optional[TraceEvaluation]:
        if interval <= 0:
            raise ValueError("interval must be positive")

        cache = {}
        eval_count = 0
        split_count = 0
        max_depth_reached = 0

        def eval_at(elapsed: float) -> TraceEvaluation:
            nonlocal eval_count
            elapsed = float(elapsed)
            if elapsed not in cache:
                eval_count += 1
                cache[elapsed] = self.evaluate_at(elapsed)
            return cache[elapsed]

        def tracked_search(
            left: float,
            right: float,
            depth: int,
            right_eval: TraceEvaluation,
        ) -> Optional[TraceEvaluation]:
            nonlocal split_count, max_depth_reached
            max_depth_reached = max(max_depth_reached, depth)

            state_change = right_eval.state_change
            if self._is_nontrivial_atomic(state_change):
                if depth >= self.max_depth or (right - left) <= self.time_tolerance:
                    return right_eval

                split_count += 1
                mid = 0.5 * (left + right)
                mid_eval = eval_at(mid)
                if self._is_trivial_atomic(mid_eval.state_change):
                    return tracked_search(mid, right, depth + 1, right_eval)
                return tracked_search(left, mid, depth + 1, mid_eval)

            if self._is_trivial_atomic(state_change):
                return None

            if depth >= self.max_depth or (right - left) <= self.time_tolerance:
                raise MaxRecursionDepthError(
                    state_change,
                    level=depth,
                    adaptive_dt=right - left,
                )

            split_count += 1
            mid = 0.5 * (left + right)
            mid_eval = eval_at(mid)
            if self._is_trivial_atomic(mid_eval.state_change):
                return tracked_search(mid, right, depth + 1, right_eval)
            return tracked_search(left, mid, depth + 1, mid_eval)

        right_eval = eval_at(interval)
        try:
            result = tracked_search(0.0, interval, 0, right_eval)
            return result
        finally:
            self.last_diagnostics = TraceDiagnostics(
                evaluation_count=eval_count,
                split_count=split_count,
                max_recursion_depth=max_depth_reached,
            )

    def _search(
        self,
        left: float,
        right: float,
        depth: int,
        right_eval: TraceEvaluation,
        eval_at: Callable[[float], TraceEvaluation],
    ) -> Optional[TraceEvaluation]:
        state_change = right_eval.state_change

        if self._is_nontrivial_atomic(state_change):
            if depth >= self.max_depth or (right - left) <= self.time_tolerance:
                return right_eval

            mid = 0.5 * (left + right)
            mid_eval = eval_at(mid)
            if self._is_trivial_atomic(mid_eval.state_change):
                return self._search(mid, right, depth + 1, right_eval, eval_at)
            return self._search(left, mid, depth + 1, mid_eval, eval_at)

        if self._is_trivial_atomic(state_change):
            return None

        if depth >= self.max_depth or (right - left) <= self.time_tolerance:
            raise MaxRecursionDepthError(
                state_change,
                level=depth,
                adaptive_dt=right - left,
            )

        mid = 0.5 * (left + right)
        mid_eval = eval_at(mid)
        if self._is_trivial_atomic(mid_eval.state_change):
            return self._search(mid, right, depth + 1, right_eval, eval_at)
        return self._search(left, mid, depth + 1, mid_eval, eval_at)


def propagate_sensor_constant_velocity(
    sensor: Sensor,
    velocity: np.ndarray,
    dt: float,
    sensor_network: SensorNetwork,
    ambient_motion: Optional[BilliardMotion] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate one mobile sensor for dt with billiard-style boundary reflection."""

    def _reflect_in_axis_aligned_box(sensor_obj: Sensor, domain_obj) -> bool:
        if not (hasattr(domain_obj, "min") and hasattr(domain_obj, "max")):
            return False
        lo = np.asarray(domain_obj.min, dtype=float)
        hi = np.asarray(domain_obj.max, dtype=float)
        pos = np.asarray(sensor_obj.pos, dtype=float).copy()
        vel = np.asarray(sensor_obj.vel, dtype=float).copy()

        # Repeatedly mirror out-of-bound coordinates back into the box.
        for _ in range(8):
            moved = False
            for dim in range(len(pos)):
                if pos[dim] < lo[dim]:
                    pos[dim] = 2.0 * lo[dim] - pos[dim]
                    vel[dim] *= -1.0
                    moved = True
                elif pos[dim] > hi[dim]:
                    pos[dim] = 2.0 * hi[dim] - pos[dim]
                    vel[dim] *= -1.0
                    moved = True
            if not moved:
                sensor_obj.pos = pos
                sensor_obj.vel = vel
                return True
        sensor_obj.pos = np.minimum(np.maximum(pos, lo), hi)
        sensor_obj.vel = vel
        return bool(sensor_obj.pos in domain_obj)

    billiard = BilliardMotion() if ambient_motion is None else ambient_motion
    velocity_arr = np.asarray(velocity, dtype=float)
    start_pos = np.asarray(sensor.pos, dtype=float)

    temp = Sensor(
        np.array(start_pos, dtype=float),
        np.array(velocity_arr, dtype=float),
        sensor.radius,
        boundary_sensor=False,
    )
    temp.old_pos = np.array(start_pos, dtype=float)
    temp.old_vel = np.array(velocity_arr, dtype=float)
    temp.vel = np.array(velocity_arr, dtype=float)
    billiard.local_update(temp, float(dt))

    if temp.pos not in sensor_network.domain:
        try:
            billiard.reflect(sensor_network.domain, temp)
        except AssertionError:
            # Numerical drift may place old_pos slightly outside; use a robust box fallback.
            if not _reflect_in_axis_aligned_box(temp, sensor_network.domain):
                raise

    return np.asarray(temp.pos, dtype=float), np.asarray(temp.vel, dtype=float)


class RLEventSimulation:
    """
    Event-driven replacement for EvasionPathSimulation where a controller sets
    all mobile sensor velocities and the simulator advances to the earliest
    non-trivial atomic topological change in each control interval.
    """

    def __init__(
        self,
        sensor_network: SensorNetwork,
        dt: float,
        *,
        end_time: float = 0,
        outer_winding_sign: int = -1,
        max_event_recursion: int = 25,
        event_time_tolerance: float = 1e-6,
    ) -> None:
        self.dt = float(dt)
        self.Tend = float(end_time)
        self.time = 0.0

        self.sensor_network = sensor_network
        self._ambient_motion = BilliardMotion()
        self._fence_node_count = len(sensor_network.fence_sensors)
        self._interior_point = self._domain_interior_point(sensor_network)
        self._outer_winding_sign = -1 if outer_winding_sign < 0 else 1
        self.max_event_recursion = max_event_recursion
        self.event_time_tolerance = event_time_tolerance
        self._validate_network_configuration()

        point_radii = self._point_radii()
        self.topology = self._build_topology(points=np.asarray(self.sensor_network.points, dtype=float), point_radii=point_radii)
        self.cycle_label = CycleLabelling(self.topology)

    def _validate_network_configuration(self) -> None:
        if getattr(self.sensor_network, "domain", None) is None:
            raise ValueError("Sensor network must include a domain for billiard reflections.")
        if not getattr(self.sensor_network, "fence_sensors", None):
            raise ValueError("Sensor network must include static fence sensors.")

        non_boundary = [sensor for sensor in self.sensor_network.fence_sensors if not sensor.boundary_flag]
        if non_boundary:
            raise ValueError("All fence sensors must be marked as boundary sensors (boundary_flag=True).")

    @staticmethod
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

    def _point_radii(self):
        if not getattr(self.sensor_network, "use_weighted_alpha", False):
            return None
        return self.sensor_network.point_radii

    def _build_topology(self, *, points, point_radii=None, topology_cache=None) -> Topology:
        points = np.asarray(points, dtype=float)

        cache_key = None
        radii_key = None
        if point_radii is not None:
            point_radii = np.asarray(point_radii, dtype=float)
            radii_key = point_radii.tobytes()

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

    def _validate_action(self, mobile_velocities) -> np.ndarray:
        velocities = np.asarray(mobile_velocities, dtype=float)
        mobile_count = len(self.sensor_network.mobile_sensors)
        if velocities.shape != (mobile_count, self.topology.dim):
            raise ValueError(
                f"Expected action shape {(mobile_count, self.topology.dim)}, got {velocities.shape}."
            )
        return velocities

    def _evaluate_elapsed(
        self,
        elapsed: float,
        mobile_velocities: Optional[np.ndarray],
        *,
        point_radii,
        topology_cache,
    ) -> TraceEvaluation:
        if mobile_velocities is None:
            mobile_velocities = np.asarray(
                [np.asarray(sensor.vel, dtype=float) for sensor in self.sensor_network.mobile_sensors],
                dtype=float,
            )

        predicted_positions = []
        predicted_velocities = []

        for sensor, velocity in zip(self.sensor_network.mobile_sensors, mobile_velocities):
            pos, vel = propagate_sensor_constant_velocity(
                sensor,
                velocity,
                elapsed,
                self.sensor_network,
                ambient_motion=self._ambient_motion,
            )
            predicted_positions.append(pos)
            predicted_velocities.append(vel)

        fence_positions = [np.asarray(sensor.pos, dtype=float) for sensor in self.sensor_network.fence_sensors]
        all_points = np.asarray(fence_positions + predicted_positions, dtype=float)
        topology = self._build_topology(points=all_points, point_radii=point_radii, topology_cache=topology_cache)

        state_change = StateChange(topology, self.topology)
        return TraceEvaluation(
            elapsed=float(elapsed),
            topology=topology,
            state_change=state_change,
            mobile_positions=np.asarray(predicted_positions, dtype=float),
            mobile_velocities=np.asarray(predicted_velocities, dtype=float),
        )

    def _commit(self, evaluation: TraceEvaluation) -> None:
        for sensor, pos, vel in zip(
            self.sensor_network.mobile_sensors,
            evaluation.mobile_positions,
            evaluation.mobile_velocities,
        ):
            sensor.pos = np.asarray(pos, dtype=float)
            sensor.vel = np.asarray(vel, dtype=float)

        self.time += evaluation.elapsed
        self.cycle_label.update(evaluation.state_change, self.time)
        self.topology = evaluation.topology
        self.sensor_network.update()

    def step_with_velocities(self, mobile_velocities, interval: Optional[float] = None) -> RLStepResult:
        """
        Advance by exactly `interval` using one fixed actor decision for the whole control step.

        The commanded velocities are applied once at the start of the outer interval.
        Topology integration is then performed adaptively: if the topology change over a
        subinterval is non-atomic, the simulator recursively bisects that subinterval until
        atomic changes can be committed safely. The actor is not queried during this internal
        subdivision, so control updates happen only once per outer `dt`.
        """

        interval = self.dt if interval is None else float(interval)
        if interval <= 0:
            raise ValueError("interval must be positive")

        validated_velocities = self._validate_action(mobile_velocities)
        point_radii = self._point_radii()
        topology_cache = {}
        initial_topology = self.topology
        diagnostics = TraceDiagnostics()
        event_found = False

        for sensor, velocity in zip(self.sensor_network.mobile_sensors, validated_velocities):
            sensor.vel = np.asarray(velocity, dtype=float)

        def commit_interval(subinterval: float, depth: int) -> None:
            nonlocal diagnostics, event_found
            diagnostics = TraceDiagnostics(
                evaluation_count=diagnostics.evaluation_count + 1,
                split_count=diagnostics.split_count,
                max_recursion_depth=max(diagnostics.max_recursion_depth, depth),
                recursion_limit_hit=diagnostics.recursion_limit_hit,
            )

            evaluation = self._evaluate_elapsed(
                subinterval,
                None,
                point_radii=point_radii,
                topology_cache=topology_cache,
            )
            state_change = evaluation.state_change

            if state_change.is_atomic_change():
                if not AtomicEventTracer._is_trivial_atomic(state_change):
                    event_found = True
                self._commit(evaluation)
                return

            if depth >= self.max_event_recursion or subinterval <= self.event_time_tolerance:
                diagnostics = TraceDiagnostics(
                    evaluation_count=diagnostics.evaluation_count,
                    split_count=diagnostics.split_count,
                    max_recursion_depth=diagnostics.max_recursion_depth,
                    recursion_limit_hit=True,
                )
                self._commit(evaluation)
                return

            diagnostics = TraceDiagnostics(
                evaluation_count=diagnostics.evaluation_count,
                split_count=diagnostics.split_count + 1,
                max_recursion_depth=diagnostics.max_recursion_depth,
                recursion_limit_hit=diagnostics.recursion_limit_hit,
            )
            half = 0.5 * subinterval
            commit_interval(half, depth + 1)
            commit_interval(half, depth + 1)

        commit_interval(interval, 0)

        final_positions = np.asarray([np.asarray(sensor.pos, dtype=float) for sensor in self.sensor_network.mobile_sensors], dtype=float)
        final_velocities = np.asarray([np.asarray(sensor.vel, dtype=float) for sensor in self.sensor_network.mobile_sensors], dtype=float)
        final_state_change = StateChange(self.topology, initial_topology)

        return RLStepResult(
            elapsed=float(interval),
            event_found=event_found,
            topology=self.topology,
            state_change=final_state_change,
            mobile_positions=final_positions,
            mobile_velocities=final_velocities,
            trace_diagnostics=diagnostics,
        )

    def has_intruder(self) -> bool:
        return self.cycle_label.has_intruder()
