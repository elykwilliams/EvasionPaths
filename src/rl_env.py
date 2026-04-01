# ******************************************************************************
#  Copyright (c) 2026, Marco Campos - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

_GYM_IMPORT_ERROR = None
try:
    from gymnasium import Env, spaces
except ModuleNotFoundError:
    try:
        from gym import Env, spaces  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        Env = object  # type: ignore
        spaces = None  # type: ignore
        _GYM_IMPORT_ERROR = exc

from rl_simulation import RLEventSimulation
from sensor_network import SensorNetwork


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights for the RL environment."""

    true_cycle_closed_reward: float = 3.0
    true_cycle_added_penalty: float = 4.0
    time_penalty: float = 0.03
    # Deprecated compatibility fields from earlier experiments; currently unused.
    time_penalty_growth: float = 0.0
    time_penalty_power: float = 1.0
    control_effort_penalty: float = 0.05
    clear_bonus: float = 12.0
    timeout_penalty: float = 12.0
    disconnection_penalty: float = 100.0
    interface_edge_loss_penalty_weight: float = 0.0
    interface_edge_stretch_penalty_weight: float = 0.0
    success_time_bonus_weight: float = 0.0
    one_hole_linger_penalty_weight: float = 0.0
    one_hole_area_scale_alpha: float = 0.0
    one_hole_perimeter_scale_alpha: float = 0.0
    # Soft nearest-neighbor spacing band around sensing radius.
    # Desired lower bound (not hard-constrained): d_nn >= min_ratio * r.
    neighbor_min_distance_ratio: float = 0.8660254037844386  # sqrt(3)/2
    # Discourage excessive spread: d_nn <= max_ratio * r.
    neighbor_max_distance_ratio: float = 2.0
    neighbor_close_penalty_weight: float = 0.0
    neighbor_far_penalty_weight: float = 0.0
    mobile_overlap_penalty_weight: float = 0.0
    hard_close_distance_ratio: float = 0.5
    hard_close_mobile_penalty_weight: float = 0.0
    hard_close_fence_penalty_weight: float = 0.0
    # Apply the same min/max ratio band to nearest fence distance as well.
    fence_close_penalty_weight: float = 0.0
    fence_far_penalty_weight: float = 0.0
    # Penalize lineage events where parents include both false and true
    # and the resulting child cycle is true (false + true -> true).
    merge_hazard_penalty_weight: float = 0.0
    # Optional geometry shaping (default-off for ablation compatibility).
    area_progress_reward_weight: float = 0.0
    perimeter_progress_reward_weight: float = 0.0
    largest_area_progress_reward_weight: float = 0.0
    largest_perimeter_progress_reward_weight: float = 0.0
    area_regress_penalty_weight: float = 0.0
    perimeter_regress_penalty_weight: float = 0.0
    largest_area_regress_penalty_weight: float = 0.0
    largest_perimeter_regress_penalty_weight: float = 0.0
    area_residual_penalty_weight: float = 0.0
    perimeter_residual_penalty_weight: float = 0.0
    largest_area_residual_penalty_weight: float = 0.0
    largest_perimeter_residual_penalty_weight: float = 0.0


@dataclass(frozen=True)
class PhaseRewardMultipliers:
    """Per-phase reward multipliers applied to the base RewardConfig."""

    true_cycle_closed_reward: float = 1.0
    true_cycle_added_penalty: float = 1.0
    time_penalty: float = 1.0
    control_effort_penalty: float = 1.0
    clear_bonus: float = 1.0
    timeout_penalty: float = 1.0
    disconnection_penalty: float = 1.0
    interface_edge_loss_penalty_weight: float = 1.0
    interface_edge_stretch_penalty_weight: float = 1.0
    success_time_bonus_weight: float = 1.0
    one_hole_linger_penalty_weight: float = 1.0
    one_hole_area_scale_alpha: float = 1.0
    one_hole_perimeter_scale_alpha: float = 1.0
    neighbor_close_penalty_weight: float = 1.0
    neighbor_far_penalty_weight: float = 1.0
    mobile_overlap_penalty_weight: float = 1.0
    hard_close_mobile_penalty_weight: float = 1.0
    hard_close_fence_penalty_weight: float = 1.0
    fence_close_penalty_weight: float = 1.0
    fence_far_penalty_weight: float = 1.0
    merge_hazard_penalty_weight: float = 1.0
    area_progress_reward_weight: float = 1.0
    perimeter_progress_reward_weight: float = 1.0
    largest_area_progress_reward_weight: float = 1.0
    largest_perimeter_progress_reward_weight: float = 1.0
    area_regress_penalty_weight: float = 1.0
    perimeter_regress_penalty_weight: float = 1.0
    largest_area_regress_penalty_weight: float = 1.0
    largest_perimeter_regress_penalty_weight: float = 1.0
    area_residual_penalty_weight: float = 1.0
    perimeter_residual_penalty_weight: float = 1.0
    largest_area_residual_penalty_weight: float = 1.0
    largest_perimeter_residual_penalty_weight: float = 1.0


@dataclass(frozen=True)
class PhaseRewardSchedule:
    """State-dependent reward multipliers keyed by interior true-cycle count."""

    simplify: PhaseRewardMultipliers = field(default_factory=PhaseRewardMultipliers)
    consolidate: PhaseRewardMultipliers = field(default_factory=PhaseRewardMultipliers)
    compress: PhaseRewardMultipliers = field(default_factory=PhaseRewardMultipliers)


@dataclass(frozen=True)
class EventLogRecord:
    """Structured log payload for one environment step."""

    episode_index: int
    step_index: int
    sim_time: float
    event_time: float
    event_found: bool
    alpha_change: Tuple
    boundary_change: Tuple
    trace_evaluation_count: int
    trace_split_count: int
    trace_max_recursion_depth: int
    trace_recursion_limit_hit: bool
    true_cycle_count_before: int
    true_cycle_count_after: int
    true_cycles_closed: int
    true_cycles_added: int
    reward: float
    terminated: bool
    truncated: bool

    def as_dict(self) -> Dict:
        return {
            "episode_index": self.episode_index,
            "step_index": self.step_index,
            "sim_time": self.sim_time,
            "event_time": self.event_time,
            "event_found": self.event_found,
            "alpha_change": list(self.alpha_change),
            "boundary_change": list(self.boundary_change),
            "trace_evaluation_count": self.trace_evaluation_count,
            "trace_split_count": self.trace_split_count,
            "trace_max_recursion_depth": self.trace_max_recursion_depth,
            "trace_recursion_limit_hit": self.trace_recursion_limit_hit,
            "true_cycle_count_before": self.true_cycle_count_before,
            "true_cycle_count_after": self.true_cycle_count_after,
            "true_cycles_closed": self.true_cycles_closed,
            "true_cycles_added": self.true_cycles_added,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


class RLEvasionEnv(Env):
    """
    Gymnasium environment wrapper around RLEventSimulation.

    One environment step applies one velocity action for all mobile sensors,
    then advances the simulator to the earliest non-trivial atomic event
    (or to full dt when no event is detected).
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        *,
        dt: float,
        sensor_network_builder: Optional[Callable[[], SensorNetwork]] = None,
        simulation_builder: Optional[Callable[[], RLEventSimulation]] = None,
        state_mode: str = "vector",
        max_speed: Optional[float] = None,
        max_speed_scale: float = 0.1,
        end_time: float = 0.0,
        max_steps: Optional[int] = None,
        coordinate_free: bool = True,
        enforce_2d: bool = True,
        enable_event_logging: bool = True,
        event_log_path: Optional[str] = None,
        reward_config: Optional[RewardConfig] = None,
        phase_reward_schedule: Optional[PhaseRewardSchedule] = None,
        sim_kwargs: Optional[Dict] = None,
    ) -> None:
        if spaces is None:  # pragma: no cover
            raise ModuleNotFoundError(
                "RLEvasionEnv requires gymnasium (preferred) or gym. "
                "Install with: pip install gymnasium"
            ) from _GYM_IMPORT_ERROR

        if simulation_builder is None and sensor_network_builder is None:
            raise ValueError("Provide either simulation_builder or sensor_network_builder.")

        self.dt = float(dt)
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        self._user_max_speed = max_speed
        self.max_speed_scale = float(max_speed_scale)
        if self.max_speed_scale <= 0:
            raise ValueError("max_speed_scale must be positive.")
        self.max_speed = 0.0
        self.end_time = float(end_time)
        self.max_steps = max_steps
        self.state_mode = str(state_mode).lower()
        if self.state_mode not in {"vector", "graph", "cycle_graph"}:
            raise ValueError("state_mode must be one of: 'vector', 'graph', 'cycle_graph'.")
        self.coordinate_free = bool(coordinate_free)
        self.enforce_2d = bool(enforce_2d)
        self.enable_event_logging = bool(enable_event_logging)
        self.event_log_path = event_log_path
        self.reward_config = reward_config or RewardConfig()
        self.phase_reward_schedule = phase_reward_schedule or self._default_phase_reward_schedule()
        self.sim_kwargs = {} if sim_kwargs is None else dict(sim_kwargs)

        self._sensor_network_builder = sensor_network_builder
        self._simulation_builder = simulation_builder
        self._episode_index = 0
        self.event_log = []

        self._simulation = self._build_simulation()
        self._step_count = 0
        self._one_hole_entry_area_norm: Optional[float] = None
        self._one_hole_entry_perimeter_norm: Optional[float] = None
        self._one_hole_entry_time: Optional[float] = None

        self.max_speed = self._resolve_max_speed(self._simulation)
        self._set_action_space(self._simulation)
        self.observation_space = self._build_observation_space(self._simulation)

    @property
    def simulation(self) -> RLEventSimulation:
        if self._simulation is None:
            raise RuntimeError("Environment is closed.")
        return self._simulation

    @staticmethod
    def _default_phase_reward_schedule() -> PhaseRewardSchedule:
        return PhaseRewardSchedule()

    def _build_simulation(self) -> RLEventSimulation:
        if self._simulation_builder is not None:
            return self._simulation_builder()

        if self._sensor_network_builder is None:
            raise RuntimeError("No simulation builder available.")

        sensor_network = self._sensor_network_builder()
        return RLEventSimulation(
            sensor_network=sensor_network,
            dt=self.dt,
            end_time=self.end_time,
            **self.sim_kwargs,
        )

    def _append_event_log_file(self, record: EventLogRecord) -> None:
        if self.event_log_path is None:
            return
        path = Path(self.event_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.as_dict(), sort_keys=True))
            handle.write("\n")

    def _build_observation_space(self, simulation: RLEventSimulation):
        if self.state_mode in {"graph", "cycle_graph"}:
            total_nodes = len(simulation.sensor_network.fence_sensors) + len(simulation.sensor_network.mobile_sensors)
            payload = {
                "edge_x": spaces.Box(low=-np.inf, high=np.inf, shape=(total_nodes, total_nodes, 5), dtype=np.float32),
                "edge_mask": spaces.Box(low=0.0, high=1.0, shape=(total_nodes, total_nodes), dtype=np.float32),
            }
            if self.state_mode == "graph":
                payload["node_x"] = spaces.Box(low=-np.inf, high=np.inf, shape=(total_nodes, 7), dtype=np.float32)
                payload["global_x"] = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
            else:
                payload["node_x"] = spaces.Box(low=-np.inf, high=np.inf, shape=(total_nodes, 8), dtype=np.float32)
                payload["global_x"] = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
                payload["cycle_node_index"] = spaces.Box(
                    low=-1,
                    high=max(total_nodes - 1, 0),
                    shape=(total_nodes, total_nodes),
                    dtype=np.int64,
                )
                payload["cycle_node_mask"] = spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(total_nodes, total_nodes),
                    dtype=np.float32,
                )
                payload["cycle_mask"] = spaces.Box(low=0.0, high=1.0, shape=(total_nodes,), dtype=np.float32)
                payload["cycle_is_true"] = spaces.Box(low=0.0, high=1.0, shape=(total_nodes,), dtype=np.float32)
                payload["node_cycle_token_x"] = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(total_nodes, total_nodes, 7),
                    dtype=np.float32,
                )
                payload["node_cycle_token_mask"] = spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(total_nodes, total_nodes),
                    dtype=np.float32,
                )
            return spaces.Dict(payload)

        obs = self._observation_vector(simulation)
        return spaces.Box(
            low=np.full(obs.shape, -np.inf, dtype=np.float32),
            high=np.full(obs.shape, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    @staticmethod
    def _true_cycle_count(simulation: RLEventSimulation) -> int:
        return int(sum(1 for _, value in simulation.cycle_label.label.items() if bool(value)))

    @classmethod
    def _interior_true_cycle_count(cls, simulation: RLEventSimulation) -> int:
        labels = simulation.cycle_label.label
        try:
            outer_cycle = simulation.topology.outer_cycle
            return int(sum(1 for cycle, value in labels.items() if bool(value) and cycle != outer_cycle))
        except Exception:
            return cls._true_cycle_count(simulation)

    @classmethod
    def _interior_true_cycles(cls, simulation: RLEventSimulation):
        labels = simulation.cycle_label.label
        try:
            outer_cycle = simulation.topology.outer_cycle
            return [cycle for cycle, value in labels.items() if bool(value) and cycle != outer_cycle]
        except Exception:
            return [cycle for cycle, value in labels.items() if bool(value)]

    @staticmethod
    def _phase_name_from_true_cycle_count(true_cycle_count: int) -> str:
        if int(true_cycle_count) >= 3:
            return "simplify"
        if int(true_cycle_count) == 2:
            return "consolidate"
        return "compress"

    def _phase_one_hot(self, true_cycle_count: int) -> np.ndarray:
        phase = self._phase_name_from_true_cycle_count(true_cycle_count)
        if phase == "simplify":
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if phase == "consolidate":
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def _phase_weighted_reward_config(self, true_cycle_count: int) -> RewardConfig:
        phase = self._phase_name_from_true_cycle_count(true_cycle_count)
        multipliers = getattr(self.phase_reward_schedule, phase)
        weighted = {}
        for field_name in RewardConfig.__dataclass_fields__:
            base_value = getattr(self.reward_config, field_name)
            multiplier = getattr(multipliers, field_name, 1.0)
            weighted[field_name] = base_value * multiplier
        return RewardConfig(**weighted)

    def _update_one_hole_reference(
        self,
        *,
        true_cycle_count: int,
        area_current_norm: float,
        perimeter_current_norm: float,
    ) -> None:
        if int(true_cycle_count) == 1:
            if self._one_hole_entry_area_norm is None:
                self._one_hole_entry_area_norm = float(area_current_norm)
                self._one_hole_entry_perimeter_norm = float(perimeter_current_norm)
                self._one_hole_entry_time = float(self.simulation.time)
            return
        self._one_hole_entry_area_norm = None
        self._one_hole_entry_perimeter_norm = None
        self._one_hole_entry_time = None

    def _one_hole_dwell_time_norm(self, simulation: RLEventSimulation) -> float:
        if self._one_hole_entry_time is None:
            return 0.0
        horizon = float(simulation.Tend) if getattr(simulation, "Tend", 0.0) > 0 else max(self.dt, 1.0)
        return max(0.0, float(simulation.time) - float(self._one_hole_entry_time)) / max(horizon, 1e-8)

    def _resolve_max_speed(self, simulation: RLEventSimulation) -> float:
        if self._user_max_speed is not None:
            max_speed = float(self._user_max_speed)
            if max_speed <= 0:
                raise ValueError("max_speed must be positive.")
            return max_speed

        sensing_radius = getattr(simulation.sensor_network, "sensing_radius", None)
        if sensing_radius is None:
            raise ValueError("Unable to derive max_speed: sensor_network.sensing_radius is missing.")
        max_speed = self.max_speed_scale * float(sensing_radius) / self.dt
        if max_speed <= 0:
            raise ValueError("Derived max_speed must be positive; check sensing_radius and dt.")
        return max_speed

    def _set_action_space(self, simulation: RLEventSimulation) -> None:
        mobile_count = len(simulation.sensor_network.mobile_sensors)
        dim = simulation.topology.dim
        if self.enforce_2d and dim != 2:
            raise ValueError(f"RLEvasionEnv currently enforces 2D; got dim={dim}.")

        action_shape = (mobile_count, dim)
        max_val = np.float32(self.max_speed)
        self.action_space = spaces.Box(low=-max_val, high=max_val, shape=action_shape, dtype=np.float32)

    def _postprocess_action(self, action: np.ndarray) -> np.ndarray:
        # Per-sensor radial normalization preserves direction while enforcing ||v_i|| <= max_speed.
        norms = np.linalg.norm(action, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.max_speed / np.maximum(norms, 1e-8))
        return (action * scale).astype(np.float32)

    @staticmethod
    def _simplex_nodes(simplex) -> Tuple[int, ...]:
        if hasattr(simplex, "nodes"):
            return tuple(simplex.nodes)
        return tuple(simplex)

    def _cycle_node_dart_sets(
        self,
        simulation: RLEventSimulation,
    ) -> Tuple[Set[int], Set[int], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        labels = simulation.cycle_label.label
        try:
            outer_cycle = simulation.topology.outer_cycle
        except Exception:
            outer_cycle = None

        true_nodes: Set[int] = set()
        false_nodes: Set[int] = set()
        true_darts: Set[Tuple[int, int]] = set()
        false_darts: Set[Tuple[int, int]] = set()

        for cycle, value in labels.items():
            if outer_cycle is not None and cycle == outer_cycle:
                continue

            cycle_is_true = bool(value)
            extracted_directed = False

            try:
                iterator = iter(cycle)
            except TypeError:
                iterator = None

            if iterator is not None:
                for face in iterator:
                    try:
                        nodes = tuple(int(node) for node in self._simplex_nodes(face))
                    except Exception:
                        continue

                    if cycle_is_true:
                        true_nodes.update(nodes)
                    else:
                        false_nodes.update(nodes)

                    if len(nodes) == 2 and nodes[0] != nodes[1]:
                        extracted_directed = True
                        dart = (nodes[0], nodes[1])
                        if cycle_is_true:
                            true_darts.add(dart)
                        else:
                            false_darts.add(dart)

            if extracted_directed:
                continue

            # Fallback path: recover directed traversal from combinatorial map when available.
            try:
                simplex = next(iter(cycle))
                node_order = list(simulation.topology.cmap.get_cycle_nodes(simplex))
            except Exception:
                node_order = []

            if len(node_order) < 2:
                continue

            if cycle_is_true:
                true_nodes.update(int(node) for node in node_order)
            else:
                false_nodes.update(int(node) for node in node_order)

            for idx in range(len(node_order)):
                u = int(node_order[idx])
                v = int(node_order[(idx + 1) % len(node_order)])
                if u == v:
                    continue
                dart = (u, v)
                if cycle_is_true:
                    true_darts.add(dart)
                else:
                    false_darts.add(dart)

        return true_nodes, false_nodes, true_darts, false_darts

    def _ordered_cycle_index_tensors(
        self,
        simulation: RLEventSimulation,
        total_nodes: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cycle_index = np.full((total_nodes, total_nodes), -1, dtype=np.int64)
        cycle_node_mask = np.zeros((total_nodes, total_nodes), dtype=np.float32)
        cycle_mask = np.zeros((total_nodes,), dtype=np.float32)
        cycle_is_true = np.zeros((total_nodes,), dtype=np.float32)

        labels = simulation.cycle_label.label
        try:
            outer_cycle = simulation.topology.outer_cycle
        except Exception:
            outer_cycle = None
        row = 0
        for cycle, value in labels.items():
            if outer_cycle is not None and cycle == outer_cycle:
                continue
            if row >= total_nodes:
                break
            edge_set: Set[Tuple[int, int]] = set()
            try:
                iterator = iter(cycle)
            except TypeError:
                iterator = None
            if iterator is not None:
                for face in iterator:
                    nodes = tuple(int(node) for node in self._simplex_nodes(face))
                    if len(nodes) == 2 and nodes[0] != nodes[1]:
                        edge_set.add((nodes[0], nodes[1]))
            ordered_nodes = self._ordered_cycle_nodes_from_edges(edge_set)
            if ordered_nodes is None and hasattr(cycle, "nodes"):
                ordered_nodes = tuple(int(node) for node in cycle.nodes)
            if not ordered_nodes:
                continue
            count = min(len(ordered_nodes), total_nodes)
            cycle_index[row, :count] = np.asarray(ordered_nodes[:count], dtype=np.int64)
            cycle_node_mask[row, :count] = 1.0
            cycle_mask[row] = 1.0
            cycle_is_true[row] = 1.0 if bool(value) else 0.0
            row += 1
        return cycle_index, cycle_node_mask, cycle_mask, cycle_is_true

    @staticmethod
    def _rotate_clockwise(vec: np.ndarray) -> np.ndarray:
        return np.array([vec[1], -vec[0]], dtype=float)

    def _node_cycle_token_tensors(
        self,
        simulation: RLEventSimulation,
        total_nodes: int,
        positions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        token_x = np.zeros((total_nodes, total_nodes, 7), dtype=np.float32)
        token_mask = np.zeros((total_nodes, total_nodes), dtype=np.float32)
        counts = np.zeros((total_nodes,), dtype=np.int64)
        eps = 1e-12

        labels = simulation.cycle_label.label
        try:
            outer_cycle = simulation.topology.outer_cycle
        except Exception:
            outer_cycle = None

        for cycle, value in labels.items():
            if outer_cycle is not None and cycle == outer_cycle:
                continue

            edge_set: Set[Tuple[int, int]] = set()
            try:
                iterator = iter(cycle)
            except TypeError:
                iterator = None

            if iterator is not None:
                for face in iterator:
                    nodes = tuple(int(node) for node in self._simplex_nodes(face))
                    if len(nodes) == 2 and nodes[0] != nodes[1]:
                        edge_set.add((nodes[0], nodes[1]))

            ordered_nodes = self._ordered_cycle_nodes_from_edges(edge_set)
            if ordered_nodes is None and hasattr(cycle, "nodes"):
                ordered_nodes = tuple(int(node) for node in cycle.nodes)
            if not ordered_nodes:
                continue

            nodes = list(ordered_nodes)
            if len(nodes) > 1 and nodes[0] == nodes[-1]:
                nodes = nodes[:-1]
            if len(nodes) < 3:
                continue

            size_feat = float(np.log1p(len(nodes)))
            is_true = 1.0 if bool(value) else 0.0
            is_false = 1.0 - is_true

            for idx, node in enumerate(nodes):
                prev_node = nodes[(idx - 1) % len(nodes)]
                next_node = nodes[(idx + 1) % len(nodes)]
                slot = int(counts[node])
                if slot >= total_nodes:
                    continue

                d_in = positions[node] - positions[prev_node]
                d_out = positions[next_node] - positions[node]
                inward = -0.5 * self._rotate_clockwise(d_in + d_out)

                in_norm = float(np.linalg.norm(d_in))
                out_norm = float(np.linalg.norm(d_out))
                denom = max(in_norm * out_norm, eps)
                cos_turn = float(np.dot(d_in, d_out) / denom)
                sin_turn = float((d_in[0] * d_out[1] - d_in[1] * d_out[0]) / denom)

                token_x[node, slot] = np.array(
                    [
                        float(inward[0]),
                        float(inward[1]),
                        cos_turn,
                        sin_turn,
                        size_feat,
                        is_true,
                        is_false,
                    ],
                    dtype=np.float32,
                )
                token_mask[node, slot] = 1.0
                counts[node] += 1

        return token_x, token_mask

    def _graph_observation_dict(self, simulation: RLEventSimulation, *, include_cycle_context: bool = False) -> Dict[str, np.ndarray]:
        fence_sensors = list(simulation.sensor_network.fence_sensors)
        mobile_sensors = list(simulation.sensor_network.mobile_sensors)
        fence_count = len(fence_sensors)
        sensors = fence_sensors + mobile_sensors
        total_nodes = len(sensors)
        node_dim = 8 if include_cycle_context else 7
        node_x = np.zeros((total_nodes, node_dim), dtype=np.float32)
        edge_x = np.zeros((total_nodes, total_nodes, 5), dtype=np.float32)
        edge_mask = np.zeros((total_nodes, total_nodes), dtype=np.float32)

        true_nodes, false_nodes, true_darts, false_darts = self._cycle_node_dart_sets(simulation)
        positions = np.asarray([sensor.pos for sensor in sensors], dtype=float)
        max_incident_scale = float(max(1, 2 * max(1, total_nodes - 1)))

        for i, sensor in enumerate(sensors):
            is_fence = float(i < fence_count)
            is_mobile = 1.0 - is_fence
            on_true_cycle = float(i in true_nodes)
            on_false_cycle = float(i in false_nodes)
            incident_true_darts = float(sum(1 for u, v in true_darts if u == i or v == i)) / max_incident_scale
            incident_false_darts = float(sum(1 for u, v in false_darts if u == i or v == i)) / max_incident_scale
            if include_cycle_context:
                vel = np.asarray(sensor.vel, dtype=float)
                node_x[i] = np.array(
                    [
                        is_fence,
                        is_mobile,
                        float(vel[0]) if vel.size > 0 else 0.0,
                        float(vel[1]) if vel.size > 1 else 0.0,
                        on_true_cycle,
                        on_false_cycle,
                        incident_true_darts,
                        incident_false_darts,
                    ],
                    dtype=np.float32,
                )
            else:
                speed_mag = float(np.linalg.norm(sensor.vel))
                node_x[i] = np.array(
                    [
                        is_fence,
                        is_mobile,
                        speed_mag,
                        on_true_cycle,
                        on_false_cycle,
                        incident_true_darts,
                        incident_false_darts,
                    ],
                    dtype=np.float32,
                )

        for simplex in simulation.topology.simplices(1):
            nodes = self._simplex_nodes(simplex)
            if len(nodes) != 2:
                continue
            i, j = int(nodes[0]), int(nodes[1])
            if i == j:
                continue

            edge_mask[i, j] = 1.0
            edge_mask[j, i] = 1.0

            disp_ij = positions[j] - positions[i]
            dist_ij = float(np.linalg.norm(disp_ij))
            if dist_ij > 1e-12:
                direction_ij = disp_ij / dist_ij
            else:
                direction_ij = np.zeros(2, dtype=float)
            edge_x[i, j] = np.array(
                [
                    dist_ij,
                    float(direction_ij[0]),
                    float(direction_ij[1]),
                    float((i, j) in true_darts),
                    float((i, j) in false_darts),
                ],
                dtype=np.float32,
            )

            disp_ji = -disp_ij
            dist_ji = dist_ij
            if dist_ji > 1e-12:
                direction_ji = disp_ji / dist_ji
            else:
                direction_ji = np.zeros(2, dtype=float)
            edge_x[j, i] = np.array(
                [
                    dist_ji,
                    float(direction_ji[0]),
                    float(direction_ji[1]),
                    float((j, i) in true_darts),
                    float((j, i) in false_darts),
                ],
                dtype=np.float32,
            )

        if simulation.Tend > 0:
            time_norm = float(simulation.time / simulation.Tend)
        else:
            time_norm = float(simulation.time / max(self.dt, 1.0))
        true_cycle_count = self._interior_true_cycle_count(simulation)
        if include_cycle_context:
            global_x = np.concatenate(
                [
                    np.array(
                        [
                            time_norm,
                            float(true_cycle_count),
                            float(self._one_hole_dwell_time_norm(simulation)),
                        ],
                        dtype=np.float32,
                    ),
                    self._phase_one_hot(true_cycle_count),
                ]
            ).astype(np.float32)
            cycle_index, cycle_node_mask, cycle_mask, cycle_is_true = self._ordered_cycle_index_tensors(simulation, total_nodes)
            node_cycle_token_x, node_cycle_token_mask = self._node_cycle_token_tensors(simulation, total_nodes, positions)
            return {
                "node_x": node_x,
                "edge_x": edge_x,
                "edge_mask": edge_mask,
                "global_x": global_x,
                "cycle_node_index": cycle_index,
                "cycle_node_mask": cycle_node_mask,
                "cycle_mask": cycle_mask,
                "cycle_is_true": cycle_is_true,
                "node_cycle_token_x": node_cycle_token_x,
                "node_cycle_token_mask": node_cycle_token_mask,
            }

        global_x = np.concatenate(
            [
                np.array([time_norm, float(true_cycle_count)], dtype=np.float32),
                self._phase_one_hot(true_cycle_count),
            ]
        ).astype(np.float32)

        return {
            "node_x": node_x,
            "edge_x": edge_x,
            "edge_mask": edge_mask,
            "global_x": global_x,
        }

    def _domain_area_perimeter_scales(self, simulation: RLEventSimulation) -> Tuple[float, float]:
        domain = getattr(simulation.sensor_network, "domain", None)
        if domain is None:
            return 1.0, 1.0
        if hasattr(domain, "min") and hasattr(domain, "max"):
            lo = np.asarray(domain.min, dtype=float)
            hi = np.asarray(domain.max, dtype=float)
            if lo.shape == hi.shape and lo.size == 2:
                extent = np.maximum(hi - lo, 1e-12)
                area_scale = float(extent[0] * extent[1])
                perimeter_scale = float(2.0 * (extent[0] + extent[1]))
                return max(area_scale, 1e-12), max(perimeter_scale, 1e-12)
        # Conservative fallback; keeps terms numerically bounded.
        return 1.0, 1.0

    @staticmethod
    def _ordered_cycle_nodes_from_edges(edges: Set[Tuple[int, int]]) -> Optional[Tuple[int, ...]]:
        if len(edges) < 3:
            return None
        adjacency: Dict[int, Set[int]] = {}
        for i, j in edges:
            if i == j:
                continue
            adjacency.setdefault(i, set()).add(j)
            adjacency.setdefault(j, set()).add(i)
        if len(adjacency) < 3:
            return None
        if any(len(neighbors) != 2 for neighbors in adjacency.values()):
            return None

        start = min(adjacency)
        prev = None
        current = start
        ordered = [start]
        for _ in range(len(adjacency) + 1):
            neighbors = adjacency[current]
            candidates = [node for node in neighbors if node != prev]
            if not candidates:
                return None
            nxt = min(candidates)
            if nxt == start:
                if len(ordered) >= 3 and len(ordered) == len(adjacency):
                    return tuple(ordered)
                return None
            if nxt in ordered:
                return None
            ordered.append(nxt)
            prev, current = current, nxt
        return None

    def _true_cycle_area_perimeter(self, simulation: RLEventSimulation) -> Tuple[float, float]:
        sensors = list(simulation.sensor_network.fence_sensors) + list(simulation.sensor_network.mobile_sensors)
        positions = np.asarray([sensor.pos for sensor in sensors], dtype=float)
        total_area = 0.0
        total_perimeter = 0.0

        for cycle in self._interior_true_cycles(simulation):
            edge_set: Set[Tuple[int, int]] = set()
            try:
                iterator = iter(cycle)
            except TypeError:
                continue
            for face in iterator:
                try:
                    nodes = tuple(int(node) for node in self._simplex_nodes(face))
                except Exception:
                    continue
                if len(nodes) != 2:
                    continue
                i, j = nodes
                if i == j:
                    continue
                edge_set.add((min(i, j), max(i, j)))

            ordered_nodes = self._ordered_cycle_nodes_from_edges(edge_set)
            if ordered_nodes is None:
                continue
            if any(node < 0 or node >= len(positions) for node in ordered_nodes):
                continue

            poly = positions[list(ordered_nodes)]
            shifted = np.roll(poly, -1, axis=0)
            perimeter = float(np.sum(np.linalg.norm(shifted - poly, axis=1)))
            area = float(
                0.5
                * np.abs(
                    np.sum(poly[:, 0] * shifted[:, 1] - shifted[:, 0] * poly[:, 1])
                )
            )
            total_perimeter += perimeter
            total_area += area

        return total_area, total_perimeter

    def _largest_true_cycle_area_perimeter(self, simulation: RLEventSimulation) -> Tuple[float, float]:
        sensors = list(simulation.sensor_network.fence_sensors) + list(simulation.sensor_network.mobile_sensors)
        positions = np.asarray([sensor.pos for sensor in sensors], dtype=float)
        largest_area = 0.0
        largest_perimeter = 0.0

        for cycle in self._interior_true_cycles(simulation):
            edge_set: Set[Tuple[int, int]] = set()
            try:
                iterator = iter(cycle)
            except TypeError:
                continue
            for face in iterator:
                try:
                    nodes = tuple(int(node) for node in self._simplex_nodes(face))
                except Exception:
                    continue
                if len(nodes) != 2:
                    continue
                i, j = nodes
                if i == j:
                    continue
                edge_set.add((min(i, j), max(i, j)))

            ordered_nodes = self._ordered_cycle_nodes_from_edges(edge_set)
            if ordered_nodes is None:
                continue
            if any(node < 0 or node >= len(positions) for node in ordered_nodes):
                continue

            poly = positions[list(ordered_nodes)]
            shifted = np.roll(poly, -1, axis=0)
            perimeter = float(np.sum(np.linalg.norm(shifted - poly, axis=1)))
            area = float(
                0.5
                * np.abs(
                    np.sum(poly[:, 0] * shifted[:, 1] - shifted[:, 0] * poly[:, 1])
                )
            )
            if area > largest_area:
                largest_area = area
                largest_perimeter = perimeter

        return largest_area, largest_perimeter

    def _topology_summary_features(self, simulation: RLEventSimulation) -> np.ndarray:
        simplices_per_dim = np.asarray(
            [len(simulation.topology.simplices(k)) for k in range(1, simulation.topology.dim + 1)],
            dtype=float,
        )

        homology_generators = simulation.topology.homology_generators
        connected_homology = float(sum(1 for cycle in homology_generators if simulation.topology.is_connected_cycle(cycle)))

        summary = np.asarray(
            [
                float(simulation.time),
                float(len(simulation.topology.boundary_cycles)),
                float(len(homology_generators)),
                connected_homology,
                float(self._interior_true_cycle_count(simulation)),
            ],
            dtype=float,
        )
        phase = self._phase_one_hot(int(summary[-1]))
        return np.concatenate([simplices_per_dim, summary, phase]).astype(np.float32)

    def _coordinate_free_observation_vector(self, simulation: RLEventSimulation) -> np.ndarray:
        mobile_sensors = simulation.sensor_network.mobile_sensors
        mobile_count = len(mobile_sensors)
        fence_count = len(simulation.sensor_network.fence_sensors)

        speed_magnitudes = np.asarray([np.linalg.norm(sensor.vel) for sensor in mobile_sensors], dtype=float)
        edge_degree = np.zeros(mobile_count, dtype=float)
        triangle_incidence = np.zeros(mobile_count, dtype=float)

        for edge in simulation.topology.simplices(1):
            for node in self._simplex_nodes(edge):
                mobile_idx = node - fence_count
                if 0 <= mobile_idx < mobile_count:
                    edge_degree[mobile_idx] += 1.0

        if simulation.topology.dim >= 2:
            for triangle in simulation.topology.simplices(2):
                for node in self._simplex_nodes(triangle):
                    mobile_idx = node - fence_count
                    if 0 <= mobile_idx < mobile_count:
                        triangle_incidence[mobile_idx] += 1.0

        summary = self._topology_summary_features(simulation)
        return np.concatenate([speed_magnitudes, edge_degree, triangle_incidence, summary]).astype(np.float32)

    def _coordinate_observation_vector(self, simulation: RLEventSimulation) -> np.ndarray:
        mobile_positions = np.asarray([s.pos for s in simulation.sensor_network.mobile_sensors], dtype=float).reshape(-1)
        mobile_velocities = np.asarray([s.vel for s in simulation.sensor_network.mobile_sensors], dtype=float).reshape(-1)
        summary = self._topology_summary_features(simulation)
        return np.concatenate([mobile_positions, mobile_velocities, summary]).astype(np.float32)

    def _observation_vector(self, simulation: RLEventSimulation) -> np.ndarray:
        if self.coordinate_free:
            return self._coordinate_free_observation_vector(simulation)
        return self._coordinate_observation_vector(simulation)

    def _observation(self, simulation: RLEventSimulation):
        if self.state_mode == "graph":
            return self._graph_observation_dict(simulation)
        if self.state_mode == "cycle_graph":
            return self._graph_observation_dict(simulation, include_cycle_context=True)
        return self._observation_vector(simulation)

    def _reward(
        self,
        phase_true_cycle_count: int,
        true_cycles_closed: int,
        true_cycles_added: int,
        merge_hazard_count: int,
        interface_edge_loss_count: int,
        interface_edge_stretch: float,
        elapsed: float,
        action: np.ndarray,
        neighbor_close_violation: float,
        neighbor_far_violation: float,
        mobile_overlap_count: int,
        hard_close_mobile_violation: float,
        fence_close_violation: float,
        fence_far_violation: float,
        hard_close_fence_violation: float,
        area_delta_norm: float,
        perimeter_delta_norm: float,
        largest_area_delta_norm: float,
        largest_perimeter_delta_norm: float,
        area_current_norm: float,
        perimeter_current_norm: float,
        largest_area_current_norm: float,
        largest_perimeter_current_norm: float,
        final_time: float,
        cleared: bool,
        timed_out: bool,
        disconnected: bool,
    ) -> float:
        phase_config = self._phase_weighted_reward_config(phase_true_cycle_count)
        effort = float(np.mean(np.sum(action * action, axis=1))) if action.size else 0.0
        area_regress = max(0.0, -float(area_delta_norm))
        perimeter_regress = max(0.0, -float(perimeter_delta_norm))
        largest_area_regress = max(0.0, -float(largest_area_delta_norm))
        largest_perimeter_regress = max(0.0, -float(largest_perimeter_delta_norm))
        one_hole_area_scale = 1.0
        one_hole_perimeter_scale = 1.0
        if int(phase_true_cycle_count) == 1:
            if self._one_hole_entry_area_norm is not None and self._one_hole_entry_area_norm > 1e-8:
                area_ratio = float(area_current_norm) / float(self._one_hole_entry_area_norm)
                one_hole_area_scale = 1.0 + phase_config.one_hole_area_scale_alpha * max(area_ratio, 0.0)
            if self._one_hole_entry_perimeter_norm is not None and self._one_hole_entry_perimeter_norm > 1e-8:
                perimeter_ratio = float(perimeter_current_norm) / float(self._one_hole_entry_perimeter_norm)
                one_hole_perimeter_scale = 1.0 + phase_config.one_hole_perimeter_scale_alpha * max(perimeter_ratio, 0.0)

        reward = phase_config.true_cycle_closed_reward * float(true_cycles_closed)
        reward -= phase_config.true_cycle_added_penalty * float(true_cycles_added)
        reward -= phase_config.merge_hazard_penalty_weight * float(merge_hazard_count)
        reward -= phase_config.interface_edge_loss_penalty_weight * float(interface_edge_loss_count)
        reward -= phase_config.interface_edge_stretch_penalty_weight * float(interface_edge_stretch)
        reward -= phase_config.time_penalty * float(elapsed)
        reward -= phase_config.control_effort_penalty * effort
        reward -= phase_config.neighbor_close_penalty_weight * float(neighbor_close_violation)
        reward -= phase_config.neighbor_far_penalty_weight * float(neighbor_far_violation)
        reward -= phase_config.mobile_overlap_penalty_weight * float(mobile_overlap_count)
        reward -= phase_config.hard_close_mobile_penalty_weight * float(hard_close_mobile_violation)
        reward -= phase_config.fence_close_penalty_weight * float(fence_close_violation)
        reward -= phase_config.fence_far_penalty_weight * float(fence_far_violation)
        reward -= phase_config.hard_close_fence_penalty_weight * float(hard_close_fence_violation)
        reward += phase_config.area_progress_reward_weight * one_hole_area_scale * float(area_delta_norm)
        reward += phase_config.perimeter_progress_reward_weight * one_hole_perimeter_scale * float(perimeter_delta_norm)
        reward += phase_config.largest_area_progress_reward_weight * float(largest_area_delta_norm)
        reward += phase_config.largest_perimeter_progress_reward_weight * float(largest_perimeter_delta_norm)
        reward -= phase_config.area_regress_penalty_weight * one_hole_area_scale * float(area_regress)
        reward -= phase_config.perimeter_regress_penalty_weight * one_hole_perimeter_scale * float(perimeter_regress)
        reward -= phase_config.largest_area_regress_penalty_weight * float(largest_area_regress)
        reward -= phase_config.largest_perimeter_regress_penalty_weight * float(largest_perimeter_regress)
        reward -= phase_config.area_residual_penalty_weight * float(area_current_norm)
        reward -= phase_config.perimeter_residual_penalty_weight * float(perimeter_current_norm)
        reward -= phase_config.largest_area_residual_penalty_weight * float(largest_area_current_norm)
        reward -= phase_config.largest_perimeter_residual_penalty_weight * float(largest_perimeter_current_norm)
        if int(phase_true_cycle_count) == 1 and not cleared:
            reward -= phase_config.one_hole_linger_penalty_weight * float(elapsed)
        if cleared:
            reward += phase_config.clear_bonus
            reward += phase_config.success_time_bonus_weight * (1.0 / max(float(final_time), 1e-8))
        if disconnected:
            reward -= phase_config.disconnection_penalty
        if timed_out and not cleared:
            reward -= phase_config.timeout_penalty
        return float(reward)

    def _reward_terms(
        self,
        *,
        phase_true_cycle_count: int,
        true_cycles_closed: int,
        true_cycles_added: int,
        merge_hazard_count: int,
        interface_edge_loss_count: int,
        interface_edge_stretch: float,
        elapsed: float,
        effort: float,
        neighbor_close_violation: float,
        neighbor_far_violation: float,
        mobile_overlap_count: int,
        hard_close_mobile_violation: float,
        fence_close_violation: float,
        fence_far_violation: float,
        hard_close_fence_violation: float,
        area_delta_norm: float,
        perimeter_delta_norm: float,
        largest_area_delta_norm: float,
        largest_perimeter_delta_norm: float,
        area_current_norm: float,
        perimeter_current_norm: float,
        largest_area_current_norm: float,
        largest_perimeter_current_norm: float,
        final_time: float,
        cleared: bool,
        timed_out: bool,
        disconnected: bool,
    ) -> Dict[str, float]:
        phase_config = self._phase_weighted_reward_config(phase_true_cycle_count)
        area_regress = max(0.0, -float(area_delta_norm))
        perimeter_regress = max(0.0, -float(perimeter_delta_norm))
        largest_area_regress = max(0.0, -float(largest_area_delta_norm))
        largest_perimeter_regress = max(0.0, -float(largest_perimeter_delta_norm))
        one_hole_area_scale = 1.0
        one_hole_perimeter_scale = 1.0
        if int(phase_true_cycle_count) == 1:
            if self._one_hole_entry_area_norm is not None and self._one_hole_entry_area_norm > 1e-8:
                area_ratio = float(area_current_norm) / float(self._one_hole_entry_area_norm)
                one_hole_area_scale = 1.0 + phase_config.one_hole_area_scale_alpha * max(area_ratio, 0.0)
            if self._one_hole_entry_perimeter_norm is not None and self._one_hole_entry_perimeter_norm > 1e-8:
                perimeter_ratio = float(perimeter_current_norm) / float(self._one_hole_entry_perimeter_norm)
                one_hole_perimeter_scale = 1.0 + phase_config.one_hole_perimeter_scale_alpha * max(perimeter_ratio, 0.0)
        return {
            "true_cycles_closed": phase_config.true_cycle_closed_reward * float(true_cycles_closed),
            "true_cycles_added": -phase_config.true_cycle_added_penalty * float(true_cycles_added),
            "merge_hazard_count": -phase_config.merge_hazard_penalty_weight * float(merge_hazard_count),
            "interface_edge_loss": -phase_config.interface_edge_loss_penalty_weight * float(interface_edge_loss_count),
            "interface_edge_stretch": -phase_config.interface_edge_stretch_penalty_weight * float(interface_edge_stretch),
            "elapsed": -phase_config.time_penalty * float(elapsed),
            "effort": -phase_config.control_effort_penalty * float(effort),
            "neighbor_close_violation": -phase_config.neighbor_close_penalty_weight * float(neighbor_close_violation),
            "neighbor_far_violation": -phase_config.neighbor_far_penalty_weight * float(neighbor_far_violation),
            "mobile_overlap_count": -phase_config.mobile_overlap_penalty_weight * float(mobile_overlap_count),
            "hard_close_mobile_violation": -phase_config.hard_close_mobile_penalty_weight * float(hard_close_mobile_violation),
            "fence_close_violation": -phase_config.fence_close_penalty_weight * float(fence_close_violation),
            "fence_far_violation": -phase_config.fence_far_penalty_weight * float(fence_far_violation),
            "hard_close_fence_violation": -phase_config.hard_close_fence_penalty_weight * float(hard_close_fence_violation),
            "area_progress_norm": phase_config.area_progress_reward_weight * one_hole_area_scale * float(area_delta_norm),
            "perimeter_progress_norm": phase_config.perimeter_progress_reward_weight * one_hole_perimeter_scale * float(perimeter_delta_norm),
            "largest_area_progress_norm": phase_config.largest_area_progress_reward_weight * float(largest_area_delta_norm),
            "largest_perimeter_progress_norm": phase_config.largest_perimeter_progress_reward_weight * float(largest_perimeter_delta_norm),
            "area_regress_norm": -phase_config.area_regress_penalty_weight * one_hole_area_scale * float(area_regress),
            "perimeter_regress_norm": -phase_config.perimeter_regress_penalty_weight * one_hole_perimeter_scale * float(perimeter_regress),
            "largest_area_regress_norm": -phase_config.largest_area_regress_penalty_weight * float(largest_area_regress),
            "largest_perimeter_regress_norm": -phase_config.largest_perimeter_regress_penalty_weight * float(largest_perimeter_regress),
            "area_residual_norm": -phase_config.area_residual_penalty_weight * float(area_current_norm),
            "perimeter_residual_norm": -phase_config.perimeter_residual_penalty_weight * float(perimeter_current_norm),
            "largest_area_residual_norm": -phase_config.largest_area_residual_penalty_weight * float(largest_area_current_norm),
            "largest_perimeter_residual_norm": -phase_config.largest_perimeter_residual_penalty_weight * float(largest_perimeter_current_norm),
            "one_hole_linger_penalty": (
                -phase_config.one_hole_linger_penalty_weight * float(elapsed)
                if int(phase_true_cycle_count) == 1 and not cleared
                else 0.0
            ),
            "one_hole_area_scale": float(one_hole_area_scale),
            "one_hole_perimeter_scale": float(one_hole_perimeter_scale),
            "clear_indicator": phase_config.clear_bonus * float(1.0 if cleared else 0.0),
            "success_time_bonus": (
                phase_config.success_time_bonus_weight * (1.0 / max(float(final_time), 1e-8))
                if cleared
                else 0.0
            ),
            "disconnection_indicator": -phase_config.disconnection_penalty * float(1.0 if disconnected else 0.0),
            "timeout_indicator": -phase_config.timeout_penalty * float(1.0 if (timed_out and not cleared) else 0.0),
        }

    def _mobile_neighbor_band_violations(self, simulation: RLEventSimulation) -> Tuple[float, float, float]:
        mobile = list(simulation.sensor_network.mobile_sensors)
        if len(mobile) < 2:
            return 0.0, 0.0, 0.0

        positions = np.asarray([sensor.pos for sensor in mobile], dtype=float)
        deltas = positions[:, None, :] - positions[None, :, :]
        dists = np.linalg.norm(deltas, axis=2)
        np.fill_diagonal(dists, np.inf)
        nearest = np.min(dists, axis=1)

        sensing_radius = float(getattr(simulation.sensor_network, "sensing_radius", 1.0))
        sensing_radius = max(sensing_radius, 1e-8)
        min_ratio = max(0.0, float(self.reward_config.neighbor_min_distance_ratio))
        max_ratio = max(min_ratio, float(self.reward_config.neighbor_max_distance_ratio))
        d_min = min_ratio * sensing_radius
        d_max = max_ratio * sensing_radius

        close_violation = np.maximum(0.0, (d_min - nearest) / sensing_radius)
        far_violation = np.maximum(0.0, (nearest - d_max) / sensing_radius)
        close_loss = float(np.mean(close_violation * close_violation))
        far_loss = float(np.mean(far_violation * far_violation))
        nearest_mean = float(np.mean(nearest))
        return close_loss, far_loss, nearest_mean

    def _mobile_fence_band_violations(self, simulation: RLEventSimulation) -> Tuple[float, float, float]:
        mobile = list(simulation.sensor_network.mobile_sensors)
        fence = list(simulation.sensor_network.fence_sensors)
        if not mobile or not fence:
            return 0.0, 0.0, 0.0

        mobile_pos = np.asarray([sensor.pos for sensor in mobile], dtype=float)
        fence_pos = np.asarray([sensor.pos for sensor in fence], dtype=float)
        deltas = mobile_pos[:, None, :] - fence_pos[None, :, :]
        dists = np.linalg.norm(deltas, axis=2)
        nearest = np.min(dists, axis=1)

        sensing_radius = float(getattr(simulation.sensor_network, "sensing_radius", 1.0))
        sensing_radius = max(sensing_radius, 1e-8)
        min_ratio = max(0.0, float(self.reward_config.neighbor_min_distance_ratio))
        max_ratio = max(min_ratio, float(self.reward_config.neighbor_max_distance_ratio))
        d_min = min_ratio * sensing_radius
        d_max = max_ratio * sensing_radius

        close_violation = np.maximum(0.0, (d_min - nearest) / sensing_radius)
        far_violation = np.maximum(0.0, (nearest - d_max) / sensing_radius)
        close_loss = float(np.mean(close_violation * close_violation))
        far_loss = float(np.mean(far_violation * far_violation))
        nearest_mean = float(np.mean(nearest))
        return close_loss, far_loss, nearest_mean

    def _hard_close_violations(self, simulation: RLEventSimulation) -> Tuple[float, float]:
        sensing_radius = float(getattr(simulation.sensor_network, "sensing_radius", 1.0))
        sensing_radius = max(sensing_radius, 1e-8)
        threshold = max(0.0, float(self.reward_config.hard_close_distance_ratio)) * sensing_radius

        mobile = list(simulation.sensor_network.mobile_sensors)
        fence = list(simulation.sensor_network.fence_sensors)

        mobile_close_loss = 0.0
        if len(mobile) >= 2:
            positions = np.asarray([sensor.pos for sensor in mobile], dtype=float)
            deltas = positions[:, None, :] - positions[None, :, :]
            dists = np.linalg.norm(deltas, axis=2)
            tri = np.triu_indices(len(mobile), k=1)
            pair_dists = dists[tri]
            if pair_dists.size > 0:
                violation = np.maximum(0.0, (threshold - pair_dists) / sensing_radius)
                mobile_close_loss = float(np.mean(violation * violation))

        fence_close_loss = 0.0
        if mobile and fence:
            mobile_pos = np.asarray([sensor.pos for sensor in mobile], dtype=float)
            fence_pos = np.asarray([sensor.pos for sensor in fence], dtype=float)
            deltas = mobile_pos[:, None, :] - fence_pos[None, :, :]
            dists = np.linalg.norm(deltas, axis=2)
            violation = np.maximum(0.0, (threshold - dists) / sensing_radius)
            fence_close_loss = float(np.mean(violation * violation))

        return mobile_close_loss, fence_close_loss

    def _mobile_overlap_count(self, simulation: RLEventSimulation) -> int:
        mobile = list(simulation.sensor_network.mobile_sensors)
        if len(mobile) < 2:
            return 0

        sensing_radius = float(getattr(simulation.sensor_network, "sensing_radius", 1.0))
        sensing_radius = max(sensing_radius, 1e-8)
        threshold = sensing_radius
        positions = np.asarray([sensor.pos for sensor in mobile], dtype=float)
        deltas = positions[:, None, :] - positions[None, :, :]
        dists = np.linalg.norm(deltas, axis=2)
        np.fill_diagonal(dists, np.inf)
        nearest = np.min(dists, axis=1)
        return int(np.sum(nearest <= threshold))

    def _cycle_nodes_for_lineage(self, cycle) -> Set[int]:
        if hasattr(cycle, "nodes"):
            try:
                return set(int(node) for node in cycle.nodes)
            except Exception:
                pass
        nodes: Set[int] = set()
        try:
            iterator = iter(cycle)
        except TypeError:
            return nodes
        for face in iterator:
            try:
                face_nodes = tuple(int(node) for node in self._simplex_nodes(face))
            except Exception:
                continue
            nodes.update(face_nodes)
        return nodes

    def _interior_cycle_maps(self, labels: Dict, outer_cycle) -> Tuple[Dict[str, bool], Dict[str, Set[int]]]:
        outer_key = str(outer_cycle) if outer_cycle is not None else None
        label_by_key: Dict[str, bool] = {}
        nodeset_by_key: Dict[str, Set[int]] = {}
        for cycle, value in labels.items():
            key = str(cycle)
            if outer_key is not None and key == outer_key:
                continue
            label_by_key[key] = bool(value)
            nodeset_by_key[key] = self._cycle_nodes_for_lineage(cycle)
        return label_by_key, nodeset_by_key

    def _is_true_cycle_disconnected(self, simulation: RLEventSimulation) -> bool:
        true_cycle_count = self._interior_true_cycle_count(simulation)
        if true_cycle_count <= 0:
            return False
        fence_count = len(simulation.sensor_network.fence_sensors)
        mobile_indices = set(range(fence_count, fence_count + len(simulation.sensor_network.mobile_sensors)))
        true_nodes, _, _, _ = self._cycle_node_dart_sets(simulation)
        return len(true_nodes & mobile_indices) == 0

    def _merge_hazard_false_true_to_true(
        self,
        previous_labels: Dict,
        current_labels: Dict,
        outer_cycle,
    ) -> int:
        prev_label_by_key, prev_nodeset_by_key = self._interior_cycle_maps(previous_labels, outer_cycle)
        curr_label_by_key, curr_nodeset_by_key = self._interior_cycle_maps(current_labels, outer_cycle)
        born = set(curr_label_by_key) - set(prev_label_by_key)
        died = set(prev_label_by_key) - set(curr_label_by_key)

        parents_for_child: Dict[str, List[str]] = {k: [] for k in born}
        for child in born:
            child_nodes = curr_nodeset_by_key.get(child, set())
            overlap_scores: List[Tuple[str, int]] = []
            for parent in died:
                parent_nodes = prev_nodeset_by_key.get(parent, set())
                score = len(parent_nodes & child_nodes)
                if score > 0:
                    overlap_scores.append((parent, score))
            if not overlap_scores:
                continue

            overlap_scores.sort(key=lambda item: item[1], reverse=True)
            best_parent, best_score = overlap_scores[0]
            selected_parents = [best_parent]
            if best_score >= 2:
                tied = [p for p, s in overlap_scores[1:] if s >= 2 and s >= best_score - 1]
                selected_parents.extend(tied)
            parents_for_child[child].extend(selected_parents)

        hazards = 0
        for child in born:
            if not curr_label_by_key.get(child, False):
                continue
            parents = parents_for_child.get(child, [])
            if len(parents) < 2:
                continue
            parent_labels = [bool(prev_label_by_key.get(parent, False)) for parent in parents]
            if any(parent_labels) and not all(parent_labels):
                hazards += 1
        return int(hazards)

    @staticmethod
    def _sensor_positions(simulation: RLEventSimulation) -> np.ndarray:
        sensors = list(simulation.sensor_network.fence_sensors) + list(simulation.sensor_network.mobile_sensors)
        if not sensors:
            return np.zeros((0, 2), dtype=float)
        return np.asarray([np.asarray(sensor.pos, dtype=float) for sensor in sensors], dtype=float)

    @staticmethod
    def _interface_edge_set(true_darts: Set[Tuple[int, int]], false_darts: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        true_edges = {tuple(sorted((int(u), int(v)))) for u, v in true_darts if int(u) != int(v)}
        false_edges = {tuple(sorted((int(u), int(v)))) for u, v in false_darts if int(u) != int(v)}
        return true_edges & false_edges

    def _interface_edge_metrics(
        self,
        *,
        previous_true_darts: Set[Tuple[int, int]],
        previous_false_darts: Set[Tuple[int, int]],
        previous_positions: np.ndarray,
        current_true_darts: Set[Tuple[int, int]],
        current_false_darts: Set[Tuple[int, int]],
        current_positions: np.ndarray,
        sensing_radius: float,
    ) -> Tuple[int, float]:
        prev_interface = self._interface_edge_set(previous_true_darts, previous_false_darts)
        curr_interface = self._interface_edge_set(current_true_darts, current_false_darts)
        lost_count = int(len(prev_interface - curr_interface))

        surviving = prev_interface & curr_interface
        if not surviving:
            return lost_count, 0.0

        radius = max(float(sensing_radius), 1e-8)
        stretch_terms = []
        for u, v in surviving:
            if u >= previous_positions.shape[0] or v >= previous_positions.shape[0]:
                continue
            if u >= current_positions.shape[0] or v >= current_positions.shape[0]:
                continue
            prev_len = float(np.linalg.norm(previous_positions[v] - previous_positions[u]))
            curr_len = float(np.linalg.norm(current_positions[v] - current_positions[u]))
            stretch = max(0.0, (curr_len - prev_len) / radius)
            stretch_terms.append(stretch * stretch)

        if not stretch_terms:
            return lost_count, 0.0
        return lost_count, float(np.mean(stretch_terms))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        _ = options  # reserved for future env options
        self._simulation = self._build_simulation()
        self.max_speed = self._resolve_max_speed(self._simulation)
        self._set_action_space(self._simulation)
        if self.enforce_2d and self._simulation.topology.dim != 2:
            raise ValueError(f"RLEvasionEnv currently enforces 2D; got dim={self._simulation.topology.dim}.")
        self._episode_index += 1
        self._step_count = 0
        self._one_hole_entry_area_norm = None
        self._one_hole_entry_perimeter_norm = None
        self._one_hole_entry_time = None

        self.observation_space = self._build_observation_space(self._simulation)
        observation = self._observation(self.simulation)
        initial_true_cycle_count = self._interior_true_cycle_count(self.simulation)
        initial_area, initial_perimeter = self._true_cycle_area_perimeter(self.simulation)
        area_scale, perimeter_scale = self._domain_area_perimeter_scales(self.simulation)
        self._update_one_hole_reference(
            true_cycle_count=initial_true_cycle_count,
            area_current_norm=(initial_area / area_scale),
            perimeter_current_norm=(initial_perimeter / perimeter_scale),
        )
        info = {
            "time": float(self.simulation.time),
            "intruder_count": initial_true_cycle_count,
            "true_cycle_count": initial_true_cycle_count,
            "phase": self._phase_name_from_true_cycle_count(initial_true_cycle_count),
            "phase_one_hot": self._phase_one_hot(initial_true_cycle_count).tolist(),
            "episode_index": self._episode_index,
            "max_speed": float(self.max_speed),
        }
        return observation, info

    def step(self, action):
        sim = self.simulation

        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.shape != self.action_space.shape:
            raise ValueError(f"Expected action shape {self.action_space.shape}, got {action_arr.shape}.")

        if not np.isfinite(action_arr).all():
            raise ValueError("Action contains NaN or inf.")
        processed_action = self._postprocess_action(action_arr)

        previous_labels = dict(sim.cycle_label.label)
        previous_true_cycle_count = self._interior_true_cycle_count(sim)
        _, _, previous_true_darts, previous_false_darts = self._cycle_node_dart_sets(sim)
        previous_positions = self._sensor_positions(sim)
        prev_area, prev_perimeter = self._true_cycle_area_perimeter(sim)
        prev_largest_true_area, prev_largest_true_perimeter = self._largest_true_cycle_area_perimeter(sim)
        result = sim.step_with_velocities(processed_action, interval=self.dt)
        current_true_cycle_count = self._interior_true_cycle_count(sim)
        _, _, current_true_darts, current_false_darts = self._cycle_node_dart_sets(sim)
        current_positions = self._sensor_positions(sim)
        curr_area, curr_perimeter = self._true_cycle_area_perimeter(sim)
        largest_true_area, largest_true_perimeter = self._largest_true_cycle_area_perimeter(sim)
        true_cycles_closed = max(previous_true_cycle_count - current_true_cycle_count, 0)
        true_cycles_added = max(current_true_cycle_count - previous_true_cycle_count, 0)
        area_scale, perimeter_scale = self._domain_area_perimeter_scales(sim)
        area_delta_norm = (prev_area - curr_area) / area_scale
        perimeter_delta_norm = (prev_perimeter - curr_perimeter) / perimeter_scale
        area_current_norm = curr_area / area_scale
        perimeter_current_norm = curr_perimeter / perimeter_scale
        largest_area_delta_norm = (prev_largest_true_area - largest_true_area) / area_scale
        largest_perimeter_delta_norm = (prev_largest_true_perimeter - largest_true_perimeter) / perimeter_scale
        largest_area_current_norm = largest_true_area / area_scale
        largest_perimeter_current_norm = largest_true_perimeter / perimeter_scale
        self._update_one_hole_reference(
            true_cycle_count=current_true_cycle_count,
            area_current_norm=area_current_norm,
            perimeter_current_norm=perimeter_current_norm,
        )
        neighbor_close_violation, neighbor_far_violation, nearest_neighbor_mean = self._mobile_neighbor_band_violations(sim)
        fence_close_violation, fence_far_violation, nearest_fence_mean = self._mobile_fence_band_violations(sim)
        mobile_overlap_count = self._mobile_overlap_count(sim)
        hard_close_mobile_violation, hard_close_fence_violation = self._hard_close_violations(sim)
        merge_hazard_count = self._merge_hazard_false_true_to_true(
            previous_labels=previous_labels,
            current_labels=sim.cycle_label.label,
            outer_cycle=sim.topology.outer_cycle,
        )
        interface_edge_loss_count, interface_edge_stretch = self._interface_edge_metrics(
            previous_true_darts=previous_true_darts,
            previous_false_darts=previous_false_darts,
            previous_positions=previous_positions,
            current_true_darts=current_true_darts,
            current_false_darts=current_false_darts,
            current_positions=current_positions,
            sensing_radius=float(getattr(sim.sensor_network, "sensing_radius", 1.0)),
        )

        cleared = not sim.has_intruder()
        disconnected = self._is_true_cycle_disconnected(sim)
        terminated = cleared or disconnected
        self._step_count += 1

        truncated = False
        if not terminated and self.max_steps is not None and self._step_count >= self.max_steps:
            truncated = True
        if not terminated and sim.Tend > 0 and sim.time >= sim.Tend:
            truncated = True

        effort = float(np.mean(np.sum(processed_action * processed_action, axis=1))) if processed_action.size else 0.0

        reward = self._reward(
            current_true_cycle_count,
            true_cycles_closed,
            true_cycles_added,
            merge_hazard_count,
            interface_edge_loss_count,
            interface_edge_stretch,
            result.elapsed,
            processed_action,
            neighbor_close_violation,
            neighbor_far_violation,
            mobile_overlap_count,
            hard_close_mobile_violation,
            fence_close_violation,
            fence_far_violation,
            hard_close_fence_violation,
            area_delta_norm,
            perimeter_delta_norm,
            largest_area_delta_norm,
            largest_perimeter_delta_norm,
            area_current_norm,
            perimeter_current_norm,
            largest_area_current_norm,
            largest_perimeter_current_norm,
            sim.time,
            cleared=cleared,
            timed_out=truncated,
            disconnected=disconnected,
        )
        reward_terms = self._reward_terms(
            phase_true_cycle_count=current_true_cycle_count,
            true_cycles_closed=true_cycles_closed,
            true_cycles_added=true_cycles_added,
            merge_hazard_count=merge_hazard_count,
            interface_edge_loss_count=interface_edge_loss_count,
            interface_edge_stretch=interface_edge_stretch,
            elapsed=result.elapsed,
            effort=effort,
            neighbor_close_violation=neighbor_close_violation,
            neighbor_far_violation=neighbor_far_violation,
            mobile_overlap_count=mobile_overlap_count,
            hard_close_mobile_violation=hard_close_mobile_violation,
            fence_close_violation=fence_close_violation,
            fence_far_violation=fence_far_violation,
            hard_close_fence_violation=hard_close_fence_violation,
            area_delta_norm=area_delta_norm,
            perimeter_delta_norm=perimeter_delta_norm,
            largest_area_delta_norm=largest_area_delta_norm,
            largest_perimeter_delta_norm=largest_perimeter_delta_norm,
            area_current_norm=area_current_norm,
            perimeter_current_norm=perimeter_current_norm,
            largest_area_current_norm=largest_area_current_norm,
            largest_perimeter_current_norm=largest_perimeter_current_norm,
            final_time=sim.time,
            cleared=cleared,
            timed_out=truncated,
            disconnected=disconnected,
        )

        record = EventLogRecord(
            episode_index=self._episode_index,
            step_index=self._step_count,
            sim_time=float(sim.time),
            event_time=float(result.elapsed),
            event_found=bool(result.event_found),
            alpha_change=result.state_change.alpha_complex_change(),
            boundary_change=result.state_change.boundary_cycle_change(),
            trace_evaluation_count=int(result.trace_diagnostics.evaluation_count),
            trace_split_count=int(result.trace_diagnostics.split_count),
            trace_max_recursion_depth=int(result.trace_diagnostics.max_recursion_depth),
            trace_recursion_limit_hit=bool(result.trace_diagnostics.recursion_limit_hit),
            true_cycle_count_before=int(previous_true_cycle_count),
            true_cycle_count_after=int(current_true_cycle_count),
            true_cycles_closed=int(true_cycles_closed),
            true_cycles_added=int(true_cycles_added),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        if self.enable_event_logging:
            self.event_log.append(record)
            self._append_event_log_file(record)

        observation = self._observation(sim)
        info = {
            "time": float(sim.time),
            "elapsed": float(result.elapsed),
            "event_found": bool(result.event_found),
            "intruder_count": current_true_cycle_count,
            "true_cycle_count": current_true_cycle_count,
            "phase": self._phase_name_from_true_cycle_count(current_true_cycle_count),
            "phase_one_hot": self._phase_one_hot(current_true_cycle_count).tolist(),
            "one_hole_dwell_time_norm": float(self._one_hole_dwell_time_norm(sim)),
            "true_cycles_closed": int(true_cycles_closed),
            "true_cycles_added": int(true_cycles_added),
            "merge_hazard_count": int(merge_hazard_count),
            "interface_edge_loss_count": int(interface_edge_loss_count),
            "interface_edge_stretch": float(interface_edge_stretch),
            "disconnected": bool(disconnected),
            "termination_reason": (
                "disconnection"
                if disconnected
                else ("clear" if cleared else ("timeout" if truncated else "running"))
            ),
            "alpha_complex_change": result.state_change.alpha_complex_change(),
            "boundary_cycle_change": result.state_change.boundary_cycle_change(),
            "trace_evaluation_count": int(result.trace_diagnostics.evaluation_count),
            "trace_split_count": int(result.trace_diagnostics.split_count),
            "trace_max_recursion_depth": int(result.trace_diagnostics.max_recursion_depth),
            "trace_recursion_limit_hit": bool(result.trace_diagnostics.recursion_limit_hit),
            "true_cycle_area": float(curr_area),
            "true_cycle_perimeter": float(curr_perimeter),
            "largest_true_cycle_area": float(largest_true_area),
            "largest_true_cycle_perimeter": float(largest_true_perimeter),
            "true_cycle_area_norm": float(area_current_norm),
            "true_cycle_perimeter_norm": float(perimeter_current_norm),
            "true_cycle_area_delta_norm": float(area_delta_norm),
            "true_cycle_perimeter_delta_norm": float(perimeter_delta_norm),
            "largest_true_cycle_area_norm": float(largest_area_current_norm),
            "largest_true_cycle_perimeter_norm": float(largest_perimeter_current_norm),
            "largest_true_cycle_area_delta_norm": float(largest_area_delta_norm),
            "largest_true_cycle_perimeter_delta_norm": float(largest_perimeter_delta_norm),
            "one_hole_entry_area_norm": (
                None if self._one_hole_entry_area_norm is None else float(self._one_hole_entry_area_norm)
            ),
            "one_hole_entry_perimeter_norm": (
                None if self._one_hole_entry_perimeter_norm is None else float(self._one_hole_entry_perimeter_norm)
            ),
            "neighbor_close_violation": float(neighbor_close_violation),
            "neighbor_far_violation": float(neighbor_far_violation),
            "mobile_overlap_count": int(mobile_overlap_count),
            "hard_close_mobile_violation": float(hard_close_mobile_violation),
            "neighbor_nearest_mean_distance": float(nearest_neighbor_mean),
            "fence_close_violation": float(fence_close_violation),
            "fence_far_violation": float(fence_far_violation),
            "hard_close_fence_violation": float(hard_close_fence_violation),
            "fence_nearest_mean_distance": float(nearest_fence_mean),
            "reward_terms": reward_terms,
            "event_record": record.as_dict(),
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        sim = self.simulation
        return {
            "time": float(sim.time),
            "intruder_count": self._interior_true_cycle_count(sim),
            "true_cycle_count": self._interior_true_cycle_count(sim),
            "mobile_positions": np.asarray([s.pos for s in sim.sensor_network.mobile_sensors], dtype=float),
        }

    def close(self) -> None:
        self._simulation = None
