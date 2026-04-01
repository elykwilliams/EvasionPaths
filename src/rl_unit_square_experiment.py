# ******************************************************************************
#  Copyright (c) 2026, Marco Campos - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np

from boundary_geometry import RectangularDomain
from motion_model import BilliardMotion
from rl_attention_logging import AttentionLogConfig
from rl_env import (
    PhaseRewardMultipliers,
    PhaseRewardSchedule,
    RLEvasionEnv,
    RewardConfig,
)
from sensor_network import SensorNetwork, generate_fence_sensors, generate_mobile_sensors


def _default_train_seeds() -> Tuple[int, ...]:
    return tuple(range(1000, 1200))


def _default_eval_seeds() -> Tuple[int, ...]:
    return tuple(range(5000, 5050))


@dataclass(frozen=True)
class UnitSquareRLConfig:
    # Fixed v1 environment contract
    num_mobile_sensors: int = 12
    sensing_radius: float = 0.2
    fence_sensing_radius: float | None = None
    fence_offset_ratio: float | None = None
    use_weighted_alpha: bool = False
    dt: float = 0.1
    tmax: float = 10.0

    # Initialization and control
    initial_sensor_speed: float = 1.0
    max_speed_scale: float = 0.1
    coordinate_free: bool = True
    state_mode: str = "graph"

    # Reward contract (outer cycle excluded by env reward logic)
    reward_config: RewardConfig = field(
        default_factory=lambda: RewardConfig(
            true_cycle_closed_reward=3.0,
            true_cycle_added_penalty=4.0,
            time_penalty=0.03,
            control_effort_penalty=0.05,
            clear_bonus=12.0,
            timeout_penalty=12.0,
            disconnection_penalty=40.0,
            largest_area_progress_reward_weight=0.0,
            largest_perimeter_progress_reward_weight=0.0,
            area_regress_penalty_weight=0.0,
            perimeter_regress_penalty_weight=0.0,
            largest_area_regress_penalty_weight=0.0,
            largest_perimeter_regress_penalty_weight=0.0,
            neighbor_min_distance_ratio=0.8660254037844386,  # sqrt(3)/2
            neighbor_max_distance_ratio=2.0,
            neighbor_close_penalty_weight=0.8,
            neighbor_far_penalty_weight=0.2,
            mobile_overlap_penalty_weight=0.02,
            hard_close_distance_ratio=0.5,
            hard_close_mobile_penalty_weight=0.1,
            hard_close_fence_penalty_weight=0.1,
            fence_close_penalty_weight=0.8,
            fence_far_penalty_weight=0.2,
            merge_hazard_penalty_weight=6.0,
            interface_edge_loss_penalty_weight=10.0,
            interface_edge_stretch_penalty_weight=2.0,
            largest_area_residual_penalty_weight=0.0,
            largest_perimeter_residual_penalty_weight=0.0,
        )
    )
    phase_reward_schedule: PhaseRewardSchedule = field(
        default_factory=lambda: PhaseRewardSchedule(
            simplify=PhaseRewardMultipliers(
                true_cycle_closed_reward=1.5,
                true_cycle_added_penalty=1.2,
                time_penalty=0.9,
                control_effort_penalty=0.35,
                clear_bonus=1.1,
                timeout_penalty=1.1,
                merge_hazard_penalty_weight=1.35,
                neighbor_close_penalty_weight=0.3,
                neighbor_far_penalty_weight=0.0,
                fence_close_penalty_weight=0.3,
                fence_far_penalty_weight=0.0,
                area_progress_reward_weight=0.0,
                perimeter_progress_reward_weight=0.0,
                largest_area_progress_reward_weight=0.0,
                largest_perimeter_progress_reward_weight=0.0,
                area_regress_penalty_weight=0.0,
                perimeter_regress_penalty_weight=0.0,
                largest_area_regress_penalty_weight=0.0,
                largest_perimeter_regress_penalty_weight=0.0,
                area_residual_penalty_weight=0.0,
                perimeter_residual_penalty_weight=0.0,
                largest_area_residual_penalty_weight=0.0,
                largest_perimeter_residual_penalty_weight=0.0,
            ),
            consolidate=PhaseRewardMultipliers(
                true_cycle_closed_reward=1.7,
                true_cycle_added_penalty=1.35,
                time_penalty=0.95,
                control_effort_penalty=0.45,
                clear_bonus=1.15,
                timeout_penalty=1.15,
                merge_hazard_penalty_weight=1.5,
                neighbor_close_penalty_weight=0.4,
                neighbor_far_penalty_weight=0.0,
                fence_close_penalty_weight=0.4,
                fence_far_penalty_weight=0.0,
                area_progress_reward_weight=0.0,
                perimeter_progress_reward_weight=0.0,
                largest_area_progress_reward_weight=0.0,
                largest_perimeter_progress_reward_weight=0.0,
                area_regress_penalty_weight=0.0,
                perimeter_regress_penalty_weight=0.0,
                largest_area_regress_penalty_weight=0.0,
                largest_perimeter_regress_penalty_weight=0.0,
                area_residual_penalty_weight=0.0,
                perimeter_residual_penalty_weight=0.0,
                largest_area_residual_penalty_weight=0.0,
                largest_perimeter_residual_penalty_weight=0.0,
            ),
            compress=PhaseRewardMultipliers(
                true_cycle_closed_reward=1.0,
                true_cycle_added_penalty=1.4,
                time_penalty=1.0,
                control_effort_penalty=0.8,
                clear_bonus=1.35,
                timeout_penalty=1.2,
                merge_hazard_penalty_weight=1.6,
                neighbor_close_penalty_weight=0.55,
                neighbor_far_penalty_weight=0.2,
                fence_close_penalty_weight=0.55,
                fence_far_penalty_weight=0.2,
                area_progress_reward_weight=1.5,
                perimeter_progress_reward_weight=1.5,
                largest_area_progress_reward_weight=1.7,
                largest_perimeter_progress_reward_weight=1.7,
                area_regress_penalty_weight=1.8,
                perimeter_regress_penalty_weight=1.8,
                largest_area_regress_penalty_weight=2.0,
                largest_perimeter_regress_penalty_weight=2.0,
                area_residual_penalty_weight=1.6,
                perimeter_residual_penalty_weight=1.6,
                largest_area_residual_penalty_weight=1.8,
                largest_perimeter_residual_penalty_weight=1.8,
            ),
        )
    )
    attention_log_config: AttentionLogConfig = field(
        default_factory=lambda: AttentionLogConfig(
            enabled=True,
            topk=40,
            every_n_steps_train=10,
            every_n_steps_eval=1,
            log_full_attention=False,
            path="",
        )
    )

    # Deterministic seed schedules
    train_seeds: Tuple[int, ...] = field(default_factory=_default_train_seeds)
    eval_seeds: Tuple[int, ...] = field(default_factory=_default_eval_seeds)

    def as_serializable(self) -> Dict:
        payload = asdict(self)
        payload["train_seeds"] = list(self.train_seeds)
        payload["eval_seeds"] = list(self.eval_seeds)
        return payload


class SeededUnitSquareBuilder:
    """Deterministic per-episode sensor network generator over fixed unit square + fixed fence rule."""

    def __init__(self, config: UnitSquareRLConfig, seeds: Sequence[int]):
        if not seeds:
            raise ValueError("seeds must not be empty")
        self.config = config
        self.seeds = tuple(int(seed) for seed in seeds)
        self._index = 0

    def __call__(self) -> SensorNetwork:
        seed = self.seeds[self._index % len(self.seeds)]
        self._index += 1
        np.random.seed(seed)

        domain = RectangularDomain()
        motion_model = BilliardMotion()
        fence_radius = (
            float(self.config.sensing_radius)
            if self.config.fence_sensing_radius is None
            else float(self.config.fence_sensing_radius)
        )
        fence_offset_distance = None
        if self.config.fence_offset_ratio is not None:
            fence_offset_distance = float(self.config.fence_offset_ratio) * fence_radius
        fence = generate_fence_sensors(
            domain,
            fence_radius,
            offset_distance=fence_offset_distance,
        )
        mobile_sensors = generate_mobile_sensors(
            domain,
            self.config.num_mobile_sensors,
            self.config.sensing_radius,
            self.config.initial_sensor_speed,
        )

        return SensorNetwork(
            mobile_sensors=mobile_sensors,
            motion_model=motion_model,
            fence=fence,
            sensing_radius=self.config.sensing_radius,
            domain=domain,
            use_weighted_alpha=bool(self.config.use_weighted_alpha),
        )


def make_unit_square_env(
    config: UnitSquareRLConfig,
    *,
    seeds: Sequence[int],
    event_log_path: str | None = None,
    enable_event_logging: bool = True,
    max_steps_override: int | None = None,
) -> RLEvasionEnv:
    builder = SeededUnitSquareBuilder(config, seeds)
    return RLEvasionEnv(
        dt=config.dt,
        sensor_network_builder=builder,
        max_speed=None,
        max_speed_scale=config.max_speed_scale,
        end_time=config.tmax,
        max_steps=max_steps_override,
        state_mode=config.state_mode,
        coordinate_free=config.coordinate_free,
        enforce_2d=True,
        reward_config=config.reward_config,
        phase_reward_schedule=config.phase_reward_schedule,
        enable_event_logging=enable_event_logging,
        event_log_path=event_log_path,
    )


def make_training_env(
    config: UnitSquareRLConfig,
    *,
    event_log_path: str | None = None,
    enable_event_logging: bool = True,
    max_steps_override: int | None = None,
) -> RLEvasionEnv:
    return make_unit_square_env(
        config,
        seeds=config.train_seeds,
        event_log_path=event_log_path,
        enable_event_logging=enable_event_logging,
        max_steps_override=max_steps_override,
    )


def make_eval_env(
    config: UnitSquareRLConfig,
    *,
    event_log_path: str | None = None,
    enable_event_logging: bool = True,
    max_steps_override: int | None = None,
) -> RLEvasionEnv:
    return make_unit_square_env(
        config,
        seeds=config.eval_seeds,
        event_log_path=event_log_path,
        enable_event_logging=enable_event_logging,
        max_steps_override=max_steps_override,
    )
