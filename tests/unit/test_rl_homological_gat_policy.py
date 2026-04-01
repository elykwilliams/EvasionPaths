import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_env import RLEvasionEnv
from rl_homological_gat_policy import HomologicalGATActorCriticPolicy, HomologicalGraphAttentionExtractor
from rl_simulation import RLStepResult, TraceDiagnostics
from sensor_network import Sensor


class _Face:
    def __init__(self, nodes):
        self.nodes = tuple(nodes)


class _Cycle(frozenset):
    @property
    def nodes(self):
        return frozenset(node for face in self for node in face.nodes)


class _Topology:
    dim = 2

    def __init__(self):
        self.outer_cycle = _Cycle({_Face((0, 1)), _Face((1, 0))})
        self.true_cycle = _Cycle({_Face((0, 2)), _Face((2, 1)), _Face((1, 0))})
        self.false_cycle = _Cycle({_Face((0, 3)), _Face((3, 1)), _Face((1, 0))})
        self.boundary_cycles = {self.outer_cycle, self.true_cycle, self.false_cycle}
        self.homology_generators = {self.outer_cycle, self.true_cycle, self.false_cycle}

    def simplices(self, dim):
        if dim == 1:
            return {(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)}
        if dim == 2:
            return set()
        return set()

    @staticmethod
    def is_connected_cycle(_):
        return True


class _CycleLabel:
    def __init__(self, topology):
        self.label = {
            topology.outer_cycle: True,
            topology.true_cycle: True,
            topology.false_cycle: False,
        }

    def has_intruder(self):
        return True


class _SensorNetwork:
    def __init__(self):
        self.sensing_radius = 0.2
        self.fence_sensors = [
            Sensor(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.2, True),
            Sensor(np.array([1.0, 0.0]), np.array([0.0, 0.0]), 0.2, True),
        ]
        self.mobile_sensors = [
            Sensor(np.array([0.45, 0.25]), np.array([0.1, 0.0]), 0.2, False),
            Sensor(np.array([0.55, 0.55]), np.array([0.0, -0.1]), 0.2, False),
        ]


class _Simulation:
    def __init__(self):
        self.dt = 0.1
        self.Tend = 2.0
        self.time = 0.0
        self.sensor_network = _SensorNetwork()
        self.topology = _Topology()
        self.cycle_label = _CycleLabel(self.topology)

    def step_with_velocities(self, mobile_velocities, interval=None):
        elapsed = self.dt if interval is None else float(interval)
        self.time += elapsed
        return RLStepResult(
            elapsed=elapsed,
            event_found=False,
            topology=self.topology,
            state_change=type(
                "_StateChange",
                (),
                {
                    "alpha_complex_change": lambda self: (0, 0, 0, 0),
                    "boundary_cycle_change": lambda self: (0, 0),
                },
            )(),
            mobile_positions=np.asarray([s.pos for s in self.sensor_network.mobile_sensors], dtype=float),
            mobile_velocities=np.asarray(mobile_velocities, dtype=float),
            trace_diagnostics=TraceDiagnostics(),
        )

    def has_intruder(self):
        return True


def _make_env():
    return RLEvasionEnv(
        dt=0.1,
        simulation_builder=_Simulation,
        max_speed=0.5,
        max_speed_scale=0.1,
        state_mode="cycle_graph",
        coordinate_free=True,
    )


def test_homological_gat_policy_emits_mobile_velocity_actions():
    env = _make_env()
    obs, _ = env.reset()
    policy = HomologicalGATActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 3e-4,
        features_extractor_class=HomologicalGraphAttentionExtractor,
        features_extractor_kwargs={
            "token_hidden_dim": 16,
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 2,
            "dropout": 0.0,
        },
        net_arch={"pi": [64], "vf": [64]},
    )

    obs_tensor, _ = policy.obs_to_tensor(obs)
    actions, values, log_prob = policy.forward(obs_tensor, deterministic=True)

    assert actions.shape == (1, 2, 2)
    assert values.shape == (1, 1)
    assert log_prob.shape == (1,)

    extractor = policy.features_extractor
    assert "token_pool_true" in extractor.last_token_attention
    assert "token_pool_false" in extractor.last_token_attention
