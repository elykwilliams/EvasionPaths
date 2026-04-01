import numpy as np
import pytest

pytest.importorskip("gymnasium")

from rl_env import RLEvasionEnv, RewardConfig
from rl_simulation import RLStepResult, TraceDiagnostics
from sensor_network import Sensor


class _FakeFace:
    def __init__(self, nodes):
        self.nodes = tuple(nodes)


class _FakeCycle(frozenset):
    @property
    def nodes(self):
        return frozenset(node for face in self for node in face.nodes)


class _FakeStateChange:
    def alpha_complex_change(self):
        return (1, 0, 0, 0)

    def boundary_cycle_change(self):
        return (1, 0)


class _FakeTopology:
    dim = 2

    def __init__(self):
        self.outer_cycle = _FakeCycle({_FakeFace((0, 1))})
        self.inner_cycle = _FakeCycle({_FakeFace((0, 1))})
        self.boundary_cycles = {self.outer_cycle, self.inner_cycle}
        self.homology_generators = {self.outer_cycle, self.inner_cycle}

    def simplices(self, dim):
        if dim == 1:
            return {(0, 1)}
        if dim == 2:
            return set()
        return set()

    @staticmethod
    def is_connected_cycle(_):
        return True


class _FakeCycleLabel:
    def __init__(self, topology):
        self.label = {topology.outer_cycle: True, topology.inner_cycle: True}

    def has_intruder(self):
        return sum(1 for v in self.label.values() if v) > 1


class _FakeSensorNetwork:
    def __init__(self, *, fence_count=0):
        self.sensing_radius = 0.2
        self.fence_sensors = [
            Sensor(np.array([float(i), 0.0]), np.array([0.0, 0.0]), 0.2, True)
            for i in range(fence_count)
        ]
        self.mobile_sensors = [
            Sensor(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.2, False),
            Sensor(np.array([1.0, 0.0]), np.array([0.0, 0.0]), 0.2, False),
        ]


class _FakeSimulation:
    def __init__(self, dt=0.1, clear_after_steps=1, end_time=0.0, *, fence_count=0):
        self.dt = float(dt)
        self.Tend = float(end_time)
        self.time = 0.0

        self.sensor_network = _FakeSensorNetwork(fence_count=fence_count)
        self.topology = _FakeTopology()
        self.cycle_label = _FakeCycleLabel(self.topology)

        self._clear_after_steps = clear_after_steps
        self._step_count = 0

    def step_with_velocities(self, mobile_velocities, interval=None):
        elapsed = self.dt if interval is None else float(interval)
        self._step_count += 1

        for sensor, velocity in zip(self.sensor_network.mobile_sensors, mobile_velocities):
            vel = np.asarray(velocity, dtype=float)
            sensor.vel = vel
            sensor.pos = sensor.pos + elapsed * vel

        self.time += elapsed

        if self._clear_after_steps is not None and self._step_count >= self._clear_after_steps:
            self.cycle_label.label = {self.topology.outer_cycle: True}

        return RLStepResult(
            elapsed=elapsed,
            event_found=False,
            topology=self.topology,
            state_change=_FakeStateChange(),
            mobile_positions=np.asarray([s.pos for s in self.sensor_network.mobile_sensors], dtype=float),
            mobile_velocities=np.asarray([s.vel for s in self.sensor_network.mobile_sensors], dtype=float),
            trace_diagnostics=TraceDiagnostics(
                evaluation_count=3,
                split_count=2,
                max_recursion_depth=2,
            ),
        )

    def has_intruder(self):
        return self.cycle_label.has_intruder()


def _make_env(
    clear_after_steps=1,
    max_steps=None,
    coordinate_free=True,
    max_speed=0.5,
    max_speed_scale=0.1,
    state_mode="vector",
    fence_count=0,
    simulation_builder=None,
):
    def builder():
        if simulation_builder is not None:
            return simulation_builder()
        return _FakeSimulation(dt=0.1, clear_after_steps=clear_after_steps, fence_count=fence_count)

    return RLEvasionEnv(
        dt=0.1,
        simulation_builder=builder,
        max_speed=max_speed,
        max_speed_scale=max_speed_scale,
        max_steps=max_steps,
        state_mode=state_mode,
        coordinate_free=coordinate_free,
        reward_config=RewardConfig(
            true_cycle_closed_reward=2.0,
            true_cycle_added_penalty=4.0,
            time_penalty=0.0,
            clear_bonus=5.0,
            control_effort_penalty=0.0,
            timeout_penalty=0.0,
        ),
    )


def test_reset_returns_valid_observation_and_info():
    env = _make_env(clear_after_steps=2)
    obs, info = env.reset(seed=7)

    assert obs.shape == env.observation_space.shape
    assert info["time"] == 0.0
    assert info["intruder_count"] == 1
    assert info["true_cycle_count"] == 1


def test_coordinate_free_is_default_and_has_expected_observation_size():
    env = _make_env(clear_after_steps=2)
    obs, _ = env.reset(seed=7)
    assert env.coordinate_free is True
    assert obs.shape == (16,)


def test_coordinate_free_toggle_changes_observation_shape():
    env = _make_env(clear_after_steps=2, coordinate_free=False)
    obs, _ = env.reset(seed=7)
    assert obs.shape == (18,)


def test_graph_state_mode_returns_expected_observation_schema():
    env = _make_env(clear_after_steps=2, state_mode="graph")
    obs, _ = env.reset(seed=7)
    assert set(obs.keys()) == {"node_x", "edge_x", "edge_mask", "global_x"}
    assert obs["node_x"].shape == (2, 7)
    assert obs["edge_x"].shape == (2, 2, 5)
    assert obs["edge_mask"].shape == (2, 2)
    assert obs["global_x"].shape == (5,)
    assert obs["node_x"][0, 3] == pytest.approx(1.0)
    assert obs["node_x"][0, 4] == pytest.approx(0.0)


def test_cycle_graph_state_mode_returns_expected_observation_schema():
    env = _make_env(clear_after_steps=2, state_mode="cycle_graph")
    obs, _ = env.reset(seed=7)
    assert set(obs.keys()) == {
        "node_x",
        "edge_x",
        "edge_mask",
        "global_x",
        "cycle_node_index",
        "cycle_node_mask",
        "cycle_mask",
        "cycle_is_true",
        "node_cycle_token_x",
        "node_cycle_token_mask",
    }
    assert obs["node_x"].shape == (2, 8)
    assert obs["global_x"].shape == (6,)
    assert obs["cycle_node_index"].shape == (2, 2)
    assert obs["cycle_node_mask"][0, :2] == pytest.approx(np.array([1.0, 1.0]))
    assert obs["cycle_mask"][0] == pytest.approx(1.0)
    assert obs["cycle_is_true"][0] == pytest.approx(1.0)
    assert obs["node_cycle_token_x"].shape == (2, 2, 7)
    assert obs["node_cycle_token_mask"].shape == (2, 2)


def test_default_max_speed_is_derived_from_sensing_radius_and_dt():
    env = _make_env(clear_after_steps=1, max_speed=None, max_speed_scale=0.1)
    env.reset()
    assert env.max_speed == pytest.approx(0.2)

    action = np.array([[2.0, 0.0], [-2.0, 0.0]], dtype=np.float32)
    env.step(action)
    sim = env.simulation
    assert sim.sensor_network.mobile_sensors[0].vel == pytest.approx(np.array([0.2, 0.0]))
    assert sim.sensor_network.mobile_sensors[1].vel == pytest.approx(np.array([-0.2, 0.0]))


def test_step_clips_action_and_terminates_on_clear():
    env = _make_env(clear_after_steps=1)
    env.reset()

    action = np.array([[2.0, 0.0], [-2.0, 0.0]], dtype=np.float32)
    _, reward, terminated, truncated, _ = env.step(action)

    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(7.0)

    sim = env.simulation
    assert sim.sensor_network.mobile_sensors[0].vel == pytest.approx(np.array([0.5, 0.0]))
    assert sim.sensor_network.mobile_sensors[1].vel == pytest.approx(np.array([-0.5, 0.0]))
    assert len(env.event_log) == 1
    record = env.event_log[0]
    assert record.trace_evaluation_count == 3
    assert record.trace_split_count == 2
    assert record.trace_max_recursion_depth == 2
    assert record.true_cycles_closed == 1
    assert record.true_cycles_added == 0


def test_mobile_overlap_count_penalizes_each_crowded_mobile():
    sim = _FakeSimulation(dt=0.1, clear_after_steps=None)
    sim.sensor_network.mobile_sensors[0].pos = np.array([0.0, 0.0], dtype=float)
    sim.sensor_network.mobile_sensors[1].pos = np.array([0.1, 0.0], dtype=float)

    env = RLEvasionEnv(
        dt=0.1,
        simulation_builder=lambda: sim,
        max_speed=0.5,
        state_mode="vector",
        coordinate_free=True,
        reward_config=RewardConfig(
            true_cycle_closed_reward=0.0,
            true_cycle_added_penalty=0.0,
            time_penalty=0.0,
            clear_bonus=0.0,
            control_effort_penalty=0.0,
            timeout_penalty=0.0,
            mobile_overlap_penalty_weight=1.5,
        ),
    )
    env.reset()

    _, reward, terminated, truncated, info = env.step(np.zeros((2, 2), dtype=np.float32))

    assert terminated is False
    assert truncated is False
    assert info["mobile_overlap_count"] == 2
    assert info["reward_terms"]["mobile_overlap_count"] == pytest.approx(-3.0)
    assert reward == pytest.approx(-3.0)


def test_step_raises_value_error_for_invalid_action_shape():
    env = _make_env(clear_after_steps=2)
    env.reset()

    with pytest.raises(ValueError):
        env.step(np.array([1.0, 2.0], dtype=np.float32))


def test_step_truncates_on_max_steps_without_clear():
    env = _make_env(clear_after_steps=None, max_steps=2)
    env.reset()

    action = np.zeros((2, 2), dtype=np.float32)
    _, _, terminated_1, truncated_1, _ = env.step(action)
    _, _, terminated_2, truncated_2, _ = env.step(action)

    assert terminated_1 is False
    assert truncated_1 is False
    assert terminated_2 is False
    assert truncated_2 is True


class _DisconnectedTopology(_FakeTopology):
    def __init__(self):
        self.outer_cycle = _FakeCycle({_FakeFace((0, 1))})
        self.inner_cycle = _FakeCycle({_FakeFace((0, 1))})
        self.boundary_cycles = {self.outer_cycle, self.inner_cycle}
        self.homology_generators = {self.outer_cycle, self.inner_cycle}

    def simplices(self, dim):
        if dim == 1:
            return {(0, 1), (1, 2)}
        if dim == 2:
            return set()
        return set()


class _DisconnectedSimulation(_FakeSimulation):
    def __init__(self):
        super().__init__(dt=0.1, clear_after_steps=None, fence_count=2)
        self.topology = _DisconnectedTopology()
        self.cycle_label = _FakeCycleLabel(self.topology)
        self.sensor_network.mobile_sensors = [Sensor(np.array([2.0, 0.0]), np.array([0.0, 0.0]), 0.2, False)]


def test_disconnection_is_terminal_failure_with_penalty():
    env = _make_env(
        clear_after_steps=None,
        state_mode="graph",
        simulation_builder=_DisconnectedSimulation,
    )
    env.reset()
    action = np.zeros((1, 2), dtype=np.float32)
    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "disconnection"
    assert info["disconnected"] is True
    assert reward <= -90.0


def test_interface_edge_metrics_penalize_loss_and_stretch():
    env = _make_env(clear_after_steps=None)

    previous_true_darts = {(0, 1)}
    previous_false_darts = {(1, 0)}
    previous_positions = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float)

    current_true_darts = {(0, 1)}
    current_false_darts = {(1, 0)}
    current_positions = np.asarray([[0.0, 0.0], [1.3, 0.0]], dtype=float)

    loss_count, stretch = env._interface_edge_metrics(
        previous_true_darts=previous_true_darts,
        previous_false_darts=previous_false_darts,
        previous_positions=previous_positions,
        current_true_darts=current_true_darts,
        current_false_darts=current_false_darts,
        current_positions=current_positions,
        sensing_radius=1.0,
    )

    assert loss_count == 0
    assert stretch == pytest.approx(0.09)

    loss_count, stretch = env._interface_edge_metrics(
        previous_true_darts=previous_true_darts,
        previous_false_darts=previous_false_darts,
        previous_positions=previous_positions,
        current_true_darts=set(),
        current_false_darts=set(),
        current_positions=current_positions,
        sensing_radius=1.0,
    )

    assert loss_count == 1
    assert stretch == pytest.approx(0.0)


def test_hard_close_violations_trigger_below_half_radius():
    env = _make_env(clear_after_steps=None)
    env.reset()
    sim = env.simulation
    sim.sensor_network.sensing_radius = 1.0
    sim.sensor_network.mobile_sensors[0].pos = np.array([0.0, 0.0])
    sim.sensor_network.mobile_sensors[1].pos = np.array([0.4, 0.0])

    mobile_close, fence_close = env._hard_close_violations(sim)
    assert mobile_close == pytest.approx(0.01)
    assert fence_close == pytest.approx(0.0)

    sim.sensor_network.fence_sensors = [Sensor(np.array([0.3, 0.0]), np.array([0.0, 0.0]), 1.0, True)]
    mobile_close, fence_close = env._hard_close_violations(sim)
    assert mobile_close == pytest.approx(0.01)
    assert fence_close > 0.0
