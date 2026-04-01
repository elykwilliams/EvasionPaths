import numpy as np
import pytest

from boundary_geometry import RectangularDomain
from rl_simulation import AtomicEventTracer, TraceEvaluation, propagate_sensor_constant_velocity
from sensor_network import Sensor, SensorNetwork


class _DummyTopology:
    pass


class _DummyStateChange:
    def __init__(self, kind: str):
        self.kind = kind

    def is_atomic_change(self):
        return self.kind != "non_atomic"

    def alpha_complex_change(self):
        if self.kind == "trivial":
            return (0, 0, 0, 0)
        return (1, 0, 0, 0)

    def boundary_cycle_change(self):
        if self.kind == "trivial":
            return (0, 0)
        return (1, 0)


def _make_eval(elapsed: float, kind: str) -> TraceEvaluation:
    return TraceEvaluation(
        elapsed=elapsed,
        topology=_DummyTopology(),
        state_change=_DummyStateChange(kind),
        mobile_positions=np.zeros((0, 2), dtype=float),
        mobile_velocities=np.zeros((0, 2), dtype=float),
    )


def test_atomic_event_tracer_returns_earliest_nontrivial_atomic_change():
    threshold = 0.3

    def evaluate_at(elapsed: float) -> TraceEvaluation:
        kind = "trivial" if elapsed < threshold else "atomic"
        return _make_eval(elapsed, kind)

    tracer = AtomicEventTracer(evaluate_at, max_depth=30, time_tolerance=1e-5)
    result = tracer.find_first_nontrivial_atomic(1.0)

    assert result is not None
    assert result.elapsed == pytest.approx(threshold, abs=2e-5)


def test_atomic_event_tracer_returns_none_when_interval_is_trivial():
    tracer = AtomicEventTracer(lambda elapsed: _make_eval(elapsed, "trivial"), max_depth=20, time_tolerance=1e-6)

    result = tracer.find_first_nontrivial_atomic(0.8)
    assert result is None


def test_atomic_event_tracer_finds_first_event_even_if_later_region_is_nonatomic():
    first_event = 0.25
    second_event = 0.75

    def evaluate_at(elapsed: float) -> TraceEvaluation:
        if elapsed < first_event:
            kind = "trivial"
        elif elapsed < second_event:
            kind = "atomic"
        else:
            kind = "non_atomic"
        return _make_eval(elapsed, kind)

    tracer = AtomicEventTracer(evaluate_at, max_depth=30, time_tolerance=1e-5)
    result = tracer.find_first_nontrivial_atomic(1.0)

    assert result is not None
    assert result.elapsed == pytest.approx(first_event, abs=2e-5)
    assert tracer.last_diagnostics.evaluation_count >= 1
    assert tracer.last_diagnostics.split_count >= 1
    assert tracer.last_diagnostics.max_recursion_depth >= 1


def test_propagate_sensor_constant_velocity_reflects_at_wall():
    domain = RectangularDomain(0, 1, 0, 1)
    sensing_radius = 0.2
    mobile = [Sensor(np.array([0.9, 0.5]), np.array([0.0, 0.0]), sensing_radius, False)]
    network = SensorNetwork(
        mobile_sensors=mobile,
        motion_model=None,
        fence=[],
        sensing_radius=sensing_radius,
        domain=domain,
    )

    new_pos, new_vel = propagate_sensor_constant_velocity(
        mobile[0],
        velocity=np.array([1.0, 0.2]),
        dt=0.2,
        sensor_network=network,
    )

    assert new_pos == pytest.approx(np.array([0.9, 0.54]))
    assert new_vel == pytest.approx(np.array([-1.0, 0.2]))
