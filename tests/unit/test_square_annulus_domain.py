# ******************************************************************************
#  Copyright (c) 2026, Contributors - All Rights Reserved.
# ******************************************************************************

import numpy as np

from boundary_geometry import SquareAnnulusDomain
from combinatorial_map import BoundaryCycle, OrientedSimplex
from cycle_labelling import CycleLabelling
from motion_model import BilliardMotion
from sensor_network import Sensor, generate_annulus_fence_sensors
from topology import Topology


class _FakeAlphaComplex:
    dim = 2

    @staticmethod
    def simplices(_dim):
        return set()


class _FakeRotationInfo:
    incident_simplices = {}
    points = np.zeros((0, 2), dtype=float)


class _FakeMap:
    def __init__(self, cycles):
        self.boundary_cycles = set(cycles)
        self.rotation_info = _FakeRotationInfo()

    @staticmethod
    def get_cycle(_simplex):
        raise KeyError


def _make_cycle(edges):
    return BoundaryCycle(frozenset(OrientedSimplex(edge) for edge in edges))


def test_square_annulus_membership_and_intersections():
    domain = SquareAnnulusDomain(sensor_radius=1.0)

    assert np.array([4.5, 0.0]) in domain
    assert np.array([1.0, 0.0]) not in domain
    assert np.array([6.0, 0.0]) not in domain

    inner_hit = domain.get_intersection_point(np.array([3.0, 0.0]), np.array([0.0, 0.0]))
    outer_hit = domain.get_intersection_point(np.array([4.0, 0.0]), np.array([6.0, 0.0]))

    np.testing.assert_allclose(inner_hit, np.array([2.0, 0.0]))
    np.testing.assert_allclose(outer_hit, np.array([5.0, 0.0]))


def test_square_annulus_reflects_off_inner_hole():
    domain = SquareAnnulusDomain(sensor_radius=1.0)
    sensor = Sensor(np.array([3.0, 0.0]), np.array([-1.0, 0.0]), 1.0, False)
    sensor.old_pos = np.array([3.0, 0.0])
    sensor.pos = np.array([0.0, 0.0])
    sensor.old_vel = np.array([-1.0, 0.0])
    sensor.vel = np.array([-1.0, 0.0])

    BilliardMotion.reflect(domain, sensor)

    np.testing.assert_allclose(sensor.pos, np.array([4.0, 0.0]))
    np.testing.assert_allclose(sensor.vel, np.array([1.0, 0.0]))


def test_generate_annulus_fence_sensors_builds_outer_and_inner_groups():
    radius = 1.0
    domain = SquareAnnulusDomain(sensor_radius=radius)
    fence_bundle = generate_annulus_fence_sensors(domain, radius)

    assert len(fence_bundle.fence_groups) == 2
    assert fence_bundle.excluded_fence_groups == (0, 1)

    outer_group, inner_group = fence_bundle.fence_groups
    assert outer_group == tuple(range(len(outer_group)))
    assert inner_group == tuple(range(len(outer_group), len(fence_bundle.sensors)))

    dx = radius * np.sin(np.pi / 3)
    outer_points = np.asarray([fence_bundle.sensors[idx].pos for idx in outer_group], dtype=float)
    inner_points = np.asarray([fence_bundle.sensors[idx].pos for idx in inner_group], dtype=float)

    assert np.allclose(np.max(np.abs(outer_points), axis=1), domain.outer_half_side + dx)
    assert np.allclose(np.max(np.abs(inner_points), axis=1), domain.inner_half_side - dx)
    assert all(point not in domain for point in outer_points)
    assert all(point not in domain for point in inner_points)


def test_topology_can_exclude_outer_and_inner_fence_cycles():
    outer_cycle = _make_cycle(((0, 1), (1, 2), (2, 3), (3, 0)))
    inner_cycle = _make_cycle(((4, 5), (5, 6), (6, 7), (7, 4)))
    interior_cycle = _make_cycle(((8, 9), (9, 10), (10, 8)))

    topology = Topology(
        _FakeAlphaComplex(),
        _FakeMap((outer_cycle, inner_cycle, interior_cycle)),
        fence_node_count=8,
        fence_node_groups=((0, 1, 2, 3), (4, 5, 6, 7)),
        excluded_fence_groups=(0, 1),
    )

    assert topology.outer_cycle == outer_cycle
    assert topology.excluded_cycles == (outer_cycle, inner_cycle)
    assert topology.is_excluded_cycle(outer_cycle)
    assert topology.is_excluded_cycle(inner_cycle)
    assert not topology.is_excluded_cycle(interior_cycle)


def test_cycle_labelling_ignores_excluded_cycles_for_intruder_detection():
    outer_cycle = _make_cycle(((0, 1), (1, 2), (2, 3), (3, 0)))
    inner_cycle = _make_cycle(((4, 5), (5, 6), (6, 7), (7, 4)))
    interior_cycle = _make_cycle(((8, 9), (9, 10), (10, 8)))

    topology = Topology(
        _FakeAlphaComplex(),
        _FakeMap((outer_cycle, inner_cycle, interior_cycle)),
        fence_node_count=8,
        fence_node_groups=((0, 1, 2, 3), (4, 5, 6, 7)),
        excluded_fence_groups=(0, 1),
    )

    cycle_label = CycleLabelling(topology)
    cycle_label.label = {
        outer_cycle: True,
        inner_cycle: True,
    }
    assert not cycle_label.has_intruder()

    cycle_label.label[interior_cycle] = True
    assert cycle_label.has_intruder()

