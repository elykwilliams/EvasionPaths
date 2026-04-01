# ******************************************************************************
#  Copyright (c) 2026, Contributors - All Rights Reserved.
# ******************************************************************************

import random

import networkx as nx
import numpy as np

from alpha_complex import Simplex
from boundary_geometry import UnitCube, get_unitcube_fence
from combinatorial_map import BoundaryCycle, OrientedSimplex
from motion_model import BilliardMotion
from sensor_network import SensorNetwork, generate_mobile_sensors
from time_stepping import EvasionPathSimulation
from topology import Topology, generate_topology


def _build_topology(*, seed: int, radius: float, num_mobile: int):
    np.random.seed(seed)
    random.seed(seed)

    domain = UnitCube()
    fence = get_unitcube_fence(radius)
    mobile = generate_mobile_sensors(domain, num_mobile, radius, 1.0) if num_mobile else []
    network = SensorNetwork(mobile, BilliardMotion(), fence, radius, domain)
    topology = generate_topology(
        network.points,
        radius,
        fence_node_count=len(fence),
        interior_point=np.array([0.5, 0.5, 0.5], dtype=float),
    )
    return topology, network


def test_outer_cycle_is_fence_dominant_without_mobile_sensors():
    topology, network = _build_topology(seed=1, radius=0.35, num_mobile=0)
    outer = topology.outer_cycle
    fence_ids = set(range(len(network.fence_sensors)))

    assert outer in topology.homology_generators
    assert len(outer) > 0
    assert set(outer.nodes).issubset(fence_ids)


def test_unitcube_previous_crash_case_initializes():
    np.random.seed(2)
    random.seed(2)
    domain = UnitCube()
    radius = 0.4
    fence = get_unitcube_fence(radius)
    mobile = generate_mobile_sensors(domain, 8, radius, 1.0)
    sim = EvasionPathSimulation(
        SensorNetwork(mobile, BilliardMotion(), fence, radius, domain),
        dt=0.01,
        end_time=0.01,
    )

    assert sim.topology.outer_cycle is not None
    assert len(sim.cycle_label.label) > 0


def test_component_id_connectivity_matches_has_path_logic():
    topology, _network = _build_topology(seed=1, radius=0.35, num_mobile=8)
    graph = topology.face_connectivity_graph
    anchor = topology._outer_anchor_face()

    for cycle in topology.homology_generators:
        source = Simplex(next(iter(cycle)).nodes)
        expected = bool(anchor is not None and source in graph and nx.has_path(graph, source, anchor))
        assert topology.is_connected_cycle(cycle) == expected


def test_winding_breaks_orientation_tie_between_shell_twins():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    interior = np.array([0.1, 0.1, 0.1], dtype=float)

    faces = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)]
    outer_like = BoundaryCycle(frozenset(OrientedSimplex(face) for face in faces))
    inner_like = BoundaryCycle(frozenset(OrientedSimplex(face[::-1]) for face in faces))

    class _FakeAlphaComplex:
        dim = 3

        @staticmethod
        def simplices(dim):
            return set()

    class _FakeRotationInfo:
        def __init__(self, pts):
            self.points = pts
            self.incident_simplices = {}

    class _FakeMap:
        def __init__(self, pts, cycles):
            self.rotation_info = _FakeRotationInfo(pts)
            self.boundary_cycles = set(cycles)

        @staticmethod
        def get_cycle(_simplex):
            raise KeyError

    topology = Topology(
        _FakeAlphaComplex(),
        _FakeMap(points, [outer_like, inner_like]),
        fence_node_count=4,
        interior_point=interior,
    )

    w_outer = topology._signed_winding_number(outer_like, interior)
    w_inner = topology._signed_winding_number(inner_like, interior)
    assert w_outer * w_inner < 0
    assert abs(abs(w_outer) - 1.0) < 1e-6
    assert abs(abs(w_inner) - 1.0) < 1e-6

    selected = topology.outer_cycle
    assert topology._signed_winding_number(selected, interior) < 0


def test_winding_sign_switch_can_select_positive_orientation():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    interior = np.array([0.1, 0.1, 0.1], dtype=float)

    faces = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)]
    neg_cycle = BoundaryCycle(frozenset(OrientedSimplex(face) for face in faces))
    pos_cycle = BoundaryCycle(frozenset(OrientedSimplex(face[::-1]) for face in faces))

    class _FakeAlphaComplex:
        dim = 3

        @staticmethod
        def simplices(dim):
            return set()

    class _FakeRotationInfo:
        def __init__(self, pts):
            self.points = pts
            self.incident_simplices = {}

    class _FakeMap:
        def __init__(self, pts, cycles):
            self.rotation_info = _FakeRotationInfo(pts)
            self.boundary_cycles = set(cycles)

        @staticmethod
        def get_cycle(_simplex):
            raise KeyError

    topology = Topology(
        _FakeAlphaComplex(),
        _FakeMap(points, [neg_cycle, pos_cycle]),
        fence_node_count=4,
        interior_point=interior,
        outer_winding_sign=1,
    )
    selected = topology.outer_cycle
    assert topology._signed_winding_number(selected, interior) > 0
