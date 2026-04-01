# ******************************************************************************
#  Copyright (c) 2026, Contributors - All Rights Reserved.
# ******************************************************************************

from unittest import mock

from combinatorial_map import BoundaryCycle, OrientedSimplex
from state_change import StateChange
from topology import Topology, generate_topology


def _simplex(name):
    s = mock.Mock()
    s.name = name
    s.nodes = tuple(name)
    return s


def _cycle(nodes):
    oriented_edges = []
    for i in range(len(nodes)):
        oriented_edges.append(OrientedSimplex((nodes[i], nodes[(i + 1) % len(nodes)])))
    return BoundaryCycle(frozenset(oriented_edges))


def _build_topology(*, cycles, simplices_by_dim, dim=2):
    ac = mock.Mock()
    ac.dim = dim
    ac.simplices.side_effect = lambda d: simplices_by_dim.get(d, set())

    cmap = mock.Mock()
    cmap.boundary_cycles = set(cycles)

    return Topology(ac, cmap)


def test_topology_exposes_simplices_and_cycles():
    simplex_c = _simplex("c")
    cycle_a = _cycle((0, 1, 2))
    cycle_b = _cycle((1, 3, 4))
    cycle_c = _cycle((2, 4, 5))
    topology = _build_topology(
        cycles={cycle_a, cycle_b, cycle_c},
        simplices_by_dim={1: {"ab", "bc"}, 2: {simplex_c}},
    )

    assert topology.simplices(2) == {simplex_c}
    assert cycle_a in topology.boundary_cycles


def test_state_change_counts_match_expected_deltas():
    simplex_a = _simplex("a")
    simplex_b = _simplex("b")
    cycle_x = _cycle((0, 1, 2))
    cycle_y = _cycle((1, 2, 3))
    cycle_z = _cycle((0, 3, 4))

    old_topology = _build_topology(
        cycles={cycle_x, cycle_y},
        simplices_by_dim={1: {"ab", "bc"}, 2: {simplex_a}},
    )
    new_topology = _build_topology(
        cycles={cycle_y, cycle_z},
        simplices_by_dim={1: {"ab", "cd"}, 2: {simplex_b}},
    )

    state_change = StateChange(new_topology, old_topology)

    # (+1-simplices, -1-simplices, +2-simplices, -2-simplices)
    assert state_change.alpha_complex_change() == (1, 1, 1, 1)
    # (+boundary generators, -boundary generators)
    assert state_change.boundary_cycle_change() == (1, 1)


def test_generate_topology_constructs_2d_alpha_complex():
    points = [(1.5, 0.5), (1.0, 1.0), (0.5, 1.6), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0)]
    radius = 0.5

    topology = generate_topology(points, radius)

    assert topology.dim == 2
    assert len(topology.simplices(1)) > 0
