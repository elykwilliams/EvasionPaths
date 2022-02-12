from typing import FrozenSet
from unittest import mock

import pytest
from dataclasses import dataclass

from alpha_complex import Simplex
from combinatorial_map import CombinatorialMap2D, RotationInfo2D
from cycle_labelling import CycleLabellingDict
from state_change import StateChange2D
from topology import ConnectedTopology2D
from update_data import LabelUpdateFactory, Remove1SimplexUpdate2D


@pytest.fixture
def mock_rotinfo():
    adj = {0: [1, 5],
           1: [2, 3, 5, 0],
           2: [3, 1],
           3: [4, 1, 2],
           4: [5, 3],
           5: [4, 0, 1]}

    def next_dart(dart):
        v1, v2 = dart
        index = adj[v1].index(v2)
        return v1, adj[v1][(index + 1) % len(adj[v1])]

    s = mock.Mock()
    s.next.side_effect = next_dart

    s.all_darts = sum(([(v1, v2) for v2 in adj[v1]] for v1 in adj), [])
    return s


@dataclass
class Simplex:
    nodes: FrozenSet[int]

    def is_subface(self, subface):
        return subface.nodes.issubset(self.nodes)

    def to_cycle(self, boundary_cycles):
        cycles_found = [cycle for cycle in boundary_cycles if self.nodes == cycle.nodes]
        return cycles_found.pop()

    def __len__(self):
        return len(self.nodes) - 1

    def __eq__(self, other):
        return self.nodes == other.nodes

    def __hash__(self):
        return hash(self.nodes)


def mock_alphacomplex(simplices, edges):
    ac = mock.Mock()
    ac.simplices.side_effect = \
        lambda dim: [Simplex(frozenset(edge)) for edge in edges] if dim == 1 \
            else [Simplex(frozenset(simplex)) for simplex in simplices]
    ac.nodes = {0, 1, 2, 3, 4, 5}
    return ac


@pytest.mark.fixture
def topology(simplices, edges, points):
    ac = mock_alphacomplex(simplices, edges)
    rotinfo = RotationInfo2D(points, ac)
    cmap = CombinatorialMap2D(rotinfo)
    return ConnectedTopology2D(ac, cmap)


class TestIntegrateCMap:
    result = {
        frozenset({(0, 5), (5, 1), (1, 0)}),
        frozenset({(5, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)}),
        frozenset({(2, 1), (1, 3), (3, 2)}),
        frozenset({(3, 1), (1, 5), (5, 4), (4, 3)})
    }

    simplices = [(0, 1, 5)]
    edges = [(5, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (1, 3)]
    points = [(2, 0), (1, 1), (0, 2), (0, 1), (0, 0), (1, 0)]

    def test_mockrotinfo(self, mock_rotinfo):
        cmap = CombinatorialMap2D(mock_rotinfo)
        assert set(cycle.darts for cycle in cmap.boundary_cycles) == self.result

    def test_mock_alphacomplex(self):
        rotinfo = RotationInfo2D(self.points, mock_alphacomplex(self.simplices, self.edges))
        cmap = CombinatorialMap2D(rotinfo)
        assert set(cycle.darts for cycle in cmap.boundary_cycles) == self.result

    def test_integrate_topology(self):
        topology1 = topology(self.simplices, self.edges, self.points)
        assert set(cycle.darts for cycle in topology1.boundary_cycles) == self.result

    def test_integrate_labelling(self):
        topology1 = topology(self.simplices, self.edges, self.points)
        labelling = CycleLabellingDict(topology1)
        assert len(labelling.dict) == 4

    def test_integrate_labelling_false(self):
        topology1 = topology(self.simplices, self.edges, self.points)
        labelling = CycleLabellingDict(topology1)
        cycle = topology1.cmap.get_cycle((0, 5))
        assert not labelling[cycle]

    def test_integrate_sc(self):
        topology1 = topology(self.simplices, self.edges, self.points)

        edges = [(5, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]
        topology2 = topology(self.simplices, edges, self.points)

        sc = StateChange2D(topology2, topology1)
        assert sc.case == (0, 1, 0, 0, 1, 2)

    def test_integrate_labelupdate(self):
        topology1 = topology(self.simplices, self.edges, self.points)
        labelling = CycleLabellingDict(topology1)

        edges = [(5, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]
        topology2 = topology(self.simplices, edges, self.points)

        sc = StateChange2D(topology2, topology1)
        label_update = LabelUpdateFactory().get_update(sc, labelling)
        assert type(label_update) == Remove1SimplexUpdate2D

    def test_integrate_labelling_update(self):
        topology1 = topology(self.simplices, self.edges, self.points)
        labelling = CycleLabellingDict(topology1)

        edges = [(5, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]
        topology2 = topology(self.simplices, edges, self.points)

        sc = StateChange2D(topology2, topology1)
        label_update = LabelUpdateFactory().get_update(sc, labelling)
        labelling.update(label_update)
        assert len(labelling.dict) == 3

    def test_check_labels(self):
        topology1 = topology(self.simplices, self.edges, self.points)
        labelling = CycleLabellingDict(topology1)

        edges = [(5, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]
        topology2 = topology(self.simplices, edges, self.points)

        sc = StateChange2D(topology2, topology1)
        label_update = LabelUpdateFactory().get_update(sc, labelling)
        labelling.update(label_update)

        cycle1 = topology2.cmap.get_cycle((0, 5))
        cycle2 = topology2.cmap.get_cycle((3, 2))
        assert (not labelling[cycle1]) and labelling[cycle2]
