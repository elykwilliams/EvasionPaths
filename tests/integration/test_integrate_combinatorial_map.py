from itertools import chain
from unittest import mock

import pytest

from alpha_complex import Simplex
from combinatorial_map import CombinatorialMap2D, RotationInfo2D, OrientedSimplex, BoundaryCycle
from cycle_labelling import CycleLabellingDict
from state_change import StateChange
from topology import Topology
from update_data import LabelUpdateFactory, Remove1SimplexUpdate2D


@pytest.fixture
def mock_rotinfo():
    mock_rotinfo = mock.Mock()
    mock_rotinfo.rotinfo = {
        OrientedSimplex((0,)): [OrientedSimplex((0, n)) for n in (5, 1)],
        OrientedSimplex((1,)): [OrientedSimplex((1, n)) for n in (0, 5, 3, 2)],
        OrientedSimplex((2,)): [OrientedSimplex((2, n)) for n in (1, 3)],
        OrientedSimplex((3,)): [OrientedSimplex((3, n)) for n in (2, 1, 4)],
        OrientedSimplex((4,)): [OrientedSimplex((4, n)) for n in (3, 5)],
        OrientedSimplex((5,)): [OrientedSimplex((5, n)) for n in (1, 0, 4)]
    }

    mock_rotinfo.next.side_effect = lambda cell, node: RotationInfo2D.next(mock_rotinfo, cell, node)
    mock_rotinfo.oriented_simplices = set(chain.from_iterable(mock_rotinfo.rotinfo.values()))

    return mock_rotinfo


def mock_alphacomplex(simplices, edges):
    ac = mock.Mock()
    ac.simplices.side_effect = \
        lambda dim: [Simplex(frozenset(edge)) for edge in edges] if dim == 1 \
            else [Simplex(frozenset(simplex)) for simplex in simplices]
    ac.nodes = {0, 1, 2, 3, 4, 5}
    ac.dim = 2
    return ac


@pytest.mark.fixture
def topology(simplices, edges, points):
    ac = mock_alphacomplex(simplices, edges)
    rotinfo = RotationInfo2D(points, ac)
    cmap = CombinatorialMap2D(rotinfo)
    return Topology(ac, cmap)


class TestIntegrateCMap:
    cycle1 = {OrientedSimplex(simplex) for simplex in {(0, 5), (5, 1), (1, 0)}}
    cycle2 = {OrientedSimplex(simplex) for simplex in {(5, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)}}
    cycle3 = {OrientedSimplex(simplex) for simplex in {(2, 1), (1, 3), (3, 2)}}
    cycle4 = {OrientedSimplex(simplex) for simplex in {(3, 1), (1, 5), (5, 4), (4, 3)}}

    result = {
        BoundaryCycle(frozenset(cycle1)), BoundaryCycle(frozenset(cycle2)),
        BoundaryCycle(frozenset(cycle3)), BoundaryCycle(frozenset(cycle4))
    }

    simplices = [(0, 1, 5)]
    edges = [(5, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (1, 3)]
    points = [(2, 0), (1, 1), (0, 2), (0, 1), (0, 0), (1, 0)]

    def test_from_mockrotinfo(self, mock_rotinfo):
        cmap = CombinatorialMap2D(mock_rotinfo)
        assert cmap.boundary_cycles == self.result

    def test_mock_alphacomplex(self):
        rotinfo = RotationInfo2D(self.points, mock_alphacomplex(self.simplices, self.edges))
        cmap = CombinatorialMap2D(rotinfo)
        assert cmap.boundary_cycles == self.result

    def test_integrate_topology(self):
        topology1 = topology(self.simplices, self.edges, self.points)
        assert topology1.boundary_cycles == self.result

    def test_integrate_labelling(self):
        topology1 = topology(self.simplices, self.edges, self.points)
        labelling = CycleLabellingDict(topology1)
        assert len(labelling.dict) == 4

    def test_integrate_labelling_false(self):
        topology1 = topology(self.simplices, self.edges, self.points)
        labelling = CycleLabellingDict(topology1)
        cycle = topology1.cmap.get_cycle(OrientedSimplex((0, 5)))
        assert not labelling[cycle]

    def test_integrate_sc(self):
        topology1 = topology(self.simplices, self.edges, self.points)

        edges = [(5, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]
        topology2 = topology(self.simplices, edges, self.points)

        sc = StateChange(topology2, topology1)
        assert sc.case == (0, 1, 0, 0, 1, 2)

    def test_integrate_labelupdate(self):
        topology1 = topology(self.simplices, self.edges, self.points)
        labelling = CycleLabellingDict(topology1)

        edges = [(5, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]
        topology2 = topology(self.simplices, edges, self.points)

        label_update = LabelUpdateFactory().get_update(topology2, topology1, labelling)
        assert type(label_update) == Remove1SimplexUpdate2D

    def test_integrate_labelling_update(self):
        topology1 = topology(self.simplices, self.edges, self.points)
        labelling = CycleLabellingDict(topology1)

        edges = [(5, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]
        topology2 = topology(self.simplices, edges, self.points)

        label_update = LabelUpdateFactory().get_update(topology2, topology1, labelling)
        labelling.update(label_update)
        assert len(labelling.dict) == 3

    def test_check_labels(self):
        topology1 = topology(self.simplices, self.edges, self.points)
        labelling = CycleLabellingDict(topology1)

        edges = [(5, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]
        topology2 = topology(self.simplices, edges, self.points)

        label_update = LabelUpdateFactory().get_update(topology2, topology1, labelling)
        labelling.update(label_update)

        cycle1 = topology2.cmap.get_cycle(OrientedSimplex((0, 5)))
        cycle2 = topology2.cmap.get_cycle(OrientedSimplex((2, 3)))
        assert (not labelling[cycle1]) and (not labelling[cycle2])
