from itertools import chain
from unittest import mock

import pytest

from combinatorial_map import BoundaryCycle, CombinatorialMap2D, RotationInfo2D, OrientedSimplex


class TestBoundaryCycle:
    darts = [(1, 2), (2, 3), (3, 1)]

    @pytest.fixture
    def single_cycle(self):
        oriented_simplices = frozenset({OrientedSimplex(edge) for edge in self.darts})
        return BoundaryCycle(oriented_simplices)

    def test_init(self, single_cycle):
        assert single_cycle.oriented_simplices is not None

    def test_iter(self, single_cycle):
        oriented_simplices = frozenset({OrientedSimplex(edge) for edge in self.darts})
        assert all(s in oriented_simplices for s in single_cycle)

    def test_nodes(self, single_cycle):
        assert single_cycle.nodes == {1, 2, 3}


class TestCombinatorialMap2D:
    def test_init(self):
        mock_rotinfo = mock.Mock()
        cmap = CombinatorialMap2D(mock_rotinfo)
        assert not cmap._boundary_cycles

    def test_alpha(self):
        assert CombinatorialMap2D.alpha(OrientedSimplex((1, 0))) == OrientedSimplex((0, 1))

    def test_phi(self):
        mock_rotinfo = mock.Mock()
        mock_rotinfo.next.return_value = OrientedSimplex((1, 5))
        cmap = CombinatorialMap2D(mock_rotinfo)
        assert cmap.phi(OrientedSimplex((1, 0)), OrientedSimplex((1,))) == OrientedSimplex((5, 1))

    def test_get_cycle(self):
        expected_cycle = [OrientedSimplex((0, 5)), OrientedSimplex((5, 1)), OrientedSimplex((1, 0))]
        mock_rotinfo = mock.Mock()
        mock_rotinfo.next.side_effect = [OrientedSimplex((1, 5)), OrientedSimplex((5, 0)), OrientedSimplex((0, 1))]

        cmap = CombinatorialMap2D(mock_rotinfo)
        cycle = cmap.get_cycle(OrientedSimplex((1, 0)))
        assert cycle == BoundaryCycle(frozenset(expected_cycle))

    def test_get_cycle_cached_all_simplices(self):
        expected_cycle = [OrientedSimplex((0, 5)), OrientedSimplex((5, 1)), OrientedSimplex((1, 0))]
        mock_rotinfo = mock.Mock()
        mock_rotinfo.next.side_effect = [simplex.alpha() for simplex in expected_cycle]
        cmap = CombinatorialMap2D(mock_rotinfo)
        cmap.get_cycle(OrientedSimplex((1, 0)))
        assert all(s in cmap._simplices_map for s in expected_cycle)

    def test_boundary_cycles(self):
        mock_rotinfo = mock.Mock()
        mock_rotinfo.rotinfo = {
            OrientedSimplex((0,)): [OrientedSimplex((0, n)) for n in (1, 5)],
            OrientedSimplex((1,)): [OrientedSimplex((1, n)) for n in (2, 3, 5, 0)],
            OrientedSimplex((2,)): [OrientedSimplex((2, n)) for n in (3, 1)],
            OrientedSimplex((3,)): [OrientedSimplex((3, n)) for n in (4, 1, 2)],
            OrientedSimplex((4,)): [OrientedSimplex((4, n)) for n in (5, 3)],
            OrientedSimplex((5,)): [OrientedSimplex((5, n)) for n in (4, 0, 1)]
        }

        mock_rotinfo.next.side_effect = lambda cell, node: RotationInfo2D.next(mock_rotinfo, cell, node)
        mock_rotinfo.oriented_simplices = set(chain.from_iterable(mock_rotinfo.rotinfo.values()))

        cmap = CombinatorialMap2D(mock_rotinfo)
        cycles = cmap.boundary_cycles
        assert len(cycles) == 4


class TestRotInfo:
    points = [(2, 0), (1, 1), (0, 2), (0, 1), (0, 0), (1, 0)]
    edges = [(5, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (1, 3)]

    @pytest.mark.fixture
    def Simplex(self, nodes):
        m = mock.Mock()
        m.nodes = nodes
        return m

    @pytest.fixture
    def alpha_complex(self):
        ac = mock.Mock()
        ac.simplices.side_effect = lambda dim: {
            0: {self.Simplex((n,)) for n in range(6)},
            1: {self.Simplex(edge) for edge in self.edges}
        }[dim]
        ac.nodes = [0, 1, 2, 3, 4, 5]
        return ac

    def test_init(self, alpha_complex):
        ri = RotationInfo2D(self.points, alpha_complex)
        assert ri.rotinfo is not None

    def test_adjacency(self, alpha_complex):
        ri = RotationInfo2D(self.points, alpha_complex)

        adj = {
            OrientedSimplex((0,)): [OrientedSimplex((0, n)) for n in (5, 1)],
            OrientedSimplex((1,)): [OrientedSimplex((1, n)) for n in (0, 5, 3, 2)],
            OrientedSimplex((2,)): [OrientedSimplex((2, n)) for n in (1, 3)],
            OrientedSimplex((3,)): [OrientedSimplex((3, n)) for n in (2, 1, 4)],
            OrientedSimplex((4,)): [OrientedSimplex((4, n)) for n in (3, 5)],
            OrientedSimplex((5,)): [OrientedSimplex((5, n)) for n in (1, 0, 4)]
        }
        assert all(set(ri.rotinfo[n]) == set(adj[n]) for n in adj)

    def test_next(self, alpha_complex):
        ri = RotationInfo2D(self.points, alpha_complex)
        dart = ri.next(OrientedSimplex((5, 0)), OrientedSimplex((5,)))
        assert dart == OrientedSimplex((5, 4))
