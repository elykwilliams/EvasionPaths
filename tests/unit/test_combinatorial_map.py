from unittest import mock
from unittest.mock import patch

import pytest

from combinatorial_map2 import BoundaryCycle2D, CombinatorialMap2D


class TestBoundaryCycle2D:
    def test_init(self):
        darts = [(1, 2), (2, 3), (3, 1)]
        cycle = BoundaryCycle2D(frozenset(darts))
        assert cycle.darts is not None

    def test_iter(self):
        darts = [(1, 2), (2, 3), (3, 1)]
        cycle = BoundaryCycle2D(frozenset(darts))
        assert all(d in darts for d in cycle)

    def test_nodes(self):
        darts = [(1, 2), (2, 3), (3, 1)]
        cycle = BoundaryCycle2D(frozenset(darts))
        assert cycle.nodes == {1, 2, 3}


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


class TestCombinatorialMap2D:
    @patch("combinatorial_map2.CombinatorialMap2D.__post_init__", return_value=None)
    def test_init(self, mock_rotinfo):
        cmap = CombinatorialMap2D(mock_rotinfo)
        assert cmap._boundary_cycles is not None

    @patch("combinatorial_map2.CombinatorialMap2D.__post_init__", return_value=None)
    def test_alpha(self, m, mock_rotinfo):
        cmap = CombinatorialMap2D(mock_rotinfo)
        assert cmap.alpha((1, 0)) == (0, 1)

    @patch("combinatorial_map2.CombinatorialMap2D.__post_init__", return_value=None)
    def test_sigma(self, _, mock_rotinfo):
        cmap = CombinatorialMap2D(mock_rotinfo)
        assert cmap.sigma((1, 0)) == (1, 2)

    @patch("combinatorial_map2.CombinatorialMap2D.__post_init__", return_value=None)
    def test_phi(self, _, mock_rotinfo):
        cmap = CombinatorialMap2D(mock_rotinfo)
        assert cmap.phi((1, 0)) == (0, 5)

    @patch("combinatorial_map2.CombinatorialMap2D.__post_init__", return_value=None)
    def test_generate_cycle_darts(self, _, mock_rotinfo):
        cmap = CombinatorialMap2D(mock_rotinfo)
        darts = cmap._generate_cycle_darts((1, 0))
        assert set(darts) == {(1, 0), (0, 5), (5, 1)}

    def test_post_init(self, mock_rotinfo):
        cmap = CombinatorialMap2D(mock_rotinfo)
        assert len(cmap._boundary_cycles) == 4

    def test_get_cycle(self, mock_rotinfo):
        cmap = CombinatorialMap2D(mock_rotinfo)
        cycle = cmap.get_cycle((5, 1))
        assert cycle == BoundaryCycle2D(frozenset({(5, 1), (1, 0), (0, 5)}))

    def test_boundary_cycles(self, mock_rotinfo):
        cmap = CombinatorialMap2D(mock_rotinfo)
        c1 = frozenset({(0, 5), (5, 1), (1, 0)})
        c2 = frozenset({(5, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)})
        c3 = frozenset({(2, 1), (1, 3), (3, 2)})
        c4 = frozenset({(3, 1), (1, 5), (5, 4), (4, 3)})
        assert set(cycle.darts for cycle in cmap.boundary_cycles) == {c1, c2, c3, c4}
