from unittest import mock

import pytest

from alpha_complex import Simplex, AlphaComplex


def mock_BoundaryCycle(nodes):
    bc = mock.Mock()
    bc.nodes = nodes
    return bc


class TestSimplex:
    def test_init(self):
        nodes = frozenset({1, 2, 3, 4})
        s = Simplex(nodes)
        assert s.nodes == {1, 2, 3, 4}

    def test_is_subface(self):
        simplex = Simplex(frozenset({1, 2, 3}))
        edge = Simplex(frozenset({1, 2}))
        assert simplex.is_subface(edge)

    def test_is_not_subface(self):
        simplex = Simplex(frozenset({1, 2, 3}))
        edge = Simplex(frozenset({1, 4}))
        assert not simplex.is_subface(edge)

    def test_to_cycle(self):
        bc1 = mock_BoundaryCycle({1, 2, 3})
        bc2 = mock_BoundaryCycle({2, 3, 4})
        bc3 = mock_BoundaryCycle({3, 4, 1})
        boundary_cycles = [bc1, bc2, bc3]

        simplex = Simplex(frozenset({1, 2, 3}))
        assert simplex.to_cycle(boundary_cycles) is bc1

    def test_to_cycle_nonunique(self):
        bc1 = mock_BoundaryCycle({1, 2, 3})
        bc2 = mock_BoundaryCycle({2, 3, 4})
        bc3 = mock_BoundaryCycle({1, 2, 3})
        boundary_cycles = [bc1, bc2, bc3]

        simplex = Simplex(frozenset({1, 2, 3}))
        pytest.raises(ValueError, simplex.to_cycle, boundary_cycles)

    def test_to_cycle_notfound(self):
        bc2 = mock_BoundaryCycle({2, 3, 4})
        bc3 = mock_BoundaryCycle({4, 2, 3})
        boundary_cycles = [bc2, bc3]

        simplex = Simplex(frozenset({1, 2, 3}))
        pytest.raises(ValueError, simplex.to_cycle, boundary_cycles)


class TestAlphaComplex:
    points = [(0, 0), (0, 1), (1, 0), (1, 1)]
    radius = 0.5 ** 0.5

    def test_init(self):
        ac = AlphaComplex(self.points, self.radius)
        assert ac.simplex_tree is not None

    def test_init_3d(self):
        points = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0), (0.5, 0.5, 1)]
        ac = AlphaComplex(points, 1 / 2)
        assert ac.simplex_tree is not None

    def test_simplices_correct_length(self):
        ac = AlphaComplex(self.points, self.radius)
        assert len(ac.simplices(1)[0]) == 1 and len(ac.simplices(2)[0]) == 2

    def test_num_nodes(self):
        ac = AlphaComplex(self.points, self.radius)
        assert ac.nodes == {0, 1, 2, 3}
