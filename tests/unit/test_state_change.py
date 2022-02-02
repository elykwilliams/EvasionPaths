from unittest import mock
from unittest.mock import patch

import pytest

from state_change import StateChange2D


## All changes done with respect to initial topology: topology1
# Edges as shown, FAB and ABC are 2-simplices
#        D
#      /  \
#    /      \
#   E        \
#  |  \       \
#  |    \      \
#  F --- A ----C
#   \    |    /
#     \  |   /
#       \| /
#        B


@pytest.fixture
def concrete_topology():
    topology = mock.Mock()
    topology.alpha_cycle = 'bcdef'
    return topology


@pytest.fixture
def initial_topology():
    topology = concrete_topology()
    topology.boundary_cycles.return_value = ['abc', 'cdea', 'efa', 'fab']
    topology.simplices.side_effect \
        = lambda dim: ["abc", "fab"] if dim == 2 else ['ab', 'bc', 'ac', 'cd', 'de', 'ef', 'fa', 'ea', 'fb']
    return topology


@pytest.fixture
def trivial_topology():
    topology = concrete_topology()
    topology.boundary_cycles.return_value = ['fedcb']
    topology.simplices.side_effect = lambda dim: []
    return topology


@pytest.fixture
def remove_1simplex_topology():
    # remove EA
    topology = concrete_topology()
    topology.boundary_cycles.return_value = ['acdef', 'fab', 'abc']
    topology.simplices.side_effect \
        = lambda dim: ["abc", "fab"] if dim == 2 else ['ab', 'bc', 'ac', 'cd', 'de', 'ef', 'fa', 'fb']
    return topology


@pytest.fixture
def remove_2simplex_topology():
    # remove ABC
    topology = concrete_topology()
    topology.boundary_cycles.return_value = ['abc', 'cdea', 'efa', 'fab']
    topology.simplices.side_effect = \
        lambda dim: ["fab"] if dim == 2 else ['ab', 'bc', 'ac', 'cd', 'de', 'ef', 'fa', 'ea', 'fb']
    return topology


@pytest.fixture
def remove_simplex_pair_topology():
    # remove BC and ABC
    topology = concrete_topology()
    topology.boundary_cycles.return_value = ['abcde', 'efa', 'fab']
    topology.simplices.side_effect \
        = lambda dim: ["fab"] if dim == 2 else ['ab', 'ac', 'cd', 'de', 'ef', 'fa', 'ea', 'fb']
    return topology


@pytest.fixture
def add_1simplex_topology():
    # add CE
    topology = concrete_topology()
    topology.boundary_cycles.return_value = ['abc', 'cde', 'ace', 'efa', 'fab']
    topology.simplices.side_effect \
        = lambda dim: ["abc", "fab"] if dim == 2 else ['ab', 'bc', 'ac', 'cd', 'de', 'ef', 'fa', 'ea', 'fb', 'ce']
    return topology


@pytest.fixture
def add_2simplex_topology():
    # add EFA
    topology = concrete_topology()
    topology.boundary_cycles.return_value = ['abc', 'cdea', 'efa', 'fab']
    topology.simplices.side_effect \
        = lambda dim: ["abc", "fab", 'efa'] if dim == 2 else ['ab', 'bc', 'ac', 'cd', 'de', 'ef', 'fa', 'ea', 'fb']
    return topology


@pytest.fixture
def add_simplex_pair_topology():
    # add CE CDE
    topology = concrete_topology()
    topology.boundary_cycles.return_value = ['abc', 'cde', 'ace', 'efa', 'fab']
    topology.simplices.side_effect \
        = lambda dim: ["abc", "fab", 'cde'] if dim == 2 \
        else ['ab', 'bc', 'ac', 'cd', 'de', 'ef', 'fa', 'ea', 'fb', 'ce']
    return topology


@pytest.fixture
def delauny_flip_topology():
    # switch ab with fc
    topology = concrete_topology()
    topology.boundary_cycles.return_value = ["acf", "fbc", 'cdea', 'efa']
    topology.simplices.side_effect \
        = lambda dim: ["acf", "fbc"] if dim == 2 else ['fc', 'bc', 'ac', 'cd', 'de', 'ef', 'fa', 'ea', 'fb']
    return topology


@pytest.fixture
def non_atomic_topology():
    # remove EA and AC
    topology = concrete_topology()
    topology.boundary_cycles.return_value = ['abcdef', 'bfa']
    topology.simplices.side_effect \
        = lambda dim: ["abc", "fab"] if dim == 2 else ['ab', 'bc', 'cd', 'de', 'ef', 'fa', 'fb']
    return topology


@pytest.mark.fixture
def Simplex(name, edges=(), nodes=()):
    s = mock.Mock()
    s.to_cycle.return_value = name
    s.is_subface.side_effect = lambda e: True if e in edges else False
    s.nodes = nodes
    return s


class TestStateChange:
    @pytest.fixture
    def topology1(self):
        t = mock.Mock()
        t.simplices.side_effect = lambda dim: ["B", "C"] if dim == 2 else ["bc"]
        t.boundary_cycles.return_value = ["B", "C"]
        return t

    @pytest.fixture
    def topology2(self):
        t = mock.Mock()
        t.simplices.side_effect = lambda dim: ["B", "C", "D"] if dim == 2 else ["bc", "cd"]
        t.boundary_cycles.return_value = ["B", "C", "D"]
        return t

    def test_init(self):
        t1 = mock.Mock()
        t2 = mock.Mock()
        sc = StateChange2D(t1, t2)
        assert sc.old_topology is not None and sc.new_topology is not None

    @patch("state_change.SetDifference")
    def test_simplices2(self, mock_set_diff, topology1, topology2):
        sc = StateChange2D(topology2, topology1)
        _ = sc.simplices(2)
        mock_set_diff.assert_called_once_with(["B", "C", "D"], ["B", "C"])

    @patch("state_change.SetDifference")
    def test_simplices1(self, mock_set_diff, topology1, topology2):
        sc = StateChange2D(topology2, topology1)
        _ = sc.simplices(1)
        mock_set_diff.assert_called_once_with(["bc", "cd"], ["bc"])

    @patch("state_change.SetDifference")
    def test_boundary_cycels(self, mock_set_diff, topology1, topology2):
        sc = StateChange2D(topology2, topology1)
        _ = sc.boundary_cycles
        mock_set_diff.assert_called_once_with(["B", "C", "D"], ["B", "C"])

    def test_is_valid(self, topology1, topology2):
        topology2.simplices.side_effect = lambda dim: [Simplex("B"), Simplex("C"), Simplex("D")]
        topology1.simplices.side_effect = lambda dim: [Simplex("B"), Simplex("C")]
        sc = StateChange2D(topology2, topology1)
        assert sc.is_valid()

    def test_invalid_remove(self, topology1, topology2):
        topology2.simplices.side_effect = lambda dim: [Simplex("B"), Simplex("C"), Simplex("D")]
        topology1.simplices.side_effect = lambda dim: [Simplex("B"), Simplex("F")]
        sc = StateChange2D(topology2, topology1)
        assert not sc.is_valid()

    def test_invalid_add(self, topology1, topology2):
        topology2.simplices.side_effect = lambda dim: [Simplex("B"), Simplex("C"), Simplex("D")]
        topology1.simplices.side_effect = lambda dim: [Simplex("B"), Simplex("C")]
        topology2.boundary_cycles.return_value = ["B", "C", "E"]
        sc = StateChange2D(topology2, topology1)
        assert not sc.is_valid()

    def test_simplex_len(self, topology1, topology2):
        simplexB = Simplex("B")
        simplexC = Simplex("C")
        topology2.simplices.side_effect = lambda dim: [simplexB, simplexC, Simplex("D")]
        topology1.simplices.side_effect = lambda dim: [simplexB, simplexC]
        sc = StateChange2D(topology2, topology1)
        assert len(sc.simplices(2).added()) == 1 and len(sc.simplices(2).removed()) == 0

    def test_case(self, topology1, topology2):
        simplexB = Simplex("B")
        simplexC = Simplex("C")
        topology2.simplices.side_effect = lambda dim: [simplexB, simplexC, Simplex("D")]
        topology1.simplices.side_effect = lambda dim: [simplexB, simplexC]
        topology2.boundary_cycles.return_value = ["B", "C", "D"]
        sc = StateChange2D(topology2, topology1)
        assert sc.case == (1, 0, 1, 0, 1, 0)
