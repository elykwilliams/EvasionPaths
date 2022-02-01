from unittest import mock

import pytest

from update_data2 import StateChange


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


def Simplex(name):
    s = mock.Mock()
    s.to_cycle.return_value = name
    return s


def SetDifference(new, old):
    s = mock.Mock()
    s.added.return_value = new
    s.removed.return_value = old
    return s


class TestStateChange:
    def test_case(self):
        edges = SetDifference([None] * 1, [None] * 2)
        simplices = SetDifference([None] * 3, [None] * 4)
        boundary_cycles = SetDifference([None] * 5, [None] * 6)
        sc = StateChange(edges, simplices, boundary_cycles)

        assert sc.case == (1, 2, 3, 4, 5, 6)

    def test_is_valid_added(self):
        simplex = Simplex("A")

        simplices = mock.Mock()
        simplices.added.return_value = [simplex]
        simplices.removed.return_value = []

        b_cycles = mock.Mock()
        b_cycles.new_list = ["A", "B"]
        b_cycles.old_list = ["B"]

        edges = mock.Mock()

        sc = StateChange(edges, simplices, b_cycles)
        assert sc.is_valid()

    def test_is_valid_removed(self):
        simplex = Simplex("A")

        simplices = mock.Mock()
        simplices.added.return_value = []
        simplices.removed.return_value = [simplex]

        b_cycles = mock.Mock()
        b_cycles.new_list = ["B"]
        b_cycles.old_list = ["A", "B"]

        edges = mock.Mock()

        sc = StateChange(edges, simplices, b_cycles)
        assert sc.is_valid()

    def test_invalid_add_simplex(self):
        simplex = Simplex("A")

        simplices = mock.Mock()
        simplices.added.return_value = [simplex]
        simplices.removed.return_value = []

        b_cycles = mock.Mock()
        b_cycles.new_list = ["C"]
        b_cycles.old_list = []

        edges = mock.Mock()

        sc = StateChange(edges, simplices, b_cycles)
        assert not sc.is_valid()

    def test_invalid_remove_simplex(self):
        simplex = Simplex("A")

        simplices = mock.Mock()
        simplices.added.return_value = []
        simplices.removed.return_value = [simplex]

        b_cycles = mock.Mock()
        b_cycles.new_list = []
        b_cycles.old_list = ["C"]

        edges = mock.Mock()

        sc = StateChange(edges, simplices, b_cycles)
        assert not sc.is_valid()
