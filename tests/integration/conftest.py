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

from unittest import mock

import pytest


@pytest.fixture
def connected_topology():
    topology = mock.Mock()
    topology.alpha_cycle = 'bcdef'
    return topology


@pytest.fixture
def topology1(connected_topology):
    def simplices(dim):
        return ["abc", "fab"] if dim == 2 else ['ab', 'bc', 'ac', 'cd', 'de', 'ef', 'fa', 'ea', 'fb']

    connected_topology.boundary_cycles.return_value = ['abc', 'cdea', 'efa', 'fab']
    connected_topology.simplices.side_effect = simplices
    return connected_topology


@pytest.fixture
def remove_1simplex(connected_topology):
    # remove EA
    def simplices(dim):
        return ["abc", "fab"] if dim == 2 else ['ab', 'bc', 'ac', 'cd', 'de', 'ef', 'fa', 'fb']

    connected_topology.boundary_cycles.return_value = ['acdef', 'efa', 'abc']
    connected_topology.simplices.side_effect = simplices
    return connected_topology


@pytest.fixture
def remove_2simplex(connected_topology):
    # remove ABC
    def simplices(dim):
        return ["fab"] if dim == 2 else ['ab', 'bc', 'ac', 'cd', 'de', 'ef', 'fa', 'ea', 'fb']

    connected_topology.boundary_cycles.return_value = ['abc', 'cdea', 'efa', 'fab']
    connected_topology.simplices.side_effect = simplices
    return connected_topology


@pytest.fixture
def remove_simplex_pair(connected_topology):
    # remove BC and ABC
    def simplices(dim):
        return ["fab"] if dim == 2 else ['ab', 'ac', 'cd', 'de', 'ef', 'fa', 'ea', 'fb']

    connected_topology.boundary_cycles.return_value = ['abcde', 'efa', 'fab']
    connected_topology.simplices.side_effect = simplices
    return connected_topology


@pytest.fixture
def add_1simplex(connected_topology):
    # add CE
    def simplices(dim):
        return ["abc", "fab"] if dim == 2 else ['ab', 'bc', 'ac', 'cd', 'de', 'ef', 'fa', 'ea', 'fb', 'ce']

    connected_topology.boundary_cycles.return_value = ['abc', 'cde', 'ace', 'efa', 'fab']
    connected_topology.simplices.side_effect = simplices
    return connected_topology


@pytest.fixture
def add_2simplex(connected_topology):
    # add EFA
    def simplices(dim):
        return ["abc", "fab", 'efa'] if dim == 2 else ['ab', 'bc', 'ac', 'cd', 'de', 'ef', 'fa', 'ea', 'fb']

    connected_topology.boundary_cycles.return_value = ['abc', 'cdea', 'efa', 'fab']
    connected_topology.simplices.side_effect = simplices
    return connected_topology


@pytest.fixture
def add_simplex_pair(connected_topology):
    # add CE CDE
    def simplices(dim):
        return ["abc", "fab", 'cde'] if dim == 2 else ['ab', 'bc', 'ac', 'cd', 'de', 'ef', 'fa', 'ea', 'fb', 'ce']

    connected_topology.boundary_cycles.return_value = ['abc', 'cde', 'ace', 'efa', 'fab']
    connected_topology.simplices.side_effect = simplices
    return connected_topology


@pytest.fixture
def delauny_flip(connected_topology):
    # switch ab with fc
    def simplices(dim):
        return ["acf", "fbc"] if dim == 2 else ['fc', 'bc', 'ac', 'cd', 'de', 'ef', 'fa', 'ea', 'fb']

    connected_topology.boundary_cycles.return_value = ["acf", "fbc", 'cdea', 'efa']
    connected_topology.simplices.side_effect = simplices
    return connected_topology
