from unittest import mock

import pytest

from cycle_labelling import CycleLabellingTree
from update_data import get_label_update, InvalidStateChange


@pytest.fixture
def initial_topology():
    topology = mock.Mock()
    topology.alpha_cycle = 'bcdef'
    topology.boundary_cycles.return_value = ['abc', 'cdea', 'efa', 'fab']
    topology.simplices.side_effect \
        = lambda dim: ["abc", "fab"] if dim == 2 else ['ab', 'bc', 'ac', 'cd', 'de', 'ef', 'fa', 'ea', 'fb']
    return topology


@pytest.fixture
def remove_1simplex_topology():
    # remove EA
    topology = mock.Mock()
    topology.alpha_cycle = 'bcdef'
    topology.boundary_cycles.return_value = ['acdef', 'fab', 'abc']
    topology.simplices.side_effect \
        = lambda dim: ["abc", "fab"] if dim == 2 else ['ab', 'bc', 'ac', 'cd', 'de', 'ef', 'fa', 'fb']
    return topology


@pytest.fixture
def non_atomic_topology():
    # remove EA and AC
    topology = mock.Mock()
    topology.alpha_cycle = 'bcdef'
    topology.boundary_cycles.return_value = ['abcdef', 'bfa']
    topology.simplices.side_effect \
        = lambda dim: ["abc", "fab"] if dim == 2 else ['ab', 'bc', 'cd', 'de', 'ef', 'fa', 'fb']
    return topology


class TestGetLabelUpdate:
    # label_update will produce a valid dictionary or object of type InvalidStateChange

    # TODO Paramterize
    def test_doesnt_return_none(self, initial_topology, remove_1simplex_topology):
        topology1 = initial_topology
        topology2 = remove_1simplex_topology
        cycle_labelling = CycleLabellingTree(initial_topology)
        update = get_label_update(cycle_labelling, topology1, topology2)
        assert update is not None and not isinstance(update, InvalidStateChange)

    def test_nonatomic_returns_InvalidStateChange(self, initial_topology, non_atomic_topology):
        topology1 = initial_topology
        topology2 = non_atomic_topology
        cycle_labelling = CycleLabellingTree(initial_topology)
        label_update = get_label_update(cycle_labelling, topology1, topology2)
        assert isinstance(label_update, InvalidStateChange)

    def test_invalid_returnInvalidStateChange(self, initial_topology, invalid_topology):
        assert False

    # TODO Paramaterize
    def test_returns_correct_case(self, initial_topology, valid_topology):
        assert False
