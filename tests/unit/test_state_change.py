from unittest import mock

import pytest

from cycle_labelling import CycleLabellingTree
from update_data import *
from utilities import UpdateError


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


@pytest.mark.fixture
def concrete_topology():
    topology = mock.Mock()
    topology.alpha_cycle = 'bcdef'
    return topology


@pytest.mark.fixture
def initial_topology():
    topology = concrete_topology()
    topology.boundary_cycles.return_value = ['abc', 'cdea', 'efa', 'fab']
    topology.simplices.side_effect \
        = lambda dim: ["abc", "fab"] if dim == 2 else ['ab', 'bc', 'ac', 'cd', 'de', 'ef', 'fa', 'ea', 'fb']
    return topology


@pytest.mark.fixture
def trivial_topology():
    topology = concrete_topology()
    topology.boundary_cycles.return_value = ['fedcb']
    topology.simplices.side_effect = lambda dim: []
    return topology


@pytest.mark.fixture
def remove_1simplex_topology():
    # remove EA
    topology = concrete_topology()
    topology.boundary_cycles.return_value = ['acdef', 'fab', 'abc']
    topology.simplices.side_effect \
        = lambda dim: ["abc", "fab"] if dim == 2 else ['ab', 'bc', 'ac', 'cd', 'de', 'ef', 'fa', 'fb']
    return topology


@pytest.mark.fixture
def remove_2simplex_topology():
    # remove ABC
    topology = concrete_topology()
    topology.boundary_cycles.return_value = ['abc', 'cdea', 'efa', 'fab']
    topology.simplices.side_effect = \
        lambda dim: ["fab"] if dim == 2 else ['ab', 'bc', 'ac', 'cd', 'de', 'ef', 'fa', 'ea', 'fb']
    return topology


@pytest.mark.fixture
def remove_simplex_pair_topology():
    # remove BC and ABC
    topology = concrete_topology()
    topology.boundary_cycles.return_value = ['abcde', 'efa', 'fab']
    topology.simplices.side_effect \
        = lambda dim: ["fab"] if dim == 2 else ['ab', 'ac', 'cd', 'de', 'ef', 'fa', 'ea', 'fb']
    return topology


@pytest.mark.fixture
def add_1simplex_topology():
    # add CE
    topology = concrete_topology()
    topology.boundary_cycles.return_value = ['abc', 'cde', 'ace', 'efa', 'fab']
    topology.simplices.side_effect \
        = lambda dim: ["abc", "fab"] if dim == 2 else ['ab', 'bc', 'ac', 'cd', 'de', 'ef', 'fa', 'ea', 'fb', 'ce']
    return topology


@pytest.mark.fixture
def add_2simplex_topology():
    # add EFA
    topology = concrete_topology()
    topology.boundary_cycles.return_value = ['abc', 'cdea', 'efa', 'fab']
    topology.simplices.side_effect \
        = lambda dim: ["abc", "fab", 'efa'] if dim == 2 else ['ab', 'bc', 'ac', 'cd', 'de', 'ef', 'fa', 'ea', 'fb']
    return topology


@pytest.mark.fixture
def add_simplex_pair_topology():
    # add CE CDE
    topology = concrete_topology()
    topology.boundary_cycles.return_value = ['abc', 'cde', 'ace', 'efa', 'fab']
    topology.simplices.side_effect \
        = lambda dim: ["abc", "fab", 'cde'] if dim == 2 \
        else ['ab', 'bc', 'ac', 'cd', 'de', 'ef', 'fa', 'ea', 'fb', 'ce']
    return topology


@pytest.mark.fixture
def delauny_flip_topology():
    # switch ab with fc
    topology = concrete_topology()
    topology.boundary_cycles.return_value = ["acf", "fbc", 'cdea', 'efa']
    topology.simplices.side_effect \
        = lambda dim: ["acf", "fbc"] if dim == 2 else ['fc', 'bc', 'ac', 'cd', 'de', 'ef', 'fa', 'ea', 'fb']
    return topology


@pytest.mark.fixture
def non_atomic_topology():
    # remove EA and AC
    topology = concrete_topology()
    topology.boundary_cycles.return_value = ['abcdef', 'bfa']
    topology.simplices.side_effect \
        = lambda dim: ["abc", "fab"] if dim == 2 else ['ab', 'bc', 'cd', 'de', 'ef', 'fa', 'fb']
    return topology


class TestGetLabelUpdate:
    topology1 = initial_topology()
    cycle_labelling = CycleLabellingTree(topology1)

    # label_update will produce a valid dictionary or object of type InvalidStateChange
    @pytest.mark.parametrize('new_state', [remove_1simplex_topology,
                                           remove_2simplex_topology,
                                           remove_simplex_pair_topology,
                                           add_1simplex_topology,
                                           add_2simplex_topology,
                                           add_simplex_pair_topology,
                                           delauny_flip_topology])
    def test_doesnt_return_none(self, new_state):
        update = get_label_update(self.cycle_labelling, self.topology1, new_state())
        assert update is not None and not isinstance(update, InvalidStateChange)

    def test_nonatomic_returns_InvalidStateChange(self):
        label_update = get_label_update(self.cycle_labelling, self.topology1, non_atomic_topology())
        assert isinstance(label_update, InvalidStateChange)

    def test_invalid_returns_InvalidStateChange(self):
        cycle_labelling = CycleLabellingTree(trivial_topology())
        with pytest.raises(UpdateError):
            get_label_update(cycle_labelling, self.topology1, remove_1simplex_topology())

    @pytest.mark.parametrize('new_state, expected', [(remove_1simplex_topology, Remove1Simplex),
                                                     (remove_2simplex_topology, Remove2Simplices),
                                                     (remove_simplex_pair_topology, RemoveSimplexPair),
                                                     (add_1simplex_topology, Add1Simplex),
                                                     (add_2simplex_topology, Add2Simplices),
                                                     (add_simplex_pair_topology, AddSimplexPair),
                                                     (delauny_flip_topology, DelaunyFlip)])
    def test_returns_correct_case(self, new_state, expected):
        label_update = get_label_update(self.cycle_labelling, self.topology1, new_state())
        assert type(label_update) == expected
