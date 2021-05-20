from unittest import mock

import pytest

from cycle_labelling import CycleLabellingTree


@pytest.fixture
def connected_topology():
    def simplices(dim):
        return ["B", "C"] if dim == 2 else None

    topology = mock.Mock()
    topology.boundary_cycles = mock.Mock(return_value=['A', 'B', 'C', 'D', 'E'])
    topology.simplices = mock.Mock(side_effect=simplices)
    topology.alpha_cycle = 'alpha'
    return topology


@pytest.fixture
def connected_labelling(connected_topology):
    return CycleLabellingTree(connected_topology, policy="connected")


@pytest.fixture
def abstract_remove_1simplex():
    state_change = mock.Mock()
    state_change.removed_cycles = ['E', 'D']
    state_change.added_cycles = ['F']
    state_change.added_simplices = []
    state_change.removed_simplices = []
    state_change.case = (0, 1, 0, 0, 1, 2)
    state_change.is_atomic.return_value = True
    return state_change


@pytest.fixture
def abstract_add_1simplex():
    state_change = mock.Mock()
    state_change.removed_cycles = ['E']
    state_change.added_cycles = ['F', 'G']
    state_change.added_simplices = []
    state_change.removed_simplices = []
    state_change.case = (1, 0, 0, 0, 2, 1)
    state_change.is_atomic.return_value = True
    return state_change


@pytest.fixture
def abstract_add_2simplex():
    state_change = mock.Mock()
    state_change.removed_cycles = []
    state_change.added_cycles = []
    state_change.added_simplices = ['D']
    state_change.removed_simplices = []
    state_change.case = (0, 0, 1, 0, 0, 0)
    state_change.is_atomic.return_value = True
    return state_change


@pytest.fixture
def abstract_remove_2simplex():
    state_change = mock.Mock()
    state_change.removed_cycles = []
    state_change.added_cycles = []
    state_change.removed_simplices = ['C']
    state_change.case = (0, 0, 0, 1, 0, 0)
    state_change.is_atomic.return_value = True
    return state_change


@pytest.fixture
def abstract_remove_simplex_pair():
    state_change = mock.Mock()
    state_change.removed_cycles = ['C', 'D']
    state_change.added_cycles = ['F']
    state_change.removed_simplices = ['C']
    state_change.case = (0, 1, 0, 1, 1, 2)
    state_change.is_atomic.return_value = True
    return state_change


@pytest.fixture
def abstract_add_simplex_pair():
    state_change = mock.Mock()
    state_change.removed_cycles = ['E']
    state_change.added_cycles = ['F', 'G']
    state_change.added_simplices = ['G']
    state_change.case = (1, 0, 1, 0, 2, 1)
    state_change.is_atomic.return_value = True
    return state_change


@pytest.fixture
def abstract_delauny_flip():
    state_change = mock.Mock()
    state_change.removed_cycles = ['B', 'C']
    state_change.added_cycles = ['F', 'G']
    state_change.added_simplices = ['F', 'G']
    state_change.removed_simplices = ['B', 'C']
    state_change.case = (1, 1, 2, 2, 2, 2)
    state_change.is_atomic.return_value = True
    return state_change
