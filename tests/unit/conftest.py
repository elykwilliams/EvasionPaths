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
def remove_1simplex():
    state_change = mock.Mock()
    state_change.cycles_removed = ['E', 'D']
    state_change.cycles_added = ['F']
    state_change.simplices_added = []
    state_change.simplices_removed = []
    state_change.case = (0, 1, 0, 0, 1, 2)
    state_change.is_atomic.return_value = True
    return state_change


@pytest.fixture
def add_1simplex():
    state_change = mock.Mock()
    state_change.cycles_removed = ['E']
    state_change.cycles_added = ['F', 'G']
    state_change.simplices_added = []
    state_change.simplices_removed = []
    state_change.case = (1, 0, 0, 0, 2, 1)
    state_change.is_atomic.return_value = True
    return state_change


@pytest.fixture
def add_2simplices():
    state_change = mock.Mock()
    state_change.cycles_removed = []
    state_change.cycles_added = []
    state_change.simplices_added = ['D']
    state_change.simplices_removed = []
    state_change.case = (0, 0, 1, 0, 0, 0)
    state_change.is_atomic.return_value = True
    return state_change


@pytest.fixture
def remove_2simplices():
    state_change = mock.Mock()
    state_change.cycles_removed = []
    state_change.cycles_added = []
    state_change.simplices_removed = ['C']
    state_change.case = (0, 0, 0, 1, 0, 0)
    state_change.is_atomic.return_value = True
    return state_change


@pytest.fixture
def remove_simplex_pair():
    state_change = mock.Mock()
    state_change.cycles_removed = ['C', 'D']
    state_change.cycles_added = ['F']
    state_change.simplices_removed = ['C']
    state_change.case = (0, 1, 0, 1, 1, 2)
    state_change.is_atomic.return_value = True
    return state_change


@pytest.fixture
def add_simplex_pair():
    state_change = mock.Mock()
    state_change.cycles_removed = ['E']
    state_change.cycles_added = ['F', 'G']
    state_change.simplices_added = ['G']
    state_change.case = (1, 0, 1, 0, 2, 1)
    state_change.is_atomic.return_value = True
    return state_change


@pytest.fixture
def delauny_flip():
    state_change = mock.Mock()
    state_change.cycles_removed = ['B', 'C']
    state_change.cycles_added = ['F', 'G']
    state_change.simplices_added = ['F', 'G']
    state_change.simplices_removed = ['B', 'C']
    state_change.case = (1, 1, 2, 2, 2, 2)
    state_change.is_atomic.return_value = True
    return state_change


@pytest.fixture
def sample_update_obj():
    update_obj = mock.Mock()
    update_obj.cycles_removed = ['C', 'B', 'A']
    update_obj.cycles_added = ['F', 'G']
    update_obj.simplices_added = []
    update_obj.simplices_removed = []
    update_obj.label_update = {'F': False, 'G': True}
    return update_obj
