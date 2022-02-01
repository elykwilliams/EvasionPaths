from unittest import mock

import pytest


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
    retval = {cycle: True for cycle in connected_topology.boundary_cycles()}
    retval.update({cycle: False for cycle in connected_topology.simplices(2)})
    return retval
    # return CycleLabellingTree(connected_topology, policy="connected")


@pytest.fixture
def default_state_change():
    sc = mock.MagicMock()
    sc.is_valid.return_value = True
    return sc


@pytest.fixture
def default_labelling():
    return mock.MagicMock()


@pytest.fixture
def trivial_state_change(default_state_change):
    return default_state_change


@pytest.fixture
def add_2simplex_state_change(default_state_change):
    simplex = mock.Mock()
    simplex.to_cycle.return_value = "A"
    default_state_change.boundary_cycles.new_list = ["A", "B", "C", "D", "E"]
    default_state_change.simplices.added.return_value = [simplex]
    return default_state_change


@pytest.fixture
def remove_2simplex_state_change(default_state_change):
    simplex = mock.Mock()
    simplex.to_cycle.return_value = "B"
    default_state_change.boundary_cycles.old_list = ["A", "B", "C", "D", "E"]
    default_state_change.simplices.removed.return_value = [simplex]
    return default_state_change


@pytest.fixture
def add_1simplex_state_change(default_state_change):
    default_state_change.boundary_cycles.added.return_value = ["F", "G"]
    default_state_change.boundary_cycles.removed.return_value = ["B"]
    return default_state_change


@pytest.fixture
def remove_1simplex_state_change(default_state_change):
    default_state_change.boundary_cycles.removed.return_value = ["A", "B"]
    default_state_change.boundary_cycles.added.return_value = ["F"]
    return default_state_change


@pytest.fixture
def add_simplex_pair_state_change(default_state_change):
    simplex = mock.Mock()
    simplex.to_cycle.return_value = "G"
    simplex.is_subface.side_effect = lambda e: True if e == "fg" else False
    default_state_change.boundary_cycles.old_list = ["A", "B", "C", "D", "E"]
    default_state_change.simplices.added.return_value = [simplex]

    default_state_change.boundary_cycles.removed.return_value = ["D"]
    default_state_change.boundary_cycles.added.return_value = ["F", "G"]
    default_state_change.edges.added.return_value = ["fg"]
    return default_state_change


@pytest.fixture
def remove_simplex_pair_state_change(default_state_change):
    default_state_change.boundary_cycles.removed.return_value = ["A", "B"]
    default_state_change.boundary_cycles.added.return_value = ["F"]

    simplex = mock.Mock()
    simplex.to_cycle.return_value = "B"
    simplex.is_subface.side_effect = lambda e: True if e == "ab" else False
    default_state_change.simplices.removed.return_value = [simplex]
    default_state_change.edges.removed.return_value = ["ab"]
    return default_state_change


@pytest.fixture
def delauny_state_change(default_state_change):
    simplexA = mock.Mock()
    simplexA.to_cycle.return_value = "A"
    simplexA.is_subface.side_effect = lambda e: True if e == "ab" else False
    simplexA.nodes = (1, 2, 3)

    simplexB = mock.Mock()
    simplexB.to_cycle.return_value = "B"
    simplexB.is_subface.side_effect = lambda e: True if e == "ab" else False
    simplexB.nodes = (2, 3, 4)

    simplexF = mock.Mock()
    simplexF.to_cycle.return_value = "F"
    simplexF.is_subface.side_effect = lambda e: True if e == "fg" else False
    simplexF.nodes = (3, 4, 1)

    simplexG = mock.Mock()
    simplexG.to_cycle.return_value = "G"
    simplexG.is_subface.side_effect = lambda e: True if e == "fg" else False
    simplexG.nodes = (2, 4, 1)

    default_state_change.boundary_cycles.removed.return_value = ["A", "B"]
    default_state_change.boundary_cycles.added.return_value = ["F", "G"]
    default_state_change.simplices.removed.return_value = [simplexA, simplexB]
    default_state_change.simplices.added.return_value = [simplexF, simplexG]
    default_state_change.edges.added.return_value = ['fg']
    default_state_change.edges.removed.return_value = ['ab']
    return default_state_change
