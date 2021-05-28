from unittest import mock

import pytest

from cycle_labelling import CycleLabellingTree
from topological_state import StateChange
from update_data import UpdateError


@pytest.fixture
def non_atomic(connected_topology):
    # TODO fixme
    def simplices(dim):
        return ["acf", "fbc"] if dim == 2 else ['fc', 'bc', 'ac', 'cd', 'de', 'ef', 'fa', 'ea', 'fb']

    connected_topology.boundary_cycles = mock.Mock(return_value=["acf", "fbc", 'cdea', 'efa'])
    connected_topology.simplices = mock.Mock(side_effect=simplices)
    return connected_topology


class TestCycleLabelStateChange:
    # Test that the interface for each allowed method is accessible
    connected_cases = ['topology1', 'remove_1simplex', 'remove_2simplex', 'remove_simplex_pair',
                       'add_1simplex', 'add_2simplex', 'add_simplex_pair', 'delauny_flip']

    @pytest.mark.parametrize('topology2', connected_cases, ids=connected_cases)
    def test_interface(self, topology1, topology2, request):
        connected_labelling = CycleLabellingTree(topology1)
        topology2 = request.getfixturevalue(topology2)

        state_change = StateChange(topology1, topology2)
        # print(state_change.case)
        # connected_labelling._tree.show(data_property="real")
        # connected_labelling._tree.show()

        connected_labelling.update(state_change)
        # connected_labelling._tree.show(data_property="real")
        # connected_labelling._tree.show()

    # Test that non atomic transitions raise a known error
    def test_raises_non_atomic(self, topology1, non_atomic):
        pass


class TestCycleLabellingUpdateData:

    def test_add_1simplex(self, connected_labelling, add_1simplex):
        connected_labelling.update(add_1simplex)
        expected = {'A': True, 'B': False, 'C': False, 'D': True, 'F': True, 'G': True, 'alpha': False}

        assert all([connected_labelling[cycle] == val for cycle, val in expected.items()])
        assert all(cycle in expected for cycle in connected_labelling)

    def test_add_2simplex(self, connected_labelling, add_2simplices):
        connected_labelling.update(add_2simplices)
        expected = {'A': True, 'B': False, 'C': False, 'D': False, 'E': True, 'alpha': False}

        assert all(connected_labelling[cycle] == val for cycle, val in expected.items())
        assert all(cycle in expected for cycle in connected_labelling)

    def test_add_simplex_pair(self, connected_labelling, add_simplex_pair):
        connected_labelling.update(add_simplex_pair)
        expected = {'A': True, 'B': False, 'C': False, 'D': True, 'F': True, 'G': False, 'alpha': False}

        assert all(connected_labelling[cycle] == val for cycle, val in expected.items())
        assert all(cycle in expected for cycle in connected_labelling)

    def test_remove_1simplex(self, connected_labelling, remove_1simplex):
        connected_labelling.update(remove_1simplex)
        expected = {'A': True, 'B': False, 'C': False, 'F': True, 'alpha': False}

        assert all(connected_labelling[cycle] == val for cycle, val in expected.items())
        assert all(cycle in expected for cycle in connected_labelling)

    def test_remove_2simplex(self, connected_labelling, remove_2simplices):
        connected_labelling.update(remove_2simplices)
        expected = {'A': True, 'B': False, 'C': False, 'D': True, 'E': True, 'alpha': False}

        assert all([connected_labelling[cycle] == val for cycle, val in expected.items()])
        assert all([cycle in expected for cycle in connected_labelling])

    def test_remove_simplex_pair(self, connected_labelling, remove_simplex_pair):
        connected_labelling.update(remove_simplex_pair)
        expected = {'A': True, 'B': False, 'E': True, 'F': True, 'alpha': False}

        assert all([connected_labelling[cycle] == val for cycle, val in expected.items()])
        assert all([cycle in expected for cycle in connected_labelling])

    def test_delauny_flip(self, connected_labelling, delauny_flip):
        connected_labelling.update(delauny_flip)
        expected = {'A': True, 'D': True, 'E': True, 'F': False, 'G': False, 'alpha': False}

        assert all([connected_labelling[cycle] == val for cycle, val in expected.items()])
        assert all([cycle in expected for cycle in connected_labelling])

    def test_raises_non_atomic(self, connected_labelling):
        state_change = mock.Mock()
        state_change.is_atomic.return_value = False
        pytest.raises(UpdateError, connected_labelling.update, state_change)


class TestCycleLabellingTopology:
    pass


class TestCycleLabellingSimulation:
    pass
