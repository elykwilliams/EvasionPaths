from unittest import mock

import pytest

from cycle_labelling import CycleLabellingTree
from topological_state import StateChange


@pytest.fixture
def non_atomic(connected_topology):
    # TODO n
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


class TestCycleLabelTopology:
    pass


class TestCycleLabelEvasionPath:
    pass
