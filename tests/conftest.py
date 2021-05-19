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
    return CycleLabellingTree(connected_topology)
