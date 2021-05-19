from unittest import mock

import pytest

from cycle_labelling import CycleLabellingTree


@pytest.fixture
def simple_topology():
    def simplices(dim):
        return ["B", "C"] if dim == 2 else None

    topology = mock.Mock()
    topology.boundary_cycles = mock.Mock(return_value=['A', 'B', 'C', 'D', 'E'])
    topology.simplices = mock.Mock(side_effect=simplices)
    topology.alpha_cycle = 'alpha'
    return topology


@pytest.fixture
def cycle_labelling(simple_topology):
    return CycleLabellingTree(simple_topology)
