from unittest import mock

import pytest


@pytest.fixture
def connected_topology():
    topology = mock.Mock()
    topology.boundary_cycles.return_value = ['A', 'B', 'C', 'D', 'E']
    topology.simplices.side_effect = lambda dim: ["B", "C"] if dim == 2 else None
    topology.alpha_cycle = 'alpha'
    return topology


@pytest.fixture
def connected_labelling(connected_topology):
    result = {cycle: True for cycle in connected_topology.boundary_cycles()}
    result.update({cycle: False for cycle in connected_topology.simplices(2)})
    return result
