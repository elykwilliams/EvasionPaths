# ******************************************************************************
#  Copyright (c) 2023, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************

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
    result = {cycle: True for cycle in connected_topology.boundary_cycles_difference()}
    result.update({cycle: False for cycle in connected_topology.simplices_difference(2)})
    return result
