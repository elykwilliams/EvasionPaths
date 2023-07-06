from unittest import mock

import pytest
from update_data import LabelUpdateFactory, Add2SimplicesUpdate2D, RemoveSimplexPairUpdate2D

from state_change import StateChange


@pytest.mark.fixture
def Simplex(name, edges=(), nodes=()):
    s = mock.Mock()
    s.to_cycle.return_value = name
    s.is_subface.side_effect = lambda e: True if e in edges else False
    s.nodes = nodes
    return s


@pytest.mark.fixture
def topology(cycles, simplices=(), edges=()):
    new_topology = mock.Mock()
    new_topology.boundary_cycles = cycles
    new_topology.simplices.side_effect = lambda dim: simplices if dim == 2 else edges
    new_topology.dim = 2
    return new_topology


class TestAdd2Simplex:
    simplexB = Simplex("B")
    simplexC = Simplex("C")
    simplexD = Simplex("D")
    cycles = ['A', 'B', 'C', 'D', 'E']
    new_topology = topology(cycles, simplices=[simplexB, simplexC, simplexD])
    old_topology = topology(cycles, simplices=[simplexB, simplexC])

    def test_new_simplices(self, connected_labelling):
        state_change = StateChange(self.new_topology, self.old_topology)
        assert state_change.simplices_difference[2].new_list == [self.simplexB, self.simplexC, self.simplexD]

    def test_added_simplices(self, connected_labelling):
        state_change = StateChange(self.new_topology, self.old_topology)
        assert state_change.simplices_difference[2].added() == {self.simplexD}

    def test_case(self, connected_labelling):
        state_change = StateChange(self.new_topology, self.old_topology)
        assert state_change.case == (0, 0, 1, 0, 0, 0)

    def test_factory(self, connected_labelling):
        label_update = LabelUpdateFactory.get_update(self.new_topology, self.old_topology, connected_labelling)
        assert type(label_update) == Add2SimplicesUpdate2D

    def test_update(self, connected_labelling):
        label_update = LabelUpdateFactory.get_update(self.new_topology, self.old_topology, connected_labelling)
        assert label_update.mapping == {"D": False}


class TestRemoveSimplexPair:
    simplexB = Simplex("B")
    simplexC = Simplex("C")
    cycles = ['A', 'B', 'C', 'D', 'E']
    new_topology = topology(['A', 'B', 'E', 'F'], simplices=[simplexB], edges=[])
    old_topology = topology(cycles, simplices=[simplexB, simplexC], edges=["cd"])

    def test_case(self, connected_labelling):
        state_change = StateChange(self.new_topology, self.old_topology)
        assert state_change.case == (0, 1, 0, 1, 1, 2)

    def test_factory(self, connected_labelling):
        label_update = LabelUpdateFactory.get_update(self.new_topology, self.old_topology, connected_labelling)
        assert type(label_update) == RemoveSimplexPairUpdate2D

    def test_update(self, connected_labelling):
        label_update = LabelUpdateFactory.get_update(self.new_topology, self.old_topology, connected_labelling)
        assert label_update.mapping == {"F": True}
