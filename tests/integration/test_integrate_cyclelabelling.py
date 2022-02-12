from unittest import mock

import pytest

from cycle_labelling2 import CycleLabellingDict
from state_change import StateChange2D
from topology2 import ConnectedTopology
from update_data import LabelUpdateFactory
from utilities import SetDifference


@pytest.mark.fixture
def Simplex(name, edges=(), nodes=()):
    s = mock.Mock()
    s.to_cycle.return_value = name
    s.is_subface.side_effect = lambda e: True if e in edges else False
    s.nodes = nodes
    return s


@pytest.mark.fixture
def mock_AlphaComplex(simplices, edges):
    ac = mock.Mock()
    ac.simplices.side_effect = lambda dim: simplices if dim == 2 else edges
    return ac


@pytest.mark.fixture
def mock_CombinatorialMap(cycle_dict):
    m = mock.Mock()
    m.dict = cycle_dict
    m.boundary_cycles = m.dict.keys()
    m.alpha.side_effect = lambda dart: dart[-1::-1]
    m.get_cycle.side_effect = lambda dart: next(key for key, value in m.dict.items() if dart in value)
    return m


@pytest.mark.fixture
def mock_topology(cycles, simplices=(), edges=()):
    new_topology = mock.Mock()
    new_topology.boundary_cycles = cycles
    new_topology.simplices.side_effect = lambda dim: simplices if dim == 2 else edges
    return new_topology


simplexB = Simplex("B")
simplexC = Simplex("C")
simplexD = Simplex("D")


class TestIntegrateCycleLabellingDict:
    cycles = ['A', 'B', 'C', 'D', 'E']

    def test_mock_label_update(self):
        label_update = mock.Mock()
        label_update.cycles_added = []
        label_update.cycles_removed = []
        label_update.mapping = {"D": False}

        topology = mock_topology(self.cycles, simplices=[simplexB, simplexC])

        labelling = CycleLabellingDict(topology)
        labelling.update(label_update)
        assert labelling.dict == {"A": True, "B": False, "C": False, "D": False, "E": True}

    def test_mock_state_change(self):
        sc = mock.Mock()
        sc.case = (0, 0, 1, 0, 0, 0)
        sc.is_valid.return_value = True
        sc.boundary_cycles = SetDifference(self.cycles, self.cycles)
        sc.simplices.return_value = SetDifference([simplexB, simplexC, simplexD], [simplexB, simplexC])

        topology = mock_topology(self.cycles, simplices=[simplexB, simplexC])
        labelling = CycleLabellingDict(topology)
        label_update = LabelUpdateFactory().get_update(sc, labelling)

        labelling.update(label_update)
        assert labelling.dict == {"A": True, "B": False, "C": False, "D": False, "E": True}

    def test_mock_topology(self):
        old_topology = mock_topology(self.cycles, simplices=[simplexB, simplexC])
        labelling = CycleLabellingDict(old_topology)

        new_topology = mock_topology(self.cycles, simplices=[simplexB, simplexC, simplexD])
        sc = StateChange2D(new_topology, old_topology)
        label_update = LabelUpdateFactory().get_update(sc, labelling)

        labelling.update(label_update)
        assert labelling.dict == {"A": True, "B": False, "C": False, "D": False, "E": True}

    def test_mock_cmap(self):
        d = {"A": None, "B": None, "C": None, "D": None, "E": None}
        cmap = mock_CombinatorialMap(d)

        alpha_complex = mock_AlphaComplex({simplexB, simplexC}, {})
        top1 = ConnectedTopology(alpha_complex, cmap)
        labelling = CycleLabellingDict(top1)

        alpha_complex = mock_AlphaComplex({simplexB, simplexC, simplexD}, {})
        top2 = ConnectedTopology(alpha_complex, cmap)

        sc = StateChange2D(top2, top1)
        label_update = LabelUpdateFactory().get_update(sc, labelling)

        labelling.update(label_update)
        assert labelling.dict == {"A": True, "B": False, "C": False, "D": False, "E": True}
