from unittest import mock
from unittest.mock import patch

import pytest

from cycle_labelling import CycleLabellingDict
from utilities import LabellingError


@pytest.mark.fixture
def Simplex(name, edges=(), nodes=()):
    s = mock.Mock()
    s.to_cycle.return_value = name
    s.is_subface.side_effect = lambda e: True if e in edges else False
    s.nodes = nodes
    return s


class TestCycleLabellingDict:
    @pytest.fixture
    def topology(self):
        t = mock.Mock()
        t.boundary_cycles = ["A", "B", "C", "D", "E"]
        t.simplices.return_value = []
        return t

    @pytest.fixture
    def label_update(self):
        update = mock.Mock()
        update.cycles_added = []
        update.cycles_removed = []
        update.mapping = {"B": True, "D": False}
        return update

    def test_init_cycles(self, topology):
        labelling = CycleLabellingDict(topology)
        assert set(labelling.dict.keys()) == {"A", "B", "C", "D", "E"}

    def test_init_default_true(self, topology):
        labelling = CycleLabellingDict(topology)
        assert all(value for _, value in labelling.dict.items())

    def test_init_simplices_false(self, topology):
        simplexB = Simplex("B")
        topology.simplices.return_value = [simplexB]

        labelling = CycleLabellingDict(topology)
        assert not labelling.dict["B"]

    def test_contains(self, topology):
        labelling = CycleLabellingDict(topology)
        assert "B" in labelling and "F" not in labelling

    def test_iter(self, topology):
        labelling = CycleLabellingDict(topology)
        assert any(labelling[cycle] for cycle in labelling)

    def test_get_item(self, topology):
        labelling = CycleLabellingDict(topology)
        assert labelling["A"]

    def test_get_item_raises(self, topology):
        labelling = CycleLabellingDict(topology)
        pytest.raises(LabellingError, lambda: labelling["F"])

    def test_update_change_values(self, topology, label_update):
        labelling = CycleLabellingDict(topology)
        labelling.update(label_update)
        assert labelling.dict["B"] and not labelling.dict["D"]

    def test_is_valid(self, topology, label_update):
        labelling = CycleLabellingDict(topology)
        assert labelling.is_valid(label_update)

    def test_invalid_remove(self, topology, label_update):
        label_update.cycles_removed = ["F"]
        labelling = CycleLabellingDict(topology)
        assert not labelling.is_valid(label_update)

    def test_invalid_edit(self, topology, label_update):
        label_update.mapping = {"B": True, "F": False}
        labelling = CycleLabellingDict(topology)
        assert not labelling.is_valid(label_update)

    @patch("cycle_labelling.CycleLabellingDict.is_valid", return_value=False)
    def test_update_raises_invalid(self, topology, label_update):
        labelling = CycleLabellingDict(topology)
        pytest.raises(LabellingError, labelling.update, label_update)
