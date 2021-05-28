from unittest import mock

import pytest

from cycle_labelling import CycleLabellingTree
from utilities import LabellingError


class TestInitCycleLabelling:

    def test_init_labelling(self, connected_topology):
        CycleLabellingTree(connected_topology)

    # All boundary cycles are added to the tree
    def test_all_cycles_in_tree(self, connected_topology):
        expected_cycles = ['A', 'B', 'C', 'D', 'E', 'alpha']
        labelling = CycleLabellingTree(connected_topology)
        assert all([cycle in labelling for cycle in expected_cycles])
        assert all([cycle in expected_cycles for cycle in labelling])

    # boundary cycles in label <=> bcycle in tree
    def test_cycle_in_label_equiv_label_in_tree(self, connected_topology):
        labelling = CycleLabellingTree(connected_topology)

        tree_nodes = labelling.tree.expand_tree('alpha')
        assert all([node in labelling for node in tree_nodes])
        assert all([labelling.tree.contains(cycle) for cycle in labelling])

    # Make sure alpha cycle is set as root
    def test_alpha_cycle_is_root(self, connected_topology):
        labelling = CycleLabellingTree(connected_topology)
        assert labelling.tree.root == 'alpha'

    ## test all simplices are false
    def test_simplices_false(self, connected_topology):
        labelling = CycleLabellingTree(connected_topology)
        assert not any([labelling[cycle] for cycle in ['B', 'C']])

    ## Test all non-simplices are true
    def test_non_simplices_true(self, connected_topology):
        labelling = CycleLabellingTree(connected_topology)
        assert all([labelling[cycle] for cycle in ['A', 'D', 'E']])

    ## Cycles all added under alphacomplex if connected
    def test_connected_init_depth(self, connected_topology):
        connected_topology.is_connected_cycle = mock.Mock(return_value=True)

        assert CycleLabellingTree(connected_topology, "power-down").tree.DEPTH == 1
        assert CycleLabellingTree(connected_topology, "connected").tree.DEPTH == 1
        assert CycleLabellingTree(connected_topology, "power-on").tree.DEPTH == 1

    # TODO test assertions


class TestUpdateTree:
    cycles_removed = ['C', 'B', 'A']
    cycles_added = ['F', 'G']
    cycle_dict = {'F': False, 'G': True}

    def test_cycle_update(self, connected_labelling, sample_update_data):
        connected_labelling.update_tree(sample_update_data)

    def test_removes_old_cycles(self, connected_labelling, sample_update_data):
        connected_labelling.update_tree(sample_update_data)
        assert all(cycle not in connected_labelling for cycle in self.cycles_removed)

    def test_adds_new_cycles(self, connected_labelling, sample_update_data):
        connected_labelling.update_tree(sample_update_data)
        assert all([cycle in connected_labelling for cycle in self.cycles_added])

    def test_correct_updates(self, connected_labelling, sample_update_data):
        expected_values = {'D': True, 'E': True, 'F': False, 'G': True, 'alpha': False}
        connected_labelling.update_tree(sample_update_data)
        assert all([connected_labelling[cycle] == expected_values[cycle] for cycle in connected_labelling])

    def test_raises_new_cycles_not_labled(self, connected_labelling, sample_update_data):
        sample_update_data.cycles_added = ['F', 'G', 'H']
        pytest.raises(LabellingError, connected_labelling.update_tree, sample_update_data)


class TestUpdate:

    def test_update(self, connected_topology):
        state_change = mock.MagicMock()
        state_change.update_data.cycles_added = []
        state_change.update_data.cycles_removed = []
        state_change.update_data.simplices_added = []
        state_change.update_data.simplices_removed = []
        state_change.update_data.label_update = dict()
        state_change.case = (0, 0, 0, 0, 0, 0)

        labelling = CycleLabellingTree(connected_topology)

        labelling.update(state_change)
