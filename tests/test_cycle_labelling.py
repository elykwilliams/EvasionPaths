import unittest
from unittest import mock
from unittest.mock import patch

import pytest

from cycle_labelling import CycleLabellingTree


class TestInitCycleLabelling(unittest.TestCase):
    def setUp(self) -> None:
        def simplices(dim):
            return ["B", "C"] if dim == 2 else None

        self.topology = mock.Mock()
        self.topology.boundary_cycles = mock.Mock(return_value=list("ABCDE"))
        self.topology.simplices = mock.Mock(side_effect=simplices)
        self.topology.alpha_cycle = 'alpha'

    # All boundary cycles are added to the tree
    def test_all_cycles_in_tree(self):
        labelling = CycleLabellingTree(self.topology)
        expected_cycles = list("ABCDE") + ['alpha']

        assert all([cycle in labelling for cycle in expected_cycles])
        assert all([cycle in expected_cycles for cycle in labelling])

    # boundary cycles in label <=> bcycle in tree
    def test_cycle_in_label_equiv_label_in_tree(self):
        labelling = CycleLabellingTree(self.topology)
        tree_nodes = labelling._tree.expand_tree(self.topology.alpha_cycle)

        assert all([node in labelling for node in tree_nodes])
        assert all([labelling._tree.contains(cycle) for cycle in labelling])

    # Make sure alpha cycle is set as root
    def test_alpha_cycle_is_root(self):
        labelling = CycleLabellingTree(self.topology)
        assert labelling._tree.root == self.topology.alpha_cycle

    ## test all simplices are false
    def test_simplices_false(self):
        labelling = CycleLabellingTree(self.topology)
        simplex_labels = [labelling._tree.get_node(cycle).data for cycle in self.topology.simplices(2)]
        assert not any(simplex_labels)

    ## Test all non-simplices are true
    def test_non_simplices_true(self):
        labelling = CycleLabellingTree(self.topology)
        hole_labels = [node_id for node_id in labelling
                       if node_id not in self.topology.simplices(2) and node_id != self.topology.alpha_cycle]
        assert all(hole_labels)

    ## Cycles all added under alphacomplex if connected
    def test_connected_init_depth(self):
        self.topology.is_connected_cycle = mock.Mock(return_value=True)
        assert CycleLabellingTree(self.topology, "power-down")._tree.DEPTH == 1
        assert CycleLabellingTree(self.topology, "connected")._tree.DEPTH == 1
        assert CycleLabellingTree(self.topology, "power-on")._tree.DEPTH == 1


## Test behavior when a connected 2simplex is added
# In these tast cases, We have boundary cycles A, B, C, D, E. With B and C as simplices.
# D then becomes a simplex
class TestAddConnectedSimplex(unittest.TestCase):
    def setUp(self) -> None:
        def simplices(dim):
            return ["B", "C"] if dim == 2 else None

        self.topology = mock.Mock()
        self.topology.boundary_cycles = mock.Mock(return_value=list("ABCDE"))
        self.topology.simplices = mock.Mock(side_effect=simplices)
        self.topology.alpha_cycle = 'alpha'

        self.cycle_labelling = CycleLabellingTree(self.topology)
        self.simplices_added = ["D"]

    def test_add_2simplices(self):
        self.cycle_labelling.add_2simplices(self.simplices_added)

    # simplices are set to false
    def test_added_2simplices_false(self):
        self.cycle_labelling.add_2simplices(self.simplices_added)
        assert not any([self.cycle_labelling[cycle] for cycle in self.simplices_added])

    # cycles havent changed
    @patch('cycle_labelling.Tree.remove_node')
    @patch('cycle_labelling.Tree.add_node')
    def test_no_cycles_added_or_removed(self, mock_add_node, mock_remove_node):
        self.cycle_labelling.add_2simplices(self.simplices_added)
        assert mock_add_node.call_count == 0
        assert mock_remove_node.call_count == 0

    # setting non-cycle raises keyerror
    def test_raises_on_cycle_not_found(self):
        pytest.raises(KeyError, self.cycle_labelling.add_2simplices, ['Z'])


## Test behavior when an connected 2simplex is removed
class TestRemoveSimplex(unittest.TestCase):

    def setUp(self) -> None:
        def simplices(dim):
            return ["B", "C"] if dim == 2 else None

        self.topology = mock.Mock()
        self.topology.boundary_cycles = mock.Mock(return_value=list("ABCDE"))
        self.topology.simplices = mock.Mock(side_effect=simplices)
        self.topology.alpha_cycle = 'alpha'

        self.cycle_labelling = CycleLabellingTree(self.topology)
        self.simplices_removed = ['C']

    def test_remove_2simplices(self):
        self.cycle_labelling.remove_2simplices(self.simplices_removed)

    # cycles haven't changed
    @patch('cycle_labelling.Tree.remove_node')
    @patch('cycle_labelling.Tree.add_node')
    def test_no_cycles_added_or_removed(self, mock_add_node, mock_remove_node):
        self.cycle_labelling.remove_2simplices(self.simplices_removed)
        assert mock_add_node.call_count == 0
        assert mock_remove_node.call_count == 0

    # setting non-cycle raises keyerror
    def test_raises_on_cycle_not_found(self):
        pytest.raises(KeyError, self.cycle_labelling.remove_2simplices, ['Z'])

    ## cycle lables havent changed
    def test_cycle_labels_unchanged(self):
        self.cycle_labelling.remove_2simplices(self.simplices_removed)
        for cycle in self.cycle_labelling:
            if cycle in "ADE":
                assert self.cycle_labelling[cycle]
            else:
                assert not self.cycle_labelling[cycle]


## Test behavior when an edge is added using the "power-down/connected" approach
# In this case, an edge is  added splitting E into boundary cycles F and G
# This does NOT count the case where any sort of re-connection happens
class TestAddConnected1Simplex(unittest.TestCase):
    def setUp(self):
        def simplices(dim):
            return ["B", "C"] if dim == 2 else None

        self.topology = mock.Mock()
        self.topology.boundary_cycles = mock.Mock(return_value=list("ABCDE"))
        self.topology.simplices = mock.Mock(side_effect=simplices)
        self.topology.alpha_cycle = 'alpha'

        self.cycle_labelling = CycleLabellingTree(self.topology)
        self.cycles_added = ['F', 'G']
        self.cycles_removed = ['E']

    def test_add_1simplex(self):
        self.cycle_labelling.add_1simplex(self.cycles_removed, self.cycles_added)

    def test_correct_cycles(self):
        self.cycle_labelling.add_1simplex(self.cycles_removed, self.cycles_added)
        expected_cycles = list("ABCDFG") + ['alpha']

        assert all([cycle in self.cycle_labelling for cycle in expected_cycles])
        assert all([cycle in expected_cycles for cycle in self.cycle_labelling])

    @patch('cycle_labelling.Tree.update_node')
    @patch('cycle_labelling.Tree.remove_node')
    @patch('cycle_labelling.Tree.add_node')
    def test_two_remove_one_add(self, mock_add_node, mock_remove_node, _):
        self.cycle_labelling.add_1simplex(self.cycles_removed, self.cycles_added)
        assert mock_add_node.call_count == 2
        assert mock_remove_node.call_count == 1

    # If E has intrude, so to F and G
    # Else F and G are clear
    def test_correct_update_false(self):
        self.cycle_labelling.set('E', False)
        self.cycle_labelling.add_1simplex(self.cycles_removed, self.cycles_added)
        assert not any([self.cycle_labelling[cycle] for cycle in 'FG'])

    def test_correct_update_true(self):
        self.cycle_labelling.set('E', True)
        self.cycle_labelling.add_1simplex(self.cycles_removed, self.cycles_added)
        assert all([self.cycle_labelling[cycle] for cycle in 'FG'])

    def test_other_labels_unchanged(self):
        expected_dict = {'A': True, 'B': False, 'C': False, 'D': True}
        self.cycle_labelling.add_1simplex(self.cycles_removed, self.cycles_added)
        assert all([expected_dict[cycle] == self.cycle_labelling[cycle] for cycle in 'ABCD'])

    def test_old_cycles_removed(self):
        self.cycle_labelling.add_1simplex(self.cycles_removed, self.cycles_added)
        assert not any([cycle in self.cycle_labelling for cycle in self.cycles_removed])
