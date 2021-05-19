from unittest import mock
from unittest.mock import patch

import pytest

from cycle_labelling import CycleLabellingTree


class TestInitCycleLabelling:

    def test_init_labelling(self, simple_topology):
        CycleLabellingTree(simple_topology)

    # All boundary cycles are added to the tree
    def test_all_cycles_in_tree(self, simple_topology):
        expected_cycles = ['A', 'B', 'C', 'D', 'E', 'alpha']
        cycle_labelling = CycleLabellingTree(simple_topology)
        assert all([cycle in cycle_labelling for cycle in expected_cycles])
        assert all([cycle in expected_cycles for cycle in cycle_labelling])

    # boundary cycles in label <=> bcycle in tree
    def test_cycle_in_label_equiv_label_in_tree(self, simple_topology):
        cycle_labelling = CycleLabellingTree(simple_topology)

        tree_nodes = cycle_labelling._tree.expand_tree('alpha')
        assert all([node in cycle_labelling for node in tree_nodes])
        assert all([cycle_labelling._tree.contains(cycle) for cycle in cycle_labelling])

    # Make sure alpha cycle is set as root
    def test_alpha_cycle_is_root(self, simple_topology):
        cycle_labelling = CycleLabellingTree(simple_topology)

        assert cycle_labelling._tree.root == 'alpha'

    ## test all simplices are false
    def test_simplices_false(self, simple_topology):
        cycle_labelling = CycleLabellingTree(simple_topology)

        assert not any([cycle_labelling[cycle] for cycle in ['B', 'C']])

    ## Test all non-simplices are true
    def test_non_simplices_true(self, simple_topology):
        cycle_labelling = CycleLabellingTree(simple_topology)

        assert all([cycle_labelling[cycle] for cycle in ['A', 'D', 'E']])

    ## Cycles all added under alphacomplex if connected
    def test_connected_init_depth(self, simple_topology):
        simple_topology.is_connected_cycle = mock.Mock(return_value=True)

        assert CycleLabellingTree(simple_topology, "power-down")._tree.DEPTH == 1
        assert CycleLabellingTree(simple_topology, "connected")._tree.DEPTH == 1
        assert CycleLabellingTree(simple_topology, "power-on")._tree.DEPTH == 1

    # TODO test assertions


## Test behavior when a connected 2simplex is added
# In these test cases, We have boundary cycles A, B, C, D, E. With B and C as simplices.
# D then becomes a simplex
class TestAddConnected2Simplex:
    simplices_added = ["D"]

    def test_add_2simplices(self, cycle_labelling):
        cycle_labelling.add_2simplices(self.simplices_added)

    # simplices are set to false
    def test_added_2simplices_false(self, cycle_labelling):
        cycle_labelling.add_2simplices(self.simplices_added)

        assert not any([cycle_labelling[cycle] for cycle in self.simplices_added])

    # cycles havent changed
    @patch('cycle_labelling.Tree.remove_node')
    @patch('cycle_labelling.Tree.add_node')
    def test_no_cycles_added_or_removed(self, mock_add_node, mock_remove_node, cycle_labelling):
        cycle_labelling.add_2simplices(self.simplices_added)

        assert mock_add_node.call_count == 0
        assert mock_remove_node.call_count == 0

    # setting non-cycle raises keyerror
    def test_raises_on_cycle_not_found(self, cycle_labelling):
        pytest.raises(KeyError, cycle_labelling.add_2simplices, ['Z'])

    # TODO test more assertions


## Test behavior when an connected 2simplex is removed
class TestRemove2Simplex:
    simplices_removed = ['C']

    def test_remove_2simplices(self, cycle_labelling):
        cycle_labelling.remove_2simplices(self.simplices_removed)

    # cycles haven't changed
    @patch('cycle_labelling.Tree.remove_node')
    @patch('cycle_labelling.Tree.add_node')
    def test_no_cycles_added_or_removed(self, mock_add_node, mock_remove_node, cycle_labelling):
        cycle_labelling.remove_2simplices(self.simplices_removed)

        assert mock_add_node.call_count == 0
        assert mock_remove_node.call_count == 0

    # setting non-cycle raises keyerror
    def test_raises_on_cycle_not_found(self, cycle_labelling):
        pytest.raises(KeyError, cycle_labelling.remove_2simplices, ['Z'])

    ## other cycle labels havent changed
    def test_cycle_labels_unchanged(self, cycle_labelling):
        cycle_labelling.remove_2simplices(self.simplices_removed)

        assert all(cycle_labelling[cycle] for cycle in ['A', 'D', 'E'])
        assert not cycle_labelling['B']

    # TODO Test assertions


## Test behavior when an edge is added using the "power-down/connected" approach
# In this case, an edge is  added splitting E into boundary cycles F and G
# This does NOT count the case where any sort of re-connection happens
class TestAddConnected1Simplex:
    cycles_added = ['F', 'G']
    cycles_removed = ['E']
    expected_cycles = ['A', 'B', 'C', 'D', 'F', 'G', 'alpha']

    def test_add_1simplex(self, cycle_labelling):
        cycle_labelling.add_1simplex(self.cycles_removed, self.cycles_added)

    def test_correct_cycles(self, cycle_labelling):
        cycle_labelling.add_1simplex(self.cycles_removed, self.cycles_added)

        assert all([cycle in cycle_labelling for cycle in self.expected_cycles])
        assert all([cycle in self.expected_cycles for cycle in cycle_labelling])

    @patch('cycle_labelling.Tree.update_node')
    @patch('cycle_labelling.Tree.remove_node')
    @patch('cycle_labelling.Tree.add_node')
    def test_two_add_one_remove(self, mock_add_node, mock_remove_node, _, cycle_labelling):
        cycle_labelling.add_1simplex(self.cycles_removed, self.cycles_added)

        assert mock_add_node.call_count == 2
        assert mock_remove_node.call_count == 1

    # If E has intrude, so to F and G
    # Else F and G are clear
    @pytest.mark.parametrize("cycleE", [True, False])
    def test_correct_update(self, cycleE, cycle_labelling):
        cycle_labelling.set('E', cycleE)

        cycle_labelling.add_1simplex(self.cycles_removed, self.cycles_added)

        assert all([cycle_labelling[cycle] == cycleE for cycle in 'FG'])

    def test_other_labels_unchanged(self, cycle_labelling):
        expected_dict = {'A': True, 'B': False, 'C': False, 'D': True}
        cycle_labelling.add_1simplex(self.cycles_removed, self.cycles_added)
        assert all([expected_dict[cycle] == cycle_labelling[cycle] for cycle in 'ABCD'])

    def test_old_cycles_removed(self, cycle_labelling):
        cycle_labelling.add_1simplex(self.cycles_removed, self.cycles_added)
        assert not any([cycle in cycle_labelling for cycle in self.cycles_removed])

    # Cannot remove more than one boundary cycle
    def test_raises_too_many_removed(self, cycle_labelling):
        pytest.raises(AssertionError, cycle_labelling.add_1simplex, ['D', 'E'], self.cycles_added)

    # TODO Test more assertions


## Test behavior when an edge is removed using the "power-down/connected" approach
# In this case, an edge is  removed joining E and D into boundary cycles F
# This does NOT count the case where any sort of disconnection happens
class TestRemoveConnected1Simplex:
    cycles_removed = ['E', 'D']
    cycles_added = ['F']
    expected_cycles = ['A', 'B', 'C', 'F', 'alpha']

    def test_remove_1simplex(self, cycle_labelling):
        cycle_labelling.remove_1simplex(self.cycles_removed, self.cycles_added)

    def test_correct_cycles(self, cycle_labelling):
        cycle_labelling.remove_1simplex(self.cycles_removed, self.cycles_added)

        assert all([cycle in cycle_labelling for cycle in self.expected_cycles])
        assert all([cycle in self.expected_cycles for cycle in cycle_labelling])

    @patch('cycle_labelling.Tree.update_node')
    @patch('cycle_labelling.Tree.remove_node')
    @patch('cycle_labelling.Tree.add_node')
    def test_two_remove_one_add(self, mock_add_node, mock_remove_node, _, cycle_labelling):
        cycle_labelling.remove_1simplex(self.cycles_removed, self.cycles_added)

        assert mock_add_node.call_count == 1
        assert mock_remove_node.call_count == 2

    @pytest.mark.parametrize("cycleD,cycleE", [(True, True), (True, False), (False, False)])
    def test_correct_update(self, cycleD, cycleE, cycle_labelling):
        cycle_labelling.set("D", cycleD)
        cycle_labelling.set("E", cycleE)

        cycle_labelling.remove_1simplex(self.cycles_removed, self.cycles_added)

        assert cycle_labelling['F'] == cycleD or cycleE

    def test_other_labels_unchanged(self, cycle_labelling):
        cycle_labelling.remove_1simplex(self.cycles_removed, self.cycles_added)

        assert all([cycle_labelling[cycle] for cycle in ['A']])
        assert not any([cycle_labelling[cycle] for cycle in ['B', 'C']])

    def test_old_cycles_removed(self, cycle_labelling):
        cycle_labelling.remove_1simplex(self.cycles_removed, self.cycles_added)

        assert all([cycle not in cycle_labelling for cycle in self.cycles_removed])

    # TODO test assertion

    ## test_raises_bad_number_cycles

    ## test_raises_cycles_not_found


## Test behavior when an edge is added using the "power-down/connected" approach
# In this case, an edge is added splitting E into cycles F and G where F is also a 2simplex
class TestAddConnectedSimplexPair:
    cycles_removed = ['E']
    cycles_added = ['F', 'G']
    added_simplices = ['F']
    expected_cycles = ['A', 'B', 'C', 'D', 'F', 'G', 'alpha']

    def test_add_simplex_pair(self, cycle_labelling):
        cycle_labelling.add_simplex_pair(self.cycles_removed, self.cycles_added, self.added_simplices)

    def test_correct_cycles(self, cycle_labelling):
        cycle_labelling.add_simplex_pair(self.cycles_removed, self.cycles_added, self.added_simplices)

        assert all([cycle in cycle_labelling for cycle in self.expected_cycles])
        assert all([cycle in self.expected_cycles for cycle in cycle_labelling])

    @patch('cycle_labelling.Tree.update_node')
    @patch('cycle_labelling.Tree.remove_node')
    @patch('cycle_labelling.Tree.add_node')
    def test_two_add_one_remove(self, mock_add_node, mock_remove_node, _, cycle_labelling):
        cycle_labelling.add_simplex_pair(self.cycles_removed, self.cycles_added, self.added_simplices)

        assert mock_add_node.call_count == 2
        assert mock_remove_node.call_count == 1

    # If E has intrude, so to F and G
    # Else F and G are clear
    @pytest.mark.parametrize("cycleE", [True, False])
    def test_correct_update(self, cycleE, cycle_labelling):
        cycle_labelling.set('E', cycleE)

        cycle_labelling.add_simplex_pair(self.cycles_removed, self.cycles_added, self.added_simplices)

        assert cycle_labelling['G'] == cycleE
        assert cycle_labelling['F'] is False

    def test_other_labels_unchanged(self, cycle_labelling):
        expected_dict = {'A': True, 'B': False, 'C': False, 'D': True}

        cycle_labelling.add_simplex_pair(self.cycles_removed, self.cycles_added, self.added_simplices)

        assert all([expected_dict[cycle] == cycle_labelling[cycle] for cycle in 'ABCD'])

    def test_old_cycles_removed(self, cycle_labelling):
        cycle_labelling.add_simplex_pair(self.cycles_removed, self.cycles_added, self.added_simplices)
        assert not any([cycle in cycle_labelling for cycle in self.cycles_removed])

    # Cannot remove more than one boundary cycle
    def test_raises_too_many_removed(self, cycle_labelling):
        pytest.raises(AssertionError,
                      cycle_labelling.add_simplex_pair, ['D', 'E'], self.cycles_added, self.added_simplices)

    def test_raises_not_subset(self, cycle_labelling):
        pytest.raises(AssertionError, cycle_labelling.add_simplex_pair, self.cycles_removed, self.cycles_added, ['A'])

    def test_raises_bad_cycle(self, cycle_labelling):
        pytest.raises(KeyError, cycle_labelling.add_simplex_pair, ['Z'], self.cycles_added, self.added_simplices)


## Test behavior when an edge is removed using the "power-down/connected" approach
# In this case, an edge is  removed joining E and D into boundary cycles F
# cycle E is a simplex.
# This does NOT count the case where any sort of disconnection happens
class TestRemoveConnectedSimplexPair:
    cycles_removed = ['C', 'D']
    cycles_added = ['F']
    removed_simplices = ['C']
    expected_cycles = ['A', 'B', 'E', 'F', 'alpha']

    def test_remove_simplex_pair(self, cycle_labelling):
        cycle_labelling.remove_simplex_pair(self.cycles_removed, self.cycles_added, self.removed_simplices)

    def test_correct_cycles(self, cycle_labelling):
        cycle_labelling.remove_simplex_pair(self.cycles_removed, self.cycles_added, self.removed_simplices)

        assert all([cycle in cycle_labelling for cycle in self.expected_cycles])
        assert all([cycle in self.expected_cycles for cycle in cycle_labelling])

    @patch('cycle_labelling.Tree.update_node')
    @patch('cycle_labelling.Tree.remove_node')
    @patch('cycle_labelling.Tree.add_node')
    def test_two_remove_one_add(self, mock_add_node, mock_remove_node, _, cycle_labelling):
        cycle_labelling.remove_simplex_pair(self.cycles_removed, self.cycles_added, self.removed_simplices)

        assert mock_add_node.call_count == 1
        assert mock_remove_node.call_count == 2

    @pytest.mark.parametrize("cycleD", [True, False])
    def test_correct_update(self, cycleD, cycle_labelling):
        cycle_labelling.set("D", cycleD)

        cycle_labelling.remove_simplex_pair(self.cycles_removed, self.cycles_added, self.removed_simplices)

        assert cycle_labelling['F'] == cycleD

    def test_other_labels_unchanged(self, cycle_labelling):
        cycle_labelling.remove_simplex_pair(self.cycles_removed, self.cycles_added, self.removed_simplices)

        assert all([cycle_labelling[cycle] for cycle in ['A', 'E']])
        assert cycle_labelling['B'] is False

    def test_old_cycles_removed(self, cycle_labelling):
        cycle_labelling.remove_simplex_pair(self.cycles_removed, self.cycles_added, self.removed_simplices)

        assert all([cycle not in cycle_labelling for cycle in self.cycles_removed])

    def test_raises_too_many_cycles(self, cycle_labelling):
        pytest.raises(AssertionError,
                      cycle_labelling.remove_simplex_pair, self.cycles_removed, self.cycles_added, self.cycles_removed)

    def test_raises_not_subset(self, cycle_labelling):
        pytest.raises(AssertionError,
                      cycle_labelling.remove_simplex_pair, self.cycles_removed, self.cycles_added, ['B'])

    def test_raises_too_many_added(self, cycle_labelling):
        pytest.raises(AssertionError,
                      cycle_labelling.remove_simplex_pair, self.cycles_removed, ['F', 'G'], self.removed_simplices)
