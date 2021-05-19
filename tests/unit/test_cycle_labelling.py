from unittest import mock
from unittest.mock import patch

import pytest

from cycle_labelling import CycleLabellingTree


class TestInitCycleLabelling:

    def test_init_labelling(self, connected_topology):
        CycleLabellingTree(connected_topology)

    # All boundary cycles are added to the tree
    def test_all_cycles_in_tree(self, connected_topology):
        expected_cycles = ['A', 'B', 'C', 'D', 'E', 'alpha']
        cycle_labelling = CycleLabellingTree(connected_topology)
        assert all([cycle in cycle_labelling for cycle in expected_cycles])
        assert all([cycle in expected_cycles for cycle in cycle_labelling])

    # boundary cycles in label <=> bcycle in tree
    def test_cycle_in_label_equiv_label_in_tree(self, connected_topology):
        cycle_labelling = CycleLabellingTree(connected_topology)

        tree_nodes = cycle_labelling._tree.expand_tree('alpha')
        assert all([node in cycle_labelling for node in tree_nodes])
        assert all([cycle_labelling._tree.contains(cycle) for cycle in cycle_labelling])

    # Make sure alpha cycle is set as root
    def test_alpha_cycle_is_root(self, connected_topology):
        cycle_labelling = CycleLabellingTree(connected_topology)
        assert cycle_labelling._tree.root == 'alpha'

    ## test all simplices are false
    def test_simplices_false(self, connected_topology):
        cycle_labelling = CycleLabellingTree(connected_topology)
        assert not any([cycle_labelling[cycle] for cycle in ['B', 'C']])

    ## Test all non-simplices are true
    def test_non_simplices_true(self, connected_topology):
        cycle_labelling = CycleLabellingTree(connected_topology)
        assert all([cycle_labelling[cycle] for cycle in ['A', 'D', 'E']])

    ## Cycles all added under alphacomplex if connected
    def test_connected_init_depth(self, connected_topology):
        connected_topology.is_connected_cycle = mock.Mock(return_value=True)

        assert CycleLabellingTree(connected_topology, "power-down")._tree.DEPTH == 1
        assert CycleLabellingTree(connected_topology, "connected")._tree.DEPTH == 1
        assert CycleLabellingTree(connected_topology, "power-on")._tree.DEPTH == 1

    # TODO test assertions


## Test behavior when a connected 2simplex is added
# In these test cases, We have boundary cycles A, B, C, D, E. With B and C as simplices.
# D then becomes a simplex
class TestAddConnected2Simplex:
    simplices_added = ["D"]

    def test_add_2simplices(self, connected_labelling):
        connected_labelling.add_2simplices(self.simplices_added)

    # simplices are set to false
    def test_added_2simplices_false(self, connected_labelling):
        cycle_label = connected_labelling.add_2simplices(self.simplices_added)
        assert not any([cycle_label[cycle] for cycle in self.simplices_added])

    # setting non-cycle raises keyerror
    def test_raises_on_cycle_not_found(self, connected_labelling):
        pytest.raises(KeyError, connected_labelling.add_2simplices, ['Z'])

    # TODO test more assertions


## Test behavior when an connected 2simplex is removed
class TestRemove2Simplex:
    simplices_removed = ['C']

    def test_remove_2simplices(self, connected_labelling):
        connected_labelling.remove_2simplices(self.simplices_removed)

    # setting non-cycle raises keyerror
    def test_raises_on_cycle_not_found(self, connected_labelling):
        pytest.raises(KeyError, connected_labelling.remove_2simplices, ['Z'])

    # TODO Test assertions


## Test behavior when an edge is added using the "power-down/connected" approach
# In this case, an edge is  added splitting E into boundary cycles F and G
# This does NOT count the case where any sort of re-connection happens
class TestAddConnected1Simplex:
    cycles_added = ['F', 'G']
    cycles_removed = ['E']

    def test_add_1simplex(self, connected_labelling):
        connected_labelling.add_1simplex(self.cycles_removed, self.cycles_added)

    # If E has intrude, so to F and G
    # Else F and G are clear
    @pytest.mark.parametrize("cycleE", [True, False])
    def test_correct_update(self, cycleE, connected_labelling):
        connected_labelling.set('E', cycleE)
        cycle_label = connected_labelling.add_1simplex(self.cycles_removed, self.cycles_added)
        assert cycle_label == {'F': cycleE, 'G': cycleE}

    # Cannot remove more than one boundary cycle
    def test_raises_too_many_removed(self, connected_labelling):
        pytest.raises(AssertionError, connected_labelling.add_1simplex, ['D', 'E'], self.cycles_added)

    # TODO Test more assertions


## Test behavior when an edge is removed using the "power-down/connected" approach
# In this case, an edge is  removed joining E and D into boundary cycles F
# This does NOT count the case where any sort of disconnection happens
class TestRemoveConnected1Simplex:
    cycles_removed = ['E', 'D']
    cycles_added = ['F']

    def test_remove_1simplex(self, connected_labelling):
        connected_labelling.remove_1simplex(self.cycles_removed, self.cycles_added)

    def test_correct_cycles(self, connected_labelling):
        cycle_label = connected_labelling.remove_1simplex(self.cycles_removed, self.cycles_added)
        assert len(cycle_label) == 1 and 'F' in cycle_label

    @pytest.mark.parametrize("cycleD,cycleE", [(True, True), (True, False), (False, False)])
    def test_correct_update(self, cycleD, cycleE, connected_labelling):
        connected_labelling.set("D", cycleD)
        connected_labelling.set("E", cycleE)
        cycle_label = connected_labelling.remove_1simplex(self.cycles_removed, self.cycles_added)
        assert cycle_label == {'F': cycleD or cycleE}


    # TODO test assertion

    ## test_raises_bad_number_cycles

    ## test_raises_cycles_not_found


## Test behavior when an edge is added using the "power-down/connected" approach
# In this case, an edge is added splitting E into cycles F and G where F is also a 2simplex
class TestAddConnectedSimplexPair:
    cycles_removed = ['E']
    cycles_added = ['F', 'G']
    added_simplices = ['F']

    def test_add_simplex_pair(self, connected_labelling):
        connected_labelling.add_simplex_pair(self.cycles_removed, self.cycles_added, self.added_simplices)

    # If E has intrude, so to F and G
    # Else F and G are clear
    @pytest.mark.parametrize("cycleE", [True, False])
    def test_correct_update(self, cycleE, connected_labelling):
        connected_labelling.set('E', cycleE)
        cycle_label = connected_labelling.add_simplex_pair(self.cycles_removed, self.cycles_added, self.added_simplices)
        assert cycle_label == {'F': False, 'G': cycleE}

    # Cannot remove more than one boundary cycle
    def test_raises_too_many_removed(self, connected_labelling):
        pytest.raises(AssertionError,
                      connected_labelling.add_simplex_pair, ['D', 'E'], self.cycles_added, self.added_simplices)

    def test_raises_not_subset(self, connected_labelling):
        pytest.raises(AssertionError,
                      connected_labelling.add_simplex_pair, self.cycles_removed, self.cycles_added, ['A'])

    def test_raises_bad_cycle(self, connected_labelling):
        pytest.raises(KeyError, connected_labelling.add_simplex_pair, ['Z'], self.cycles_added, self.added_simplices)


## Test behavior when an edge is removed using the "power-down/connected" approach
# In this case, an edge is  removed joining E and D into boundary cycles F
# cycle E is a simplex.
# This does NOT count the case where any sort of disconnection happens
class TestRemoveConnectedSimplexPair:
    cycles_removed = ['C', 'D']
    cycles_added = ['F']
    removed_simplices = ['C']
    expected_cycles = ['A', 'B', 'E', 'F', 'alpha']

    def test_remove_simplex_pair(self, connected_labelling):
        connected_labelling.remove_simplex_pair(self.cycles_removed, self.cycles_added)

    @pytest.mark.parametrize("cycleD", [True, False])
    def test_correct_update(self, cycleD, connected_labelling):
        connected_labelling.set("D", cycleD)
        cycle_label = connected_labelling.remove_simplex_pair(self.cycles_removed, self.cycles_added)
        assert cycle_label == {'F': cycleD}

    def test_raises_too_many_added(self, connected_labelling):
        pytest.raises(AssertionError,
                      connected_labelling.remove_simplex_pair, self.cycles_removed, ['F', 'G'])


## Delauny flip where the edge between B and C flips resulting in simplices F and G
class TestConnectedDelaunyFlip:
    cycles_added = ['F', 'G']
    cycles_removed = ['B', 'C']

    def test_delauny_flip(self, connected_labelling):
        connected_labelling.delauny_flip(self.cycles_removed, self.cycles_added)

    def test_correct_update(self, connected_labelling):
        cycle_label = connected_labelling.delauny_flip(self.cycles_removed, self.cycles_added)
        assert cycle_label == {'F': False, 'G': False}

    def test_raises_only_two(self, connected_labelling):
        pytest.raises(AssertionError, connected_labelling.delauny_flip, ['A', 'B', 'C'], ['F', 'G', 'H'])
        pytest.raises(AssertionError, connected_labelling.delauny_flip, ['B'], ['F'])

    def test_raises_same_length(self, connected_labelling):
        pytest.raises(AssertionError, connected_labelling.delauny_flip, self.cycles_removed, ['F', 'G', 'H'])
        pytest.raises(AssertionError, connected_labelling.delauny_flip, self.cycles_removed, ['F'])
