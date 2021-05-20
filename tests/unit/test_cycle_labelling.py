from unittest import mock
from unittest.mock import patch

import pytest

import cycle_labelling
from cycle_labelling import CycleLabellingTree


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

        tree_nodes = labelling._tree.expand_tree('alpha')
        assert all([node in labelling for node in tree_nodes])
        assert all([labelling._tree.contains(cycle) for cycle in labelling])

    # Make sure alpha cycle is set as root
    def test_alpha_cycle_is_root(self, connected_topology):
        labelling = CycleLabellingTree(connected_topology)
        assert labelling._tree.root == 'alpha'

    ## test all simplices are false
    def test_simplices_false(self, connected_topology):
        labelling = CycleLabellingTree(connected_topology)
        assert not any([labelling[cycle] for cycle in ['B', 'C']])

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

    def test_add_2simplices(self, connected_labelling, abstract_add_2simplex):
        connected_labelling.add_2simplices(abstract_add_2simplex.simplices_added)

    # simplices are set to false
    def test_added_2simplices_false(self, connected_labelling, abstract_add_2simplex):
        cycle_label = connected_labelling.add_2simplices(abstract_add_2simplex.simplices_added)
        assert not any([cycle_label[cycle] for cycle in abstract_add_2simplex.simplices_added])

    # setting non-cycle raises keyerror
    def test_raises_on_cycle_not_found(self, connected_labelling):
        pytest.raises(KeyError, connected_labelling.add_2simplices, ['Z'])

    # TODO test more assertions


## Test behavior when an connected 2simplex is removed
class TestRemove2Simplex:
    simplices_removed = ['C']

    def test_remove_2simplices(self, connected_labelling, abstract_remove_2simplex):
        connected_labelling.remove_2simplices(abstract_remove_2simplex.simplices_removed)

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

    def test_add_1simplex(self, connected_labelling, abstract_add_1simplex):
        connected_labelling.add_1simplex(abstract_add_1simplex.cycles_removed, abstract_add_1simplex.cycles_added)

    # If E has intrude, so to F and G
    # Else F and G are clear
    @pytest.mark.parametrize("cycleE", [True, False])
    def test_correct_update(self, cycleE, connected_labelling, abstract_add_1simplex):
        connected_labelling.set('E', cycleE)
        cycle_label = connected_labelling.add_1simplex(abstract_add_1simplex.cycles_removed,
                                                       abstract_add_1simplex.cycles_added)
        assert cycle_label == {'F': cycleE, 'G': cycleE}

    # Cannot remove more than one boundary cycle
    def test_raises_too_many_removed(self, connected_labelling, abstract_add_1simplex):
        pytest.raises(AssertionError, connected_labelling.add_1simplex, ['D', 'E'], abstract_add_1simplex.cycles_added)

    # TODO Test more assertions


## Test behavior when an edge is removed using the "power-down/connected" approach
# In this case, an edge is  removed joining E and D into boundary cycles F
# This does NOT count the case where any sort of disconnection happens
class TestRemoveConnected1Simplex:
    cycles_removed = ['E', 'D']
    cycles_added = ['F']

    def test_remove_1simplex(self, connected_labelling, abstract_remove_1simplex):
        connected_labelling.remove_1simplex(abstract_remove_1simplex.cycles_removed,
                                            abstract_remove_1simplex.cycles_added)

    def test_correct_cycles(self, connected_labelling, abstract_remove_1simplex):
        cycle_label = connected_labelling.remove_1simplex(abstract_remove_1simplex.cycles_removed,
                                                          abstract_remove_1simplex.cycles_added)
        assert len(cycle_label) == 1 and 'F' in cycle_label

    @pytest.mark.parametrize("cycleD,cycleE", [(True, True), (True, False), (False, False)])
    def test_correct_update(self, cycleD, cycleE, connected_labelling, abstract_remove_1simplex):
        connected_labelling.set("D", cycleD)
        connected_labelling.set("E", cycleE)
        cycle_label = connected_labelling.remove_1simplex(abstract_remove_1simplex.cycles_removed,
                                                          abstract_remove_1simplex.cycles_added)
        assert cycle_label == {'F': cycleD or cycleE}


    # TODO test assertion

    ## test_raises_bad_number_cycles

    ## test_raises_cycles_not_found


## Test behavior when an edge is added using the "power-down/connected" approach
# In this case, an edge is added splitting E into cycles F and G where F is also a 2simplex
class TestAddConnectedSimplexPair:
    cycles_removed = ['E']
    cycles_added = ['F', 'G']
    simplices_added = ['F']

    def test_add_simplex_pair(self, connected_labelling):
        connected_labelling.add_simplex_pair(self.cycles_removed, self.cycles_added, self.simplices_added)

    # If E has intrude, so to F and G
    # Else F and G are clear
    @pytest.mark.parametrize("cycleE", [True, False])
    def test_correct_update(self, cycleE, connected_labelling):
        connected_labelling.set('E', cycleE)
        cycle_label = connected_labelling.add_simplex_pair(self.cycles_removed, self.cycles_added, self.simplices_added)
        assert cycle_label == {'F': False, 'G': cycleE}

    # Cannot remove more than one boundary cycle
    def test_raises_too_many_removed(self, connected_labelling):
        pytest.raises(AssertionError,
                      connected_labelling.add_simplex_pair, ['D', 'E'], self.cycles_added, self.simplices_added)

    def test_raises_not_subset(self, connected_labelling):
        pytest.raises(AssertionError,
                      connected_labelling.add_simplex_pair, self.cycles_removed, self.cycles_added, ['A'])

    def test_raises_bad_cycle(self, connected_labelling):
        pytest.raises(KeyError, connected_labelling.add_simplex_pair, ['Z'], self.cycles_added, self.simplices_added)


## Test behavior when an edge is removed using the "power-down/connected" approach
# In this case, an edge is  removed joining E and D into boundary cycles F
# cycle E is a simplex.
# This does NOT count the case where any sort of disconnection happens
class TestRemoveConnectedSimplexPair:
    cycles_removed = ['C', 'D']
    cycles_added = ['F']
    simplices_removed = ['C']

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


class TestUpdateTree:
    cycles_removed = ['C', 'B', 'A']
    cycles_added = ['F', 'G']
    cycle_dict = {'F': False, 'G': True}

    def test_cycle_update(self, connected_labelling):
        cycles_removed, cycles_added = [], []
        cycle_dict = dict()
        connected_labelling.update_tree(cycles_removed, cycles_added, cycle_dict)

    def test_removes_old_cycles(self, connected_labelling):
        connected_labelling.update_tree(self.cycles_removed, self.cycles_added, self.cycle_dict)

        assert all(cycle not in connected_labelling for cycle in self.cycles_removed)

    def test_adds_new_cycles(self, connected_labelling):
        connected_labelling.update_tree(self.cycles_removed, self.cycles_added, self.cycle_dict)

        assert all([cycle in connected_labelling for cycle in self.cycles_added])

    def test_correct_updates(self, connected_labelling):
        expected_values = {'D': True, 'E': True, 'F': False, 'G': True, 'alpha': False}
        connected_labelling.update_tree(self.cycles_removed, self.cycles_added, self.cycle_dict)

        assert all([connected_labelling[cycle] == expected_values[cycle] for cycle in connected_labelling])

    def test_raises_new_cycles_not_labled(self, connected_labelling):
        pytest.raises(AssertionError,
                      connected_labelling.update_tree, self.cycles_removed, ['F', 'G', 'H'], self.cycle_dict)


class TestGetCycleLabelUpdate:

    @patch.object(cycle_labelling.CycleLabellingTree, 'remove_1simplex', return_value=dict())
    def test_remove_1simplex(self, _, connected_topology, abstract_remove_1simplex):
        labelling = CycleLabellingTree(connected_topology)
        labelling.get_label_update(abstract_remove_1simplex)
        assert labelling.remove_1simplex.called

    @patch.object(cycle_labelling.CycleLabellingTree, 'remove_2simplices', return_value=dict())
    def test_remove_2simplex(self, _, connected_topology, abstract_remove_2simplex):
        labelling = CycleLabellingTree(connected_topology)
        labelling.get_label_update(abstract_remove_2simplex)
        assert labelling.remove_2simplices.called

    @patch.object(cycle_labelling.CycleLabellingTree, 'remove_simplex_pair', return_value=dict())
    def test_remove_simplex_pair(self, _, connected_topology, abstract_remove_simplex_pair):
        labelling = CycleLabellingTree(connected_topology)
        labelling.get_label_update(abstract_remove_simplex_pair)
        assert labelling.remove_simplex_pair.called

    @patch.object(cycle_labelling.CycleLabellingTree, 'add_1simplex', return_value=dict())
    def test_add_1simplex(self, _, connected_topology, abstract_add_1simplex):
        labelling = CycleLabellingTree(connected_topology)
        labelling.get_label_update(abstract_add_1simplex)
        assert labelling.add_1simplex.called

    @patch.object(cycle_labelling.CycleLabellingTree, 'add_2simplices', return_value=dict())
    def test_add_2simplices(self, _, connected_topology, abstract_add_2simplex):
        labelling = CycleLabellingTree(connected_topology)
        labelling.get_label_update(abstract_add_2simplex)
        assert labelling.add_2simplices.called

    @patch.object(cycle_labelling.CycleLabellingTree, 'add_simplex_pair', return_value=dict())
    def test_add_simplex_pair(self, _, connected_topology, abstract_add_simplex_pair):
        labelling = CycleLabellingTree(connected_topology)
        labelling.get_label_update(abstract_add_simplex_pair)
        assert labelling.add_simplex_pair.called

    @patch.object(cycle_labelling.CycleLabellingTree, 'delauny_flip', return_value=dict())
    def test_add_simplex_pair(self, _, connected_topology, abstract_delauny_flip):
        labelling = CycleLabellingTree(connected_topology)
        labelling.get_label_update(abstract_delauny_flip)
        assert labelling.delauny_flip.called

    def test_raises_non_atomic(self, connected_labelling):
        state_change = mock.Mock()
        state_change.is_atomic.return_value = False

        pytest.raises(AssertionError, connected_labelling.get_label_update, state_change)

    def test_raises_is_atomic_not_found(self, connected_labelling):
        state_change = mock.Mock()
        state_change.case = (1, 0, 2, 0, 2, 1)
        pytest.raises(AssertionError, connected_labelling.get_label_update, state_change)


class TestUpdate:

    @patch.object(cycle_labelling.CycleLabellingTree, 'get_label_update', return_value=dict())
    def test_update(self, connected_topology):
        state_change = mock.Mock()
        state_change.cycles_added = []
        state_change.cycles_removed = []
        labelling = CycleLabellingTree(connected_topology)

        labelling.update(state_change)

    def test_add_1simplex(self, connected_labelling, abstract_add_1simplex):
        connected_labelling.update(abstract_add_1simplex)
        expected = {'A': True, 'B': False, 'C': False, 'D': True, 'F': True, 'G': True, 'alpha': False}

        assert all(connected_labelling[cycle] == val for cycle, val in expected.items())
        assert all(cycle in expected for cycle in connected_labelling)

    def test_add_2simplex(self, connected_labelling, abstract_add_2simplex):
        connected_labelling.update(abstract_add_2simplex)
        expected = {'A': True, 'B': False, 'C': False, 'D': False, 'E': True, 'alpha': False}

        assert all(connected_labelling[cycle] == val for cycle, val in expected.items())
        assert all(cycle in expected for cycle in connected_labelling)

    def test_add_simplex_pair(self, connected_labelling, abstract_add_simplex_pair):
        connected_labelling.update(abstract_add_simplex_pair)
        expected = {'A': True, 'B': False, 'C': False, 'D': True, 'F': True, 'G': False, 'alpha': False}

        assert all(connected_labelling[cycle] == val for cycle, val in expected.items())
        assert all(cycle in expected for cycle in connected_labelling)

    def test_remove_1simplex(self, connected_labelling, abstract_remove_1simplex):
        connected_labelling.update(abstract_remove_1simplex)
        expected = {'A': True, 'B': False, 'C': False, 'F': True, 'alpha': False}

        assert all(connected_labelling[cycle] == val for cycle, val in expected.items())
        assert all(cycle in expected for cycle in connected_labelling)

    def test_remove_2simplex(self, connected_labelling, abstract_remove_2simplex):
        connected_labelling.update(abstract_remove_2simplex)
        expected = {'A': True, 'B': False, 'C': False, 'D': True, 'E': True, 'alpha': False}

        assert all([connected_labelling[cycle] == val for cycle, val in expected.items()])
        assert all([cycle in expected for cycle in connected_labelling])

    def test_remove_simplex_pair(self, connected_labelling, abstract_remove_simplex_pair):
        connected_labelling.update(abstract_remove_simplex_pair)
        expected = {'A': True, 'B': False, 'E': True, 'F': True, 'alpha': False}

        assert all([connected_labelling[cycle] == val for cycle, val in expected.items()])
        assert all([cycle in expected for cycle in connected_labelling])

    def test_delauny_flip(self, connected_labelling, abstract_delauny_flip):
        connected_labelling.update(abstract_delauny_flip)
        expected = {'A': True, 'D': True, 'E': True, 'F': False, 'G': False, 'alpha': False}

        assert all([connected_labelling[cycle] == val for cycle, val in expected.items()])
        assert all([cycle in expected for cycle in connected_labelling])

    def test_raises_non_atomic(self, connected_labelling):
        state_change = mock.Mock()
        state_change.is_atomic.return_value = False
        pytest.raises(AssertionError, connected_labelling.update, state_change)
