from unittest import mock

import pytest

from update_data import *


## Test behavior when a connected 2simplex is added
# In these test cases, We have boundary cycles A, B, C, D, E. With B and C as simplices.
# D then becomes a simplex
class TestAddConnected2Simplex:
    simplices_added = ["D"]

    def test_add_2simplices(self, connected_labelling, add_2simplices):
        _ = Add2Simplices(connected_labelling, add_2simplices).label_update

    # simplices are set to false
    # TODO test result vs expected dict
    def test_added_2simplices_false(self, connected_labelling, add_2simplices):
        cycle_label = Add2Simplices(connected_labelling, add_2simplices).label_update
        assert not any([cycle_label[cycle] for cycle in self.simplices_added])

    # setting non-cycle raises keyerror
    def test_raises_on_cycle_not_found(self, connected_labelling, add_2simplices):
        add_2simplices.simplices_added = ['Z']
        with pytest.raises(UpdateError):
            Add2Simplices(connected_labelling, add_2simplices).is_valid()

    # TODO test more assertions


## Test behavior when an connected 2simplex is removed
class TestRemove2Simplex:
    simplices_removed = ['C']

    def test_remove_2simplices(self, connected_labelling, remove_2simplices):
        _ = Remove2Simplices(connected_labelling, remove_2simplices).label_update

    # setting non-cycle raises keyerror
    def test_raises_on_cycle_not_found(self, connected_labelling, remove_2simplices):
        remove_2simplices.simplices_removed = ['Z']
        with pytest.raises(UpdateError):
            Remove2Simplices(connected_labelling, remove_2simplices).is_valid()

    # TODO Test assertions


## Test behavior when an edge is added using the "power-down/connected" approach
# In this case, an edge is  added splitting E into boundary cycles F and G
# This does NOT count the case where any sort of re-connection happens
class TestAddConnected1Simplex:
    cycles_added = ['F', 'G']
    cycles_removed = ['E']

    def test_add_1simplex(self, connected_labelling, add_1simplex):
        _ = Add1Simplex(connected_labelling, add_1simplex).label_update

    # If E has intrude, so to F and G
    # Else F and G are clear
    @pytest.mark.parametrize("cycleE", [True, False])
    def test_correct_update(self, cycleE, connected_labelling, add_1simplex):
        connected_labelling['E'] = cycleE
        cycle_label = Add1Simplex(connected_labelling, add_1simplex).label_update
        assert cycle_label == {'F': cycleE, 'G': cycleE}

    # Cannot remove more than one boundary cycle
    def test_raises_too_many_removed(self, connected_labelling, add_1simplex):
        add_1simplex.cycles_removed = ['D', 'E']
        with pytest.raises(UpdateError):
            Add1Simplex(connected_labelling, add_1simplex).is_valid()

    # TODO Test more assertions


## Test behavior when an edge is removed using the "power-down/connected" approach
# In this case, an edge is  removed joining E and D into boundary cycles F
# This does NOT count the case where any sort of disconnection happens
class TestRemoveConnected1Simplex:
    cycles_removed = ['E', 'D']
    cycles_added = ['F']

    def test_remove_1simplex(self, connected_labelling, remove_1simplex):
        _ = Remove1Simplex(connected_labelling, remove_1simplex).label_update

    @pytest.mark.parametrize("cycleD,cycleE", [(True, True), (True, False), (False, False)])
    def test_correct_update(self, cycleD, cycleE, connected_labelling, remove_1simplex):
        connected_labelling["D"] = cycleD
        connected_labelling["E"] = cycleE
        cycle_label = Remove1Simplex(connected_labelling, remove_1simplex).label_update
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

    def test_add_simplex_pair(self, connected_labelling, add_simplex_pair):
        _ = AddSimplexPair(connected_labelling, add_simplex_pair).label_update

    # If E has intrude, so to F and G
    # G is simplex
    # Else F and G are clear
    @pytest.mark.parametrize("cycleE", [True, False])
    def test_correct_update(self, cycleE, connected_labelling, add_simplex_pair):
        connected_labelling['E'] = cycleE
        cycle_label = AddSimplexPair(connected_labelling, add_simplex_pair).label_update
        assert cycle_label == {'F': cycleE, 'G': False}

    # Cannot remove more than one boundary cycle
    def test_raises_too_many_removed(self, connected_labelling, add_simplex_pair):
        add_simplex_pair.cycles_removed = ['D', 'E']
        with pytest.raises(UpdateError):
            AddSimplexPair(connected_labelling, add_simplex_pair).is_valid()

    def test_raises_not_subset(self, connected_labelling, add_simplex_pair):
        add_simplex_pair.simplices_added = ['A']
        with pytest.raises(UpdateError):
            AddSimplexPair(connected_labelling, add_simplex_pair).is_valid()

    def test_raises_bad_cycle(self, connected_labelling, add_simplex_pair):
        add_simplex_pair.cycles_removed = ['Z']
        with pytest.raises(UpdateError):
            AddSimplexPair(connected_labelling, add_simplex_pair).is_valid()


## Test behavior when an edge is removed using the "power-down/connected" approach
# In this case, an edge is  removed joining E and D into boundary cycles F
# cycle E is a simplex.
# This does NOT count the case where any sort of disconnection happens
class TestRemoveConnectedSimplexPair:
    cycles_removed = ['C', 'D']
    cycles_added = ['F']
    simplices_removed = ['C']

    def test_remove_simplex_pair(self, connected_labelling, remove_simplex_pair):
        _ = RemoveSimplexPair(connected_labelling, remove_simplex_pair).label_update

    @pytest.mark.parametrize("cycleD", [True, False])
    def test_correct_update(self, cycleD, connected_labelling, remove_simplex_pair):
        connected_labelling["D"] = cycleD
        cycle_label = RemoveSimplexPair(connected_labelling, remove_simplex_pair).label_update
        assert cycle_label == {'F': cycleD}

    def test_raises_too_many_added(self, connected_labelling, remove_simplex_pair):
        remove_simplex_pair.cycles_added = ['F', 'G']
        with pytest.raises(UpdateError):
            RemoveSimplexPair(connected_labelling, remove_simplex_pair).is_valid()


## Delauny flip where the edge between B and C flips resulting in simplices F and G
class TestConnectedDelaunyFlip:
    cycles_added = ['F', 'G']
    cycles_removed = ['B', 'C']

    def test_delauny_flip(self, connected_labelling, delauny_flip):
        _ = DelaunyFlip(connected_labelling, delauny_flip).label_update

    def test_correct_update(self, connected_labelling, delauny_flip):
        cycle_label = DelaunyFlip(connected_labelling, delauny_flip).label_update
        assert cycle_label == {'F': False, 'G': False}

    def test_raises_too_many(self, connected_labelling, delauny_flip):
        delauny_flip.simplices_added = ['F', 'G', 'H']
        delauny_flip.simplices_removed = ['A', 'B', 'C']
        with pytest.raises(UpdateError):
            DelaunyFlip(connected_labelling, delauny_flip).is_valid()

    def test_raises_not_enough(self, connected_labelling, delauny_flip):
        delauny_flip.simplices_added = ['F']
        delauny_flip.simplices_removed = ['B']
        with pytest.raises(UpdateError):
            DelaunyFlip(connected_labelling, delauny_flip).is_valid()

    def test_raises_same_length(self, connected_labelling, delauny_flip):
        delauny_flip.simplices_added = ['F', 'G', 'H']
        with pytest.raises(UpdateError):
            DelaunyFlip(connected_labelling, delauny_flip).is_valid()

        delauny_flip.simplices_added = ['F']
        with pytest.raises(UpdateError):
            DelaunyFlip(connected_labelling, delauny_flip).is_valid()


class TestGetUpdateData:

    def test_raises_non_atomic(self, connected_labelling):
        state_change = mock.Mock()
        state_change.is_atomic.return_value = False
        pytest.raises(UpdateError, get_update_data, connected_labelling, state_change)

    def test_raises_is_atomic_not_found(self, connected_labelling):
        state_change = mock.Mock()
        state_change.case = (1, 0, 2, 0, 2, 1)
        pytest.raises(UpdateError, get_update_data, connected_labelling, state_change)
