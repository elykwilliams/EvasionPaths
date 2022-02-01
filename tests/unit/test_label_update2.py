from unittest import mock

import pytest

from update_data2 import \
    LabelUpdate, NonAtomicUpdate, Add2Simplices, \
    Remove2Simplices, Add1Simplex, AddSimplexPair, \
    RemoveSimplexPair, DelaunyFlip, LabelUpdateFactory


class TestLabelUpdate:

    @pytest.fixture
    def labelling(self, default_labelling):
        return default_labelling

    def test_init(self, trivial_state_change, labelling):
        update = LabelUpdate(trivial_state_change, labelling)
        assert update.is_atomic()

    def test_default_mapping(self, trivial_state_change, labelling):
        update = LabelUpdate(trivial_state_change, labelling)
        assert len(update.mapping) == 0

    def test_nodes_added(self, trivial_state_change, labelling):
        trivial_state_change.boundary_cycles.added.return_value = ["M", "J"]
        update = LabelUpdate(trivial_state_change, labelling)
        assert ["M", "J"] == update.nodes_added

    def test_nodes_removed(self, trivial_state_change, labelling):
        trivial_state_change.boundary_cycles.removed.return_value = ["M", "J"]
        update = LabelUpdate(trivial_state_change, labelling)
        assert ["M", "J"] == update.nodes_removed

    def test_default_is_valid(self, trivial_state_change, labelling):
        update = LabelUpdate(trivial_state_change, labelling)
        assert update.is_valid()

    def test_is_valid(self, default_state_change, connected_labelling):
        default_state_change.boundary_cycles.removed.return_value = ["A"]
        update = LabelUpdate(default_state_change, connected_labelling)
        assert update.is_valid()

    def test_invalid(self, default_state_change, connected_labelling):
        default_state_change.boundary_cycles.removed.return_value = ["G"]
        update = LabelUpdate(default_state_change, connected_labelling)
        assert not update.is_valid()


class TestNonAtomicUpdate:
    def test_is_not_atomic(self, default_state_change):
        labelling = mock.Mock()
        update = NonAtomicUpdate(default_state_change, labelling)
        assert not update.is_atomic()


class TestAdd2Simplices:
    def test_add_simplex(self, add_2simplex_state_change, connected_labelling):
        update = Add2Simplices(add_2simplex_state_change, connected_labelling)
        assert update.mapping == {"A": False}

    def test_add_multiple_simplices(self, add_2simplex_state_change, connected_labelling):
        simplexA = mock.Mock()
        simplexA.to_cycle.return_value = "A"
        simplexD = mock.Mock()
        simplexD.to_cycle.return_value = "D"
        add_2simplex_state_change.simplices.added.return_value = [simplexA, simplexD]
        update = Add2Simplices(add_2simplex_state_change, connected_labelling)
        assert update.mapping == {"A": False, "D": False}

    def test_updates_single_element(self, add_2simplex_state_change, connected_labelling):
        update = Add2Simplices(add_2simplex_state_change, connected_labelling)
        assert len(update.mapping) == 1

    def test_updates_correct_number_multiple(self, add_2simplex_state_change, connected_labelling):
        simplexA = mock.Mock()
        simplexA.to_cycle.return_value = "A"
        simplexB = mock.Mock()
        simplexB.to_cycle.return_value = "B"
        add_2simplex_state_change.simplices.added.return_value = [simplexA, simplexB]
        update = Add2Simplices(add_2simplex_state_change, connected_labelling)
        assert len(update.mapping) == 2


class TestRemove2Simplices:
    def test_valid_simplex(self, remove_2simplex_state_change, connected_labelling):
        update = Remove2Simplices(remove_2simplex_state_change, connected_labelling)
        assert update.mapping == {"B": False}

    def test_correct_length(self, remove_2simplex_state_change, connected_labelling):
        update = Remove2Simplices(remove_2simplex_state_change, connected_labelling)
        assert len(update.mapping) == 1


class TestAdd1Simplex:
    def test_added_nodes(self, add_1simplex_state_change, connected_labelling):
        update = Add1Simplex(add_1simplex_state_change, connected_labelling)
        assert update.nodes_added == ["F", "G"]

    def test_split_false(self, add_1simplex_state_change, connected_labelling):
        update = Add1Simplex(add_1simplex_state_change, connected_labelling)
        assert update.mapping == {"F": False, "G": False}

    def test_split_true(self, add_1simplex_state_change, connected_labelling):
        add_1simplex_state_change.boundary_cycles.removed.return_value = ["A"]
        update = Add1Simplex(add_1simplex_state_change, connected_labelling)
        assert update.mapping == {"F": True, "G": True}

    def test_correct_size(self, add_1simplex_state_change, connected_labelling):
        update = Add1Simplex(add_1simplex_state_change, connected_labelling)
        assert len(update.mapping) == 2


class TestRemove1Simplex:
    def test_valid_true_false(self, remove_1simplex_state_change, connected_labelling):
        update = Add1Simplex(remove_1simplex_state_change, connected_labelling)
        assert update.mapping == {"F": True}

    def test_valid_true_true(self, remove_1simplex_state_change, connected_labelling):
        remove_1simplex_state_change.boundary_cycles.removed.return_value = ["D", "E"]
        update = Add1Simplex(remove_1simplex_state_change, connected_labelling)
        assert update.mapping == {"F": True}

    def test_valid_false_false(self, remove_1simplex_state_change, connected_labelling):
        remove_1simplex_state_change.boundary_cycles.removed.return_value = ["B", "C"]
        update = Add1Simplex(remove_1simplex_state_change, connected_labelling)
        assert update.mapping == {"F": False}

    def test_correct_length(self, remove_1simplex_state_change, connected_labelling):
        update = Add1Simplex(remove_1simplex_state_change, connected_labelling)
        assert len(update.mapping) == 1


class TestAddSimplexPair:
    def test_split_true(self, add_simplex_pair_state_change, connected_labelling):
        update = AddSimplexPair(add_simplex_pair_state_change, connected_labelling)
        assert update.mapping == {"G": False, "F": True}

    def test_split_false(self, add_simplex_pair_state_change, connected_labelling):
        add_simplex_pair_state_change.boundary_cycles.removed.return_value = ["B"]
        update = AddSimplexPair(add_simplex_pair_state_change, connected_labelling)
        assert update.mapping == {"G": False, "F": False}

    def test_is_atomic(self, add_simplex_pair_state_change, connected_labelling):
        update = AddSimplexPair(add_simplex_pair_state_change, connected_labelling)
        assert update.is_atomic()

    def test_is_not_atomic(self, add_simplex_pair_state_change, connected_labelling):
        add_simplex_pair_state_change.edges.added.return_value = ["bad_edge"]
        update = AddSimplexPair(add_simplex_pair_state_change, connected_labelling)
        assert not update.is_atomic()


class TestRemoveSimplexPair:
    def test_join_true(self, remove_simplex_pair_state_change, connected_labelling):
        update = RemoveSimplexPair(remove_simplex_pair_state_change, connected_labelling)
        assert update.mapping == {"F": True}

    def test_join_false(self, remove_simplex_pair_state_change, connected_labelling):
        remove_simplex_pair_state_change.boundary_cycles.removed.return_value = ["C", "B"]
        update = RemoveSimplexPair(remove_simplex_pair_state_change, connected_labelling)
        assert update.mapping == {"F": False}

    def test_is_atomic(self, remove_simplex_pair_state_change, connected_labelling):
        update = RemoveSimplexPair(remove_simplex_pair_state_change, connected_labelling)
        assert update.is_atomic()

    def test_is_not_atomic(self, remove_simplex_pair_state_change, connected_labelling):
        remove_simplex_pair_state_change.edges.removed.return_value = ["bad_edge"]
        update = RemoveSimplexPair(remove_simplex_pair_state_change, connected_labelling)
        assert not update.is_atomic()


class TestDelaunyFlip:
    def test_all_false(self, delauny_state_change, connected_labelling):
        update = DelaunyFlip(delauny_state_change, connected_labelling)
        assert update.mapping == {"F": False, "G": False}

    def test_is_atomic(self, delauny_state_change, connected_labelling):
        update = DelaunyFlip(delauny_state_change, connected_labelling)
        assert update.is_atomic()

    def test_non_atomic_old_edges(self, delauny_state_change, connected_labelling):
        delauny_state_change.edges.removed.return_value = ['bad_edge']
        update = DelaunyFlip(delauny_state_change, connected_labelling)
        assert not update.is_atomic()

    def test_non_atomic_new_edges(self, delauny_state_change, connected_labelling):
        delauny_state_change.edges.added.return_value = ['bad_edge']
        update = DelaunyFlip(delauny_state_change, connected_labelling)
        assert not update.is_atomic()

    def test_non_atomic_non_overlapping(self, delauny_state_change, connected_labelling):
        simplexF = mock.Mock()
        simplexF.to_cycle.return_value = "F"
        simplexF.is_subface.side_effect = lambda e: True if e == "fg" else False
        simplexF.nodes = (3, 4, 1)

        simplexG = mock.Mock()
        simplexG.to_cycle.return_value = "G"
        simplexG.is_subface.side_effect = lambda e: True if e == "fg" else False
        simplexG.nodes = (5, 4, 1)

        delauny_state_change.simplices.added.return_value = [simplexF, simplexG]
        update = DelaunyFlip(delauny_state_change, connected_labelling)
        assert not update.is_atomic()


class TestLabelFactory:
    def test_update_dict_nonempty(self):
        assert len(LabelUpdateFactory.atomic_updates) != 0

    def test_add1_simplex_in_dict(self):
        case = (1, 0, 0, 0, 2, 1)
        assert LabelUpdateFactory.atomic_updates[case] == Add1Simplex

    def test_get_add1simplex(self):
        sc = mock.Mock()
        sc.case = (1, 0, 0, 0, 2, 1)
        update = LabelUpdateFactory.get_label_update(sc)
        assert update == Add1Simplex

    def test_returns_nonatomic(self):
        sc = mock.Mock()
        sc.case = (1, 0, 0, 30, 2, 1)
        update = LabelUpdateFactory().get_label_update(sc)
        assert update == NonAtomicUpdate
