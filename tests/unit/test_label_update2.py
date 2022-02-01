from unittest import mock

import pytest

from update_data2 import \
    LabelUpdate2D, NonAtomicUpdate, Add2SimplicesUpdate2D, \
    Remove2SimplicesUpdate2D, Add1SimplexUpdate2D, AddSimplexPairUpdate2D, \
    RemoveSimplexPairUpdate2D, DelaunyFlipUpdate2D, LabelUpdateFactory
from utilities import UpdateError


@pytest.mark.fixture
def Simplex(name, edges=(), nodes=()):
    s = mock.Mock()
    s.to_cycle.return_value = name
    s.is_subface.side_effect = lambda e: True if e in edges else False
    s.nodes = nodes
    return s


class TestLabelUpdate:

    @pytest.fixture
    def labelling(self, default_labelling):
        return default_labelling

    edges = mock.Mock()
    simplices = mock.MagicMock()
    boundary_cycles = mock.MagicMock()

    def test_init(self, labelling):
        update = LabelUpdate2D(self.edges, self.simplices, self.boundary_cycles, labelling)
        assert update.is_atomic()

    def test_default_mapping(self, labelling):
        update = LabelUpdate2D(self.edges, self.simplices, self.boundary_cycles, labelling)
        assert len(update.mapping) == 0

    def test_nodes_added(self, labelling):
        self.boundary_cycles.added.return_value = {"M", "J"}
        update = LabelUpdate2D(self.edges, self.simplices, self.boundary_cycles, labelling)
        assert {"M", "J"} == update.cycles_added

    def test_nodes_removed(self, labelling):
        self.boundary_cycles.removed.return_value = {"M", "J"}
        update = LabelUpdate2D(self.edges, self.simplices, self.boundary_cycles, labelling)
        assert {"M", "J"} == update.cycles_removed

    def test_is_valid(self, connected_labelling):
        self.boundary_cycles.removed.return_value = {"A"}
        update = LabelUpdate2D(self.edges, self.simplices, self.boundary_cycles, connected_labelling)
        assert update.is_valid()

    def test_default_is_valid(self, connected_labelling):
        update = LabelUpdate2D(self.edges, self.simplices, self.boundary_cycles, connected_labelling)
        assert update.is_valid()

    def test_invalid(self, connected_labelling):
        self.boundary_cycles.removed.return_value = {"G"}
        update = LabelUpdate2D(self.edges, self.simplices, self.boundary_cycles, connected_labelling)
        assert not update.is_valid()


class TestNonAtomicUpdate:
    def test_is_not_atomic(self):
        labelling = mock.Mock()
        edges = mock.Mock()
        simplices = mock.Mock()
        boundary_cycles = mock.Mock()
        update = NonAtomicUpdate(edges, simplices, boundary_cycles, labelling)
        assert not update.is_atomic()


class TestAdd2Simplices:
    @pytest.fixture
    def edges(self):
        return mock.Mock()

    @pytest.fixture
    def simplices(self):
        simplex = Simplex("A")
        set_dif = mock.Mock()
        set_dif.added.return_value = {simplex}
        return set_dif

    @pytest.fixture
    def boundary_cycles(self):
        cycles = mock.Mock()
        return cycles

    def test_add_simplex(self, edges, simplices, boundary_cycles, connected_labelling):
        update = Add2SimplicesUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert update.mapping == {"A": False}

    def test_add_multiple_simplices(self, edges, simplices, boundary_cycles, connected_labelling):
        simplices.added.return_value = {Simplex("A"), Simplex("D")}
        update = Add2SimplicesUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert update.mapping == {"A": False, "D": False}

    def test_updates_single_element(self, edges, simplices, boundary_cycles, connected_labelling):
        update = Add2SimplicesUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert len(update.mapping) == 1

    def test_updates_correct_number_multiple(self, edges, simplices, boundary_cycles, connected_labelling):
        simplices.added.return_value = [Simplex("A"), Simplex("B")]
        update = Add2SimplicesUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert len(update.mapping) == 2


class TestRemove2Simplices:

    @pytest.fixture
    def edges(self):
        return mock.Mock()

    @pytest.fixture
    def simplices(self):
        set_dif = mock.Mock()
        set_dif.removed.return_value = {Simplex("B")}
        return set_dif

    @pytest.fixture
    def boundary_cycles(self):
        cycles = mock.Mock()
        return cycles

    def test_valid_simplex(self, edges, simplices, boundary_cycles, connected_labelling):
        update = Remove2SimplicesUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert update.mapping == {"B": False}

    def test_correct_length(self, edges, simplices, boundary_cycles, connected_labelling):
        update = Remove2SimplicesUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert len(update.mapping) == 1


class TestAdd1Simplex:
    @pytest.fixture
    def edges(self):
        return mock.Mock()

    @pytest.fixture
    def simplices(self):
        set_dif = mock.Mock()
        return set_dif

    @pytest.fixture
    def boundary_cycles(self):
        cycles = mock.Mock()
        cycles.added.return_value = {"F", "G"}
        cycles.removed.return_value = {"B"}
        return cycles

    def test_added_nodes(self, edges, simplices, boundary_cycles, connected_labelling):
        update = Add1SimplexUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert update.cycles_added == {"F", "G"}

    def test_split_false(self, edges, simplices, boundary_cycles, connected_labelling):
        update = Add1SimplexUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert update.mapping == {"F": False, "G": False}

    def test_split_true(self, edges, simplices, boundary_cycles, connected_labelling):
        boundary_cycles.removed.return_value = ["A"]
        update = Add1SimplexUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert update.mapping == {"F": True, "G": True}

    def test_correct_size(self, edges, simplices, boundary_cycles, connected_labelling):
        update = Add1SimplexUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert len(update.mapping) == 2


class TestRemove1Simplex:
    @pytest.fixture
    def edges(self):
        return mock.Mock()

    @pytest.fixture
    def simplices(self):
        set_dif = mock.Mock()
        return set_dif

    @pytest.fixture
    def boundary_cycles(self):
        cycles = mock.Mock()
        cycles.removed.return_value = {"A", "B"}
        cycles.added.return_value = {"F"}
        return cycles

    def test_valid_true_false(self, edges, simplices, boundary_cycles, connected_labelling):
        update = Add1SimplexUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert update.mapping == {"F": True}

    def test_valid_true_true(self, edges, simplices, boundary_cycles, connected_labelling):
        boundary_cycles.removed.return_value = {"D", "E"}
        update = Add1SimplexUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert update.mapping == {"F": True}

    def test_valid_false_false(self, edges, simplices, boundary_cycles, connected_labelling):
        boundary_cycles.removed.return_value = ["B", "C"]
        update = Add1SimplexUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert update.mapping == {"F": False}

    def test_correct_length(self, edges, simplices, boundary_cycles, connected_labelling):
        update = Add1SimplexUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert len(update.mapping) == 1


class TestAddSimplexPair:
    @pytest.fixture
    def edges(self):
        edge_dif = mock.Mock()
        edge_dif.added.return_value = {"fg"}
        return edge_dif

    @pytest.fixture
    def simplices(self):
        simp_dif = mock.Mock()
        simp_dif.added.return_value = [Simplex("G", edges=["fg"])]
        return simp_dif

    @pytest.fixture
    def boundary_cycles(self):
        cycles = mock.Mock()
        cycles.new_list = {"A", "B", "C", "D", "E"}
        cycles.removed.return_value = {"D"}
        cycles.added.return_value = {"F", "G"}
        return cycles

    def test_split_true(self, edges, simplices, boundary_cycles, connected_labelling):
        update = AddSimplexPairUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert update.mapping == {"G": False, "F": True}

    def test_split_false(self, edges, simplices, boundary_cycles, connected_labelling):
        boundary_cycles.removed.return_value = {"B"}
        update = AddSimplexPairUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert update.mapping == {"G": False, "F": False}

    def test_is_atomic(self, edges, simplices, boundary_cycles, connected_labelling):
        update = AddSimplexPairUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert update.is_atomic()

    def test_is_not_atomic(self, edges, simplices, boundary_cycles, connected_labelling):
        edges.added.return_value = {"bad_edge"}
        update = AddSimplexPairUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert not update.is_atomic()


class TestRemoveSimplexPair:
    @pytest.fixture
    def edges(self):
        edge_dif = mock.Mock()
        edge_dif.removed.return_value = {"ab"}
        return edge_dif

    @pytest.fixture
    def simplices(self):
        simp_dif = mock.Mock()
        simp_dif.removed.return_value = [Simplex("B", edges=["ab"])]
        return simp_dif

    @pytest.fixture
    def boundary_cycles(self):
        cycles = mock.Mock()
        cycles.old_list = {"A", "B", "C", "D", "E"}
        cycles.removed.return_value = {"A", "B"}
        cycles.added.return_value = {"F"}
        return cycles

    def test_join_true(self, edges, simplices, boundary_cycles, connected_labelling):
        update = RemoveSimplexPairUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert update.mapping == {"F": True}

    def test_join_false(self, edges, simplices, boundary_cycles, connected_labelling):
        boundary_cycles.removed.return_value = {"C", "B"}
        update = RemoveSimplexPairUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert update.mapping == {"F": False}

    def test_is_atomic(self, edges, simplices, boundary_cycles, connected_labelling):
        update = RemoveSimplexPairUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert update.is_atomic()

    def test_is_not_atomic(self, edges, simplices, boundary_cycles, connected_labelling):
        edges.removed.return_value = {"bad_edge"}
        update = RemoveSimplexPairUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert not update.is_atomic()


class TestDelaunyFlip:
    @pytest.fixture
    def edges(self):
        edge_dif = mock.Mock()
        edge_dif.removed.return_value = {"ab"}
        edge_dif.added.return_value = {"fg"}
        return edge_dif

    @pytest.fixture
    def simplices(self):
        simplexA = Simplex("A", edges=["ab"], nodes=(1, 2, 3))
        simplexB = Simplex("B", edges=["ab"], nodes=(2, 3, 4))

        simplexF = Simplex("F", edges=["fg"], nodes=(3, 4, 1))
        simplexG = Simplex("G", edges=["fg"], nodes=(2, 4, 1))

        simp_dif = mock.Mock()
        simp_dif.removed.return_value = {simplexA, simplexB}
        simp_dif.added.return_value = {simplexF, simplexG}
        return simp_dif

    @pytest.fixture
    def boundary_cycles(self):
        cycles = mock.Mock()
        cycles.old_list = {"A", "B", "C", "D", "E"}
        cycles.removed.return_value = {"A", "B"}
        cycles.added.return_value = {"F", "G"}
        return cycles

    def test_all_false(self, edges, simplices, boundary_cycles, connected_labelling):
        update = DelaunyFlipUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert update.mapping == {"F": False, "G": False}

    def test_is_atomic(self, edges, simplices, boundary_cycles, connected_labelling):
        update = DelaunyFlipUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert update.is_atomic()

    def test_non_atomic_old_edges(self, edges, simplices, boundary_cycles, connected_labelling):
        edges.removed.return_value = ['bad_edge']
        update = DelaunyFlipUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert not update.is_atomic()

    def test_non_atomic_new_edges(self, edges, simplices, boundary_cycles, connected_labelling):
        edges.added.return_value = ['bad_edge']
        update = DelaunyFlipUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert not update.is_atomic()

    def test_non_atomic_non_overlapping(self, edges, simplices, boundary_cycles, connected_labelling):
        simplexF = Simplex("F", edges=["fg"], nodes=(3, 4, 1))
        simplexG = Simplex("G", edges=["fg"], nodes=(5, 4, 1))

        simplices.added.return_value = [simplexF, simplexG]
        update = DelaunyFlipUpdate2D(edges, simplices, boundary_cycles, connected_labelling)
        assert not update.is_atomic()


class TestLabelFactory:
    def test_update_dict_nonempty(self):
        assert len(LabelUpdateFactory.atomic_updates) != 0

    def test_add1_simplex_in_dict(self):
        case = (1, 0, 0, 0, 2, 1)
        assert LabelUpdateFactory.atomic_updates[case] == Add1SimplexUpdate2D

    def test_get_add1simplex(self):
        labelling = mock.Mock()
        sc = mock.Mock()
        sc.case = (1, 0, 0, 0, 2, 1)
        update = LabelUpdateFactory.get_label_update(sc)
        assert type(update(labelling)) == Add1SimplexUpdate2D

    def test_returns_nonatomic(self):
        labelling = mock.Mock()
        sc = mock.Mock()
        sc.case = (1, 0, 0, 30, 2, 1)
        update = LabelUpdateFactory.get_label_update(sc)
        assert type(update(labelling)) == NonAtomicUpdate

    def test_raise_on_invalid_state_change(self):
        sc = mock.Mock()
        sc.is_valid.return_value = False
        assert pytest.raises(UpdateError, LabelUpdateFactory.get_label_update, sc)
