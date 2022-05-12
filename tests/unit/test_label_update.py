from unittest import mock
from unittest.mock import patch

import pytest

from update_data import \
    LabelUpdate, NonAtomicUpdate, Add2SimplicesUpdate2D, \
    Remove2SimplicesUpdate2D, Add1SimplexUpdate2D, AddSimplexPairUpdate2D, \
    RemoveSimplexPairUpdate2D, DelaunyFlipUpdate2D, LabelUpdateFactory, Remove1SimplexUpdate2D, FinUpdate3D, \
    FillTetrahedronFace, DrainTetrahedronFace
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
        update = LabelUpdate(lambda dim: self.edges if dim == 1 else self.simplices, self.boundary_cycles, labelling)
        assert update.is_atomic()

    def test_default_mapping(self, labelling):
        update = LabelUpdate(lambda dim: self.edges if dim == 1 else self.simplices, self.boundary_cycles, labelling)
        assert len(update.mapping) == 0

    def test_nodes_added(self, labelling):
        self.boundary_cycles.added.return_value = {"M", "J"}
        update = LabelUpdate(lambda dim: self.edges if dim == 1 else self.simplices, self.boundary_cycles, labelling)
        assert {"M", "J"} == update.cycles_added

    def test_nodes_removed(self, labelling):
        self.boundary_cycles.removed.return_value = {"M", "J"}
        update = LabelUpdate(lambda dim: self.edges if dim == 1 else self.simplices, self.boundary_cycles, labelling)
        assert {"M", "J"} == update.cycles_removed


class TestNonAtomicUpdate:
    def test_is_not_atomic(self):
        labelling = mock.Mock()
        simplices = mock.Mock()
        boundary_cycles = mock.Mock()
        update = NonAtomicUpdate(simplices, boundary_cycles, labelling)
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
        update = Add2SimplicesUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
        assert update.mapping == {"A": False}

    def test_add_multiple_simplices(self, edges, simplices, boundary_cycles, connected_labelling):
        simplices.added.return_value = {Simplex("A"), Simplex("D")}
        update = Add2SimplicesUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
        assert update.mapping == {"A": False, "D": False}

    def test_updates_single_element(self, edges, simplices, boundary_cycles, connected_labelling):
        update = Add2SimplicesUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
        assert len(update.mapping) == 1

    def test_updates_correct_number_multiple(self, edges, simplices, boundary_cycles, connected_labelling):
        simplices.added.return_value = [Simplex("A"), Simplex("B")]
        update = Add2SimplicesUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
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
        update = Remove2SimplicesUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
        assert update.mapping == {"B": False}

    def test_correct_length(self, edges, simplices, boundary_cycles, connected_labelling):
        update = Remove2SimplicesUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
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
        update = Add1SimplexUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
        assert update.cycles_added == {"F", "G"}

    def test_split_false(self, edges, simplices, boundary_cycles, connected_labelling):
        update = Add1SimplexUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
        assert update.mapping == {"F": False, "G": False}

    def test_split_true(self, edges, simplices, boundary_cycles, connected_labelling):
        boundary_cycles.removed.return_value = ["A"]
        update = Add1SimplexUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
        assert update.mapping == {"F": True, "G": True}

    def test_correct_size(self, edges, simplices, boundary_cycles, connected_labelling):
        update = Add1SimplexUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
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
        update = Remove1SimplexUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
        assert update.mapping == {"F": True}

    def test_valid_true_true(self, edges, simplices, boundary_cycles, connected_labelling):
        boundary_cycles.removed.return_value = {"D", "E"}
        update = Remove1SimplexUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
        assert update.mapping == {"F": True}

    def test_valid_false_false(self, edges, simplices, boundary_cycles, connected_labelling):
        boundary_cycles.removed.return_value = ["B", "C"]
        update = Remove1SimplexUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
        assert update.mapping == {"F": False}

    def test_correct_length(self, edges, simplices, boundary_cycles, connected_labelling):
        update = Remove1SimplexUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
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
        update = AddSimplexPairUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
        assert update.mapping == {"G": False, "F": True}

    def test_split_false(self, edges, simplices, boundary_cycles, connected_labelling):
        boundary_cycles.removed.return_value = {"B"}
        update = AddSimplexPairUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
        assert update.mapping == {"G": False, "F": False}

    def test_is_atomic(self, edges, simplices, boundary_cycles, connected_labelling):
        update = AddSimplexPairUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
        assert update.is_atomic()

    def test_is_not_atomic(self, edges, simplices, boundary_cycles, connected_labelling):
        edges.added.return_value = {"bad_edge"}
        update = AddSimplexPairUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
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
        update = RemoveSimplexPairUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
        assert update.mapping == {"F": True}

    def test_join_false(self, edges, simplices, boundary_cycles, connected_labelling):
        boundary_cycles.removed.return_value = {"C", "B"}
        update = RemoveSimplexPairUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
        assert update.mapping == {"F": False}

    def test_is_atomic(self, edges, simplices, boundary_cycles, connected_labelling):
        update = RemoveSimplexPairUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
        assert update.is_atomic()

    def test_is_not_atomic(self, edges, simplices, boundary_cycles, connected_labelling):
        edges.removed.return_value = {"bad_edge"}
        update = RemoveSimplexPairUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
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
        update = DelaunyFlipUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
        assert update.mapping == {"F": False, "G": False}

    def test_is_atomic(self, edges, simplices, boundary_cycles, connected_labelling):
        update = DelaunyFlipUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
        assert update.is_atomic()

    def test_non_atomic_old_edges(self, edges, simplices, boundary_cycles, connected_labelling):
        edges.removed.return_value = ['bad_edge']
        update = DelaunyFlipUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
        assert not update.is_atomic()

    def test_non_atomic_new_edges(self, edges, simplices, boundary_cycles, connected_labelling):
        edges.added.return_value = ['bad_edge']
        update = DelaunyFlipUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
        assert not update.is_atomic()

    def test_non_atomic_non_overlapping(self, edges, simplices, boundary_cycles, connected_labelling):
        simplexF = Simplex("F", edges=["fg"], nodes=(3, 4, 1))
        simplexG = Simplex("G", edges=["fg"], nodes=(5, 4, 1))

        simplices.added.return_value = [simplexF, simplexG]
        update = DelaunyFlipUpdate2D({1: edges, 2: simplices}, boundary_cycles, connected_labelling)
        assert not update.is_atomic()


@pytest.mark.fixture
def mock_state_change(case, is_valid):
    temp = mock.Mock()
    temp.is_valid.return_value = is_valid
    temp.case = case
    return temp


class TestLabelFactory:
    def test_update_dict_nonempty(self):
        assert len(LabelUpdateFactory.atomic_updates) != 0

    def test_add1_simplex_in_dict(self):
        case = (1, 0, 0, 0, 2, 1)
        assert LabelUpdateFactory.atomic_updates[case] == Add1SimplexUpdate2D

    def test_3d_in_dict(self):
        case = (0, 0, 0, 0, 1, 0, 0, 0)
        assert LabelUpdateFactory.atomic_updates[case] == Add2SimplicesUpdate2D

    @patch("update_data.StateChange")
    def test_get_add1simplex(self, StateChange):
        StateChange.return_value = mock_state_change((1, 0, 0, 0, 2, 1), True)
        labelling = mock.Mock()
        t1 = mock.Mock()
        update = LabelUpdateFactory().get_update(t1, t1, labelling)
        assert type(update) == Add1SimplexUpdate2D

    @patch("update_data.StateChange")
    def test_returns_nonatomic(self, StateChange):
        StateChange.return_value = mock_state_change((1, 0, 0, 30, 2, 1), True)
        labelling = mock.Mock()
        t1 = mock.Mock()
        update = LabelUpdateFactory().get_update(t1, t1, labelling)
        assert type(update) == NonAtomicUpdate

    @patch("update_data.StateChange")
    def test_raise_on_invalid_state_change(self, StateChange):
        labelling = mock.Mock()
        StateChange.return_value = mock_state_change((1, 0, 0, 30, 2, 1), False)
        t1 = mock.Mock()
        assert pytest.raises(UpdateError, LabelUpdateFactory.get_update, t1, t1, labelling)


class TestOnetoOne3D:
    @pytest.fixture
    def edges(self):
        return mock.Mock()

    @pytest.fixture
    def simplices(self):
        return mock.Mock()

    @pytest.fixture
    def tetrahedra(self):
        return mock.Mock()

    @pytest.fixture
    def boundary_cycles(self):
        cycles = mock.Mock()
        cycles.added.return_value = {"F"}
        cycles.removed.return_value = {"B"}
        return cycles

    def test_added_nodes(self, edges, simplices, tetrahedra, boundary_cycles, connected_labelling):
        update = FinUpdate3D({1: edges, 2: simplices, 3: tetrahedra}, boundary_cycles, connected_labelling)
        assert update.cycles_added == {"F"}

    def test_false(self, edges, simplices, tetrahedra, boundary_cycles, connected_labelling):
        update = FinUpdate3D({1: edges, 2: simplices, 3: tetrahedra}, boundary_cycles, connected_labelling)
        assert update.mapping == {"F": False}

    def test_true(self, edges, simplices, tetrahedra, boundary_cycles, connected_labelling):
        boundary_cycles.removed.return_value = {"A"}
        update = FinUpdate3D({1: edges, 2: simplices, 3: tetrahedra}, boundary_cycles, connected_labelling)
        assert update.mapping == {"F": True}


class TestAddSimplexPair3D:
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
    def tetrahedra(self):
        return mock.Mock()

    @pytest.fixture
    def boundary_cycles(self):
        cycles = mock.Mock()
        cycles.new_list = {"A", "B", "C", "E", "F", "G"}
        cycles.removed.return_value = {"D"}
        cycles.added.return_value = {"F", "G"}
        return cycles

    def test_is_atomic(self, edges, simplices, tetrahedra, boundary_cycles, connected_labelling):
        update = AddSimplexPairUpdate2D({1: edges, 2: simplices, 3: tetrahedra}, boundary_cycles, connected_labelling)
        assert update.is_atomic()

    def test_is_not_atomic(self, edges, simplices, tetrahedra, boundary_cycles, connected_labelling):
        edges.added.return_value = {"bad_edge"}
        update = AddSimplexPairUpdate2D({1: edges, 2: simplices, 3: tetrahedra}, boundary_cycles, connected_labelling)
        assert not update.is_atomic()


class TestFillTetrahedronFace:
    @pytest.fixture
    def edges(self):
        return mock.Mock()

    @pytest.fixture
    def simplices(self):
        face_dif = mock.Mock()
        face_dif.added.return_value = {"fgh"}
        return face_dif

    @pytest.fixture
    def tetrahedra(self):
        simp_dif = mock.Mock()
        simp_dif.added.return_value = [Simplex("G", edges=["fgh"])]
        return simp_dif

    @pytest.fixture
    def boundary_cycles(self):
        cycles = mock.Mock()
        cycles.new_list = {"A", "B", "C", "E", "F", "G"}
        cycles.removed.return_value = {"D"}
        cycles.added.return_value = {"F", "G"}
        return cycles

    def test_split_true(self, edges, simplices, tetrahedra, boundary_cycles, connected_labelling):
        update = FillTetrahedronFace({1: edges, 2: simplices, 3: tetrahedra}, boundary_cycles, connected_labelling)
        assert update.mapping == {"G": False, "F": True}

    def test_split_false(self, edges, simplices, tetrahedra, boundary_cycles, connected_labelling):
        boundary_cycles.removed.return_value = {"B"}
        update = FillTetrahedronFace({1: edges, 2: simplices, 3: tetrahedra}, boundary_cycles, connected_labelling)
        assert update.mapping == {"G": False, "F": False}

    def test_is_atomic(self, edges, simplices, tetrahedra, boundary_cycles, connected_labelling):
        update = FillTetrahedronFace({1: edges, 2: simplices, 3: tetrahedra}, boundary_cycles, connected_labelling)
        assert update.is_atomic()

    def test_is_not_atomic(self, edges, simplices, tetrahedra, boundary_cycles, connected_labelling):
        simplices.added.return_value = {"bad_edge"}
        update = FillTetrahedronFace({1: edges, 2: simplices, 3: tetrahedra}, boundary_cycles, connected_labelling)
        assert not update.is_atomic()


class TestDrainTetrahedronFace:
    @pytest.fixture
    def edges(self):
        return mock.Mock()

    @pytest.fixture
    def simplices(self):
        simp_dif = mock.Mock()
        simp_dif.added.return_value = ["cde"]
        return simp_dif

    @pytest.fixture
    def tetrahedra(self):
        tetra_dif = mock.Mock()
        tetra_dif.added.return_value = [Simplex("C", edges=["cde"])]
        return tetra_dif

    @pytest.fixture
    def boundary_cycles(self):
        cycles = mock.Mock()
        cycles.new_list = {"A", "B", "C", "E", "F"}
        cycles.removed.return_value = {"C", "D"}
        cycles.added.return_value = {"F"}
        return cycles

    def test_is_atomic(self, edges, simplices, tetrahedra, boundary_cycles, connected_labelling):
        update = DrainTetrahedronFace({1: edges, 2: simplices, 3: tetrahedra}, boundary_cycles, connected_labelling)
        assert update.is_atomic()

    def test_is_not_atomic(self, edges, simplices, tetrahedra, boundary_cycles, connected_labelling):
        simplices.added.return_value = {"bad_edge"}
        update = DrainTetrahedronFace({1: edges, 2: simplices, 3: tetrahedra}, boundary_cycles, connected_labelling)
        assert not update.is_atomic()


class TestTetrahedronEdgeFill:
    new_edge = "fgh"
    facef1 = Simplex("fg1", edges=[new_edge])
    facef2 = Simplex("fg1", edges=[new_edge])

    @pytest.fixture
    def edges(self):
        edges = mock.Mock()
        edges.added.return_value = [self.new_edge]
        return mock.Mock()

    @pytest.fixture
    def simplices(self):
        simp_dif = mock.Mock()
        simp_dif.added.return_value = [self.facef1, self.facef2]
        return simp_dif

    @pytest.fixture
    def tetrahedra(self):
        tetra_dif = mock.Mock()
        tetra_dif.added.return_value = [Simplex("F", edges=[self.facef1, self.facef2, self.new_edge])]
        return tetra_dif

    @pytest.fixture
    def boundary_cycles(self):
        cycles = mock.Mock()
        cycles.new_list = {"A", "B", "C", "E", "F", "G"}
        cycles.removed.return_value = {"D"}
        cycles.added.return_value = {"F", "G"}
        return cycles

    def test_is_atomic(self, edges, simplices, tetrahedra, boundary_cycles, connected_labelling):
        update = DrainTetrahedronFace({1: edges, 2: simplices, 3: tetrahedra}, boundary_cycles, connected_labelling)
        assert update.is_atomic()

    def test_is_not_atomic(self, edges, simplices, tetrahedra, boundary_cycles, connected_labelling):
        simplices.added.return_value = {"bad_edge"}
        update = DrainTetrahedronFace({1: edges, 2: simplices, 3: tetrahedra}, boundary_cycles, connected_labelling)
        assert not update.is_atomic()
