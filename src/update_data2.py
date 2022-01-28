from collections import defaultdict

from dataclasses import dataclass

from cycle_labelling import CycleLabellingTree
from topological_state import TopologicalState
from utilities import set_difference, UpdateError, is_subset


@dataclass
class StateChange:
    ## Identify Atomic States
    #
    # (#1-simplices added, #1-simpleices removed, #2-simplices added, #2-simplices removed, #boundary cycles added,
    # #boundary cycles removed)

    new_state: TopologicalState
    old_state: TopologicalState

    @property
    def case(self):
        return (len(self.edges_added), len(self.edges_removed), len(self.simplices_added),
                len(self.simplices_removed), len(self.cycles_added), len(self.cycles_removed))

    @property
    def edges_added(self):
        return set_difference(self.new_state.simplices(1), self.old_state.simplices(1))

    @property
    def edges_removed(self):
        return set_difference(self.old_state.simplices(1), self.new_state.simplices(1))

    @property
    def simplices_added(self):
        return set_difference(self.new_state.simplices(2), self.old_state.simplices(1))

    @property
    def simplices_removed(self):
        return set_difference(self.old_state.simplices(2), self.new_state.simplices(2))

    @property
    def cycles_added(self):
        return set_difference(self.new_state.boundary_cycles(), self.old_state.boundary_cycles())

    @property
    def cycles_removed(self):
        return set_difference(self.old_state.boundary_cycles(), self.new_state.boundary_cycles())

    ## Allow class to be printable.
    # Used mostly for debugging
    def __repr__(self) -> str:
        return (
            f"State Change: {self.case}\n"
            f"New edges: {self.edges_added}\n"
            f"Removed edges: {self.edges_removed}\n"
            f"New Simplices: {self.simplices_added}\n"
            f"Removed Simplices: {self.simplices_removed}\n"
            f"New cycles {self.cycles_added}\n"
            f"Removed Cycles {self.cycles_removed}"
        )


class LabelUpdate(StateChange):
    labels: CycleLabellingTree

    ## Check if update is self-consistant
    def is_valid(self):
        if not all(cycle in self.labels or cycle in self.cycles_added for cycle in self.simplices_added):
            raise UpdateError(f"You are attempting to add a simplex that has not yet been added to the tree")
        elif not all(cycle in self.labels for cycle in self.simplices_removed + self.cycles_removed):
            raise UpdateError(f"You are attempting to remove a simplex/cycle that has not yet been added to the tree")
        return True

    # Return Mapping with labelling for each cycle with a new labelling.
    @property
    def mapping(self):
        return dict()

    def is_atomic(self):
        return True


class LabelUpdateFactory:
    atomic_updates = defaultdict(default_factory=lambda *args: NonAtomicUpdate(*args))

    @classmethod
    def get_label_update(cls, state_change, labelling):
        update = cls.atomic_updates[state_change.case](state_change.new_state, state_change.old_state, labelling)
        try:
            if update.is_valid():
                return update
        except UpdateError:
            raise

    @classmethod
    def register(cls, case):
        def deco(deco_cls):
            cls.atomic_updates[case] = deco_cls
            return deco_cls

        return deco


class NonAtomicUpdate(LabelUpdate):
    def is_atomic(self):
        return False


@LabelUpdateFactory.register((0, 0, 0, 0, 0, 0))
class TrivialUpdate(LabelUpdate):
    pass


@LabelUpdateFactory.register((0, 0, 1, 0, 0, 0))
class Add2Simplices(LabelUpdate):
    ## All 2-Simplices are Labeled False
    @property
    def mapping(self):
        return {cycle: False for cycle in self.simplices_added}


@LabelUpdateFactory.register((0, 0, 0, 1, 0, 0))
class Remove2Simplices(LabelUpdate):
    #  No update needed, use default
    pass


@LabelUpdateFactory.register((1, 0, 0, 0, 2, 1))
class Add1Simplex(LabelUpdate):
    ## Add edge.
    # New cycles will match the removed cycle
    @property
    def mapping(self):
        return {cycle: self.labels[self.cycles_removed[0]] for cycle in self.cycles_added}


@LabelUpdateFactory.register((0, 1, 0, 0, 1, 2))
class Remove1Simplex(LabelUpdate):
    ## Remove edge.
    # added cycle will have intruder if either removed cycle has intruder
    @property
    def mapping(self):
        return {cycle: any([self.labels[cycle] for cycle in self.cycles_removed]) for cycle in self.cycles_added}


@LabelUpdateFactory.register((1, 0, 1, 0, 2, 1))
class AddSimplexPair(Add1Simplex):
    ## Add simplex pair
    # Will split a cycle in two with label as in Add1Simplex, then label 2simplex as False
    @property
    def mapping(self):
        return super().mapping.update({cycle: False for cycle in self.simplices_added})

    # If a 1-simplex and 2-simplex are added/removed simultaniously, then the 1-simplex
    # should be an edge of the 2-simplex
    def is_atomic(self):
        # Check consistency with definition
        simplex = self.simplices_added[0]
        edge = self.edges_added[0]
        if not is_subset(edge, simplex):
            return False
        return True


@LabelUpdateFactory.register((0, 1, 0, 1, 1, 2))
class RemoveSimplexPair(Remove1Simplex):
    ## Same as Remove1Simplex
    def is_atomic(self):
        # If a 1-simplex and 2-simplex are added/removed simultaniously, then the 1-simplex
        # should be an edge of the 2-simplex
        simplex = self.simplices_removed[0]
        edge = self.edges_removed[0]
        if not is_subset(edge, simplex):
            return False
        return True


@LabelUpdateFactory.register((1, 1, 2, 2, 2, 2))
class DelaunyFlip(Add2Simplices):
    ## Delauny Flip.
    # edge between two simplices flips resulting in two new simplices
    # Note that this can only happen with simplices.
    # Same as adding all 2-simplices

    # If a delaunay flip appears to have occurred, the removed 1-simplex should be
    # an edge of both removed 2-simplices; similarly the added 1-simplex should be
    # an edge of the added 2-simplices.

    # The set of vertices of the 1-simplices should contain the vertices of each 2-simplex
    # that is added or removed.
    def is_atomic(self):
        old_edge = self.edges_removed[0]
        new_edge = self.edges_added[0]
        if not all([is_subset(old_edge, s) for s in self.simplices_removed]):
            return False
        elif not all([is_subset(new_edge, s) for s in self.simplices_added]):
            return False

        nodes = list(set(old_edge).union(set(new_edge)))
        if not all([is_subset(s, nodes) for s in self.simplices_removed]):
            return False
        elif not all([is_subset(s, nodes) for s in self.simplices_added]):
            return False
        return True

# if __name__ == "__main__":
#     T1 = TopologicalState([])
#     cycle_labelling = CycleLabellingTree(T1)
#
#     T2 = TopologicalState([])
#
#     state_change = StateChange(T1, T2)
#     label_update = LabelUpdateFactory.get_label_update(state_change, cycle_labelling)
#
#     if label_update.is_atomic():
#         cycle_labelling.update(label_update)
