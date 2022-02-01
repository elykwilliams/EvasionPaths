from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import chain
from typing import Type, Dict

from dataclasses import dataclass

from utilities import SetDifference


#
#
# class AlphaComplex(ABC):
#     @abstractmethod
#     def simplices(self, dim) -> Sequence[Tuple[int]]:
#         ...
#
#
# class CombinatorialMap(ABC):
#     @property
#     @abstractmethod
#     def boundary_cycles(self):
#         ...
#
#     @abstractmethod
#     def nodes2cycles(self):
#         ...
#
#
# class AbstractTopology(ABC, AlphaComplex, CombinatorialMap):
#     pass
#
class Simplex:

    @property
    def nodes(self):
        return None

    def is_subface(self, edge):
        pass

    def to_cycle(self, cmap):
        pass


@dataclass
class StateChange:
    edges: SetDifference
    simplices: SetDifference
    boundary_cycles: SetDifference

    ## Identify Atomic States
    #
    # (#1-simplices added, #1-simpleices removed, #2-simplices added, #2-simplices removed, #boundary cycles added,
    # #boundary cycles removed)
    @property
    def case(self):
        return (len(self.edges.added()), len(self.edges.removed()),
                len(self.simplices.added()), len(self.simplices.removed()),
                len(self.boundary_cycles.added()), len(self.boundary_cycles.removed()))

    def __repr__(self) -> str:
        return (
            f"State Change: {self.case}\n"
            f"New edges: {self.edges.added()}\n"
            f"Removed edges: {self.edges.removed()}\n"
            f"New Simplices: {self.simplices.added()}\n"
            f"Removed Simplices: {self.simplices.removed()}\n"
            f"New cycles {self.boundary_cycles.added()}\n"
            f"Removed Cycles {self.boundary_cycles.removed()}"
        )

    def is_valid(self):
        new_simplex_cycles = [simplex.to_cycle(self.boundary_cycles.new_list) for simplex in self.simplices.added()]
        old_simplex_cycles = [simplex.to_cycle(self.boundary_cycles.old_list) for simplex in self.simplices.removed()]
        check1 = all(cycle in self.boundary_cycles.old_list for cycle in old_simplex_cycles)
        check2 = all(cycle in self.boundary_cycles.new_list for cycle in new_simplex_cycles)
        return check1 and check2


class LabelUpdate(ABC):

    @property
    @abstractmethod
    def nodes_added(self):
        ...

    @property
    @abstractmethod
    def nodes_removed(self):
        ...

    @property
    def mapping(self):
        return dict()

    @abstractmethod
    def is_valid(self):
        ...


class LabelUpdate2D(LabelUpdate):
    def __init__(self, state_change: StateChange, labels: Dict):
        self.state_change = state_change
        self.labels = labels

    ## Check if update is self-consistant
    # move to labellingTree ???
    def is_valid(self):
        return self.state_change.is_valid() and all(cycle in self.labels for cycle in self.nodes_removed)

    def is_atomic(self):
        return True

    @property
    def nodes_added(self):
        return self.state_change.boundary_cycles.added()

    @property
    def nodes_removed(self):
        return self.state_change.boundary_cycles.removed()

    ##
    @property
    def _simplex_cycles_added(self):
        new_cycles = self.state_change.boundary_cycles.new_list
        return [simplex.to_cycle(new_cycles) for simplex in self.state_change.simplices.added()]

    @property
    def _simplex_cycles_removed(self):
        old_cycles = self.state_change.boundary_cycles.old_list
        return [simplex.to_cycle(old_cycles) for simplex in self.state_change.simplices.removed()]


class LabelUpdateFactory:
    atomic_updates: Dict[tuple, Type[LabelUpdate]] = defaultdict(lambda: NonAtomicUpdate)

    @classmethod
    def get_label_update(cls, state_change: StateChange):
        return cls.atomic_updates[state_change.case]

    @classmethod
    def register(cls, case: tuple):
        def deco(deco_cls):
            cls.atomic_updates[case] = deco_cls
            return deco_cls

        return deco


class NonAtomicUpdate(LabelUpdate2D):
    def is_atomic(self):
        return False


@LabelUpdateFactory.register((0, 0, 0, 0, 0, 0))
class TrivialUpdate(LabelUpdate2D):
    pass


@LabelUpdateFactory.register((0, 0, 1, 0, 0, 0))
class Add2SimplicesUpdate2D(LabelUpdate2D):
    ## All 2-Simplices are Labeled False
    @property
    def mapping(self):
        return {cycle: False for cycle in self._simplex_cycles_added if cycle in self.labels}


@LabelUpdateFactory.register((0, 0, 0, 1, 0, 0))
class Remove2SimplicesUpdate2D(LabelUpdate2D):
    @property
    def mapping(self):
        return {cycle: self.labels[cycle] for cycle in self._simplex_cycles_removed if cycle in self.labels}


@LabelUpdateFactory.register((1, 0, 0, 0, 2, 1))
class Add1SimplexUpdate2D(LabelUpdate2D):
    ## Add edge.
    # New cycles will match the removed cycle
    @property
    def mapping(self):
        label = self.labels[next(iter(self.nodes_removed))]
        return {cycle: label for cycle in self.nodes_added}


@LabelUpdateFactory.register((0, 1, 0, 0, 1, 2))
class Remove1SimplexUpdate2D(LabelUpdate2D):
    ## Remove edge.
    # added cycle will have intruder if either removed cycle has intruder
    @property
    def mapping(self):
        label = any([self.labels[cycle] for cycle in self.nodes_removed])
        return {cycle: label for cycle in self.nodes_added}


@LabelUpdateFactory.register((1, 0, 1, 0, 2, 1))
class AddSimplexPairUpdate2D(Add1SimplexUpdate2D):

    ## Add simplex pair
    # Will split a cycle in two with label as in Add1Simplex, then label 2simplex as False
    @property
    def mapping(self):
        label = self.labels[next(iter(self.nodes_removed))]
        result = {cycle: label for cycle in self.nodes_added}

        result.update({cycle: False for cycle in self._simplex_cycles_added})
        return result

    # If a 1-simplex and 2-simplex are added/removed simultaniously, then the 1-simplex
    # should be an edge of the 2-simplex
    def is_atomic(self):
        simplex = next(iter(self.state_change.simplices.added()))
        edge = next(iter(self.state_change.edges.added()))
        return simplex.is_subface(edge)


@LabelUpdateFactory.register((0, 1, 0, 1, 1, 2))
class RemoveSimplexPairUpdate2D(Remove1SimplexUpdate2D):

    ## Same as Remove1Simplex
    def is_atomic(self):
        # If a 1-simplex and 2-simplex are added/removed simultaniously, then the 1-simplex
        # should be an edge of the 2-simplex
        simplex = next(iter(self.state_change.simplices.removed()))
        edge = next(iter(self.state_change.edges.removed()))
        return simplex.is_subface(edge)


@LabelUpdateFactory.register((1, 1, 2, 2, 2, 2))
class DelaunyFlipUpdate2D(LabelUpdate2D):
    ## Delauny Flip.
    # edge between two simplices flips resulting in two new simplices
    # Note that this can only happen with simplices.
    # Same as adding all 2-simplices

    ## All 2-Simplices are Labeled False
    @property
    def mapping(self):
        return {cycle: False for cycle in self._simplex_cycles_added}

    # If a delaunay flip appears to have occurred, the removed 1-simplex should be
    # an edge of both removed 2-simplices; similarly the added 1-simplex should be
    # an edge of the added 2-simplices.

    # The set of vertices of the 1-simplices should contain the vertices of each 2-simplex
    # that is added or removed.
    def is_atomic(self):
        old_edge = next(iter(self.state_change.edges.removed()))
        new_edge = next(iter(self.state_change.edges.added().pop()))

        if not all([simplex.is_subface(old_edge) for simplex in self.state_change.simplices.removed()]):
            return False
        elif not all([simplex.is_subface(new_edge) for simplex in self.state_change.simplices.added()]):
            return False

        old_nodes = chain(*[simplex.nodes for simplex in self.state_change.simplices.removed()])
        new_nodes = chain(*[simplex.nodes for simplex in self.state_change.simplices.added()])
        if set(old_nodes) != set(new_nodes):
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
