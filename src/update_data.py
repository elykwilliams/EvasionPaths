from abc import ABC, abstractmethod
from itertools import chain
from typing import Type, Dict, Iterable

from state_change import StateChange
from topology import Topology
from utilities import UpdateError


class LabelUpdate(ABC):

    @property
    @abstractmethod
    def cycles_added(self) -> Iterable:
        ...

    @property
    @abstractmethod
    def cycles_removed(self) -> Iterable:
        ...

    @property
    def mapping(self) -> Dict:
        return dict()

    @abstractmethod
    def is_valid(self) -> bool:
        ...


class LabelUpdate2D(LabelUpdate):
    def __init__(self, edges, simplices, boundary_cycles, labels: Dict):
        self.edges = edges
        self.simplices = simplices
        self.boundary_cycles = boundary_cycles
        self.labels = labels

    ## Check if update is self-consistant
    # move to labellingTree ???
    def is_valid(self):
        check1 = all(cycle in self.labels for cycle in self.cycles_removed)
        check2 = all(cycle in self.labels for cycle in self._simplex_cycles_removed)
        return check1 and check2

    def is_atomic(self):
        return True

    @property
    def cycles_added(self):
        return self.boundary_cycles.added()

    @property
    def cycles_removed(self):
        return self.boundary_cycles.removed()

    ##
    @property
    def _simplex_cycles_added(self):
        new_cycles = self.boundary_cycles.new_list
        return [simplex.to_cycle(new_cycles) for simplex in self.simplices.added()]

    @property
    def _simplex_cycles_removed(self):
        old_cycles = self.boundary_cycles.old_list
        return [simplex.to_cycle(old_cycles) for simplex in self.simplices.removed()]


class LabelUpdateFactory:
    atomic_updates2d: Dict[tuple, Type[LabelUpdate2D]] = dict()

    @classmethod
    def get_update(cls, new_topology: Topology, old_topology: Topology, labelling):
        state_change = StateChange(new_topology, old_topology)
        if not state_change.is_valid():
            raise UpdateError("The state_change provided is not self consistent")
        update_type = cls.atomic_updates2d.get(state_change.case, NonAtomicUpdate)
        return update_type(state_change.simplices(1), state_change.simplices(2),
                           state_change.boundary_cycles, labelling)

    @classmethod
    def register(cls, case: tuple):
        def deco(deco_cls):
            cls.atomic_updates2d[case] = deco_cls
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
        label = self.labels[next(iter(self.cycles_removed))]
        return {cycle: label for cycle in self.cycles_added}


@LabelUpdateFactory.register((0, 1, 0, 0, 1, 2))
class Remove1SimplexUpdate2D(LabelUpdate2D):
    ## Remove edge.
    # added cycle will have intruder if either removed cycle has intruder
    @property
    def mapping(self):
        label = any([self.labels[cycle] for cycle in self.cycles_removed])
        return {cycle: label for cycle in self.cycles_added}


@LabelUpdateFactory.register((1, 0, 1, 0, 2, 1))
class AddSimplexPairUpdate2D(Add1SimplexUpdate2D):

    ## Add simplex pair
    # Will split a cycle in two with label as in Add1Simplex, then label 2simplex as False
    @property
    def mapping(self):
        label = self.labels[next(iter(self.cycles_removed))]
        result = {cycle: label for cycle in self.cycles_added}

        result.update({cycle: False for cycle in self._simplex_cycles_added})
        return result

    # If a 1-simplex and 2-simplex are added/removed simultaniously, then the 1-simplex
    # should be an edge of the 2-simplex
    def is_atomic(self):
        simplex = next(iter(self.simplices.added()))
        edge = next(iter(self.edges.added()))
        return simplex.is_subface(edge)


@LabelUpdateFactory.register((0, 1, 0, 1, 1, 2))
class RemoveSimplexPairUpdate2D(Remove1SimplexUpdate2D):

    ## Same as Remove1Simplex
    def is_atomic(self):
        # If a 1-simplex and 2-simplex are added/removed simultaniously, then the 1-simplex
        # should be an edge of the 2-simplex
        simplex = next(iter(self.simplices.removed()))
        edge = next(iter(self.edges.removed()))
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
        old_edge = next(iter(self.edges.removed()))
        new_edge = next(iter(self.edges.added()))

        if not all([simplex.is_subface(old_edge) for simplex in self.simplices.removed()]):
            return False
        elif not all([simplex.is_subface(new_edge) for simplex in self.simplices.added()]):
            return False

        old_nodes = chain(*[simplex.nodes for simplex in self.simplices.removed()])
        new_nodes = chain(*[simplex.nodes for simplex in self.simplices.added()])
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
#     label_update = LabelUpdateFactory.get_label_update(state_change)(state_change, cycle_labelling)
#
#     if label_update.is_atomic():
#         cycle_labelling.update(label_update)
