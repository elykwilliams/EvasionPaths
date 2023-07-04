from dataclasses import dataclass
from itertools import chain
from typing import Dict, Iterable, Set

from topology import Topology


@dataclass
class SetDifference:
    new_list: Iterable
    old_list: Iterable

    def added(self) -> Set:
        return {item for item in self.new_list if item not in self.old_list}

    def removed(self) -> Set:
        return {item for item in self.old_list if item not in self.new_list}


trivial = [(0, 0, 0, 0), (0, 0, 0, 0, 0, 0)]

simplex_removed = [
    (0, 1, 0, 0),  (0, 1, 0, 1),
    (0, 0, 0, 1),
    (0, 1, 0, 0, 0, 0), (0, 1, 0, 1, 0, 0), (0, 1, 0, 2, 0, 1),
    (0, 0, 0, 1, 0, 0), (0, 0, 0, 1, 0, 1),
    (0, 0, 0, 0, 0, 1)
]

simplex_added = [
    (1, 0, 0, 0),
    (0, 0, 1, 0), (1, 0, 1, 0),
    (1, 0, 0, 0, 0, 0),
    (0, 0, 1, 0, 0, 0), (1, 0, 1, 0, 0, 0),
    (0, 0, 0, 0, 1, 0), (0, 0, 1, 0, 1, 0), (1, 0, 2, 0, 1, 0)
]

delaunay_change = [
    (1, 1, 2, 2), (0, 1, 1, 3, 2, 3), (1, 0, 3, 1, 3, 2)
]

atomic_change_list = trivial + simplex_removed + simplex_added + delaunay_change


@dataclass
class StateChange:
    new_topology: Topology
    old_topology: Topology


    ## Identify Atomic States
    #
    # (#1-simplices added, #1-simpleices removed, #2-simplices added, #2-simplices removed, #boundary cycles added,
    # #boundary cycles removed)
    @property
    def case(self):
        case = ()
        for dim in range(1, self.new_topology.dim + 1):
            case += (len(self.simplices[dim].added()), len(self.simplices[dim].removed()))
        return case + (len(self.boundary_cycles.added()), len(self.boundary_cycles.removed()))

    def __repr__(self) -> str:
        result = f"State Change: {self.case}\n"
        for dim in range(1, self.new_topology.dim + 1):
            result += f"{dim}-Simplices Added: {self.simplices[dim].added()}\n" \
                      f"{dim}-Simplices Removed: {self.simplices[dim].removed()}\n"
        result += f"Boundary Cycles Added: {self.boundary_cycles.added()}\n" \
                  f"Boundary Cycles Removed: {self.boundary_cycles.removed()}\n"
        return result

    def is_valid(self):
        new_simplex_cycles = [simplex.to_cycle(self.boundary_cycles.new_list)
                              for simplex in self.simplices[self.new_topology.dim].added()]
        old_simplex_cycles = [simplex.to_cycle(self.boundary_cycles.old_list)
                              for simplex in self.simplices[self.old_topology.dim].removed()]
        check1 = all(cycle in self.boundary_cycles.old_list for cycle in old_simplex_cycles)
        check2 = all(cycle in self.boundary_cycles.new_list for cycle in new_simplex_cycles)
        return check1 and check2

    def alpha_complex_change(self):
        case = ()
        for dim in range(1, self.new_topology.dim + 1):
            case += (len(self.simplices[dim].added()), len(self.simplices[dim].removed()))
        return case

    def boundary_cycle_change(self):
        new_holes = self.new_topology.homology_generators
        old_holes = self.old_topology.homology_generators
        return len(new_holes.difference(old_holes)), len(old_holes.difference(new_holes))

    @property
    def boundary_cycles(self) -> SetDifference:
        return SetDifference(self.new_topology.boundary_cycles, self.old_topology.boundary_cycles)

    @property
    def simplices(self) -> Dict[int, SetDifference]:
        dim = self.new_topology.dim
        return {i: SetDifference(self.new_topology.simplices(i), self.old_topology.simplices(i))
                for i in range(1, dim + 1)}

    def is_local_change_added(self):
        dim = len(self.simplices)
        for i in range(1, dim):
            for j in range(i, dim+1):
                for edge in self.simplices[i].added():
                    for face in self.simplices[j].added():
                        if not face.is_subface(edge):
                            return False
        return True

    def is_local_change_removed(self):
        dim = len(self.simplices)
        for i in range(1, dim):
            for j in range(i, dim+1):
                for edge in self.simplices[i].removed():
                    for face in self.simplices[j].removed():
                        if not face.is_subface(edge):
                            return False
        return True

    def is_atomic_change(self):
        case = self.alpha_complex_change()
        dim = self.new_topology.dim

        if case not in atomic_change_list:
            return False

        elif not self.is_local_change_added():
            return False

        elif not self.is_local_change_removed():
            return False

        if case in delaunay_change:
            old_nodes = chain(*[simplex.nodes for simplex in self.simplices[dim].removed()])
            new_nodes = chain(*[simplex.nodes for simplex in self.simplices[dim].added()])
            if set(old_nodes) != set(new_nodes):
                return False

        return True
