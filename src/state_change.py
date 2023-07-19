# ******************************************************************************
#  Copyright (c) 2023, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************

from dataclasses import dataclass
from itertools import chain
from typing import Dict, Iterable, Set

from topology import Topology


@dataclass
class SetDifference:
    new_list: Iterable
    old_list: Iterable

    def added(self) -> Set:
        return set(self.new_list).difference(self.old_list)

    def removed(self) -> Set:
        return set(self.old_list).difference(self.new_list)


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

    def __repr__(self) -> str:
        result = f"State Change:\n"
        for dim in range(1, self.new_topology.dim + 1):
            result += f"{dim}-Simplices Added: {self.simplices_difference[dim].added()}\n" \
                      f"{dim}-Simplices Removed: {self.simplices_difference[dim].removed()}\n"
        result += f"Boundary Cycles Added: {self.boundary_cycles_difference.added()}\n" \
                  f"Boundary Cycles Removed: {self.boundary_cycles_difference.removed()}\n"
        return result

    def alpha_complex_change(self):
        case = ()
        for dim in range(1, self.new_topology.dim + 1):
            case += (len(self.simplices_difference[dim].added()), len(self.simplices_difference[dim].removed()))
        return case

    def boundary_cycle_change(self):
        return len(self.boundary_cycles_difference.added()), len(self.boundary_cycles_difference.removed())

    @property
    def boundary_cycles_difference(self) -> SetDifference:
        return SetDifference(self.new_topology.homology_generators, self.old_topology.homology_generators)

    @property
    def simplices_difference(self) -> Dict[int, SetDifference]:
        dim = self.new_topology.dim
        return {i: SetDifference(self.new_topology.simplices(i), self.old_topology.simplices(i))
                for i in range(1, dim + 1)}

    def is_local_change_added(self):
        dim = len(self.simplices_difference)
        for i in range(1, dim):
            for j in range(i, dim+1):
                for edge in self.simplices_difference[i].added():
                    for face in self.simplices_difference[j].added():
                        if not face.is_subsimplex(edge):
                            return False
        return True

    def is_local_change_removed(self):
        dim = len(self.simplices_difference)
        for i in range(1, dim):
            for j in range(i, dim+1):
                for edge in self.simplices_difference[i].removed():
                    for face in self.simplices_difference[j].removed():
                        if not face.is_subsimplex(edge):
                            return False
        return True

    def is_atomic_change(self):
        case = self.alpha_complex_change()
        dim = self.new_topology.dim

        if case not in atomic_change_list:
            return False

        elif case in simplex_added:
            return self.is_local_change_added()

        elif case in simplex_removed:
            return self.is_local_change_removed()

        elif case in delaunay_change:
            old_nodes = chain.from_iterable(self.simplices_difference[dim].removed())
            new_nodes = chain.from_iterable(self.simplices_difference[dim].added())
            return set(old_nodes) == set(new_nodes)

        return True
