
from dataclasses import dataclass
from itertools import chain
from typing import Dict, Iterable, Set

from topology import Topology
from alpha_complex import Simplex

@dataclass
class SetDifference:
    new_list: Iterable
    old_list: Iterable

    def added(self) -> Set:
        return set(self.new_list).difference(self.old_list)

    def removed(self) -> Set:
        return set(self.old_list).difference(self.new_list)

def simplices_difference(dim, old_topology, new_topology) -> Dict[int, SetDifference]:
    return {i: SetDifference(new_topology.simplices(i), old_topology.simplices(i))
            for i in range(1, dim + 1)}

def is_local_change_added(self):
        for i in range(1, self.dim):
            for j in range(i, self.dim+1):
                for edge in self.simplices_difference[i].added():
                    for face in self.simplices_difference[j].added():
                        if not face.is_subsimplex(edge):
                            print(face, edge)
                            return False
        return True

if __name__ == "__main__":
    case = (0, 0, 0, 0, 0, 0)
    one_added = set()
    one_removed = set()
    two_added = set()
    two_removed = set()
    three_added = set()
    three_removed = set()

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
    print(case in atomic_change_list)
    print(case in trivial)
