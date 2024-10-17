
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
    case = (1, 0, 2, 0, 1, 0)
    one_added = {Simplex({732, 742})}
    one_removed = set()
    two_added = {Simplex({732, 742, 735}), Simplex({732, 733, 742})}
    two_removed = set()
    three_added = {Simplex({732, 733, 742, 735})}
    three_removed = set()

    for simplex1 in one_added:
        for simplex2 in two_added:
            print("1: ", simplex2.is_subsimplex(simplex1))

    for simplex1 in one_added:
        for simplex2 in three_added:
            print("2: ", simplex2.is_subsimplex(simplex1))
            print(simplex2, simplex1)

    for simplex1 in two_added:
        for simplex2 in three_added:
            print("3: ", simplex2.is_subsimplex(simplex1))
