from typing import FrozenSet

import gudhi
from dataclasses import dataclass


@dataclass
class Simplex:
    nodes: FrozenSet[int]

    def is_subface(self, subface):
        return subface.nodes.issubset(self.nodes)

    def to_cycle(self, boundary_cycles):
        cycles_found = [cycle for cycle in boundary_cycles if self.nodes == cycle.nodes]
        if len(cycles_found) == 0:
            raise ValueError("Simplex is not associated with any of the given boundary cycles")
        elif len(cycles_found) > 1:
            raise ValueError("Simplex cannot uniquely be represented as boudnary cycle")
        else:
            return cycles_found.pop()

    @property
    def dim(self):
        return len(self.nodes) - 1

    def __eq__(self, other):
        return self.nodes == other.nodes

    def __hash__(self):
        return hash(self.nodes)


class AlphaComplex:
    def __init__(self, points, radius):
        alpha_complex = gudhi.alpha_complex.AlphaComplex(points)
        self.simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=radius ** 2)
        self.dim = len(points[0])

    @property
    def nodes(self):
        return {simplex[0] for simplex, _ in self.simplex_tree.get_skeleton(0)}

    def simplices(self, dim):
        return {Simplex(frozenset(nodes)) for nodes, _ in self.simplex_tree.get_skeleton(dim) if len(nodes) == dim + 1}
