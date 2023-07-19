# ******************************************************************************
#  Copyright (c) 2023, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************

from collections import defaultdict
from typing import Iterable

import gudhi


class Simplex(frozenset):
    """
    A Simplex class that extends frozenset. Represents a simplex in the context of
    simplicial complexes in topological data analysis.
    """

    def __new__(cls, vertices: Iterable[int]):
        """
        Create a new Simplex.

        :param vertices: Vertices of the simplex.
        """
        return super().__new__(cls, vertices)

    def is_subsimplex(self, subsimplex: 'Simplex') -> bool:
        """
        Check if the given subface is a subface of this Simplex.

        :param subsimplex: The simplex to check.
        :return: True if subsimplex is a face of this Simplex, False otherwise.
        """
        return subsimplex.issubset(self)

    def to_cycle(self, boundary_cycles: Iterable['BoundaryCycle']) -> 'BoundaryCycle':
        """
        Find the cycle in the given boundary cycles that contains the same vertices as this Simplex.

        :param boundary_cycles: A collection of boundary cycles to check against.
        :return: The boundary cycle containing this simplex.
        :raises ValueError: If this Simplex does not match any of the given boundary cycles or matches more than one.
        """
        cycles_found = [cycle for cycle in boundary_cycles if self == cycle]
        if len(cycles_found) == 0:
            raise ValueError("The simplex is not a face of any of the given boundary cycles.")
        elif len(cycles_found) > 1:
            raise ValueError("The simplex is a face of more than one boundary cycle.")
        else:
            return cycles_found.pop()

    @property
    def dim(self) -> int:
        """
        Get the dimension of this Simplex.

        :return: The dimension of this Simplex.
        """
        return len(self) - 1




class AlphaComplex:
    def __init__(self, points, radius):
        alpha_complex = gudhi.alpha_complex.AlphaComplex(points)
        self.simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=radius ** 2)
        self.dim = len(points[0])

        self._simplices = defaultdict(set)
        self._nodes = set()
        for nodes, _ in self.simplex_tree.get_filtration():
            self._simplices[len(nodes) - 1].add(Simplex(nodes))
            if len(nodes) == 1:
                self._nodes.add(nodes[0])

    @property
    def nodes(self):
        return self._nodes

    def simplices(self, dim):
        return self._simplices[dim]
