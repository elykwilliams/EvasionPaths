# ******************************************************************************
#  Copyright (c) 2023, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************

from dataclasses import dataclass
from itertools import combinations

import networkx as nx

from alpha_complex import AlphaComplex, Simplex
from combinatorial_map import BoundaryCycle, RotationInfo2D, CombinatorialMap2D, CombinatorialMap, CombinatorialMap3D, \
    OrientedSimplex, RotationInfo3D


@dataclass
class Topology:
    alpha_complex: AlphaComplex
    cmap: CombinatorialMap
    _graph: nx.Graph = None

    def simplices(self, dim):
        return self.alpha_complex.simplices(dim)

    @property
    def boundary_cycles(self):
        return self.cmap.boundary_cycles

    @property
    def homology_generators(self):
        return {cycle for cycle in self.cmap.boundary_cycles if not self.is_boundary(cycle)}

    @property
    def alpha_cycle(self) -> BoundaryCycle:
        face = OrientedSimplex(tuple(range(self.dim)))
        try:
            cycle = self.cmap.get_cycle(face)
        except KeyError:
            raise KeyError("The simplex (0, 1, ... dim-1), is not a face of the external boundary.")
        return cycle

    @property
    def dim(self) -> int:
        return self.alpha_complex.dim

    def is_boundary(self, cycle):
        # Return True if the cycle is the boundary of a dim-simplex
        return Simplex(cycle.nodes) in self.simplices(self.dim)

    @property
    def face_connectivity_graph(self):
        if self._graph:
            return self._graph
        self._graph = nx.Graph()
        self._graph.add_nodes_from(self.simplices(self.dim - 1))
        for face_list in self.cmap.rotation_info.incident_simplices.values():  # Maybe move graph to cmap?
            self._graph.add_edges_from(map(lambda e: (Simplex(e[0]), Simplex(e[1])), combinations(face_list, 2)))
        return self._graph

    def is_connected_cycle(self, cycle):
        return nx.has_path(self.face_connectivity_graph, Simplex(next(iter(cycle)).nodes), Simplex(range(self.dim)))


def generate_topology(points, radius):
    ac = AlphaComplex(points, radius)
    if ac.dim == 2:
        rot_info = RotationInfo2D(points, ac)
        cmap = CombinatorialMap2D(rot_info)
    elif ac.dim == 3:
        rot_info = RotationInfo3D(points, ac)
        cmap = CombinatorialMap3D(rot_info)
    else:
        raise NotImplementedError(f"No Implementation for CombinatorialMap for dimension {ac.dim}")
    return Topology(ac, cmap)
