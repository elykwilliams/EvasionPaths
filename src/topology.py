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
        if self.dim <= 3:
            face = OrientedSimplex(tuple(range(self.dim)))
        else:
            raise NotImplementedError(f"No Implementation for CombinatorialMap for dimension {self.dim}")

        return self.cmap.get_cycle(face)

    @property
    def dim(self) -> int:
        return self.alpha_complex.dim

    def is_boundary(self, cycle):
        # Return True if the cycle is the boundary of a dim-simplex
        return Simplex(cycle.nodes) in self.simplices(self.dim)

    @property
    def face_connectivity_graph(self):
        graph = nx.Graph()
        graph.add_nodes_from(self.simplices(self.dim - 1))
        for face_list in self.cmap.rotation_info.incident_simplices.values():  # Maybe move graph to cmap?
            graph.add_edges_from(combinations(face_list, 2))
        return graph

    def is_connected_cycle(self, cycle):
        return nx.has_path(self.face_connectivity_graph, next(iter(cycle)), tuple(range(self.dim)))


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
