from abc import ABC, abstractmethod
from itertools import chain
from math import atan2
from typing import Set, List, FrozenSet, Dict, Sequence, Collection, Iterator

import networkx as nx
import numpy as np
from dataclasses import dataclass, field

from alpha_complex import AlphaComplex


class OrientedSimplex:
    def __init__(self, nodes):
        self.nodes = tuple(nodes)
        i = 0 if len(self.nodes) == 2 else self.nodes.index(min(self.nodes))
        self._hash = hash(self.nodes[i:] + self.nodes[:i])

    @property
    def dim(self):
        return len(self.nodes) - 1

    def alpha(self):
        return OrientedSimplex(tuple(reversed(self.nodes)))

    def is_subsimplex(self, sub_simplex):
        if not set(sub_simplex.nodes).issubset(set(self.nodes)):
            return False

        i = self.nodes.index(sub_simplex.nodes[0])
        return (self.nodes[i], self.nodes[(i + 1) % len(self.nodes)]) == sub_simplex.nodes

    @property
    def edges(self):
        result = []
        for i in range(len(self.nodes)):
            half_edge = (self.nodes[i], self.nodes[(i + 1) % len(self.nodes)])
            result.append(OrientedSimplex(half_edge))
        return result

    def vertices(self, points):
        return [points[n] for n in self.nodes]

    def orient(self, half_edge):
        i = self.nodes.index(half_edge.nodes[0])
        return OrientedSimplex(self.nodes[i:] + self.nodes[:i])

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return repr(self.nodes)


def get_oriented(simplices):
    oriented_simplices = set()
    for simplex in simplices:
        os = OrientedSimplex(simplex.nodes)
        oriented_simplices.update({os, os.alpha()})
    return oriented_simplices


@dataclass
class BoundaryCycle:
    oriented_simplices: FrozenSet[OrientedSimplex]

    def __iter__(self) -> Iterator[OrientedSimplex]:
        return iter(self.oriented_simplices)

    def __hash__(self):
        return hash(self.oriented_simplices)

    def __len__(self):
        return len(self.nodes)

    def __repr__(self):
        return f"BoundaryCycle[{','.join([repr(s) for s in self.oriented_simplices])}]"

    @property
    def nodes(self) -> Set[int]:
        return set(chain.from_iterable([face.nodes for face in self]))


@dataclass
class RotationInfo(ABC):
    points: Sequence
    alpha_complex: AlphaComplex
    rotinfo: Dict[OrientedSimplex, List[OrientedSimplex]] = field(default_factory=dict)

    def __post_init__(self):
        self.build_rotinfo()

    @abstractmethod
    def build_rotinfo(self):
        ...

    @property
    def oriented_simplices(self):
        return get_oriented(self.alpha_complex.simplices(self.dim - 1))

    @property
    def sub_simplices(self):
        return get_oriented(self.alpha_complex.simplices(self.dim - 2))

    @property
    def dim(self):
        return len(self.points[0])

    def next(self, oriented_simplex: OrientedSimplex, sub_simplex: OrientedSimplex) -> OrientedSimplex:
        index = self.rotinfo[sub_simplex].index(oriented_simplex)
        next_index = (index + 1) % len(self.rotinfo[sub_simplex])
        return self.rotinfo[sub_simplex][next_index]


class RotationInfo2D(RotationInfo):

    def build_rotinfo(self):
        graph = nx.Graph()
        graph.add_nodes_from(self.alpha_complex.nodes)
        graph.add_edges_from([edge.nodes for edge in self.alpha_complex.simplices(1)])

        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            anticlockwise, clockwise = False, True

            # sort w.r.t angle from x axis
            def theta(a, center):
                oa = (a[0] - center[0], a[1] - center[1])
                return atan2(oa[1], oa[0])

            sorted_neighbors = sorted(neighbors,
                                      key=lambda nei: theta(self.points[nei], self.points[node]),
                                      reverse=clockwise)
            self.rotinfo[OrientedSimplex((node,))] = [OrientedSimplex((node, n)) for n in sorted_neighbors]


class RotationInfo3D(RotationInfo):

    def build_rotinfo(self):
        for half_edge in self.sub_simplices:
            temp = self.incident_simplices(half_edge)
            self.rotinfo[half_edge] = sorted(temp, key=lambda simplex: self.theta(temp[0], simplex))

    def incident_simplices(self, half_edge: OrientedSimplex):
        return [simplex.orient(half_edge) for simplex in self.oriented_simplices if simplex.is_subsimplex(half_edge)]

    def theta(self, simplex1, simplex2):
        # compute angle with respect to two_simplices[0]
        vertices1 = simplex1.vertices(self.points)
        vertices2 = simplex2.vertices(self.points)
        vector1 = vertices1[2] - vertices1[0]
        vector2 = vertices2[2] - vertices2[0]
        normal = vertices1[1] - vertices1[0]

        projected_vector1 = vector1 - (np.dot(vector1, normal)) / (np.linalg.norm(normal) ** 2) * normal
        projected_vector2 = vector2 - (np.dot(vector2, normal)) / (np.linalg.norm(normal) ** 2) * normal

        dot_prod = np.dot(projected_vector1, projected_vector2) / (
                np.linalg.norm(projected_vector1) * np.linalg.norm(projected_vector2))
        if np.abs(dot_prod) - 1 > 1e-8:
            print(f"Warning, truncating dot product from {dot_prod}")
        dot_prod = np.clip(dot_prod, -1.0, 1.0)

        if np.dot(normal, (np.cross(projected_vector1, projected_vector2))) >= 0:
            return -np.arccos(dot_prod)
        else:
            return np.arccos(dot_prod)


@dataclass
class CombinatorialMap(ABC):
    rotation_info: RotationInfo
    _boundary_cycles: Collection[BoundaryCycle] = frozenset()
    _hashed_simplices: Dict[OrientedSimplex, BoundaryCycle] \
        = field(default_factory=dict)

    @abstractmethod
    def get_cycle(self, dart):
        pass

    @staticmethod
    def alpha(simplex: OrientedSimplex) -> OrientedSimplex:
        return OrientedSimplex(reversed(simplex.nodes))

    def sigma(self, simplex: OrientedSimplex,
              sub_simplex: OrientedSimplex) -> OrientedSimplex:
        return self.rotation_info.next(simplex, sub_simplex)

    def phi(self, simplex: OrientedSimplex,
            sub_simplex: OrientedSimplex) -> OrientedSimplex:
        return self.alpha(self.sigma(simplex, sub_simplex))

    @property
    def boundary_cycles(self) -> Collection[BoundaryCycle]:
        if self._boundary_cycles:
            return self._boundary_cycles

        simplices = self.rotation_info.oriented_simplices.copy()
        cycles = partition(self.get_cycle, simplices)
        self._boundary_cycles = cycles
        return self._boundary_cycles


class CombinatorialMap2D(CombinatorialMap):

    def get_cycle(self, simplex: OrientedSimplex) -> BoundaryCycle:
        if simplex in self._hashed_simplices:
            return self._hashed_simplices[simplex]

        cycle = {simplex}
        next_simplex = self.phi(simplex, OrientedSimplex((simplex.nodes[0],)))
        while next_simplex != simplex:
            cycle.add(next_simplex)
            pivot = OrientedSimplex((next_simplex.nodes[0],))
            next_simplex = self.phi(next_simplex, pivot)

        for s in cycle:
            self._hashed_simplices[s] = BoundaryCycle(frozenset(cycle))
        return self._hashed_simplices[simplex]

    def get_cycle_nodes(self, simplex):
        nodes = [simplex.nodes[0]]
        next_simplex = self.phi(simplex, OrientedSimplex((simplex.nodes[0],)))
        while next_simplex != simplex:
            pivot = OrientedSimplex((next_simplex.nodes[0],))
            nodes.append(next_simplex.nodes[0])
            next_simplex = self.phi(next_simplex, pivot)
        return nodes


@dataclass
class CombinatorialMap3D(CombinatorialMap):

    def flop(self, simplices):
        result = set(simplices)
        for simplex in simplices:
            result.update([self.phi(simplex, e) for e in simplex.edges])
        return result

    def get_cycle(self, simplex: OrientedSimplex):
        if simplex in self._hashed_simplices:
            return self._hashed_simplices[simplex]

        cycle = frozenset(fixed_point(self.flop, {simplex}))

        self._hashed_simplices.update({s: BoundaryCycle(cycle) for s in cycle})
        return self._hashed_simplices[simplex]


def partition(equiv_class, X: Set):
    part = set()
    while X:
        x = X.pop()
        equiv_set = equiv_class(x)
        X.difference_update(equiv_set)
        part.add(equiv_set)
    return part


def fixed_point(f, x0):
    x, next_x = x0, f(x0)
    while x != next_x:
        x, next_x = next_x, f(next_x)
    return x
