from abc import ABC, abstractmethod
from collections import defaultdict
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
        self._dim = len(self.nodes) - 1

    @property
    def dim(self):
        return self._dim

    def alpha(self):
        return OrientedSimplex(tuple(reversed(self.nodes)))

    def is_subsimplex(self, sub_simplex):
        if not set(sub_simplex.nodes).issubset(set(self.nodes)):
            return False

        i = self.nodes.index(sub_simplex.nodes[0])
        return (self.nodes[i], self.nodes[(i + 1) % (self.dim + 1)]) == sub_simplex.nodes

    @property
    def subsimplices(self):
        return [OrientedSimplex(self.nodes[(i + k) % (self.dim + 1)] for k in range(self.dim)) for i in
                range(self.dim + 1)]

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
    _oriented_simplices: Dict[int, Set[OrientedSimplex]] = field(default_factory=dict)
    incident_simplices: Dict[OrientedSimplex, List[OrientedSimplex]] = field(default_factory=lambda: defaultdict(list))

    def __post_init__(self):
        self._oriented_simplices[self.dim - 1] = get_oriented(self.alpha_complex.simplices(self.dim - 1))
        self._oriented_simplices[self.dim - 2] = get_oriented(self.alpha_complex.simplices(self.dim - 2))

        for simplex in self._oriented_simplices[self.dim - 1]:
            for edge in simplex.subsimplices:
                self.incident_simplices[edge].append(simplex.orient(edge))
        for edge in self._oriented_simplices[self.dim - 2]:
            if edge not in self.incident_simplices:
                self.incident_simplices[edge] = []

        self.build_rotinfo()

    @abstractmethod
    def build_rotinfo(self):
        ...

    @property
    def oriented_simplices(self):
        return self._oriented_simplices[self.dim - 1]

    @property
    def sub_simplices(self):
        return self._oriented_simplices[self.dim - 2]

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
            temp = self.incident_simplices[half_edge]
            self.rotinfo[half_edge] = sorted(temp, key=lambda s: self.theta(temp[0], s))

    def theta(self, simplex1, simplex2):
        # compute angle with respect to two_simplices[0]
        vertices1 = simplex1.vertices(self.points)
        vertices2 = simplex2.vertices(self.points)

        v1 = vertices1[2] - vertices1[0]
        v2 = vertices2[2] - vertices2[0]

        n = vertices1[1] - vertices1[0]
        norm_sqr = n[0] ** 2 + n[1] ** 2 + n[2] ** 2

        pv1 = v1 - (v1[0] * n[0] + v1[1] * n[1] + v1[2] * n[2]) * n / norm_sqr
        pv2 = v2 - (v2[0] * n[0] + v2[1] * n[1] + v2[2] * n[2]) * n / norm_sqr

        cos_theta = pv1[0] * pv2[0] + pv1[1] * pv2[1] + pv1[2] * pv2[2]
        cos_theta /= np.sqrt(pv1[0] ** 2 + pv1[1] ** 2 + pv1[2] ** 2)
        cos_theta /= np.sqrt(pv2[0] ** 2 + pv2[1] ** 2 + pv2[2] ** 2)

        if np.abs(cos_theta) > 1.0:
            # print(f"Warning, truncating dot product from {dot_prod}")
            cos_theta /= np.abs(cos_theta)

        orientation = n[0] * (pv1[1] * pv2[2] - pv1[2] * pv2[1]) \
                      + n[1] * (pv1[2] * pv2[0] - pv1[0] * pv2[2]) \
                      + n[2] * (pv1[0] * pv2[1] - pv1[1] * pv2[0])

        if orientation >= 0:
            return -np.arccos(cos_theta)
        else:
            return np.arccos(cos_theta)


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
            result.update([self.phi(simplex, e) for e in simplex.subsimplices])
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
