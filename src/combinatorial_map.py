from abc import ABC, abstractmethod
from collections import defaultdict
from functools import lru_cache
from itertools import chain
from math import atan2
from typing import Set, List, FrozenSet, Dict, Sequence, Collection, Tuple

import networkx as nx
import numpy as np
from dataclasses import dataclass, field

from alpha_complex import AlphaComplex

_memoized = {}  # MEMORY LEAK should be refreshed regularly


def memoize(f):
    def memoized(nodes: tuple):
        try:
            return _memoized[nodes]
        except KeyError:
            new_f = f(nodes)
            if len(nodes) > 2:
                for k in range(len(nodes)):
                    _memoized[nodes[k:] + nodes[:k]] = new_f
            else:
                _memoized[nodes] = new_f
            return _memoized[nodes]

    return memoized


@memoize
class OrientedSimplex:
    def __init__(self, nodes: tuple):
        self.nodes = nodes
        self.dim = len(nodes) - 1

    def is_subsimplex(self, sub_simplex: "OrientedSimplex") -> bool:
        return sub_simplex in self.subsimplices

    @property
    @lru_cache(maxsize=None)  # MEMORY LEAK should simple cache instead of lru_cache
    def subsimplices(self) -> Collection["OrientedSimplex"]:
        return [OrientedSimplex(tuple(self.nodes[(i + k) % (self.dim + 1)] for k in range(self.dim)))
                for i in range(self.dim + 1)]

    def oriented_nodes(self, half_edge: "OrientedSimplex") -> tuple:
        i = self.nodes.index(half_edge.nodes[0])
        return self.nodes[i:] + self.nodes[:i]

    def __repr__(self):
        return repr(self.nodes)


def get_oriented(simplices):
    oriented_simplices = set()
    for simplex in simplices:
        nodes = tuple(simplex.nodes)
        os = OrientedSimplex(nodes)
        os_alpha = OrientedSimplex(nodes[::-1])
        oriented_simplices.update({os, os_alpha})
    return oriented_simplices


class BoundaryCycle(frozenset):

    def __repr__(self):
        return f"BoundaryCycle[{super()}]"

    @property
    def nodes(self) -> FrozenSet[int]:
        return frozenset(chain.from_iterable([face.nodes for face in self]))


@dataclass
class RotationInfo(ABC):
    points: Sequence
    alpha_complex: AlphaComplex
    rotinfo: Dict[OrientedSimplex, List[OrientedSimplex]] = field(default_factory=dict)
    incident_simplices: Dict[OrientedSimplex, List[tuple]] = field(default_factory=lambda: defaultdict(list))
    _oriented_simplices: Set[OrientedSimplex] = field(default_factory=dict)

    def __post_init__(self):
        self._oriented_simplices = get_oriented(self.alpha_complex.simplices(self.dim - 1))

        for simplex in self._oriented_simplices:
            for edge in simplex.subsimplices:
                self.incident_simplices[edge].append(simplex.oriented_nodes(edge))

        self.build_rotinfo()

    @abstractmethod
    def build_rotinfo(self):
        ...

    @property
    def oriented_simplices(self):
        return self._oriented_simplices

    @property
    def dim(self):
        return len(self.points[0])

    def next(self, oriented_simplex: OrientedSimplex, sub_simplex: OrientedSimplex) -> OrientedSimplex:
        ordered_simplices = self.rotinfo[sub_simplex]
        index = ordered_simplices.index(oriented_simplex)
        next_index = (index + 1) % len(ordered_simplices)
        return ordered_simplices[next_index]


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
        for half_edge, temp in self.incident_simplices.items():
            self.rotinfo[half_edge] = [OrientedSimplex(nodes) for nodes in
                                       sorted(temp, key=lambda s: self.theta(temp[0], s))]

    def theta(self, simplex1_nodes, simplex2_nodes):
        # compute angle with respect to two_simplices[0]
        vertices1 = [self.points[n] for n in simplex1_nodes]
        vertices2 = [self.points[n] for n in simplex2_nodes]

        v1 = vertices1[2] - vertices1[0]
        v2 = vertices2[2] - vertices2[0]

        # shared half edge defines the normal vector
        n = vertices1[1] - vertices1[0]
        norm_sqr = n[0] ** 2 + n[1] ** 2 + n[2] ** 2

        # compute orthogonal projection onto unit normal: pv = v - (v . un) un
        pv1 = v1 - (v1[0] * n[0] + v1[1] * n[1] + v1[2] * n[2]) * n / norm_sqr
        pv2 = v2 - (v2[0] * n[0] + v2[1] * n[1] + v2[2] * n[2]) * n / norm_sqr

        # cos(theta) = (pv1 . pv2)/(|pv1| |pv2|)
        cos_theta = pv1[0] * pv2[0] + pv1[1] * pv2[1] + pv1[2] * pv2[2]
        cos_theta /= np.sqrt(pv1[0] ** 2 + pv1[1] ** 2 + pv1[2] ** 2)
        cos_theta /= np.sqrt(pv2[0] ** 2 + pv2[1] ** 2 + pv2[2] ** 2)

        # Make sure cos_theta is in domain of arccos
        if np.abs(cos_theta) > 1.0:
            # print(f"Warning, truncating dot product from {dot_prod}")
            cos_theta /= np.abs(cos_theta)

        # orientation is determined wrt normal by n . (pv1 x pv2)
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
    _simplices_map: Dict[OrientedSimplex, BoundaryCycle] = field(default_factory=dict)
    _phi_cache: Dict[Tuple, OrientedSimplex] = field(default_factory=dict)

    @abstractmethod
    def get_cycle(self, simplex: OrientedSimplex):
        pass

    @staticmethod
    def alpha(simplex: OrientedSimplex) -> OrientedSimplex:
        return OrientedSimplex(simplex.nodes[::-1])

    def sigma(self, simplex: OrientedSimplex,
              sub_simplex: OrientedSimplex) -> OrientedSimplex:
        return self.rotation_info.next(simplex, sub_simplex)

    def phi(self, simplex: OrientedSimplex,
            sub_simplex: OrientedSimplex) -> OrientedSimplex:
        try:
            return self._phi_cache[(simplex, sub_simplex)]
        except KeyError:
            self._phi_cache[(simplex, sub_simplex)] = self.alpha(self.sigma(simplex, sub_simplex))
            return self._phi_cache[(simplex, sub_simplex)]

    @property
    def boundary_cycles(self) -> Collection[BoundaryCycle]:
        if self._boundary_cycles:
            return self._boundary_cycles

        simplices = self.rotation_info.oriented_simplices.copy()
        cycles = partition(self.get_cycle, simplices)
        self._boundary_cycles = cycles
        self._phi_cache.clear()
        return self._boundary_cycles


class CombinatorialMap2D(CombinatorialMap):

    def get_cycle(self, simplex: OrientedSimplex) -> BoundaryCycle:
        if simplex in self._simplices_map:
            return self._simplices_map[simplex]

        cycle = {simplex}
        next_simplex = self.phi(simplex, OrientedSimplex((simplex.nodes[0],)))
        while next_simplex != simplex:
            cycle.add(next_simplex)
            pivot = OrientedSimplex((next_simplex.nodes[0],))
            next_simplex = self.phi(next_simplex, pivot)

        for s in cycle:
            self._simplices_map[s] = BoundaryCycle(frozenset(cycle))
        return self._simplices_map[simplex]

    def get_cycle_nodes(self, simplex):
        nodes = [simplex.nodes[0]]
        next_simplex = self.phi(simplex, OrientedSimplex((simplex.nodes[0],)))
        while next_simplex != simplex:
            pivot = OrientedSimplex((next_simplex.nodes[0],))
            nodes.append(next_simplex.nodes[0])
            next_simplex = self.phi(next_simplex, pivot)
        return nodes


class CombinatorialMap3D(CombinatorialMap):

    def flop(self, simplices):
        result = set(simplices)
        for simplex in simplices:
            result.update([self.phi(simplex, e) for e in simplex.subsimplices])
        return result

    def get_cycle(self, simplex: OrientedSimplex):
        if simplex in self._simplices_map:
            return self._simplices_map[simplex]

        cycle = frozenset(fixed_point(self.flop, frozenset({simplex})))
        self._simplices_map.update({s: BoundaryCycle(cycle) for s in cycle})
        return self._simplices_map[simplex]


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
