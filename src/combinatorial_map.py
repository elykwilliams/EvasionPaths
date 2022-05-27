from abc import ABC, abstractmethod
from itertools import chain
from math import atan2
from typing import Iterable, Set, List, FrozenSet, Dict, Sequence

import networkx as nx
import numpy as np
from dataclasses import dataclass, field

from alpha_complex import AlphaComplex


class OrientedSimplex:
    def __init__(self, nodes):
        self.nodes = tuple(nodes)

    @property
    def dim(self):
        return len(self.nodes) - 1

    def alpha(self):
        return OrientedSimplex(tuple(reversed(self.nodes)))

    def is_subsimplex(self, sub_simplex):
        if not set(sub_simplex.nodes).issubset(set(self.nodes)):
            return False

        # 3D specific Code
        i = self.nodes.index(sub_simplex.nodes[0])
        return (self.nodes[i % 3], self.nodes[(i + 1) % 3]) == sub_simplex.nodes

    @property
    def edges(self):
        # 3D specific code
        result = []
        for i in range(len(self.nodes)):
            half_edge = (self.nodes[i % 3], self.nodes[(i + 1) % 3])
            result.append(OrientedSimplex(half_edge))
        return result

    def vertices(self, points):
        return [points[n] for n in self.nodes]

    def orient(self, half_edge):
        i = self.nodes.index(half_edge.nodes[0])
        return OrientedSimplex(self.nodes[i:] + self.nodes[:i])

    def __hash__(self):
        i = 0 if self.dim == 1 else self.nodes.index(min(self.nodes))
        return hash(repr(OrientedSimplex(self.nodes[i:] + self.nodes[:i])))

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

    def __iter__(self) -> Iterable[OrientedSimplex]:
        return iter(self.oriented_simplices)

    def __hash__(self):
        return hash(self.oriented_simplices)

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
                                      reverse=anticlockwise)
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


class CombinatorialMap(ABC):

    @property
    @abstractmethod
    def boundary_cycles(self):
        ...

    @abstractmethod
    def get_cycle(self, dart):
        pass


@dataclass
class CombinatorialMap2D(CombinatorialMap):
    Dart = tuple
    rotation_info: RotationInfo2D
    _boundary_cycles: Set = field(default_factory=lambda: set())

    def __post_init__(self) -> None:
        all_darts = self.rotation_info.all_darts.copy()
        while all_darts:
            next_dart = next(iter(all_darts))
            cycle = self.generate_cycle_darts(next_dart)
            for dart in cycle:
                all_darts.remove(dart)
            self._boundary_cycles.add(BoundaryCycle(frozenset(cycle)))

    def alpha(self, dart: Dart) -> Dart:
        return self.Dart(reversed(dart))

    def sigma(self, dart: Dart) -> Dart:
        return self.rotation_info.next(dart)

    def phi(self, dart: Dart) -> Dart:
        return self.sigma(self.alpha(dart))

    def generate_cycle_darts(self, dart: Dart) -> List[Dart]:
        cycle = [dart]
        next_dart = self.phi(dart)
        while next_dart != dart:
            cycle.append(next_dart)
            next_dart = self.phi(next_dart)
        return cycle

    @property
    def boundary_cycles(self) -> Iterable[BoundaryCycle]:
        return self._boundary_cycles

    def get_cycle(self, dart: Dart) -> BoundaryCycle:
        for cycle in self._boundary_cycles:
            if dart in cycle:
                return cycle


@dataclass
class CombinatorialMap3D(CombinatorialMap):
    def __init__(self, rotation_info):
        self.rotation_info = rotation_info
        self.cached_cycles = dict()

    def get_oriented(self, simplices):
        oriented_simplices = []
        for simplex in simplices:
            os = OrientedSimplex(simplex.nodes)
            oriented_simplices.extend([os, os.alpha()])
        return oriented_simplices

    def alpha(self, face):
        return face.alpha()

    def sigma(self, face, half_edge):
        if not face.is_subsimplex(half_edge):
            return  # maybe should be error???
        return self.rotation_info.next(half_edge, face)

    def phi(self, simplex, half_edge):
        return self.alpha(self.sigma(simplex, half_edge))

    def flop(self, simplices):
        result = set(simplices)
        for simplex in simplices:
            result.update([self.phi(simplex, e) for e in simplex.edges])
        return result

    def get_cycle(self, simplex: OrientedSimplex):
        if simplex in self.cached_cycles:
            return self.cached_cycles[simplex]
        cycle = fixed_point(self.flop, {simplex})
        self.cached_cycles[simplex] = BoundaryCycle(frozenset(cycle))
        return self.cached_cycles[simplex]

    @property
    def boundary_cycles(self):
        faces = set(self.rotation_info.oriented_simplices)
        cycles = partition(self.get_cycle, faces)
        return cycles


def partition(equiv_class, X):
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
