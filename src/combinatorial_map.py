from abc import ABC, abstractmethod
from itertools import chain
from math import atan2
from typing import Iterable, Set, List, FrozenSet

import networkx as nx
import numpy as np
from dataclasses import dataclass, field


class OrientedSimplex:
    def __init__(self, nodes):
        self.nodes = tuple(nodes)

    @property
    def dim(self):
        return len(self.nodes) - 1

    def alpha(self):
        if self.dim == 1:
            return OrientedSimplex([self.nodes[1], self.nodes[0]])
        elif self.dim == 2:
            return OrientedSimplex([self.nodes[0], self.nodes[2], self.nodes[1]])

    def is_edge(self, half_edge):
        if not all([n in self.nodes for n in half_edge.nodes]):
            return False
        i = self.nodes.index(half_edge.nodes[0])
        return (self.nodes[i % 3], self.nodes[(i + 1) % 3]) == half_edge.nodes

    @property
    def edges(self):
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
        return "(" + ",".join(map(str, self.nodes)) + ")"


@dataclass
class BoundaryCycle:
    oriented_simplices: FrozenSet[OrientedSimplex]

    def __iter__(self) -> Iterable[OrientedSimplex]:
        return iter(self.oriented_simplices)

    def __hash__(self):
        return hash(self.oriented_simplices)

    @property
    def nodes(self) -> Set[int]:
        return set(chain.from_iterable([face.nodes for face in self.oriented_simplices]))


class RotationInfo2D:
    def __init__(self, points, alpha_complex):
        graph = nx.Graph()
        graph.add_nodes_from(alpha_complex.nodes)
        graph.add_edges_from([edge.nodes for edge in alpha_complex.simplices(1)])

        self.adj = dict()
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            anticlockwise, clockwise = False, True

            # sort w.r.t angle from x axis
            def theta(a, center):
                oa = (a[0] - center[0], a[1] - center[1])
                return atan2(oa[1], oa[0])

            sorted_neighbors = sorted(neighbors,
                                      key=lambda nei: theta(points[nei], points[node]),
                                      reverse=anticlockwise)
            self.adj[node] = list(sorted_neighbors)

    def next(self, dart):
        v1, v2 = dart
        index = self.adj[v1].index(v2)
        next_index = (index + 1) % len(self.adj[v1])
        return v1, self.adj[v1][next_index]

    @property
    def all_darts(self):
        return set(sum(([(v1, v2) for v2 in self.adj[v1]] for v1 in self.adj), []))


############# Combinatorial Map ####################
class CombinatorialMap(ABC):

    @property
    @abstractmethod
    def boundary_cycles(self):
        ...

    @abstractmethod
    def get_cycle(self, dart):
        pass

########## Combinatorial Map 2D ###################

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


############### 3D Combinatorial Map ##############

@dataclass
class CombinatorialMap3D(CombinatorialMap):
    def __init__(self, points, edges, simplices):
        self.points = points
        self.edges = edges
        self.simplices = simplices

        self.oriented_faces = self.get_oriented(self.simplices)
        self.half_edges = self.get_oriented(self.edges)

        self.rotinfo = dict()
        for half_edge in self.half_edges:
            temp = self.incident_simplices(half_edge)
            self.rotinfo[half_edge] = sorted(temp, key=lambda simplex: self.theta(temp[0], simplex))

        self.hashed_cycles = dict()

    def get_oriented(self, simplices):
        oriented_simplices = []
        for simplex in simplices:
            os = OrientedSimplex(simplex.nodes)
            oriented_simplices.extend([os, os.alpha()])
        return oriented_simplices

    def incident_simplices(self, half_edge: OrientedSimplex):
        return [simplex.orient(half_edge) for simplex in self.oriented_faces if simplex.is_edge(half_edge)]

    def alpha(self, face):
        return face.alpha()

    def sigma(self, face, half_edge):
        if not face.is_edge(half_edge):
            return  # maybe should be error???
        incident_faces = self.rotinfo[half_edge]

        for i in range(len(incident_faces)):
            if incident_faces[i] == face:
                return incident_faces[(i + 1) % len(incident_faces)]

    def phi(self, simplex, half_edge):
        return self.alpha(self.sigma(simplex, half_edge))

    def flop(self, simplices):
        result = set(simplices)
        for simplex in simplices:
            result.update([self.phi(simplex, e) for e in simplex.edges])
        return result

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

    def get_cycle(self, simplex: OrientedSimplex):
        if simplex in self.hashed_cycles:
            return self.hashed_cycles[simplex]
        cycle = {simplex}
        while self.flop(cycle) != cycle:
            cycle = self.flop(cycle)
        self.hashed_cycles[simplex] = BoundaryCycle(frozenset(cycle))
        return self.hashed_cycles[simplex]

    @property
    def boundary_cycles(self):
        faces = set(self.oriented_faces)
        cycles = set()
        while faces:
            f = faces.pop()
            bcycle = self.get_cycle(f)
            faces.difference_update(bcycle)
            cycles.add(bcycle)
        return cycles

