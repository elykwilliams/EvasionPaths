from abc import ABC, abstractmethod
from math import atan2
from typing import Iterable, Set, List

import networkx as nx
from dataclasses import dataclass, field


class BoundaryCycle(ABC):
    @abstractmethod
    def __iter__(self):
        ...

    @property
    @abstractmethod
    def nodes(self):
        ...


@dataclass
class BoundaryCycle2D(BoundaryCycle):
    darts: frozenset

    def __iter__(self):
        return iter(self.darts)

    def __hash__(self):
        return hash(self.darts)

    @property
    def nodes(self):
        return set(sum(self.darts, ()))


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
            cycle = self._generate_cycle_darts(next_dart)
            for dart in cycle:
                all_darts.remove(dart)
            self._boundary_cycles.add(BoundaryCycle2D(frozenset(cycle)))

    def alpha(self, dart: Dart) -> Dart:
        return self.Dart(reversed(dart))

    def sigma(self, dart: Dart) -> Dart:
        return self.rotation_info.next(dart)

    def phi(self, dart: Dart) -> Dart:
        return self.sigma(self.alpha(dart))

    def _generate_cycle_darts(self, dart: Dart) -> List[Dart]:
        cycle = [dart]
        next_dart = self.phi(dart)
        while next_dart != dart:
            cycle.append(next_dart)
            next_dart = self.phi(next_dart)
        return cycle

    @property
    def boundary_cycles(self) -> Iterable[BoundaryCycle2D]:
        return self._boundary_cycles

    def get_cycle(self, dart: Dart) -> BoundaryCycle2D:
        for cycle in self._boundary_cycles:
            if dart in cycle:
                return cycle
