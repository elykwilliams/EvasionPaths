from abc import ABC, abstractmethod
from typing import Iterable

import networkx as nx
from dataclasses import dataclass

from alpha_complex import SimplicialComplex


class Topology(ABC):

    @abstractmethod
    def simplices(self, dim):
        ...

    @property
    @abstractmethod
    def boundary_cycles(self):
        ...


class BoundaryCycle:
    def __init__(self, dart):
        pass

    def __iter__(self):
        pass

    @property
    def darts(self) -> Iterable:
        return []


class CombinatorialMap(ABC):

    @property
    @abstractmethod
    def boundary_cycles(self) -> Iterable[BoundaryCycle]:
        ...

    def alpha(self, dart):
        pass

    def get_cycle(self, param):
        pass

    @property
    def _graph(self):
        graph = nx.Graph()
        graph.add_nodes_from(self.boundary_cycles)
        for cycle in self.boundary_cycles:
            for dart in cycle.darts:
                graph.add_edge(cycle, self.get_cycle(self.alpha(dart)))
        return graph

    def is_connected(self):
        return nx.is_connected(self._graph)


@dataclass
class ConnectedTopology(Topology):
    alpha_complex: SimplicialComplex
    cmap: CombinatorialMap

    def simplices(self, dim):
        return self.alpha_complex.simplices(dim)

    @property
    def boundary_cycles(self) -> Iterable[BoundaryCycle]:
        return self.cmap.boundary_cycles


