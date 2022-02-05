from abc import ABC, abstractmethod
from typing import Iterable

import networkx as nx
from dataclasses import dataclass


class Topology(ABC):

    @abstractmethod
    def simplices(self, dim):
        ...

    @property
    @abstractmethod
    def boundary_cycles(self):
        ...


class AlphaComplex(ABC):
    @abstractmethod
    def simplices(self, dim):
        pass


class BoundaryCycle:
    def __init__(self, dart):
        pass

    def __iter__(self):
        pass

    @property
    def darts(self):
        pass


class CombinatorialMap(ABC):

    @property
    @abstractmethod
    def boundary_cycles(self) -> Iterable[BoundaryCycle]:
        ...

    def alpha(self, dart):
        pass

    def get_cycle(self, param):
        pass


@dataclass
class ConnectedTopology:
    alpha_complex: AlphaComplex
    cmap: CombinatorialMap

    def simplices(self, dim):
        return self.alpha_complex.simplices(dim)

    @property
    def boundary_cycles(self) -> Iterable[BoundaryCycle]:
        return self.cmap.boundary_cycles

    @property
    def _graph(self):
        graph = nx.Graph()
        graph.add_nodes_from(self.boundary_cycles)
        for cycle in self.boundary_cycles:
            for dart in cycle.darts:
                graph.add_edge(cycle, self.cmap.get_cycle(self.cmap.alpha(dart)))
        return graph
