from abc import ABC, abstractmethod

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


class CombinatorialMap(ABC):

    @property
    @abstractmethod
    def boundary_cycles(self):
        ...


@dataclass
class ConnectedTopology:
    alpha_complex: AlphaComplex
    cmap: CombinatorialMap

    def simplices(self, dim):
        return self.alpha_complex.simplices(dim)

    @property
    def boundary_cycles(self):
        return self.cmap.boundary_cycles

    @property
    def _graph(self):
        return None
