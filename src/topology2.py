from abc import ABC, abstractmethod
from typing import Iterable

from dataclasses import dataclass

from alpha_complex import SimplicialComplex, Simplex
from combinatorial_map2 import CombinatorialMap, BoundaryCycle


class Topology(ABC):

    @abstractmethod
    def simplices(self, dim) -> Iterable[Simplex]:
        ...

    @property
    @abstractmethod
    def boundary_cycles(self) -> Iterable[BoundaryCycle]:
        ...


@dataclass
class ConnectedTopology(Topology):
    alpha_complex: SimplicialComplex
    cmap: CombinatorialMap

    def simplices(self, dim):
        return self.alpha_complex.simplices(dim)

    @property
    def boundary_cycles(self):
        return self.cmap.boundary_cycles
