from abc import ABC, abstractmethod
from typing import Iterable

from dataclasses import dataclass

from alpha_complex import Simplex, AlphaComplex
from combinatorial_map import CombinatorialMap, BoundaryCycle, RotationInfo2D, CombinatorialMap2D


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
    alpha_complex: AlphaComplex
    cmap: CombinatorialMap

    def simplices(self, dim):
        return self.alpha_complex.simplices(dim)

    @property
    def boundary_cycles(self):
        return self.cmap.boundary_cycles

    @property
    def alpha_cycle(self):
        return self.cmap.get_cycle((0, 1))


def generate_topology(points, radius):
    ac = AlphaComplex(points, radius)
    rot_info = RotationInfo2D(points, ac)
    cmap = CombinatorialMap2D(rot_info)
    return ConnectedTopology(ac, cmap)
