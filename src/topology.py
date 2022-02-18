from dataclasses import dataclass

from alpha_complex import AlphaComplex
from combinatorial_map import BoundaryCycle, RotationInfo2D, CombinatorialMap2D, CombinatorialMap


@dataclass
class Topology:
    alpha_complex: AlphaComplex
    cmap: CombinatorialMap

    def simplices(self, dim):
        return self.alpha_complex.simplices(dim)

    @property
    def boundary_cycles(self):
        return self.cmap.boundary_cycles

    @property
    def alpha_cycle(self) -> BoundaryCycle:
        return self.cmap.get_cycle(tuple(range(self.dim)))

    @property
    def dim(self) -> int:
        return self.alpha_complex.dim


def generate_topology(points, radius):
    ac = AlphaComplex(points, radius)
    if ac.dim == 2:
        rot_info = RotationInfo2D(points, ac)
        cmap = CombinatorialMap2D(rot_info)
    else:
        raise NotImplementedError(f"No Implementation for CombinatorialMap for dimension {ac.dim}")
    return Topology(ac, cmap)
