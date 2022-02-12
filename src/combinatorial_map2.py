from abc import ABC, abstractmethod
from typing import Iterable, Set, List

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
class BoundaryCycle2D:
    darts: frozenset

    def __iter__(self):
        return iter(self.darts)

    @property
    def nodes(self):
        return set(sum(self.darts, ()))


class RotationInfo2D:
    def __init__(self):
        pass

    def next(self, dart):
        return dart

    @property
    def all_darts(self):
        return []


class CombinatorialMap(ABC):

    @property
    @abstractmethod
    def boundary_cycles(self):
        ...

    @abstractmethod
    def alpha(self, dart):
        pass

    @abstractmethod
    def get_cycle(self, dart):
        pass


@dataclass
class CombinatorialMap2D(CombinatorialMap):
    Dart = tuple
    rotation_info: RotationInfo2D
    _boundary_cycles: Set = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        all_darts = self.rotation_info.all_darts.copy()
        while all_darts:
            next_dart = all_darts[0]
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
