from abc import ABC, abstractmethod

from dataclasses import dataclass

from topological_state import TopologicalState
from utilities import SetDifference


@dataclass
class StateChange(ABC):
    new_topology: TopologicalState
    old_topology: TopologicalState

    @property
    @abstractmethod
    def case(self) -> tuple:
        ...

    @abstractmethod
    def is_valid(self) -> bool:
        ...

    @property
    @abstractmethod
    def boundary_cycles(self) -> SetDifference:
        ...

    def simplices(self, dim: int) -> SetDifference:
        ...


@dataclass
class StateChange2D(StateChange):

    def simplices(self, dim) -> SetDifference:
        return SetDifference(self.new_topology.simplices(dim), self.old_topology.simplices(dim))

    @property
    def boundary_cycles(self) -> SetDifference:
        return SetDifference(self.new_topology.boundary_cycles(), self.old_topology.boundary_cycles())

    ## Identify Atomic States
    #
    # (#1-simplices added, #1-simpleices removed, #2-simplices added, #2-simplices removed, #boundary cycles added,
    # #boundary cycles removed)
    @property
    def case(self):
        return (len(self.simplices(1).added()), len(self.simplices(1).removed()),
                len(self.simplices(2).added()), len(self.simplices(2).removed()),
                len(self.boundary_cycles.added()), len(self.boundary_cycles.removed()))

    def __repr__(self) -> str:
        return (
            f"State Change: {self.case}\n"
            f"New edges: {self.simplices(1).added()}\n"
            f"Removed edges: {self.simplices(1).removed()}\n"
            f"New Simplices: {self.simplices(2).added()}\n"
            f"Removed Simplices: {self.simplices(2).removed()}\n"
            f"New cycles {self.boundary_cycles.added()}\n"
            f"Removed Cycles {self.boundary_cycles.removed()}"
        )

    def is_valid(self):
        new_simplex_cycles = [simplex.to_cycle(self.boundary_cycles.new_list)
                              for simplex in self.simplices(2).added()]
        old_simplex_cycles = [simplex.to_cycle(self.boundary_cycles.old_list)
                              for simplex in self.simplices(2).removed()]
        check1 = all(cycle in self.boundary_cycles.old_list for cycle in old_simplex_cycles)
        check2 = all(cycle in self.boundary_cycles.new_list for cycle in new_simplex_cycles)
        return check1 and check2
