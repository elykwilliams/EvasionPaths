from typing import Dict

from dataclasses import dataclass

from topology import Topology
from utilities import SetDifference


@dataclass
class StateChange:
    new_topology: Topology
    old_topology: Topology

    @property
    def simplices(self) -> Dict[int, SetDifference]:
        dim = self.new_topology.dim
        return {i: SetDifference(self.new_topology.simplices(i), self.old_topology.simplices(i))
                for i in range(1, dim + 1)}

    @property
    def boundary_cycles(self) -> SetDifference:
        return SetDifference(self.new_topology.boundary_cycles, self.old_topology.boundary_cycles)

    ## Identify Atomic States
    #
    # (#1-simplices added, #1-simpleices removed, #2-simplices added, #2-simplices removed, #boundary cycles added,
    # #boundary cycles removed)
    @property
    def case(self):
        case = ()
        for dim in range(1, self.new_topology.dim + 1):
            case += (len(self.simplices[dim].added()), len(self.simplices[dim].removed()))
        return case + (len(self.boundary_cycles.added()), len(self.boundary_cycles.removed()))

    def __repr__(self) -> str:
        result = f"State Change: {self.case}\n"
        for dim in range(1, self.new_topology.dim + 1):
            result += f"{dim}-Simplices Added: {self.simplices[dim].added()}\n" \
                      f"{dim}-Simplices Removed: {self.simplices[dim].removed()}\n"
        result += f"Boundary Cycles Added: {self.boundary_cycles.added()}\n" \
                  f"Boundary Cycles Removed: {self.boundary_cycles.removed()}\n"
        return result

    def is_valid(self):
        new_simplex_cycles = [simplex.to_cycle(self.boundary_cycles.new_list)
                              for simplex in self.simplices[self.new_topology.dim].added()]
        old_simplex_cycles = [simplex.to_cycle(self.boundary_cycles.old_list)
                              for simplex in self.simplices[self.old_topology.dim].removed()]
        check1 = all(cycle in self.boundary_cycles.old_list for cycle in old_simplex_cycles)
        check2 = all(cycle in self.boundary_cycles.new_list for cycle in new_simplex_cycles)
        return check1 and check2
