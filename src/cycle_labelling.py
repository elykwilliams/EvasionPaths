# ******************************************************************************
#  Copyright (c) 2023, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************

from reeb_graph import ReebGraph
from state_change import StateChange
from topology import Topology


class CycleLabelling:

    def __init__(self, topology: Topology):
        self.label = {g: True for g in topology.homology_generators}
        self.history = [(self.label, (0,)*topology.dim*2, (0, 0), 0)]

        self.reeb_graph = ReebGraph(self.label)

    def update(self, state_change: StateChange, time):

        added_cycles = state_change.new_topology.homology_generators.difference(
            state_change.old_topology.homology_generators)
        removed_cycles = state_change.old_topology.homology_generators.difference(
            state_change.new_topology.homology_generators)

        for cycle in added_cycles:
            self.label[cycle] = any(self.label[cycle] for cycle in removed_cycles)

        for cycle in removed_cycles:
            del self.label[cycle]

        self.reeb_graph.update(time, self.label, self.history[-1][0])
        self.history.append((self.label, state_change.alpha_complex_change(), state_change.boundary_cycle_change(), time))

    def finalize(self, time):
        self.reeb_graph.finalize(time, self.label)

    def has_intruder(self):
        return sum(1 for _, value in self.label.items() if value) > 1

