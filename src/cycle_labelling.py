# ************************************************************
# Copyright (c) 2020, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************


from states import *
from combinatorial_map import simplex2cycle


class CycleLabelling:
    # True = possible intruder
    # False = no intruder
    def __init__(self, state):
        self.cycle_label = dict()
        for cycle in state.boundary_cycles:
            self.cycle_label[cycle] = True

        simplex_cycles \
            = [simplex2cycle(simplex, state.boundary_cycles) for simplex in state.simplices2]
        for simplex in simplex_cycles:
            self.cycle_label[simplex] = False

        for cycle in list(self.cycle_label):
            if not state.is_connected(cycle):
                del self.cycle_label[cycle]

    def __str__(self):
        res = ""
        for key in self.cycle_label:
            res += str(key) + ": " + str(self.cycle_label[key]) + "\n"
        return res

    def __contains__(self, item):
        return item in self.cycle_label

    def __getitem__(self, item):
        return self.cycle_label[item]

    def has_intruder(self):
        return any(self.cycle_label.values())

    def add_1simplex(self, removed_cycles, added_cycles):
        for cycle in added_cycles:
            self.cycle_label[cycle] = self.cycle_label[removed_cycles[0]]
        del self.cycle_label[removed_cycles[0]]

    def remove_1simplex(self, removed_cycles, added_cycles):
        self.cycle_label[added_cycles[0]] = any([self.cycle_label[s] for s in removed_cycles])
        for cycle in removed_cycles:
            del self.cycle_label[cycle]

    def add_2simplex(self, added_simplex):
        self.cycle_label[added_simplex] = False

    def add_simplex_pair(self, removed_cycles, added_cycles, added_simplex):
        self.add_1simplex(removed_cycles, added_cycles)
        self.add_2simplex(added_simplex)

    def remove_simplex_pair(self, removed_cycles, added_cycles):
        self.remove_1simplex(removed_cycles, added_cycles)

    def delaunay_flip(self, removed_cycles, added_cycles):
        for cycle in added_cycles:
            self.cycle_label[cycle] = False
        for cycle in removed_cycles:
            del self.cycle_label[cycle]

    def disconnect(self, removed_cycles, enclosing_cycle):
        self.cycle_label[enclosing_cycle] = any([self.cycle_label[cycle] for cycle in removed_cycles])
        for cycle in removed_cycles:
            del self.cycle_label[cycle]

    def reconnect(self, added_cycles, enclosing_cycle, reconnected_simplices):
        for cycle in added_cycles:
            self.cycle_label[cycle] = self.cycle_label[enclosing_cycle]
        for cycle in reconnected_simplices:
            self.add_2simplex(cycle)
        del self.cycle_label[enclosing_cycle]

    def ignore_state_change(self, state_change):
        # No Change
        if state_change.case == (0, 0, 0, 0, 0, 0):
            return

        if state_change.case == (1, 0, 0, 0, 2, 1):
            old_cycle = state_change.cycles_removed[0]
            if old_cycle not in self.cycle_label:
                return True
        elif state_change.case == (0, 1, 0, 0, 1, 2):
            if any([cell not in self.cycle_label for cell in state_change.cycles_removed]):
                return True
        elif state_change.case == (0, 0, 1, 0, 0, 0):
            simplex = state_change.simplices_added[0]
            new_cycle = simplex2cycle(simplex, state_change.new_state.boundary_cycles)
            if new_cycle not in self.cycle_label:
                return True
        elif state_change.case == (1, 0, 1, 0, 2, 1):
            old_cycle = state_change.cycles_removed[0]
            if old_cycle not in self.cycle_label:
                return True
        elif state_change.case == (0, 1, 0, 1, 1, 2):
            if any([cell not in self.cycle_label for cell in state_change.cycles_removed]):
                return True
        elif state_change.case == (1, 1, 2, 2, 2, 2):
            if not all([cycle in self.cycle_label for cycle in state_change.cycles_removed]):
                return True
        elif state_change.case == (0, 1, 0, 0, 2, 1) or state_change.case == (0, 1, 0, 0, 1, 1):
            old_cycle = state_change.cycles_removed[0]
            if old_cycle not in self.cycle_label:
                return True
        elif state_change.case == (1, 0, 0, 0, 1, 2) or state_change.case == (1, 0, 0, 0, 1, 1):
            enclosing_cycle = state_change.cycles_removed[0]
            if enclosing_cycle not in self.cycle_label and len(state_change.cycles_removed) != 1:
                enclosing_cycle = state_change.cycles_removed[1]
            if enclosing_cycle not in self.cycle_label:
                return True

        # Two isolated points connecting
        elif state_change.case == (1, 0, 0, 0, 1, 0):
            return True

        # 1-simplex becoming isolated points
        elif state_change.case == (0, 1, 0, 0, 0, 1):
            return True

        return False

    def update(self, state_change):
        if not state_change.is_valid():
            raise InvalidStateChange(state_change)

        if self.ignore_state_change(state_change):
            return

        # Remove Simplex
        elif state_change.case == (0, 0, 0, 1, 0, 0):
            return

        # Add Edge
        elif state_change.case == (1, 0, 0, 0, 2, 1):
            self.add_1simplex(state_change.cycles_removed, state_change.cycles_added)

        # Remove Edge
        elif state_change.case == (0, 1, 0, 0, 1, 2):
            self.remove_1simplex(state_change.cycles_removed, state_change.cycles_added)

        # Add Simplex
        elif state_change.case == (0, 0, 1, 0, 0, 0):
            simplex = state_change.simplices_added[0]
            added_simplex = simplex2cycle(simplex, state_change.new_state.boundary_cycles)
            self.add_2simplex(added_simplex)

        # Edge and Simplex Added
        elif state_change.case == (1, 0, 1, 0, 2, 1):
            simplex = state_change.simplices_added[0]
            added_simplex = simplex2cycle(simplex, state_change.new_state.boundary_cycles)
            self.add_simplex_pair(state_change.cycles_removed, state_change.cycles_added, added_simplex)

        # Edge and Simplex Removed
        elif state_change.case == (0, 1, 0, 1, 1, 2):
            self.remove_1simplex(state_change.cycles_removed, state_change.cycles_added)

        # Delunay Flip
        elif state_change.case == (1, 1, 2, 2, 2, 2):
            self.delaunay_flip(state_change.cycles_removed, state_change.cycles_added)

        # Disconnect
        elif state_change.case == (0, 1, 0, 0, 2, 1) or state_change.case == (0, 1, 0, 0, 1, 1):
            enclosing_cycle = state_change.cycles_added[0]
            if not state_change.new_state.is_connected(enclosing_cycle) and len(state_change.cycles_added) != 0:
                enclosing_cycle = state_change.cycles_added[1]

            disconnected_cycles = []
            for cycle in state_change.new_state.boundary_cycles:
                if not state_change.new_state.is_connected(cycle) and cycle in self.cycle_label:
                    disconnected_cycles.append(cycle)

            self.disconnect(state_change.cycles_removed + disconnected_cycles, enclosing_cycle)

        # Reconnect
        elif state_change.case == (1, 0, 0, 0, 1, 2) or state_change.case == (1, 0, 0, 0, 1, 1):
            enclosing_cycle = state_change.cycles_removed[0]
            if enclosing_cycle not in self.cycle_label and len(state_change.cycles_removed) != 1:
                enclosing_cycle = state_change.cycles_removed[1]

            reconnected_cycles = []
            for cycle in state_change.new_state.boundary_cycles:
                if state_change.new_state.is_connected(cycle) and cycle not in self.cycle_label:
                    reconnected_cycles.append(cycle)

            reconnected_simplices = []
            for simplex in state_change.new_state.simplices2:
                cycle = simplex2cycle(simplex, state_change.new_state.boundary_cycles)
                if state_change.new_state.is_connected(cycle):
                    reconnected_simplices.append(cycle)

            self.reconnect(state_change.cycles_added + reconnected_cycles, enclosing_cycle,
                           reconnected_simplices)
