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
            if state.is_connected(simplex):
                self.cycle_label[simplex] = False
            else:
                del self.cycle_label[simplex]

    def __getitem__(self, item):
        return self.cycle_label[item]

    def __setitem__(self, key, value):
        self.cycle_label[key] = value

    def __str__(self):
        res = ""
        for key in self.cycle_label:
            res += str(key) + ": " + str(self.cycle_label[key]) + "\n"
        return res

    def __contains__(self, item):
        return item in self.cycle_label

    def delete_cycle(self, cycle):
        del self.cycle_label[cycle]

    def has_intruder(self):
        return any(self.cycle_label.values())

    def delaunay_flip(self, cycles_removed, cycles_added):
        for cycle in cycles_added:
            self.cycle_label[cycle] = False
        for cycle in cycles_removed:
            del self.cycle_label[cycle]

    def remove_one_two_simplex(self, cycles_removed, new_cycle):
        # Add new boundary cycles to dictionary, they retain the same label as the old cycle
        self.cycle_label[new_cycle] = any([self.cycle_label[s] for s in cycles_removed])
        for cycle in cycles_removed:
            del self.cycle_label[cycle]

    def add_one_two_simplex(self, old_cycle, cycles_added, added_simplex):
        # Add new boundary cycles to dictionary, they retain the same label as the old cycle
        for cycle in cycles_added:
            self.cycle_label[cycle] = self.cycle_label[old_cycle]
        self.cycle_label[added_simplex] = False
        # Remove old boundary cycle from dictionary
        del self.cycle_label[old_cycle]

    def add_twosimplex(self, new_cycle):
        # Update existing boundary cycle
        self.cycle_label[new_cycle] = False

    def remove_onesimplex(self, cycles_removed, new_cycle):
        # Add new boundary cycles to dictionary, they retain the same label as the old cycle
        self.cycle_label[new_cycle] = any([self.cycle_label[s] for s in cycles_removed])
        # Remove old boundary cycle from dictionary
        for cycle in cycles_removed:
            del self.cycle_label[cycle]

    def add_onesimplex(self, old_cycle, cycles_added):
        # Add new boundary cycles to dictionary, they retain the same label as the old cycle
        for cycle in cycles_added:
            self.cycle_label[cycle] = self.cycle_label[old_cycle]
        # Remove old boundary cycle from dictionary
        del self.cycle_label[old_cycle]

    def update(self, state_change):
        if not state_change.is_valid():
            raise InvalidStateChange(state_change)

        # No Change
        if state_change.case == (0, 0, 0, 0, 0, 0):
            return

        # Add Edge
        elif state_change.case == (1, 0, 0, 0, 2, 1):
            old_cycle = state_change.cycles_removed[0]
            if old_cycle not in self.cycle_label:
                return
            self.add_onesimplex(old_cycle, state_change.cycles_added)

        # Remove Edge
        elif state_change.case == (0, 1, 0, 0, 1, 2):
            if any([cell not in self.cycle_label for cell in state_change.cycles_removed]):
                return

            new_cycle = state_change.cycles_added[0]
            self.remove_onesimplex(state_change.cycles_removed, new_cycle)

        # Add Simplex
        elif state_change.case == (0, 0, 1, 0, 0, 0):
            simplex = state_change.simplices_added[0]
            new_cycle = simplex2cycle(simplex, state_change.new_state.boundary_cycles)
            if new_cycle not in self.cycle_label:
                return ""

            self.add_twosimplex(new_cycle)

        # Remove Simplex
        elif state_change.case == (0, 0, 0, 1, 0, 0):
            # No label change needed
            return

        # Edge and Simplex Added
        elif state_change.case == (1, 0, 1, 0, 2, 1):
            old_cycle = state_change.cycles_removed[0]
            simplex = state_change.simplices_added[0]
            added_simplex = simplex2cycle(simplex, state_change.new_state.boundary_cycles)

            if old_cycle not in self.cycle_label:
                return

            self.add_one_two_simplex(old_cycle, state_change.cycles_added, added_simplex)

        # Edge and Simplex Removed
        elif state_change.case == (0, 1, 0, 1, 1, 2):
            new_cycle = state_change.cycles_added[0]

            if any([cell not in self.cycle_label for cell in state_change.cycles_removed]):
                return

            self.remove_one_two_simplex(state_change.cycles_removed, new_cycle)

        # Delunay Flip
        elif state_change.case == (1, 1, 2, 2, 2, 2):

            if not all([cycle in self.cycle_label for cycle in state_change.cycles_removed]):
                return

            self.delaunay_flip(state_change.cycles_removed, state_change.cycles_added)

        # Disconnect
        elif state_change.case == (0, 1, 0, 0, 2, 1) or state_change.case == (0, 1, 0, 0, 1, 1):
            old_cycle = state_change.cycles_removed[0]
            if old_cycle not in self.cycle_label:
                return

            enclosing_cycle = state_change.cycles_added[0]
            if not state_change.new_state.is_connected(enclosing_cycle) and len(state_change.cycles_added) != 0:
                enclosing_cycle = state_change.cycles_added[1]

            # Find labelled cycles that have become disconnected,
            # this works because all other disconnected cycles have been forgotten
            disconnected_cycles = []
            for cycle in state_change.new_state.boundary_cycles:
                if not state_change.new_state.is_connected(cycle) and cycle in self.cycle_label:
                    disconnected_cycles.append(cycle)

            # Enclosing cycle will be clear if the old enclosing cycle and all disconnected cycles are clear
            intruder_subcycle = any([self.cycle_label[cycle] for cycle in disconnected_cycles])

            self.cycle_label[enclosing_cycle] = intruder_subcycle or self.cycle_label[old_cycle]

            # Forget disconnected cycles
            for cycle in disconnected_cycles:
                self.delete_cycle(cycle)

            self.delete_cycle(old_cycle)

        # Reconnect
        elif state_change.case == (1, 0, 0, 0, 1, 2) or state_change.case == (1, 0, 0, 0, 1, 1):
            enclosing_cycle = state_change.cycles_removed[0]
            if enclosing_cycle not in self.cycle_label and len(state_change.cycles_removed) != 0:
                enclosing_cycle = state_change.cycles_removed[1]

            if enclosing_cycle not in self.cycle_label:
                return

            # Find labelled cycles that have just become connected,
            # These will be all boundary cycles that are connected and have no label
            new_cycle = state_change.cycles_added[0]

            # Newly connected cycles have label to match old enclosing cycle
            self.cycle_label[new_cycle] = self.cycle_label[enclosing_cycle]

            # Add back any forgotten cycle
            for cycle in state_change.new_state.boundary_cycles:
                if state_change.new_state.is_connected(cycle) and cycle not in self.cycle_label:
                    self.cycle_label[cycle] = self.cycle_label[enclosing_cycle]

            # Reset all connected 2-simplices to have no intruder
            for simplex in state_change.new_state.simplices2:
                cycle = simplex2cycle(simplex, state_change.new_state.boundary_cycles)
                if cycle not in self.cycle_label:
                    continue
                self.add_twosimplex(cycle)

            # Delete old boundary cycle
            self.delete_cycle(enclosing_cycle)

        # two isolated points connecting
        elif state_change.case == (1, 0, 0, 0, 1, 0):
            return

        # two points becoming isolated
        elif state_change.case == (0, 1, 0, 0, 0, 1):
            return
