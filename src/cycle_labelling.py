from states import *
from combinatorial_map import simplex2cycle

case_name = {
    (0, 0, 0, 0, 0, 0): "",
    (1, 0, 0, 0, 2, 1): "Add 1-Simplex",
    (1, 0, 0, 0, 1, 0): "Add 1-Simplex",
    (0, 1, 0, 0, 1, 2): "Remove 1-Simplex",
    (0, 1, 0, 0, 0, 1): "Remove 1-Simplex",
    (0, 0, 1, 0, 0, 0): "Add 2-Simplex",
    (0, 0, 0, 1, 0, 0): "Remove 2-Simplex",
    (1, 0, 1, 0, 2, 1): "Add 1-Simplex and 2-Simplex",
    (0, 1, 0, 1, 1, 2): "Remove 1-Simplex and 2-Simplex",
    (1, 1, 2, 2, 2, 2): "Delauney Flip",
    (0, 1, 0, 0, 2, 1): "Disconnect",
    (0, 1, 0, 0, 1, 1): "Disconnect",
    (1, 0, 0, 0, 1, 2): "Reconnect",
    (1, 0, 0, 0, 1, 1): "Reconnect"
}

class CycleLabelling:
    # True = possible intruder
    # False = no intruder
    def __init__(self, state):
        self.cell_label = dict()
        for cycle in state.boundary_cycles:
            self.cell_label[cycle] = True

        simplex_cycles = [simplex2cycle(simplex, state.boundary_cycles) for simplex in state.simplices2]
        for simplex in simplex_cycles:
            if state.is_connected(simplex):
                self.cell_label[simplex] = False
            else:
                del self.cell_label[simplex]

    def __getitem__(self, item):
        return self.cell_label[item]

    def __setitem__(self, key, value):
        self.cell_label[key] = value

    def __str__(self):
        res = ""
        for key in self.cell_label.keys():
            res += str(key)+": " + str(self.cell_label[key]) + "\n"
        return res

    def __contains__(self, item):
        return item in self.cell_label.keys()

    def delete_cycle(self, cycle):
        del self.cell_label[cycle]

    def has_intruder(self):
        return any(self.cell_label.values())

    def delaunay_flip(self, cycles_removed, cycles_added):
        # Add new boundary cycles
        for cycle in cycles_added:
            self.cell_label[cycle] = False
        # Remove old boundary cycles
        for cycle in cycles_removed:
            del self.cell_label[cycle]

    def remove_one_two_simplex(self, cycles_removed, new_cycle):
        # Add new boundary cycles to dictionary, they retain the same label as the old cycle
        self.cell_label[new_cycle] = any([self.cell_label[s] for s in cycles_removed])
        # Remove old boundary cycle from dictionary
        for cycle in cycles_removed:
            del self.cell_label[cycle]

    def add_one_two_simplex(self, old_cycle, cycles_added, added_simplex):
        # Add new boundary cycles to dictionary, they retain the same label as the old cycle
        for cycle in cycles_added:
            self.cell_label[cycle] = self.cell_label[old_cycle]
        self.cell_label[added_simplex] = False
        # Remove old boundary cycle from dictionary
        del self.cell_label[old_cycle]

    def add_twosimplex(self, new_cycle):
        # Update existing boundary cycle
        self.cell_label[new_cycle] = False

    def remove_onesimplex(self, cycles_removed, new_cycle):
        # Add new boundary cycles to dictionary, they retain the same label as the old cycle
        self.cell_label[new_cycle] = any([self.cell_label[s] for s in cycles_removed])
        # Remove old boundary cycle from dictionary
        for cycle in cycles_removed:
            del self.cell_label[cycle]

    def add_onesimplex(self, old_cycle, cycles_added):
        # Add new boundary cycles to dictionary, they retain the same label as the old cycle
        for cycle in cycles_added:
            self.cell_label[cycle] = self.cell_label[old_cycle]
        # Remove old boundary cycle from dictionary
        del self.cell_label[old_cycle]

    def update_labelling(self, state_change):
        # No Change
        if state_change.case == (0, 0, 0, 0, 0, 0):
            return ""

        # Add Edge
        elif state_change.case == (1, 0, 0, 0, 2, 1):
            old_cycle = state_change.cycles_removed.pop()
            if old_cycle not in self.cell_label:
                return ""
            self.add_onesimplex(old_cycle, state_change.cycles_added)

        # Remove Edge
        elif state_change.case == (0, 1, 0, 0, 1, 2):
            new_cycle = state_change.cycles_added.pop()
            if any([cell not in self.cell_label for cell in state_change.cycles_removed]):
                return ""

            self.remove_onesimplex(state_change.cycles_removed, new_cycle)

        # Add Simplex
        elif state_change.case == (0, 0, 1, 0, 0, 0):
            simplex = state_change.simplices_added.pop()
            new_cycle = simplex2cycle(simplex, state_change.new_state.boundary_cycles)
            if new_cycle not in self.cell_label:
                return ""

            self.add_twosimplex(new_cycle)

        # Remove Simplex
        elif state_change.case == (0, 0, 0, 1, 0, 0):
            # No label change needed
            pass

        # Edge and Simplex Added
        elif state_change.case == (1, 0, 1, 0, 2, 1):
            old_cycle = state_change.cycles_removed.pop()
            simplex = state_change.simplices_added.pop()
            added_simplex = simplex2cycle(simplex, state_change.new_state.boundary_cycles)

            if not set(state_change.edges_added.pop()).issubset(set(simplex)):
                raise InvalidStateChange(state_change)

            if old_cycle not in self.cell_label:
                return ""

            self.add_one_two_simplex(old_cycle, state_change.cycles_added, added_simplex)

        # Edge and Simplex Removed
        elif state_change.case == (0, 1, 0, 1, 1, 2):
            simplex = state_change.simplices_removed.pop()
            new_cycle = state_change.cycles_added.pop()

            if not set(state_change.edges_removed.pop()).issubset(set(simplex)):
                raise InvalidStateChange(state_change)

            if any([cell not in self.cell_label for cell in state_change.cycles_removed]):
                return

            self.remove_one_two_simplex(state_change.cycles_removed, new_cycle)

        # Delunay Flip
        elif state_change.case == (1, 1, 2, 2, 2, 2):
            oldedge = state_change.edges_removed.pop()
            newedge = state_change.edges_added.pop()

            if not all([cycle in self.cell_label for cycle in state_change.cycles_removed]):
                return ""

            # Check that edges correspond to correct boundary cycles
            if not all([set(oldedge).issubset(set(s)) for s in state_change.simplices_removed]):
                raise InvalidStateChange(state_change)
            elif not all([set(newedge).issubset(set(s)) for s in state_change.simplices_added]):
                raise InvalidStateChange(state_change)
            elif not all([set(s).issubset(set(oldedge).union(set(newedge))) for s in state_change.simplices_removed]):
                raise InvalidStateChange(state_change)
            elif not all([set(s).issubset(set(oldedge).union(set(newedge))) for s in state_change.simplices_added]):
                raise InvalidStateChange(state_change)

            self.delaunay_flip(state_change.cycles_removed, state_change.cycles_added)

        # Disconnect
        elif state_change.case == (0, 1, 0, 0, 2, 1) or state_change.case == (0, 1, 0, 0, 1, 1):
            old_cycle = state_change.cycles_removed.pop()
            if old_cycle not in self.cell_label:
                return ""

            enclosing_cycle = state_change.cycles_added.pop()
            if not state_change.new_state.is_connected(enclosing_cycle) and len(state_change.cycles_added) != 0:
                enclosing_cycle = state_change.cycles_added.pop()

            # Find labelled cycles that have become disconnected,
            # this works because all other disconnected cycles have been forgotten
            disconnected_cycles = []
            for cycle in state_change.new_state.boundary_cycles:
                if not state_change.new_state.is_connected(cycle) and cycle in self.cell_label:
                    disconnected_cycles.append(cycle)

            # Enclosing cycle will be clear if the old enclosing cycle and all disconnected cycles are clear
            intruder_subcycle = any([self.cell_label[cycle] for cycle in disconnected_cycles])

            self.cell_label[enclosing_cycle] = intruder_subcycle or self.cell_label[old_cycle]

            # Forget disconnected cycles
            for cycle in disconnected_cycles:
                self.delete_cycle(cycle)

            self.delete_cycle(old_cycle)

        # Reconnect
        elif state_change.case == (1, 0, 0, 0, 1, 2) or state_change.case == (1, 0, 0, 0, 1, 1):
            enclosing_cycle = state_change.cycles_removed.pop()
            if enclosing_cycle not in self.cell_label and len(state_change.cycles_removed) != 0:
                enclosing_cycle = state_change.cycles_removed.pop()

            if enclosing_cycle not in self.cell_label:
                return ""

            # Find labelled cycles that have just become connected,
            # These will be all boundary cycles that are connected and have no label
            new_cycle = state_change.cycles_added.pop()

            # Newly connected cycles have label to match old enclosing cycle
            self.cell_label[new_cycle] = self.cell_label[enclosing_cycle]

            # Add back any forgotten cycle
            for cycle in state_change.new_state.boundary_cycles:
                if state_change.new_state.is_connected(cycle) and cycle not in self.cell_label:
                    self.cell_label[cycle] = self.cell_label[enclosing_cycle]

            # Reset all connected 2-simplices to have no intruder
            for simplex in state_change.new_state.simplices2:
                cycle = simplex2cycle(simplex, state_change.new_state.boundary_cycles)
                if cycle not in self.cell_label:
                    continue
                self.add_twosimplex(cycle)

            # Delete old boundary cycle
            self.delete_cycle(enclosing_cycle)

        # two isolated points connecting
        elif state_change.case == (1, 0, 0, 0, 1, 0):
            return ""
        # two points becoming isolated
        elif state_change.case == (0, 1, 0, 0, 0, 1):
            return ""
        else:
            raise InvalidStateChange(state_change)

        return case_name[state_change.case] + ", "
