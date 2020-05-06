
class CycleLabelling:
    # True = possible intruder
    # False = no intruder
    def __init__(self, boundary_cycles, simplex_cycles):
        self.cell_label = dict()
        for cycle in boundary_cycles:
            self.cell_label[cycle] = True
        for simplex in simplex_cycles:
            self.cell_label[simplex] = False

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