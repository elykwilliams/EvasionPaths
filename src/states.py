from combinatorial_map import *
from gudhi import AlphaComplex
import networkx as nx


def set_difference(list1, list2):
    return list(set(list1).difference(set(list2)))


def is_subset(list1, list2):
    return set(list1).issubset(set(list2))


case2name = {
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


class State(object):
    def __init__(self, points, sensing_radius, boundary):
        alpha_complex = AlphaComplex(points)
        simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=sensing_radius ** 2)

        self.simplices0 = [simplex[0] for simplex, _ in simplex_tree.get_skeleton(0)]
        self.simplices1 = [tuple(simplex) for simplex, _ in simplex_tree.get_skeleton(1) if len(simplex) == 2]
        self.simplices2 = [tuple(simplex) for simplex, _ in simplex_tree.get_skeleton(2) if len(simplex) == 3]

        graph = nx.Graph()
        graph.add_nodes_from(self.simplices0)
        graph.add_edges_from(self.simplices1)

        self.boundary_cycles = CMap(graph, points).get_boundary_cycles()
        self.boundary_cycles.remove(boundary.alpha_cycle)

        self._connected_nodes = nx.node_connected_component(graph, 0)

    def is_connected(self, cycle):
        return set(cycle2nodes(cycle)).intersection(self._connected_nodes) != set()


class StateChange(object):
    def __init__(self, old_state, new_state):
        self.new_state = new_state
        self.edges_added = set_difference(new_state.simplices1, old_state.simplices1)
        self.edges_removed = set_difference(old_state.simplices1, new_state.simplices1)

        self.simplices_added = set_difference(new_state.simplices2, old_state.simplices2)
        self.simplices_removed = set_difference(old_state.simplices2, new_state.simplices2)

        self.cycles_added = set_difference(new_state.boundary_cycles, old_state.boundary_cycles)
        self.cycles_removed = set_difference(old_state.boundary_cycles, new_state.boundary_cycles)

        self.case = (len(self.edges_added), len(self.edges_removed), len(self.simplices_added),
                     len(self.simplices_removed), len(self.cycles_added), len(self.cycles_removed))

    def is_valid(self):
        if self.case not in case2name.keys():
            return False
        elif self.case == (1, 0, 1, 0, 2, 1):
            simplex = self.simplices_added[0]
            edge = self.edges_added[0]
            if not is_subset(edge, simplex):
                return False
        elif self.case == (0, 1, 0, 1, 1, 2):
            simplex = self.simplices_removed[0]
            edge = self.edges_removed[0]
            if not is_subset(edge, simplex):
                return False
        elif self.case == (1, 1, 2, 2, 2, 2):
            old_edge = self.edges_removed[0]
            new_edge = self.edges_added[0]
            if not all([is_subset(old_edge, s) for s in self.simplices_removed]):
                return False
            elif not all([is_subset(new_edge, s) for s in self.simplices_added]):
                return False

            nodes = set(old_edge).union(set(new_edge))
            if not all([is_subset(s, nodes) for s in self.simplices_removed]):
                return False
            elif not all([is_subset(s, nodes) for s in self.simplices_added]):
                return False
        return True

    def get_name(self):
        if self.is_valid():
            return case2name[self.case]
        else:
            return "Invalid Case"

    def __str__(self):
        return "State Change:\n" \
            + str(self.case) + "\n" \
            + "New edges:" + str(self.edges_added) + "\n" \
            + "Removed edges:" + str(self.edges_removed) + "\n" \
            + "New Simplices:" + str(self.simplices_added) + "\n" \
            + "Removed Simplices:" + str(self.simplices_removed) + "\n" \
            + "New cycles" + str(self.cycles_added) + "\n" \
            + "Removed Cycles" + str(self.cycles_removed)


class MaxRecursionDepth(Exception):
    def __init__(self, state_change):
        self.state_change = state_change

    def __str__(self):
        return "Max Recursion depth exceeded! \n\n" \
               + str(self.state_change)


class InvalidStateChange(Exception):
    def __init__(self, state_change):
        self.state_change = state_change

    def __str__(self):
        return "Invalid State Change \n\n" \
               + str(self.state_change)


class BadStateChange(Exception):
    def __init__(self, state_change):
        self.state_change = state_change

    def __str__(self):
        return "Invalid State Change not caught in is_valid \n\n" \
               + str(self.state_change)
