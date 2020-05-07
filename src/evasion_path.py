# Kyle Williams 3/8/20
from cycle_labelling import *
from combinatorial_map import *
from gudhi import AlphaComplex
import networkx as nx
from copy import deepcopy


def interpolate_points(old_points, new_points, t):
    assert (len(old_points) == len(new_points))
    return [(old_points[n][0] * (1 - t) + new_points[n][0] * t, old_points[n][1] * (1 - t) + new_points[n][1] * t)
            for n in range(len(old_points))]


def is_connected(G, cycle):
    """A cycle is connected if its vertices are a subset of those connected to node 0"""
    connected_nodes = nx.node_connected_component(G, 0)
    return set(cycle2nodes(cycle)).intersection(connected_nodes) != set()


def set_difference(list1, list2):
    return set(list1).difference(set(list2))


class InvalidStateChange(Exception):
    def __init__(self, state):
        # state = (new_1, old_1, new_2, old_2, new_bc, old_bc)
        self.state = state

    def __str__(self):
        case = tuple(len(s) for s in self.state)

        return "Invalid State Change:" + str(case) + "\n" \
               + "New edges:" + str(self.state[0]) + "\n" \
               + "Removed edges:" + str(self.state[1]) + "\n" \
               + "New Simplices:" + str(self.state[2]) + "\n" \
               + "Removed Simplices:" + str(self.state[3]) + "\n" \
               + "New cycles" + str(self.state[4]) + "\n" \
               + "Removed Cycles" + str(self.state[5])


class MaxRecursionDepth(InvalidStateChange):
    def __str__(self):
        return "Max Recursion depth exceeded \n\n" + str(super())


class GraphNotConnected(Exception):
    def __str__(self): return "Graph not connected"


class State(object):
    def __init__(self, simplices, edges, boundary_cycles, n_total_sensors):
        self.simplices2 = simplices
        self.simplices1 = edges
        self.boundary_cycles = boundary_cycles

        graph = nx.Graph()
        graph.add_nodes_from(range(n_total_sensors))
        graph.add_edges_from(self.simplices1)
        self.connected_nodes = nx.node_connected_component(graph, 0)

    def is_connected(self, cycle):
        return set(cycle2nodes(cycle)).intersection(self.connected_nodes) != set()


class EvasionPathSimulation:
    def __init__(self, boundary, motion_model, n_int_sensors, sensing_radius, dt, end_time=0):

        # Initialize Fields
        self.evasion_paths = ""
        self.boundary_cycles = []
        self.old_points = []
        self.old_cycles = []
        self.old_edges = []
        self.old_simplices = []
        self.edges = []
        self.simplices = []
        self.state_changes = None
        self.old_state = None
        self.graph = nx.Graph()

        self.motion_model = motion_model

        # Parameters
        self.dt = dt
        self.Tend = end_time

        # Internal time keeping
        self.time = 0
        self.n_steps = 0

        # Point data
        self.sensing_radius = sensing_radius
        self.points = boundary.generate_points(n_int_sensors)
        self.n_total_sensors = n_int_sensors + len(boundary)

        self.alpha_cycle = boundary.alpha_cycle

        self.state = self.get_state()

        self.cell_label = CycleLabelling(self.state)

        # Update old data
        self.update_old_data()

    def get_state(self):
        alpha_complex = AlphaComplex(self.points)
        simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=self.sensing_radius ** 2)
        self.edges = [tuple(simplex) for simplex, _ in simplex_tree.get_skeleton(1) if len(simplex) == 2]
        self.simplices = [tuple(simplex) for simplex, _ in simplex_tree.get_skeleton(2) if len(simplex) == 3]
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.n_total_sensors))
        self.graph.add_edges_from(self.edges)
        cmap = CMap(self.graph, self.points)
        self.boundary_cycles = [bc for bc in cmap.get_boundary_cycles() if bc != self.alpha_cycle]
        return State(self.simplices, self.edges, self.boundary_cycles, self.n_total_sensors)

    def get_boundary_cycles(self):
        cmap = CMap(self.graph, self.points)
        return [bc for bc in cmap.get_boundary_cycles()
                if bc != self.alpha_cycle]

    def update_old_data(self):
        self.old_points = self.points.copy()
        self.old_cycles = self.boundary_cycles.copy()
        self.old_edges = self.edges.copy()
        self.old_simplices = self.simplices.copy()
        self.old_state = deepcopy(self.state)

    def reset_current_data(self):
        self.points = self.old_points.copy()

    def run(self):
        if self.Tend != 0:
            while self.time < self.Tend:
                self.time += self.dt
                self.do_timestep()
            return self.time
        else:
            while self.cell_label.has_intruder():
                self.time += self.dt
                self.do_timestep()
            return self.time

    def do_timestep(self, new_points=(), level=0):

        if level == 0:
            t_values = [1.0]
            self.evasion_paths = ""
        else:
            self.reset_current_data()
            t_values = [0.5, 1.0]

        for t in t_values:

            # Update Points
            if level == 0:
                self.points = self.motion_model.update_points(self.points)
            else:
                self.points = interpolate_points(self.old_points, new_points, t)

            self.state = self.get_state()

            # Find Evasion Path
            try:
                self.update_labelling()
                self.n_steps += 1

            except InvalidStateChange as exception:

                if level == 20:
                    raise MaxRecursionDepth(exception)

                # Reset current level to previous step
                self.do_timestep(self.points, level=level + 1)


            # Update old data
            self.update_old_data()

    def update_labelling(self):

        edges_added = set_difference(self.edges, self.old_edges)
        edges_removed = set_difference(self.old_edges, self.edges)

        simplices_added = set_difference(self.simplices, self.old_simplices)
        simplices_removed = set_difference(self.old_simplices, self.simplices)

        cycles_added = set_difference(self.boundary_cycles, self.old_cycles)
        cycles_removed = set_difference(self.old_cycles, self.boundary_cycles)

        self.state_changes = (edges_added.copy(), edges_removed.copy(),
                              simplices_added.copy(), simplices_removed.copy(),
                              cycles_added.copy(), cycles_removed.copy())

        case = tuple(len(s) for s in self.state_changes)

        # No Change
        if case == (0, 0, 0, 0, 0, 0):
            return

        # Add Edge
        elif case == (1, 0, 0, 0, 2, 1):
            old_cycle = cycles_removed.pop()
            if old_cycle not in self.cell_label:
                return
            self.cell_label.add_onesimplex(old_cycle, cycles_added)

        # Remove Edge
        elif case == (0, 1, 0, 0, 1, 2):
            new_cycle = cycles_added.pop()
            if any([cell not in self.cell_label for cell in cycles_removed]):
                return

            self.cell_label.remove_onesimplex(cycles_removed, new_cycle)

        # Add Simplex
        elif case == (0, 0, 1, 0, 0, 0):
            simplex = simplices_added.pop()
            new_cycle = simplex2cycle(simplex, self.boundary_cycles)
            if new_cycle not in self.cell_label:
                return

            self.cell_label.add_twosimplex(new_cycle)

        # Remove Simplex
        elif case == (0, 0, 0, 1, 0, 0):
            # No label change needed
            pass

        # Edge and Simplex Added
        elif case == (1, 0, 1, 0, 2, 1):
            old_cycle = cycles_removed.pop()
            simplex = simplices_added.pop()
            added_simplex = simplex2cycle(simplex, self.boundary_cycles)

            if not set(edges_added.pop()).issubset(set(simplex)):
                raise InvalidStateChange(self.state_changes)

            if old_cycle not in self.cell_label:
                return

            self.cell_label.add_one_two_simplex(old_cycle, cycles_added, added_simplex)

        # Edge and Simplex Removed
        elif case == (0, 1, 0, 1, 1, 2):
            simplex = simplices_removed.pop()
            new_cycle = cycles_added.pop()

            if not set(edges_removed.pop()).issubset(set(simplex)):
                raise InvalidStateChange(self.state_changes)

            if any([cell not in self.cell_label for cell in cycles_removed]):
                return

            self.cell_label.remove_one_two_simplex(cycles_removed, new_cycle)

        # Delunay Flip
        elif case == (1, 1, 2, 2, 2, 2):
            oldedge = edges_removed.pop()
            newedge = edges_added.pop()

            if not all([cycle in self.cell_label for cycle in cycles_removed]):
                return

            # Check that edges correspond to correct boundary cycles
            if not all([set(oldedge).issubset(set(s)) for s in simplices_removed]):
                raise InvalidStateChange(self.state_changes)
            elif not all([set(newedge).issubset(set(s)) for s in simplices_added]):
                raise InvalidStateChange(self.state_changes)
            elif not all([set(s).issubset(set(oldedge).union(set(newedge))) for s in simplices_removed]):
                raise InvalidStateChange(self.state_changes)
            elif not all([set(s).issubset(set(oldedge).union(set(newedge))) for s in simplices_added]):
                raise InvalidStateChange(self.state_changes)

            self.cell_label.delaunay_flip(cycles_removed, cycles_added)

        # Disconnect
        elif case == (0, 1, 0, 0, 2, 1) or case == (0, 1, 0, 0, 1, 1):
            old_cycle = cycles_removed.pop()
            if old_cycle not in self.cell_label:
                return

            enclosing_cycle = cycles_added.pop()
            if not is_connected(self.graph, enclosing_cycle) and len(cycles_added) != 0:
                enclosing_cycle = cycles_added.pop()

            # Find labelled cycles that have become disconnected,
            # this works because all other disconnected cycles have been forgotten
            disconnected_cycles = []
            for cycle in self.boundary_cycles:
                if not is_connected(self.graph, cycle) and cycle in self.cell_label:
                    disconnected_cycles.append(cycle)

            # Enclosing cycle will be clear if the old enclosing cycle and all disconnected cycles are clear
            intruder_subcycle = any([self.cell_label[cycle] for cycle in disconnected_cycles])

            self.cell_label[enclosing_cycle] = intruder_subcycle or self.cell_label[old_cycle]

            # Forget disconnected cycles
            for cycle in disconnected_cycles:
                self.cell_label.delete_cycle(cycle)

            self.cell_label.delete_cycle(old_cycle)

        # Reconnect
        elif case == (1, 0, 0, 0, 1, 2) or case == (1, 0, 0, 0, 1, 1):
            enclosing_cycle = cycles_removed.pop()
            if enclosing_cycle not in self.cell_label and len(cycles_removed) != 0:
                enclosing_cycle = cycles_removed.pop()

            if enclosing_cycle not in self.cell_label:
                return

            # Find labelled cycles that have just become connected,
            # These will be all boundary cycles that are connected and have no label
            new_cycle = cycles_added.pop()

            # Newly connected cycles have label to match old enclosing cycle
            self.cell_label[new_cycle] = self.cell_label[enclosing_cycle]

            # Add back any forgotten cycle
            for cycle in self.get_boundary_cycles():
                if is_connected(self.graph, cycle) and cycle not in self.cell_label:
                    self.cell_label[cycle] = self.cell_label[enclosing_cycle]

            # Reset all connected 2-simplices to have no intruder
            for simplex in self.simplices:
                cycle = simplex2cycle(simplex, self.boundary_cycles)
                if cycle not in self.cell_label:
                    continue
                self.cell_label.add_twosimplex(cycle)

            # Delete old boundary cycle
            self.cell_label.delete_cycle(enclosing_cycle)

        # two isolated points connecting
        elif case == (1, 0, 0, 0, 1, 0):
            return
        # two points becomming isolated
        elif case == (0, 1, 0, 0, 0, 1):
            return
        else:
            raise InvalidStateChange(self.state_changes)

        self.evasion_paths += case_name[case] + ", "


if __name__ == "__main__":
    pass
