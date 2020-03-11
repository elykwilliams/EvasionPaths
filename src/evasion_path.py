# Kyle Williams 3/8/20
from brownian_motion import *
from combinatorial_map import *
from gudhi import AlphaComplex
import networkx as nx


def interpolate_points(old_points, new_points, t):
    assert (len(old_points) == len(new_points))
    return [(old_points[n][0] * (1 - t) + new_points[n][0] * t, old_points[n][1] * (1 - t) + new_points[n][1] * t)
            for n in range(len(old_points))]


class InvalidStateChange(Exception):
    def __init__(self, new_e, old_e, new_s, old_s, delta_bcycle):
        self.new_edges = new_e
        self.new_simplices = new_s
        self.removed_edges = old_e
        self.removed_simplices = old_s
        self.boundary_cycle_change = delta_bcycle

    def __str__(self):
        case = (len(self.new_edges)
                , len(self.removed_edges)
                , len(self.new_simplices)
                , len(self.removed_simplices)
                , self.boundary_cycle_change
                )

        return "Invalid State Change:" + str(case) + "\n" \
               + "New edges:" + str(self.new_edges) + "\n" \
               + "Removed edges:" + str(self.removed_edges) + "\n" \
               + "New Simplices:" + str(self.new_simplices) + "\n" \
               + "Removed Simplices:" + str(self.removed_simplices) + "\n" \
               + "Change in Number Boundary Cycles:" + str(self.boundary_cycle_change)


class MaxRecursionDepth(Exception):
    def __init__(self, invalidstatechange):
        self.state = invalidstatechange

    def __str__(self):
        return "Max Recursion depth exceeded \n\n" + str(self.state)


class GraphNotConnected(Exception):
    def __str__(self): return "Graph not connected"


class EvasionPathSimulation:
    def __init__(self, boundary, motion_model, n_sensors, sensing_radius, dt, end_time=0):

        # Initialize Fields
        self.evasion_paths = ""
        self.cell_label = dict()
        self.boundary_cycles = []
        self.old_points = []
        self.old_cycles = []
        self.old_edges = []
        self.old_simplices = []

        self.motion_model = motion_model

        # Parameters
        self.dt = dt
        self.Tend = end_time

        # Internal time keeping
        self.time = 0
        self.n_steps = 0

        # Point data
        self.sensing_radius = sensing_radius
        self.points = generate_points(boundary, n_sensors, self.sensing_radius)
        self.n_sensors = n_sensors + len(boundary)

        self.alpha_shape = list(range(len(boundary)))

        # Generate Combinatorial Map
        alpha_complex = AlphaComplex(self.points)
        simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=self.sensing_radius ** 2)

        self.edges = [tuple(simplex) for simplex, _ in simplex_tree.get_skeleton(1) if len(simplex) == 2]
        self.simplices = [tuple(simplex) for simplex, _ in simplex_tree.get_skeleton(2) if len(simplex) == 3]

        graph = nx.Graph()
        graph.add_nodes_from(range(self.n_sensors))
        graph.add_edges_from(self.edges)

        # Check if graph is connected
        if not nx.is_connected(graph):
            raise GraphNotConnected()

        self.cmap = CMap(graph, self.points)

        # Set initial labeling
        self.boundary_cycles = self.get_boundary_cycles()

        for cycle in self.boundary_cycles:
            self.cell_label[cycle] = True
        for simplex in self.simplices:
            self.cell_label[self.cmap.nodes2cycle(simplex)] = False

        # Update old data
        self.update_old_data()

    def get_boundary_cycles(self):
        return [bc for bc in self.cmap.get_boundary_cycles()
                if bc != self.cmap.nodes2cycle(self.alpha_shape)]

    def update_old_data(self):
        self.old_points = self.points.copy()
        self.old_cycles = self.boundary_cycles.copy()
        self.old_edges = self.edges.copy()
        self.old_simplices = self.simplices.copy()

    def reset_current_data(self):
        self.points = self.old_points.copy()
        self.edges = self.old_edges.copy()
        self.simplices = self.old_simplices.copy()
        self.boundary_cycles = self.old_cycles.copy()

    def run(self):
        if self.Tend != 0:
            while self.time < self.Tend:
                self.time += self.dt
                self.do_timestep()
            return self.time
        else:
            while any(self.cell_label.values()):
                self.time += self.dt
                self.do_timestep()
            return self.time

    def do_timestep(self, new_points=(), level=0):

        t_values = [0.5, 1.0]
        if level == 0:
            t_values = [1.0]

        for t in t_values:

            # Update Points
            if level == 0:
                self.points = self.motion_model.update_points(self.points)
            else:
                self.points = interpolate_points(self.old_points, new_points, t)

            # Update Alpha Complex
            alpha_complex = AlphaComplex(self.points)
            simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=self.sensing_radius ** 2)

            self.edges = [tuple(simplex) for simplex, _ in simplex_tree.get_skeleton(1) if len(simplex) == 2]
            self.simplices = [tuple(simplex) for simplex, _ in simplex_tree.get_skeleton(2) if len(simplex) == 3]

            graph = nx.Graph()
            graph.add_nodes_from(range(self.n_sensors))
            graph.add_edges_from(self.edges)

            # Check if graph is connected
            if not nx.is_connected(graph):
                raise GraphNotConnected()

            self.cmap = CMap(graph, self.points)

            # Update Holes
            self.boundary_cycles = self.get_boundary_cycles()

            # Find Evasion Path
            try:
                self.find_evasion_paths()
                self.n_steps += 1

            except InvalidStateChange as exception:

                if level == 20:
                    raise MaxRecursionDepth(exception)

                # Reset current level to previous step
                self.reset_current_data()
                self.do_timestep(self.points, level=level + 1)

            # Update old data
            self.update_old_data()

    def find_evasion_paths(self):

        def set_difference(list1, list2):
            return set(list1).difference(set(list2))

        edges_added = set_difference(self.edges, self.old_edges)
        edges_removed = set_difference(self.old_edges, self.edges)

        simplices_added = set_difference(self.simplices, self.old_simplices)
        simplices_removed = set_difference(self.old_simplices, self.simplices)

        cycles_added = set_difference(self.boundary_cycles, self.old_cycles)
        cycles_removed = set_difference(self.old_cycles, self.boundary_cycles)

        cycle_change = len(self.boundary_cycles) - len(self.old_cycles)

        case = (len(edges_added), len(edges_removed),
                len(simplices_added), len(simplices_removed),
                len(cycles_added), len(cycles_removed))

        # No Change
        if case == (0, 0, 0, 0, 0, 0):
            self.evasion_paths = "No Change"

        # Add Edge
        elif case == (1, 0, 0, 0, 2, 1):
            self.evasion_paths = "One edge added"

            old_cycle = cycles_removed.pop()

            # Add new boundary cycles to dictionary, they retain the same label as the old cycle
            for cycle in cycles_added:
                self.cell_label[cycle] = self.cell_label[old_cycle]

            # Remove old boundary cycle from dictionary
            del self.cell_label[old_cycle]

        # Remove Edge
        elif case == (0, 1, 0, 0, 1, 2):
            self.evasion_paths = "One edge removed"

            new_cycle = cycles_added.pop()

            # Add new boundary cycles to dictionary, they retain the same label as the old cycle
            self.cell_label[new_cycle] = any([self.cell_label[s] for s in cycles_removed])

            # Remove old boundary cycle from dictionary
            for cycle in cycles_removed:
                del self.cell_label[cycle]

        # Add Simplex
        elif case == (0, 0, 1, 0, 0, 0):
            self.evasion_paths = "One simplex added"

            new_simplex = simplices_added.pop()
            # Find relevant boundary cycle
            cycle = self.cmap.nodes2cycle(new_simplex)

            # Update existing boundary cycle
            self.cell_label[cycle] = False

        # Remove Simplex
        elif case == (0, 0, 0, 1, 0, 0):
            self.evasion_paths = "One simplex removed"
            # No label change needed

        # Edge and Simplex Added
        elif case == (1, 0, 1, 0, 2, 1):
            edge = edges_added.pop()
            simplex = simplices_added.pop()
            if not set(edge).issubset(set(simplex)):
                raise InvalidStateChange(edges_added,
                                         edges_removed,
                                         simplices_added,
                                         simplices_removed,
                                         cycle_change)

            self.evasion_paths = "Edge and Simplex added"

            old_cycle = cycles_removed.pop()
            added_simplex = self.cmap.nodes2cycle(simplex)

            # Add new boundary cycles to dictionary, they retain the same label as the old cycle
            for cycle in cycles_added:
                self.cell_label[cycle] = self.cell_label[old_cycle]

            self.cell_label[added_simplex] = False

            # Remove old boundary cycle from dictionary
            del self.cell_label[old_cycle]

        # Edge and Simplex Removed
        elif case == (0, 1, 0, 1, 1, 2):
            edge = edges_removed.pop()
            simplex = simplices_removed.pop()
            if not set(edge).issubset(set(simplex)):
                raise InvalidStateChange(edges_added,
                                         edges_removed,
                                         simplices_added,
                                         simplices_removed,
                                         cycle_change)

            self.evasion_paths = "Edge and simplex removed"
            new_cycle = cycles_added.pop()

            # Add new boundary cycles to dictionary, they retain the same label as the old cycle
            self.cell_label[new_cycle] = any([self.cell_label[s] for s in cycles_removed])

            # Remove old boundary cycle from dictionary
            for cycle in cycles_removed:
                del self.cell_label[cycle]

        # Delunay Flip
        elif case == (1, 1, 2, 2, 2, 2):
            # Check that edges correspond to correct boundary cycles
            oldedge = edges_removed.pop()
            if not all([set(oldedge).issubset(set(s)) for s in simplices_removed]):
                raise InvalidStateChange(edges_added, edges_removed, simplices_added,
                                         simplices_removed, cycle_change)
            newedge = edges_added.pop()
            if not all([set(newedge).issubset(set(s)) for s in simplices_added]):
                raise InvalidStateChange(edges_added, edges_removed, simplices_added,
                                         simplices_removed, cycle_change)

            if not all([set(s).issubset(set(oldedge).union(set(newedge))) for s in simplices_removed]):
                raise InvalidStateChange(edges_added, edges_removed, simplices_added,
                                         simplices_removed, cycle_change)

            if not all([set(s).issubset(set(oldedge).union(set(newedge))) for s in simplices_added]):
                raise InvalidStateChange(edges_added, edges_removed, simplices_added,
                                         simplices_removed, cycle_change)

            self.evasion_paths = "Delunay Flip"

            # Add new boundary cycles
            for cycle in cycles_added:
                self.cell_label[cycle] = False

            # Remove old boundary cycles
            for cycle in cycles_removed:
                del self.cell_label[cycle]

        else:
            raise InvalidStateChange(edges_added, edges_removed, simplices_added,
                                     simplices_removed, cycle_change)


if __name__ == "__main__":
    pass
