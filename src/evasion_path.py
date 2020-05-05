# Kyle Williams 3/8/20
from motion_model import *
from cycle_labelling import *
from combinatorial_map import *
from gudhi import AlphaComplex
import networkx as nx


def interpolate_points(old_points, new_points, t):
    assert (len(old_points) == len(new_points))
    return [(old_points[n][0] * (1 - t) + new_points[n][0] * t, old_points[n][1] * (1 - t) + new_points[n][1] * t)
            for n in range(len(old_points))]


def is_connected(G, cycle):
    """A cycle is connected if its vertices are a subset of those connected to node 0"""
    connected_nodes = nx.node_connected_component(G, 0)
    return set(cycle2nodes(cycle)).issubset(connected_nodes)


def set_difference(list1, list2):
    return set(list1).difference(set(list2))


class InvalidStateChange(Exception):
    def __init__(self, new_e, old_e, new_s, old_s, new_c, old_c):
        self.new_edges = new_e
        self.new_simplices = new_s
        self.new_bcycles = new_c
        self.removed_edges = old_e
        self.removed_simplices = old_s
        self.removed_bcycles = old_c

    def __str__(self):
        case = (len(self.new_edges)
                , len(self.removed_edges)
                , len(self.new_simplices)
                , len(self.removed_simplices)
                , len(self.new_bcycles)
                , len(self.removed_bcycles)
                )

        return "Invalid State Change:" + str(case) + "\n" \
               + "New edges:" + str(self.new_edges) + "\n" \
               + "Removed edges:" + str(self.removed_edges) + "\n" \
               + "New Simplices:" + str(self.new_simplices) + "\n" \
               + "Removed Simplices:" + str(self.removed_simplices) + "\n" \
               + "New cycles" + str(self.new_bcycles) + "\n" \
               + "Removed Cycles" + str(self.removed_bcycles)


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
        self.boundary_cycles = []
        self.old_points = []
        self.old_cycles = []
        self.old_edges = []
        self.old_simplices = []
        self.old_graph = None

        self.motion_model = motion_model

        # Parameters
        self.dt = dt
        self.Tend = end_time

        # Internal time keeping
        self.time = 0
        self.n_steps = 0

        # Point data
        self.sensing_radius = sensing_radius
        self.points = boundary.generate_points(n_sensors)
        self.n_sensors = n_sensors + len(boundary)

        self.alpha_shape = list(range(len(boundary)))

        # Generate Combinatorial Map
        alpha_complex = AlphaComplex(self.points)
        simplex_tree \
            = alpha_complex.create_simplex_tree(max_alpha_square=self.sensing_radius ** 2)

        self.edges \
            = [tuple(simplex) for simplex, _ in simplex_tree.get_skeleton(1) if len(simplex) == 2]
        self.simplices \
            = [tuple(simplex) for simplex, _ in simplex_tree.get_skeleton(2) if len(simplex) == 3]

        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.n_sensors))
        self.graph.add_edges_from(self.edges)

        # Check if graph is connected
        if not nx.is_connected(self.graph):
            raise GraphNotConnected()

        self.cmap = CMap(self.graph, self.points)

        # Set initial labeling
        self.boundary_cycles = self.get_boundary_cycles()

        simplex_cycles = [self.cmap.nodes2cycle(simplex) for simplex in self.simplices]
        self.cell_label = CycleLabelling(self.boundary_cycles, simplex_cycles)

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
        self.old_graph = self.graph.copy()

    def reset_current_data(self):
        self.points = self.old_points.copy()
        self.edges = self.old_edges.copy()
        self.simplices = self.old_simplices.copy()
        self.boundary_cycles = self.old_cycles.copy()
        self.graph = self.old_graph.copy()

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

            # Update Alpha Complex
            alpha_complex = AlphaComplex(self.points)
            simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=self.sensing_radius ** 2)

            self.edges = [tuple(simplex) for simplex, _ in simplex_tree.get_skeleton(1) if len(simplex) == 2]
            self.simplices = [tuple(simplex) for simplex, _ in simplex_tree.get_skeleton(2) if len(simplex) == 3]

            self.graph = nx.Graph()
            self.graph.add_nodes_from(range(self.n_sensors))
            self.graph.add_edges_from(self.edges)

            # Check if graph is connected
            if not nx.is_connected(self.graph):
                print("Graph dissconnected")

            self.cmap = CMap(self.graph, self.points)

            # Update Holes
            self.boundary_cycles = self.get_boundary_cycles()

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

        case = (len(edges_added), len(edges_removed),
                len(simplices_added), len(simplices_removed),
                len(cycles_added), len(cycles_removed))

        current_state = InvalidStateChange(edges_added, edges_removed,
                                           simplices_added, simplices_removed,
                                           cycles_added, cycles_removed)
        # No Change
        if case == (0, 0, 0, 0, 0, 0):
            # self.evasion_paths += "No Change, "
            pass
        # Add Edge
        elif case == (1, 0, 0, 0, 2, 1):

            old_cycle = cycles_removed.pop()

            self.evasion_paths += "1-Simplex added, "

            self.cell_label.add_onesimplex(old_cycle, cycles_added)

        # Remove Edge
        elif case == (0, 1, 0, 0, 1, 2):

            new_cycle = cycles_added.pop()

            self.evasion_paths += "1-Simplex removed, "

            self.cell_label.remove_onesimplex(cycles_removed, new_cycle)

        # Add Simplex
        elif case == (0, 0, 1, 0, 0, 0):

            new_simplex = simplices_added.pop()

            # Find relevant boundary cycle
            new_cycle = self.cmap.nodes2cycle(new_simplex)

            self.evasion_paths += "2-Simplex added, "

            self.cell_label.add_twosimplex(new_cycle)

        # Remove Simplex
        elif case == (0, 0, 0, 1, 0, 0):
            self.evasion_paths += "2-Simplex removed, "
            # No label change needed

        # Edge and Simplex Added
        elif case == (1, 0, 1, 0, 2, 1):

            old_cycle = cycles_removed.pop()
            simplex = simplices_added.pop()
            added_simplex = self.cmap.nodes2cycle(simplex)

            if not set(edges_added.pop()).issubset(set(simplex)):
                raise current_state

            self.evasion_paths += "1-Simplex and 2-Simplex added, "

            self.cell_label.add_one_two_simplex(old_cycle, cycles_added, added_simplex)

        # Edge and Simplex Removed
        elif case == (0, 1, 0, 1, 1, 2):

            simplex = simplices_removed.pop()
            new_cycle = cycles_added.pop()

            if not set(edges_removed.pop()).issubset(set(simplex)):
                raise current_state

            self.evasion_paths += "1-Simplex and 2-Simplex removed, "

            self.cell_label.remove_one_two_simplex(cycles_removed, new_cycle)

        # Delunay Flip
        elif case == (1, 1, 2, 2, 2, 2):
            # Check that edges correspond to correct boundary cycles
            oldedge = edges_removed.pop()
            newedge = edges_added.pop()

            if not all([set(oldedge).issubset(set(s)) for s in simplices_removed]):
                raise current_state
            elif not all([set(newedge).issubset(set(s)) for s in simplices_added]):
                raise current_state
            elif not all([set(s).issubset(set(oldedge).union(set(newedge))) for s in simplices_removed]):
                raise current_state
            elif not all([set(s).issubset(set(oldedge).union(set(newedge))) for s in simplices_added]):
                raise current_state

            self.evasion_paths += "Delaunay Flip, "

            self.cell_label.delaunay_flip(cycles_removed, cycles_added)

        # Disconnect
        elif case == (0, 1, 0, 0, 2, 1) or case == (0, 1, 0, 0, 1, 1):
            pass

        # Reconnect
        elif case == (1, 0, 0, 0, 2, 1) or case == (1, 0, 0, 0, 1, 1):
            raise

        else:
            raise current_state


if __name__ == "__main__":
    pass
