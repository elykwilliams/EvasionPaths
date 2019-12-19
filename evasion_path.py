# Kyle Williams 12/16/19

from brownian_motion import *
from combinatorial_map import *
from numpy import sqrt
from gudhi import AlphaComplex
import networkx as nx


class EvasionPathSimulation:
    def __init__(self, dt, end_time):
        # Parameters
        self.n_interior_sensors = 10
        self.sensing_radius = 0.15
        self.dt = dt
        self.Tend = end_time
        self.time = 0
        self.brownian_motion = BrownianMotion(self.dt, sigma=0.1)

        # Point data
        self.points = self.brownian_motion.generate_points(self.n_interior_sensors, self.sensing_radius)
        self.n_sensors = len(self.points)
        self.alpha_shape = list(range(self.brownian_motion.boundary.n_points))

        # Complex info
        self.alpha_complex = AlphaComplex(self.points)
        self.simplex_tree = self.alpha_complex.create_simplex_tree(self.sensing_radius**2)
        self.edges = [tuple(simplex) for simplex, _ in self.simplex_tree.get_skeleton(1) if len(simplex) == 2]
        self.simplices = [tuple(simplex) for simplex, _ in self.simplex_tree.get_skeleton(2) if len(simplex) == 3]

        self.old_edges = self.edges.copy()
        self.old_simplices = self.simplices.copy()

        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n_sensors))  # nodes numbered 0 though N points -1
        self.G.add_edges_from(self.edges)
        self.cmap = CMap(self.G, self.points)

        # Complex Coloring
        self.boundary_cycles = boundary_cycle_graphs(self.cmap)
        self.old_cycles = self.boundary_cycles.copy()

        self.holes = []
        self.cell_coloring = []
        self.evasion_paths = ""

    def run(self):
        if self.Tend > self.dt:
            while self.time < self.Tend:
                self.time += self.dt
                self.do_timestep()
            return bool(self.evasion_paths)
        else:
            while self.evasion_paths:
                self.time += self.dt
                self.do_timestep()
            return self.time

    def do_timestep(self):

        self.old_edges = self.edges.copy()
        self.old_simplices = self.simplices.copy()
        self.old_cycles = self.boundary_cycles.copy()

        # Update Points
        self.points = self.brownian_motion.update_points(self.points)

        # Update Alpha Complex
        self.alpha_complex = AlphaComplex(self.points)
        self.simplex_tree = self.alpha_complex.create_simplex_tree(self.sensing_radius**2)

        # Update Graph
        self.edges = [tuple(simplex) for simplex, _ in self.simplex_tree.get_skeleton(1) if len(simplex) == 2]
        self.simplices = [tuple(simplex) for simplex, _ in self.simplex_tree.get_skeleton(2) if len(simplex) == 3]

        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n_sensors))  # nodes numbered 0 though N points -1
        self.G.add_edges_from(self.edges)

        # Check if graph is connected
        if not nx.is_connected(self.G):
            raise Exception("Graph is not connected")

        # Update Combinatorial Map
        self.cmap = CMap(self.G, points=self.points)

        # Update Holes
        self.find_holes()

        # Find Evasion Path
        self.find_evasion_paths()

    def find_holes(self):
        self.boundary_cycles = boundary_cycle_graphs(self.cmap)

    def find_evasion_paths(self):
        edges_added = set(self.edges).difference(set(self.old_edges))
        edges_removed = set(self.old_edges).difference(set(self.edges))
        simplices_added = set(self.simplices).difference(set(self.old_simplices))
        simplices_removed = set(self.old_simplices).difference(set(self.simplices))

        cycle_change = len(self.boundary_cycles) - len(self.old_cycles)

        case = (len(edges_added), len(edges_removed), len(simplices_added), len(simplices_removed))
        if case == (0, 0, 0, 0) and cycle_change == 0:  # No Change
            self.evasion_paths = "No Change"
        elif case == (1, 0, 0, 0) and cycle_change == 1:  # Add Edge
            self.evasion_paths = "One edge added"
        elif case == (0, 1, 0, 0) and cycle_change == -1:  # Remove Edge
            self.evasion_paths = "One edge removed"
        elif case == (0, 0, 1, 0) and cycle_change == 0:  # Add Simplex
            self.evasion_paths = "One simplex added"
        elif case == (0, 0, 0, 1) and cycle_change == 0:  # Remove Simplex
            self.evasion_paths = "One simplex removed"
        elif case == (1, 0, 1, 0) and cycle_change == 1:  # Edge and Simplex Added
            e = list(edges_added)[0]
            s = list(simplices_added)[0]
            if not set(e).issubset(set(s)):
                raise Exception("Invalid State Change")
            self.evasion_paths = "Edge and Simplex added"
        elif case == (0, 1, 0, 1) and cycle_change == -1:  # Edge and Simplex Removed
            e = list(edges_removed)[0]
            s = list(simplices_removed)[0]
            if not set(e).issubset(set(s)):
                raise Exception("Invalid State Change")
            self.evasion_paths = "Edge and simplex removed"
        elif case == (1, 1, 2, 2) and cycle_change == 0:  # Delunay Flip
            self.evasion_paths = "Delunay Flip"
        elif case == (1, 1, 1, 1) and cycle_change == 0:  # Delunay Flip
            self.evasion_paths = "Delunay Flip"
        elif case == (1, 1, 0, 0) and cycle_change == 0:  # Delunay Flip
            self.evasion_paths = "Delunay Flip"
        else:
            raise Exception("Invalid State Change")

    def is_hole(self, graph):
        for face in self.simplices:
            if set(graph.nodes()) == set(face):
                return False
        if set(self.alpha_shape).issubset(set(graph.nodes())):
            return False
        if graph.order() >= 3:
            return True


if __name__ == "__main__":
    simplex = EvasionPathSimulation(0.00001, 1)

    for i in range(1, 1000):

        new_edges = set(simplex.edges).difference(set(simplex.old_edges))
        removed_edges = set(simplex.old_edges).difference(set(simplex.edges))

        simplex.do_timestep()
        if simplex.evasion_paths != "No Change":
            print("Time = ", i)
            print(simplex.evasion_paths)
            
        if False:
            ax = plt.gca()
            fig = plt.figure(1)
            plt.subplot(3, 3, (i // 10) + 1, title="iter "+str(i+1)+" "+simplex.evasion_paths)
            nx.draw(simplex.G, simplex.points)
            nx.draw_networkx_edges(simplex.G, simplex.points, list(new_edges), edge_color="green", ax=ax, width=3)
            nx.draw_networkx_edges(simplex.G, simplex.points, list(removed_edges), edge_color="red", ax=ax, width=3)

plt.show()
