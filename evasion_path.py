# Kyle Williams 12/16/19

from brownian_motion import *
from combinatorial_map import *
from numpy import sqrt
from gudhi import AlphaComplex
import networkx as nx


class EvasionPathSimulation:
    def __init__(self, dt, end_time):
        # Parameters
        self.n_interior_sensors = 15
        self.sensing_radius = 0.15
        self.dt = dt
        self.Tend = end_time
        self.time = 0
        self.brownian_motion = BrownianMotion(self.dt, sigma=0.01)

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
        self.boundary_cycles = []
        self.get_boundary_cycles()
        self.old_cycles = self.boundary_cycles.copy()

        self.holes = []
        self.evasion_paths = ""
        self.cell_label = dict()
        for bcycle in self.boundary_cycles:
            self.cell_label[str(sorted(tuple(bcycle.nodes())))] = True
        for simplex in self.simplices:
            self.cell_label[str(sorted(tuple(simplex)))] = False


    def run(self):
        if self.Tend > self.dt:
            while self.time < self.Tend:
                self.time += self.dt
                try:
                    self.do_timestep()
                except Exception:
                    raise Exception("Invalid state change at time " + str(self.time))
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
        self.get_boundary_cycles()

        # Find Evasion Path
        self.find_evasion_paths()

    def get_boundary_cycles(self):
        self.boundary_cycles = [cycle for cycle in boundary_cycle_graphs(self.cmap) if set(cycle.nodes) != set(self.alpha_shape)]

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
            e = set(list(edges_removed)[0])
            if not all([e.issubset(set(s)) for s in list(simplices_removed)]):
                raise Exception("Invalid State Change")
            e = set(list(edges_added)[0])
            if not all([e.issubset(set(s)) for s in list(simplices_added)]):
                raise Exception("Invalid State Change")
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

    def plot(self, ax, fig):
        nx.draw(simplex.G, dict(enumerate(simplex.points)), node_color="g", edge_color="g")
        nx.draw_networkx_labels(simplex.G, dict(enumerate(simplex.points)))

        for s in simplex.boundary_cycles:
            if simplex.is_hole(s):
                nx.draw(s, simplex.points, node_color="r", edge_color="r")

        plt.show()

if __name__ == "__main__":
    simplex = EvasionPathSimulation(0.0001, 100)
    print(simplex.cell_label)

    ax = plt.gca()
    fig = plt.figure(1)
    simplex.plot(ax, fig)

    for i in range(0, 0):

        new_edges = set(simplex.edges).difference(set(simplex.old_edges))
        removed_edges = set(simplex.old_edges).difference(set(simplex.edges))

        simplex.do_timestep()
        if simplex.evasion_paths != "No Change":
            print("Time = ", i)
            print(simplex.evasion_paths)


        # ax = plt.gca()
        # fig = plt.figure(1)
        # plt.subplot(3, 3, i+ 1, title="iter "+str(i+1)+" "+simplex.evasion_paths)
        # nx.draw(simplex.G, simplex.points)
        # nx.draw_networkx_edges(simplex.G, simplex.points, list(new_edges), edge_color="green", ax=ax, width=3)
        # nx.draw_networkx_edges(simplex.G, simplex.points, list(removed_edges), edge_color="red", ax=ax, width=3)

plt.show()
