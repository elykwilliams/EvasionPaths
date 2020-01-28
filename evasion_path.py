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
        self.old_cycles = self.boundary_cycles
        self.boundary_cycles = [cycle for cycle in boundary_cycle_graphs(self.cmap) if set(cycle.nodes) != set(self.alpha_shape)]

    def find_evasion_paths(self):
        edges_added = set(self.edges).difference(set(self.old_edges))
        edges_removed = set(self.old_edges).difference(set(self.edges))
        simplices_added = set(self.simplices).difference(set(self.old_simplices))
        simplices_removed = set(self.old_simplices).difference(set(self.simplices))

        cycle_change = len(self.boundary_cycles) - len(self.old_cycles)

        case = (len(edges_added), len(edges_removed), len(simplices_added), len(simplices_removed))

        # No Change
        if case == (0, 0, 0, 0) and cycle_change == 0:
            self.evasion_paths = "No Change"

        # Add Edge
        elif case == (1, 0, 0, 0) and cycle_change == 1:
            self.evasion_paths = "One edge added"
            # Find relevant boundary cycles
            newedge = set(edges_added.pop())
            newcycles = [str(sorted(tuple(s.nodes))) for s in self.boundary_cycles if newedge.issubset(set(s.nodes))]
            oldcycle = [str(sorted(tuple(s.nodes))) for s in self.old_cycles if newedge.issubset(set(s.nodes))].pop()

            # Add new boundary cycles
            for new_s in newcycles:
                self.cell_label[new_s] = self.cell_label[oldcycle]

            # Remove old boundary cycle
            del self.cell_label[oldcycle]

        # Remove Edge
        elif case == (0, 1, 0, 0) and cycle_change == -1:
            self.evasion_paths = "One edge removed"
            # Find relevant boundary cycles
            oldedge = set(edges_removed.pop())
            oldcycles = [str(sorted(tuple(s.nodes))) for s in self.old_cycles if oldedge.issubset(set(s.nodes))]
            newcycle = [str(sorted(tuple(s.nodes))) for s in self.boundary_cycles if oldedge.issubset(set(s.nodes))].pop()

            # Add new boundary cycle
            self.cell_label[newcycle] = any([self.cell_label[s] for s in oldcycles])

            # Remove old boundary cycles
            for s in oldcycles:
                del self.cell_label[s]

        # Add Simplex
        elif case == (0, 0, 1, 0) and cycle_change == 0:  # Add Simplex
            self.evasion_paths = "One simplex added"

            # Find relevant boundary cycle
            newcycle = simplices_added.pop()

            # Update existing boundary cycle
            self.cell_label[str(sorted(tuple(newcycle)))] = False

        # Remove Simplex
        elif case == (0, 0, 0, 1) and cycle_change == 0:
            self.evasion_paths = "One simplex removed"
            # No label change needed

        # Edge and Simplex Added
        elif case == (1, 0, 1, 0) and cycle_change == 1:
            # Check that new edge is an edge of new simplex
            newedge = set(edges_added.pop())
            newsimplex = set(simplices_added.pop())
            if not newedge.issubset(newsimplex):
                raise Exception("Invalid State Change")
            self.evasion_paths = "Edge and Simplex added"

            # Get relevant boundary cycles
            newcycles = [str(sorted(tuple(s.nodes))) for s in self.boundary_cycles if newedge.issubset(set(s.nodes))]
            oldcycle = [str(sorted(tuple(s.nodes))) for s in self.old_cycles if newedge.issubset(set(s.nodes))].pop()

            newsimplex = str(sorted(tuple(newsimplex)))
            newcycles.remove(newsimplex)
            newcycle = newcycles.pop()

            # Add new boundary cycles
            self.cell_label[newsimplex] = False
            self.cell_label[newcycle] = self.cell_label[oldcycle]

            # Remove old boundary cycle
            del self.cell_label[oldcycle]

        # Edge and Simplex Removed
        elif case == (0, 1, 0, 1) and cycle_change == -1:
            # Check that removed edge is subset of removed boundary cycle
            oldedge = set(edges_removed.pop())
            oldsimplex = set(simplices_removed.pop())
            if not oldedge.issubset(oldsimplex):
                raise Exception("Invalid State Change")

            self.evasion_paths = "Edge and simplex removed"
            # Find relevant boundary cycles
            oldcycles = [str(sorted(tuple(s.nodes))) for s in self.old_cycles if oldedge.issubset(set(s.nodes))]
            newcycle = [str(sorted(tuple(s.nodes))) for s in self.boundary_cycles if oldedge.issubset(set(s.nodes))].pop()

            # Add new boundary cycle
            self.cell_label[newcycle] = any([self.cell_label[s] for s in oldcycles])

            # Remove old boundary cycles
            for s in oldcycles:
                del self.cell_label[s]

        # Delunay Flip
        elif case == (1, 1, 2, 2) and cycle_change == 0:
            # Check that edges correspond to correct boundary cycles
            oldedge = set(edges_removed.pop())
            if not all([oldedge.issubset(set(s)) for s in simplices_removed]):
                raise Exception("Invalid State Change")
            newedge = set(edges_added.pop())
            if not all([newedge.issubset(s) for s in simplices_added]):
                raise Exception("Invalid State Change")

            self.evasion_paths = "Delunay Flip"
            # Add new boundary cycles
            for s in simplices_added:
                self.cell_label[str(sorted(tuple(s)))] = False

            # Remove old boundary cycles
            for s in simplices_removed:
                del self.cell_label[str(sorted(tuple(s)))]

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

    def plot(self):
        nx.draw(simplex.G, dict(enumerate(simplex.points)), node_color="g", edge_color="g")
        nx.draw_networkx_labels(simplex.G, dict(enumerate(simplex.points)))

        for s in simplex.boundary_cycles:
            if simplex.is_hole(s):
                nx.draw(s, simplex.points, node_color="r", edge_color="r")

if __name__ == "__main__":
    simplex = EvasionPathSimulation(0.0001, 100)
    for key in simplex.cell_label:
        print(key, simplex.cell_label[key])

    ax = plt.gca()
    fig = plt.figure(1)
    simplex.plot()

    for i in range(0, 3000):

        new_edges = set(simplex.edges).difference(set(simplex.old_edges))
        removed_edges = set(simplex.old_edges).difference(set(simplex.edges))

        simplex.do_timestep()
        if simplex.evasion_paths != "No Change":
            print("Time = ", i)
            print(simplex.evasion_paths)

    for key in simplex.cell_label:
        print(key, simplex.cell_label[key])

    ax = plt.gca()
    fig = plt.figure(2)
    simplex.plot()

        # ax = plt.gca()
        # fig = plt.figure(1)
        # plt.subplot(3, 3, i+ 1, title="iter "+str(i+1)+" "+simplex.evasion_paths)
        # nx.draw(simplex.G, simplex.points)
        # nx.draw_networkx_edges(simplex.G, simplex.points, list(new_edges), edge_color="green", ax=ax, width=3)
        # nx.draw_networkx_edges(simplex.G, simplex.points, list(removed_edges), edge_color="red", ax=ax, width=3)

plt.show()
