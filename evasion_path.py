# Kyle Williams 12/16/19
from boundary_geometry import Boundary
from brownian_motion import *
from combinatorial_map import *
from numpy import sqrt
from gudhi import AlphaComplex
import networkx as nx


def nodes2str(nodes): return str(sorted(tuple(nodes)))


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
        case = (len(self.new_edges), len(self.removed_edges),
                len(self.new_simplices), len(self.removed_simplices)
                , self.boundary_cycle_change)

        return "Invalid State Change:" + str(case) + "\n"\
               + "New edges:" + str(self.new_edges) + "\n"\
               + "Removed edges:" + str(self.removed_edges) + "\n"\
               + "New Simplices:" + str(self.new_simplices) + "\n"\
               + "Removed Simplices:" + str(self.removed_simplices) + "\n"\
               + "Change in Number Boundary Cycles:" + str(self.boundary_cycle_change)


class MaxRecursionDepth(Exception):
    def __init__(self, invalidstatechange):
        self.state = invalidstatechange

    def __str__(self):
        return "Max Recursion depth exceeded \n\n" + str(self.state)


class EvasionPathSimulation:
    def __init__(self, dt, end_time):
        # Parameters
        self.n_interior_sensors = 25
        self.sensing_radius = 0.15
        self.dt = dt
        self.Tend = end_time
        self.time = 0
        boundary = Boundary(spacing=self.sensing_radius,
                            x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)

        self.brownian_motion = BrownianMotion(dt=self.dt,
                                              sigma=0.01,
                                              sensing_radius=self.sensing_radius,
                                              boundary=boundary)

        # Point data
        self.points = self.brownian_motion.generate_points(n_interior_pts=self.n_interior_sensors)
        self.old_points = self.points.copy()
        self.n_sensors = len(self.points)
        self.alpha_shape = list(range(len(self.brownian_motion.boundary)))

        # Complex info
        alpha_complex = AlphaComplex(self.points)
        simplex_tree = alpha_complex.create_simplex_tree(self.sensing_radius ** 2)
        self.edges = [tuple(simplex) for simplex, _ in simplex_tree.get_skeleton(1) if len(simplex) == 2]
        self.simplices = [tuple(simplex) for simplex, _ in simplex_tree.get_skeleton(2) if len(simplex) == 3]

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
            self.cell_label[nodes2str(bcycle.nodes())] = True
        for simplex in self.simplices:
            self.cell_label[nodes2str(simplex)] = False

        # for key in self.cell_label:
        #     print(key, self.cell_label[key])
        # print("#####")

    def run(self):
        if self.Tend > self.dt:
            while self.time < self.Tend:
                self.time += self.dt
                self.do_timestep()
            return
        else:
            while any(self.cell_label.values()):
                self.time += self.dt
                try:
                    self.do_timestep()
                except Exception as e:
                    # print(self.time, self.dt, self.time/self.dt)
                    # nx.draw(self.G, self.points)
                    # plt.show()
                    # for cycle in self.old_cycles:
                    #     print(cycle.nodes)
                    raise e
            return self.time

    def do_timestep(self):

        # Update Points
        self.points = self.brownian_motion.update_points(self.points)

        # Update Alpha Complex
        alpha_complex = AlphaComplex(self.points)
        simplex_tree = alpha_complex.create_simplex_tree(self.sensing_radius ** 2)

        # Update Graph
        self.edges = [tuple(simplex) for simplex, _ in simplex_tree.get_skeleton(1) if len(simplex) == 2]
        self.simplices = [tuple(simplex) for simplex, _ in simplex_tree.get_skeleton(2) if len(simplex) == 3]

        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n_sensors))  # nodes numbered 0 though N points -1
        self.G.add_edges_from(self.edges)

        # Check if graph is connected
        if not nx.is_connected(self.G):
            raise Exception("Graph is not connected")

        # Update Combinatorial Map
        self.cmap = CMap(self.G, points=self.points)

        # Update Holes
        self.old_cycles = self.boundary_cycles
        self.get_boundary_cycles()

        # Find Evasion Path
        try:
            self.find_evasion_paths()
            # if self.evasion_paths != "No Change":
            #print(self.evasion_paths)
        except InvalidStateChange:
            try:
                self.do_adaptive_step(self.old_points, self.points, rec=1)
            except MaxRecursionDepth as exception:
                raise MaxRecursionDepth(exception)

        # Update old data
        self.old_edges = self.edges.copy()
        self.old_simplices = self.simplices.copy()
        self.old_cycles = self.boundary_cycles.copy()
        self.old_points = self.points.copy()

    def do_adaptive_step(self, old_points, new_points, rec=1):
        #print("Recursion level", rec)
        # Reset current level to previous step
        self.points = self.old_points.copy()
        self.edges = self.old_edges.copy()
        self.simplices = self.old_simplices.copy()
        self.boundary_cycles = self.old_cycles.copy()

        # Then to 100 substeps
        temp_dt = 1 / 2
        for t in np.arange(temp_dt, 1.01, temp_dt):
            #print("t = ", t)
            # Update Points
            self.points = interpolate_points(self.old_points, new_points, t)

            # Update Alpha Complex
            alpha_complex = AlphaComplex(self.points)
            simplex_tree = alpha_complex.create_simplex_tree(self.sensing_radius ** 2)

            # Update Graph
            self.edges = [tuple(simplex) for simplex, _ in simplex_tree.get_skeleton(1) if len(simplex) == 2]
            self.simplices = [tuple(simplex) for simplex, _ in simplex_tree.get_skeleton(2) if len(simplex) == 3]

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
            try:
                self.find_evasion_paths()
                # if self.evasion_paths != "No Change":
                #print(self.evasion_paths)
            except InvalidStateChange as exception:
                if rec > 20:
                    raise MaxRecursionDepth(exception)
                try:
                    self.do_adaptive_step(self.old_points, self.points, rec=rec + 1)
                except MaxRecursionDepth as exception:
                    raise MaxRecursionDepth(exception)


            # Update old data
            self.old_edges = self.edges.copy()
            self.old_simplices = self.simplices.copy()
            self.old_cycles = self.boundary_cycles.copy()
            self.old_points = self.points.copy()

    def get_boundary_cycles(self):
        self.boundary_cycles = [cycle for cycle in boundary_cycle_graphs(self.cmap)
                                if set(cycle.nodes) != set(self.alpha_shape)]

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
            newcycles = [nodes2str(s.nodes) for s in self.boundary_cycles
                         if newedge.issubset(set(s.nodes))]

            oldcycle = [nodes2str(s.nodes) for s in self.old_cycles
                        if newedge.issubset(set(s.nodes))].pop()

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
            oldcycles = [nodes2str(s.nodes) for s in self.old_cycles
                         if oldedge.issubset(set(s.nodes))]

            newcycle = [nodes2str(s.nodes) for s in self.boundary_cycles
                        if oldedge.issubset(set(s.nodes))].pop()

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
            self.cell_label[nodes2str(newcycle)] = False

        # Remove Simplex
        elif case == (0, 0, 0, 1) and cycle_change == 0:
            self.evasion_paths = "One simplex removed"
            # No label change needed

        # Edge and Simplex Added
        elif case == (1, 0, 1, 0) and cycle_change == 1:
            # Check that new edge is an edge of new simplex
            newedge = edges_added.pop()
            newsimplex = simplices_added.pop()
            if not set(newedge).issubset(set(newsimplex)):
                raise InvalidStateChange(edges_added, edges_removed, simplices_added,
                                         simplices_removed, cycle_change)
            self.evasion_paths = "Edge and Simplex added"

            # Get relevant boundary cycles
            newcycles = [nodes2str(s.nodes) for s in self.boundary_cycles
                         if set(newedge).issubset(set(s.nodes))]

            oldcycle = [nodes2str(s.nodes) for s in self.old_cycles
                        if set(newedge).issubset(set(s.nodes))].pop()

            newsimplex = nodes2str(newsimplex)
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
            oldedge = edges_removed.pop()
            oldsimplex = simplices_removed.pop()
            if not set(oldedge).issubset(set(oldsimplex)):
                raise InvalidStateChange(edges_added, edges_removed, simplices_added,
                                         simplices_removed, cycle_change)

            self.evasion_paths = "Edge and simplex removed"
            # Find relevant boundary cycles
            oldcycles = [nodes2str(s.nodes) for s in self.old_cycles
                         if set(oldedge).issubset(set(s.nodes))]

            newcycle = [nodes2str(s.nodes) for s in self.boundary_cycles
                        if set(oldedge).issubset(set(s.nodes))].pop()

            # Add new boundary cycle
            self.cell_label[newcycle] = any([self.cell_label[s] for s in oldcycles])

            # Remove old boundary cycles
            for s in oldcycles:
                del self.cell_label[s]

        # Delunay Flip
        elif case == (1, 1, 2, 2) and cycle_change == 0:
            # Check that edges correspond to correct boundary cycles
            oldedge = edges_removed.pop()
            if not all([set(oldedge).issubset(set(s)) for s in simplices_removed]):
                raise InvalidStateChange(edges_added, edges_removed, simplices_added,
                                         simplices_removed, cycle_change)
            newedge = edges_added.pop()
            if not all([set(newedge).issubset(set(s)) for s in simplices_added]):
                raise InvalidStateChange(edges_added, edges_removed, simplices_added,
                                         simplices_removed, cycle_change)

            self.evasion_paths = "Delunay Flip"
            # Add new boundary cycles
            for s in simplices_added:
                self.cell_label[nodes2str(s)] = False

            # Remove old boundary cycles
            for s in simplices_removed:
                del self.cell_label[nodes2str(s)]

        else:
            raise InvalidStateChange(edges_added, edges_removed, simplices_added,
                                     simplices_removed, cycle_change)

    def is_hole(self, graph):
        for face in self.simplices:
            if set(graph.nodes()) == set(face):
                return False
        if set(self.alpha_shape).issubset(set(graph.nodes())):
            return False
        if graph.order() >= 3:
            return True

    def plot(self, fig, ax):
        ax.clear()
        nx.draw_networkx_labels(self.G, dict(enumerate(self.points)))

        # for s in self.boundary_cycles:
        #     if self.is_hole(s):
        #         nx.draw(s, self.points, node_color="r", edge_color="r")

        for cycle in boundary_cycle_nodes(self.cmap):
            x_pts = [self.points[n][0] for n in cycle]
            y_pts = [self.points[n][1] for n in cycle]
            if set(cycle) == set(self.alpha_shape):
                continue
            if self.cell_label[nodes2str(cycle)]:
                ax.fill(x_pts, y_pts, 'r')
            else:
                ax.fill(x_pts, y_pts, 'g')

        nx.draw(self.G, dict(enumerate(self.points)), node_color="b", edge_color="k")


if __name__ == "__main__":
    time = []
    for _ in range(10):
        simplex = EvasionPathSimulation(0.1, 0)
        # for key in simplex.cell_label:
        #     print(key, simplex.cell_label[key])

        # ax = plt.gca()
        # fig = plt.figure(1)
        # fig.add_axes(ax)
        # simplex.plot(fig, ax)

        try:
            time.append(simplex.run())
        except Exception:
            print("Exception Caught, skipping simulaiton")
        else:
            print(time[-1])

        # for key in simplex.cell_label:
        #     print(key, simplex.cell_label[key])

        # fig = plt.figure(2)
        # ax2 = plt.gca()
        # fig.add_axes(ax2)
        # simplex.plot(fig, ax2)

        # plt.show()
