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
    def __init__(self):
        pass

    def __str__(self):
        return "Graph not connected"


class EvasionPathSimulation:
    def __init__(self, dt, end_time=0, ):
        # Parameters
        self.n_interior_sensors = 15
        self.sensing_radius = 0.15
        self.dt = dt
        self.Tend = end_time

        boundary = Boundary(spacing=self.sensing_radius)

        points = generate_points(boundary, self.n_interior_sensors, self.sensing_radius)

        self.brownian_motion = BrownianMotion(dt=self.dt,
                                              sigma=0.01,
                                              sensing_radius=self.sensing_radius,
                                              boundary=boundary)
        # Internal time keeping
        self.time = 0
        self.n_steps = 0

        # Point data
        self.points = points
        self.old_points = self.points.copy()
        self.n_sensors = len(self.points)
        self.alpha_shape = list(range(len(boundary)))

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
            return self.time
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
            raise GraphNotConnected()

        # Update Combinatorial Map
        self.cmap = CMap(self.G, points=self.points)

        # Update Holes
        self.old_cycles = self.boundary_cycles
        self.get_boundary_cycles()

        # Find Evasion Path
        try:
            self.find_evasion_paths()
            # if self.evasion_paths != "No Change":
            # print(self.evasion_paths)
            self.n_steps += 1
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
        # print("Recursion level", rec)
        # Reset current level to previous step
        self.points = self.old_points.copy()
        self.edges = self.old_edges.copy()
        self.simplices = self.old_simplices.copy()
        self.boundary_cycles = self.old_cycles.copy()

        # Then to 100 substeps
        temp_dt = 1 / 2
        for t in np.arange(temp_dt, 1.01, temp_dt):
            # print("t = ", t)
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
                raise GraphNotConnected()

            # Update Combinatorial Map
            self.cmap = CMap(self.G, points=self.points)

            # Update Holes
            self.get_boundary_cycles()

            # Find Evasion Path
            try:
                self.find_evasion_paths()
                # if self.evasion_paths != "No Change":
                # print(self.evasion_paths)
                self.n_steps += 1
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

        tempnewcycles = [nodes2str(cycle.nodes()) for cycle in self.boundary_cycles]
        tempoldcycles = [nodes2str(cycle.nodes()) for cycle in self.old_cycles]

        cycles_added = set(tempnewcycles).difference(set(tempoldcycles))
        cycles_removed = set(tempoldcycles).difference(set(tempnewcycles))

        cycle_change = len(self.boundary_cycles) - len(self.old_cycles)

        case = (len(edges_added), len(edges_removed),
                len(simplices_added), len(simplices_removed),
                len(cycles_added), len(cycles_removed))

        # No Change
        if case == (0, 0, 0, 0, 0, 0):
            self.evasion_paths = "No Change"

        # Add Edge
        elif case == (1, 0, 0, 0, 2, 1) or case == (1, 0, 0, 0, 1, 0):
            # Given:
            # edges_added is guaranteed to have length 1, with e = (n1, n2)
            # Number of boundary cycles is guaranteed to have increased by 1 from
            #   1 to 2, or from 0 ro 1

            # Assumption:
            # The above given conditions uniquely determine the case where a single boundary cycle
            #   splits in two.

            # Assumption:
            # It is assumed impossible to add an edge to that two cycles are removed and three
            #   cycles are added

            # Assumption:
            # The new boundary cycles contain n1 and n2
            # The removed boundary cycle contains n1 and n2

            # Corollary:
            # The new boundary cycles are contained in the set of all current boundary cycles that
            #   contain nodes n1 and n2
            # The removed boundary cycle is in the set of all old boundary cycles containing
            #   n1 and n2

            # Observation:
            # It is possible for a new boundary cycle to appear in the old boundary cycles
            # There may be some boundary cycles containing n1 and n2 that appear in the current and old
            #   boundary cycles .

            self.evasion_paths = "One edge added"

            # newedge  = (n1, n2)
            edge = edges_added.pop()

            # Find all current boundary cycles that contain n1 and n2
            relevant_new_cycles = [nodes2str(s.nodes) for s in self.boundary_cycles
                                   if set(edge).issubset(set(s.nodes))]

            # Find all old boundary cycles containing n1 and n2
            relevant_old_cycles = [nodes2str(s.nodes) for s in self.old_cycles
                                   if set(edge).issubset(set(s.nodes))]

            # Filter out unchanged cycles from newcycle
            new_cycles = set(relevant_new_cycles).difference(set(relevant_old_cycles))
            old_cycles = set(relevant_old_cycles).difference(set(relevant_new_cycles))

            # Check for case where cycle is formed in other cycle interior
            if len(old_cycles) == 0 and len(new_cycles) == 1:
                new_cycles = relevant_new_cycles
                old_cycles = relevant_old_cycles

            # Assumption:
            # There are exactly two new boundary cycles, and only one old cycle that was removed

            if len(new_cycles) != 2 or len(old_cycles) != 1:
                print("Old: ", relevant_old_cycles)
                print("New: ", relevant_new_cycles)
                assert (len(new_cycles) == 2 and len(old_cycles) == 1)

            removed_cycle = old_cycles.pop()

            # Add new boundary cycles to dictionary, they retain the same label as the old cycle
            for cycle in new_cycles:
                self.cell_label[cycle] = self.cell_label[removed_cycle]

            # Remove old boundary cycle from dictionary
            if removed_cycle not in new_cycles:
                del self.cell_label[removed_cycle]

        # Remove Edge
        elif case == (0, 1, 0, 0, 1, 2) or case == (0, 1, 0, 0, 0, 1):
            # Given:
            # edges_removed is guaranteed to have length 1, with e = (n1, n2)
            # There are 1 or 2 boundary cycles that have been removed from the graph
            # There is at most 1 new boundary cycle added to alpha complex

            # Assumption:
            # the only way the above guarantees uniquely determine the case where two boundary cycles
            #   merge into one new boundary cycle.

            # Corollary:
            # There will be two boundary cycles removed, and one new boundary cycle added.
            # The new boundary cycle may contain a possible intruder if either of the removed cycles
            #   contained a possible intruder.

            # Assumption:
            # The new boundary cycle contains n1 and n2
            # The old boundary cycles contain n1 and n2
            # It is not possible to

            # Corollary:
            # The new boundary cycle is a in the set of all current boundary cycles that contain nodes
            #   n1 and n2
            # The removed boundary cycles are contained in the set of all old boundary cycles containing
            #   n1 and n2

            # Observation:
            # The new boundary cycle may appear in the old boundary cycles by definition
            # The removed boundary cycles will not appear in the current boundary cycles by definition
            # There may be some boundary cycles containing n1 and n2 that appear in the current and old
            #   boundary cycles.

            self.evasion_paths = "One edge removed"
            # Find relevant boundary cycles

            # edge = (n1, n2)
            edge = edges_removed.pop()

            # Find all current boundary cycles that contain n1 and n2
            relevant_new_cycles = [nodes2str(s.nodes) for s in self.boundary_cycles
                                   if set(edge).issubset(set(s.nodes))]

            # Find all old boundary cycles containing n1 and n2
            relevant_old_cycles = [nodes2str(s.nodes) for s in self.old_cycles
                                   if set(edge).issubset(set(s.nodes))]

            # Filter out unchanged cycles from newcycle
            new_cycles = set(relevant_new_cycles).difference(set(relevant_old_cycles))
            old_cycles = set(relevant_old_cycles).difference(set(relevant_new_cycles))

            # Check for case where cycle is formed in other cycle interior
            if len(old_cycles) == 1 and len(new_cycles) == 0:
                new_cycles = relevant_new_cycles
                old_cycles = relevant_old_cycles

            # Assumption:
            # There are exactly two old boundary cycles, and only one new cycle

            if len(new_cycles) != 1 or len(old_cycles) != 2:
                print("Old: ", relevant_old_cycles)
                print("New: ", relevant_new_cycles)
                assert (len(new_cycles) == 1 and len(old_cycles) == 2)

            added_cycle = new_cycles.pop()

            # Add new boundary cycle to dictionary, label will be true if either old cycle label was true
            self.cell_label[added_cycle] = any([self.cell_label[s] for s in old_cycles])

            # Remove old boundary cycles from dictionary
            for cycle in old_cycles:
                if cycle != added_cycle:
                    del self.cell_label[cycle]

        # Add Simplex
        elif case == (0, 0, 1, 0, 0, 0):
            # Given:
            #   - Exactly one simplex has been added
            #   - There is no change in the number of boundary cycles

            # Assumption:
            # The given conditions uniquely determine the case where a boundary cycle becomes a simplex.
            # The boundary cycle nodes will remain unchanged, and can no longer have a possible intruder.

            # Corollary:
            # The new simplex is in the set of new and old boudnary cycles,

            # Corollary:
            # The new simplex is already in the dictionary

            self.evasion_paths = "One simplex added"

            # Find relevant boundary cycle
            cycle = nodes2str(simplices_added.pop())

            # Update existing boundary cycle
            self.cell_label[cycle] = False

        # Remove Simplex
        elif case == (0, 0, 0, 1, 0, 0) and cycle_change == 0:
            # Given:
            #  - Only one simplex has be removed from the alpha complex
            #  - The number of boundary cycles remains unchanged

            # Assumption:
            # The given conditions uniquely determine the case where a single boundary cycle
            #   goes from being a simplex, to not.
            # The resulting boundary cycle cannot have an intruder.

            # Assumption:
            # The label on a simplex is false

            # Corollary:
            #   The resulting boundary cycle label will also be false, and require no change.

            self.evasion_paths = "One simplex removed"
            # No label change needed

        # Edge and Simplex Added
        elif case == (1, 0, 1, 0, 2, 1) or case == (1, 0, 1, 0, 1, 0):
            # Given:
            #  - There is one new edge in the alpha-complex, e = (n1, n2)
            #  - There is one new simplex in the alpha-complex,
            #  - There number of boundary cycles has increased by one

            # Assumption:
            # If the new edge is an edge of the new simplex, then the above conditions
            #   uniquely determine the case where a boundary cycle is split in two boundary cycles
            #   with one of the new cycles being a simplex.
            # The boundary cycle which is a simplex will have a label of false
            # The other boundary cycle will retain the same label as the removed boundary cycle.

            # Corollary:
            # There case may be treated in the same was as adding an edge. With the exception that the
            #   label corresponding to simplex boundary cycle will be set to false.

            # Check that new edge is an edge of new simplex
            edge = edges_added.pop()
            simplex = simplices_added.pop()
            if not set(edge).issubset(set(simplex)):
                raise InvalidStateChange(edges_added,
                                         edges_removed,
                                         simplices_added,
                                         simplices_removed,
                                         cycle_change)

            self.evasion_paths = "Edge and Simplex added"

            # Given:
            # The added edge is an edge of the simplex.

            # Find all current boundary cycles that contain n1 and n2
            relevant_new_cycles = [nodes2str(s.nodes) for s in self.boundary_cycles
                                   if set(edge).issubset(set(s.nodes))]

            # Find all old boundary cycles containing n1 and n2
            relevant_old_cycles = [nodes2str(s.nodes) for s in self.old_cycles
                                   if set(edge).issubset(set(s.nodes))]

            # Filter out unchanged cycles from newcycle
            new_cycles = set(relevant_new_cycles).difference(set(relevant_old_cycles))
            old_cycles = set(relevant_old_cycles).difference(set(relevant_new_cycles))

            # Check for case where cycle is formed in other cycle interior
            if len(old_cycles) == 0 and len(new_cycles) == 1:
                new_cycles = relevant_new_cycles
                old_cycles = relevant_old_cycles

            # Assumption:
            # There are exactly two new boundary cycles, and only one old cycle that was removed

            if len(new_cycles) != 2 or len(old_cycles) != 1:
                print("Old: ", relevant_old_cycles)
                print("New: ", relevant_new_cycles)
                assert (len(new_cycles) == 2 and len(old_cycles) == 1)

            removed_cycle = old_cycles.pop()
            added_simplex = nodes2str(simplex)

            # Add new boundary cycles to dictionary, they retain the same label as the old cycle
            for cycle in new_cycles:
                self.cell_label[cycle] = self.cell_label[removed_cycle]

            # Set label of simplex as false
            self.cell_label[added_simplex] = False

            # Remove old boundary cycle from dictionary
            if removed_cycle not in new_cycles:
                del self.cell_label[removed_cycle]

        # Edge and Simplex Removed
        elif case == (0, 1, 0, 1, 1, 2) or case == (0, 1, 0, 1, 0, 1):
            # Given:
            #  - There is one removed edge in the alpha complex, e = (n1, n2)
            #  - There is one removed simplex in the alpha-complex,
            #  - There number of boundary cycles has decreased by one

            # Assumption:
            # If the removed edge was an edge of the removed simplex, then the above conditions
            #   uniquely determine the case where a boundary cycle and a simplex merge into one
            #   new boundary cycle.
            # Given that the edge is an edge of the simplex, this case may be treated exactly as
            #   the case where an edge is removed.

            edge = edges_removed.pop()
            simplex = simplices_removed.pop()
            if not set(edge).issubset(set(simplex)):
                raise InvalidStateChange(edges_added,
                                         edges_removed,
                                         simplices_added,
                                         simplices_removed,
                                         cycle_change)

            self.evasion_paths = "Edge and simplex removed"

            # Find all current boundary cycles that contain n1 and n2
            relevant_new_cycles = [nodes2str(s.nodes) for s in self.boundary_cycles
                                   if set(edge).issubset(set(s.nodes))]

            # Find all old boundary cycles containing n1 and n2
            relevant_old_cycles = [nodes2str(s.nodes) for s in self.old_cycles
                                   if set(edge).issubset(set(s.nodes))]

            # Filter out unchanged cycles from newcycle
            new_cycles = set(relevant_new_cycles).difference(set(relevant_old_cycles))
            old_cycles = set(relevant_old_cycles).difference(set(relevant_new_cycles))

            # Check for case where cycle is formed in other cycle interior
            if len(old_cycles) == 1 and len(new_cycles) == 0:
                new_cycles = relevant_new_cycles
                old_cycles = relevant_old_cycles

            # Assumption:
            # There are exactly two old boundary cycles, and only one new cycle

            if len(new_cycles) != 1 or len(old_cycles) != 2:
                print("Old: ", relevant_old_cycles)
                print("New: ", relevant_new_cycles)
                assert (len(new_cycles) == 1 and len(old_cycles) == 2)

            added_cycle = new_cycles.pop()

            # Add new boundary cycle to dictionary, label will be true if either old cycle label was true
            self.cell_label[added_cycle] = any([self.cell_label[s] for s in old_cycles])

            # Remove old boundary cycles from dictionary
            for s in old_cycles:
                if s != added_cycle:
                    del self.cell_label[s]

        # Delunay Flip
        elif case == (1, 1, 2, 2, 2, 2):
            # Given:
            #   - There is one new edge in the alpha-complex (n1, n2)
            #   - There is one edge removed from the alpha-complex (r1, r2)
            #   - There are two new simplices in the alpha complex
            #   - There are two simplices removed from the alpha complex

            # Assumption:
            # If the removed edge, is a mutual edge of the removed simplices,
            # and the added edge is a mutual edge of the added simplices,
            # and the added and removed simplices are subsets of {n1, n2, r1, r2}
            # Then we can uniquely determine a Delauny flip has occurred.

            # The new simplices will be labeled as false.

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
            for cycle in simplices_added:
                self.cell_label[nodes2str(cycle)] = False

            # Remove old boundary cycles
            for cycle in simplices_removed:
                del self.cell_label[nodes2str(cycle)]

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


def plot(Graph, points, fig, ax):
    ax.clear()
    nx.draw_networkx_labels(Graph, dict(enumerate(points)))

    # for s in self.boundary_cycles:
    #     if self.is_hole(s):
    #         nx.draw(s, self.points, node_color="r", edge_color="r")

    # for cycle in boundary_cycle_nodes(self.cmap):
        # x_pts = [points[n][0] for n in cycle]
        # y_pts = [points[n][1] for n in cycle]
        # if set(cycle) == set(self.alpha_shape):
        #    continue
        # if self.cell_label[nodes2str(cycle)]:
        #     ax.fill(x_pts, y_pts, 'r')
        # else:
        #     ax.fill(x_pts, y_pts, 'g')

    nx.draw(Graph, dict(enumerate(points)), node_color="b", edge_color="k")


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
