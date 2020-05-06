# Kyle Williams 2/25/20

# Note that networkx does support planar-graph
# https://networkx.github.io/documentation/stable/reference/algorithms/planarity.html
# However, it appears that construction is more manual,and that rotation information is
# stored as nodes/edges are added to the graph.

import networkx as nx
from matplotlib import pyplot as plt
from math import *


# Sort counter-clockwise w.r.t center (Thanks to Nick Closuit for idea)
def theta(a, center):
    oa = (a[0] - center[0], a[1] - center[1])
    return atan2(oa[1], oa[0])


def edge2dart(edge: tuple):
    return ",".join(map(str, edge))


def dart2edge(dart: str):
    return tuple(map(int, dart.split(",")))


def cycle2nodes(cycle: tuple):
    return [int(dart2edge(dart)[0]) for dart in cycle]


def simplex2cycle(simplex, boudnary_cycles):
    for cycle in boudnary_cycles:
        if set(simplex) == set(cycle2nodes(cycle)):
            return cycle


class CMap:

    def nodes2cycle(self, nodes):
        index = self._boundary_cycle_nodes_unique.index(set(nodes))
        return tuple(sorted(self._boundary_cycles[index]))

    def __init__(self, graph, points=(), rotation_data=()):
        self._sorted_darts = dict()
        self._boundary_cycles = []
        self._boundary_cycle_nodes_unique = []

        # Get rotational information
        if rotation_data:
            sorted_edges = rotation_data
        else:
            sorted_edges = get_rotational_data(graph, points)

        for node in graph.nodes():
            self._sorted_darts[node] = [edge2dart((e2, e1)) for (e1, e2) in sorted_edges[node]]

        self.darts = [edge2dart((e1, e2)) for e1, e2 in graph.edges]
        self.darts.extend([edge2dart((e2, e1)) for e1, e2 in graph.edges])

        self.G = graph          # only used for plotting
        self.points = points    # only used for plotting

        self.set_boundary_cycles()

    def sigma(self, dart):
        # Get node
        neighbor, node = dart2edge(dart)

        # Get index of given dart
        index = self._sorted_darts[node].index(dart)

        # Number of neighbors
        size = len(self._sorted_darts[node])

        # Get next dart, wrap-around if out of range
        return self._sorted_darts[node][(index + 1) % size]

    def alpha(self, dart):
        # get corresponding edge nodes
        e1, e2 = dart2edge(dart)

        # swap edge nodes to get corresponding dart
        return edge2dart((e2, e1))

    def phi(self, dart):
        return self.sigma(self.alpha(dart))

    def plot(self):
        temp_dict = {edge: edge2dart(edge) for edge in self.G.edges}
        temp_dict.update({reversed(edge): edge2dart(reversed(edge)) for edge in self.G.edges})
        nx.draw_networkx_labels(self.G, dict(enumerate(self.points)))
        nx.draw_networkx_edge_labels(self.G, dict(enumerate(self.points)), temp_dict, label_pos=0.15)
        nx.draw(self.G, self.points)
        plt.show()

    def set_boundary_cycles(self):
        """
            This function returns a list of the boundary cycles
            by iterating on phi(x).
            The boundary cycles are given in terms of darts.
            Boundary Cycles form a partition of the darts.
            Based on implementation by Deepjoyti Ghosh.
        """
        output = []
        all_darts = self.darts.copy()
        
        while all_darts:
            # set root
            cycle = [all_darts.pop()]

            # get next in cycle
            next_dart = self.phi(cycle[0])

            while next_dart != cycle[0]:

                # remove dart, since disjoint
                all_darts.remove(next_dart)

                # add to cycle
                cycle.append(next_dart)

                # get next dart
                next_dart = self.phi(next_dart)

            # cycle finished when next_dart is root
            output.append(cycle)

        self._boundary_cycles = output

        self._boundary_cycle_nodes_unique \
            = [set([dart2edge(dart)[0] for dart in cycle]) for cycle in output]

    def boundary_cycle_nodes_ordered(self):
        return [tuple([dart2edge(dart)[0] for dart in cycle]) for cycle in self._boundary_cycles]

    def get_boundary_cycles(self):
        return [tuple(sorted(cycle)) for cycle in self._boundary_cycles]


def get_rotational_data(graph, points):
    sorted_edges = [[] for _ in range(graph.order())]
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))

        # zip neighbors with associated coordinates for sorting
        neighbor_zip = list(zip(neighbors, [points[n] for n in neighbors]))

        anticlockwise, clockwise = False, True

        # Sort
        sorted_zip = sorted(neighbor_zip,
                            key=lambda pair: theta(pair[1], points[node]),
                            reverse=anticlockwise)

        # Extract sorted edges
        sorted_edges[node] = [(node, n) for (n, _) in sorted_zip]

    return sorted_edges


if __name__ == "__main__":
    # G = nx.house_x_graph()
    # G.remove_edge(0, 1)
    # G.remove_edge(2, 4)
    # G.remove_edge(1, 3)
    # G.add_edge(1, 0)
    # G.add_edge(4, 0)

    old_G = nx.Graph()
    old_G.add_nodes_from(range(8))
    old_G.add_edge(0, 1)
    old_G.add_edge(1, 2)
    old_G.add_edge(2, 3)
    old_G.add_edge(3, 0)
    old_G.add_edge(4, 5)
    old_G.add_edge(5, 6)
    old_G.add_edge(6, 7)
    old_G.add_edge(7, 4)
    #G.add_edge(0, 4)

    #print(len(G[0]))
    # G = nx.gnm_random_graph(10, 15)
    mypoints = [(cos(theta), sin(theta)) for theta in [2*pi*n/4 for n in range(4)]]
    mypoints += [(0.5*cos(theta), 0.5*sin(theta)) for theta in [2*pi*n/4 for n in range(4)]]



    nx.draw(old_G, mypoints)
    nx.draw_networkx_labels(old_G, dict(enumerate(mypoints)))
    plt.show()

    c_map = CMap(old_G, mypoints)

    c_map.plot()
    plt.show()


    print("Cmap internal boudnary cycle")
    print(c_map._boundary_cycles)
    print("\nEvasion Path sorted boundary cycles")
    print(c_map.get_boundary_cycles())
    print("\nget boundary cycle nodes")
    print(c_map.boundary_cycle_nodes_ordered())
    print("\nget boundary cycles from nodes")
    print(c_map.nodes2cycle([0, 2, 1, 3]))

    # boundary = [1, 2, 3, 4, 0]
    #
    # print([set(boundary) == set(G.nodes()) for G in simplices])

    # for i, simplex in enumerate(simplices, start=1):
    #     plot_num = str(2)+str(len(simplices)//2)+str(i)
    #     plt.subplot(int(plot_num))
    #     nx.draw_networkx_nodes(G, dict(enumerate(mypoints)), node_color="b")
    #     nx.draw_networkx_labels(G, dict(enumerate(mypoints)))
    #     nx.draw(simplex, mypoints, node_color="r")

    # plt.show()
