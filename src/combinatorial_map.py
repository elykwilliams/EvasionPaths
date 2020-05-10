# ************************************************************
# Copyright (c) 2020, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************

from math import atan2


# Sort counter-clockwise w.r.t center
def theta(a, center):
    oa = (a[0] - center[0], a[1] - center[1])
    return atan2(oa[1], oa[0])


def edge2dart(edge):
    return ",".join(map(str, edge))


def dart2edge(dart: str):
    return tuple(map(int, dart.split(",")))


def cycle2nodes(cycle: tuple):
    return [int(dart2edge(dart)[0]) for dart in cycle]


def nodes2cycle(simplex, boundary_cycles):
    for cycle in boundary_cycles:
        if set(simplex) == set(cycle2nodes(cycle)):
            return cycle


class CMap:

    def __init__(self, graph, points=(), rotation_data=()):
        self._sorted_darts = dict()
        self._boundary_cycles = []

        if rotation_data:
            sorted_edges = rotation_data
        else:
            sorted_edges = get_rotational_data(graph, points)

        for node in graph.nodes():
            self._sorted_darts[node] = [edge2dart((e2, e1)) for (e1, e2) in sorted_edges[node]]

        self.darts = [edge2dart((e1, e2)) for e1, e2 in graph.edges]
        self.darts.extend([edge2dart((e2, e1)) for e1, e2 in graph.edges])

        self.set_boundary_cycles()

    def sigma(self, dart):
        neighbor, node = dart2edge(dart)
        index = self._sorted_darts[node].index(dart)
        n_neigh = len(self._sorted_darts[node])

        # Get next dart, wrap-around if out of range
        return self._sorted_darts[node][(index + 1) % n_neigh]

    def alpha(self, dart):
        return edge2dart(reversed(dart2edge(dart)))

    def phi(self, dart):
        return self.sigma(self.alpha(dart))

    def set_boundary_cycles(self):
        """
            This function returns a list of the boundary cycles
            by iterating on phi(x).
            The boundary cycles are given in terms of darts.
            Boundary Cycles form a partition of the darts.
        """
        self._boundary_cycles = []
        all_darts = self.darts.copy()
        
        while all_darts:
            cycle = [all_darts.pop()]
            next_dart = self.phi(cycle[0])

            while next_dart != cycle[0]:
                all_darts.remove(next_dart)
                cycle.append(next_dart)
                next_dart = self.phi(next_dart)

            self._boundary_cycles.append(cycle)

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
    import networkx as nx
    from matplotlib import pyplot as plt
    from math import cos, sin, pi
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
