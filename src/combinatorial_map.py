# ************************************************************
# Copyright (c) 2021, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************

from math import atan2
from networkx import Graph


## Convert and edge to a dart.
# order of vertices is important. To get the dart in the other
# direction swap order of nodes or use alpha function.
def edge2dart(edge: tuple) -> str:
    return ",".join(map(str, edge))


## Get edge corresponding to given dart.
def dart2edge(dart: str) -> tuple:
    return tuple(map(int, dart.split(",")))


## Get node numbers associated with given boundary cycle.
# No specific order is guaranteed.
def cycle2nodes(cycle: tuple) -> list:
    return [int(dart2edge(dart)[0]) for dart in cycle]


## Find boundary cycle associated with given set of nodes.
# WARNING: This function is unsafe to use since the cycle
# associated with a set of nodes may be non-unique. Extra
# checks should be in place. This function will return the
# first cycle that is found with a matching set of nodes.
def nodes2cycle(node_list: list, boundary_cycles: list) -> tuple:
    for cycle in boundary_cycles:
        if set(node_list) == set(cycle2nodes(cycle)):
            return cycle


## This class implements a combinatorial map.
# A 2-dimensional combinatorial map (or 2-map) is a triplet M = (D, σ, α) such that:
#
#     D is a finite set of darts;
#     σ is a permutation on D;
#     α is an involution on D with no fixed point.
#
# Intuitively, a 2-map corresponds to a planar graph where each edge is subdivided into two
# darts (sometimes also called half-edges). The permutation σ gives, for each dart, the next
# dart by turning around the vertex in the positive orientation; the other permutation α gives,
# for each dart, the other dart of the same edge.
#
# α allows one to retrieve edges, and σ allows one to retrieve vertices. We define φ = σ o α
# which gives, for each dart, the next dart of the same face.
class CMap:

    ## Initialize combinatorial map with Graph and Points list.
    # You can optionally initialize with the rotation_data, meaning
    # that for each node, you provide the list of edges from that node
    # to each connected neighbor in counter-clockwise order. Leave
    # points empty if the feature is desired. Otherwise the rotation_data
    # will be computed from the point data.
    def __init__(self, graph: Graph, points: list = (), rotation_data: list = ()) -> None:
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

    ## Get next outgoing dart.
    # For a given outgoing dart, return the next outgoing dart in counter-clockwise
    # order.
    def sigma(self, dart: str) -> str:
        neighbor, node = dart2edge(dart)
        index = self._sorted_darts[node].index(dart)
        n_neigh = len(self._sorted_darts[node])

        # Get next dart, wrap-around if out of range
        return self._sorted_darts[node][(index + 1) % n_neigh]

    ## Get other half edge.
    # for each dart, return the other dart associated with the same edge.
    @staticmethod
    def alpha(dart: str) -> str:
        return edge2dart(tuple(reversed(dart2edge(dart))))

    ## Get next outgoing dart.
    # For a given incoming dart, return the next outgoing dart in counter-clockwise
    # direction.
    def phi(self, dart: str) -> str:
        return self.sigma(self.alpha(dart))

    ## compute boundary cycles.
    # iterate on phi until all darts have been accounted for.
    # This will produce a list of boundary cycles. These cycles
    # are stored internally and should only be accessed through
    # the public member functions.
    def set_boundary_cycles(self) -> None:
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

    ## Get boundary cycle nodes.
    # The function returns the node numbers of each boundary cycle. The nodes
    # are left in cyclic order, used mostly for plotting.
    def boundary_cycle_nodes_ordered(self) -> list:
        return [tuple([dart2edge(dart)[0] for dart in cycle]) for cycle in self._boundary_cycles]

    ## Get Boundary Cycles.
    # This will access the boundary cycles, and return them with each boundary cycle's darts
    # in a sorted order for a unique representation.
    def get_boundary_cycles(self) -> list:
        return [tuple(sorted(cycle)) for cycle in self._boundary_cycles]

    @staticmethod
    def alpha_cycle(domain):
        a = [f"{(n + 1) % len(domain)},{n}" for n in range(len(domain))]
        return tuple(sorted(a))

    ## Remove alpha-cycle
    # the alpha_cycle is the boundary cycle going counter-clockwise around the outside
    # of the domain.
    @staticmethod
    def remove_boundary(domain, boundary_cycles):
        return boundary_cycles.remove(CMap.alpha_cycle(domain))


## Get Rotational Data from points.
# This function is used to compute the rotational data from point data if not explicitly given.
def get_rotational_data(graph, points) -> list:
    sorted_edges = [[] for _ in range(graph.order())]
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))

        # zip neighbors with associated coordinates for sorting
        neighbor_zip = list(zip(neighbors, [points[n] for n in neighbors]))

        anticlockwise, clockwise = False, True

        # sort w.r.t angle from x axis
        def theta(a, center):
            oa = (a[0] - center[0], a[1] - center[1])
            return atan2(oa[1], oa[0])

        # Sort
        sorted_zip = sorted(neighbor_zip,
                            key=lambda pair: theta(pair[1], points[node]),
                            reverse=anticlockwise)

        # Extract sorted edges
        sorted_edges[node] = [(node, n) for (n, _) in sorted_zip]

    return sorted_edges
