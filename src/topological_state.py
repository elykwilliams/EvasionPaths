# ************************************************************
# Copyright (c) 2020, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************

from combinatorial_map import *
from gudhi import AlphaComplex
import networkx as nx


def set_difference(list1, list2):
    return list(set(list1).difference(set(list2)))


def is_subset(list1, list2):
    return set(list1).issubset(set(list2))


## The Topological State is the class used to encapsulate the simplicial, and combinatorial
# information. For a given set of points (and parameters), this class will provide access to the
# simplices of the alpha complex and the boundary cycles of the combinatorial map, as well as some
# minimal connectivity information about the underlying graph.
class TopologicalState(object):

    ## Compute Alpha-complex and combinatorial map and extract simplices and boundary cycles. Also
    # save connectivity information.
    def __init__(self, points, sensing_radius, boundary):
        alpha_complex = AlphaComplex(points)
        simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=sensing_radius ** 2)

        self._simplices = [[], [], []]
        self._simplices[0] = [simplex[0] for simplex, _ in simplex_tree.get_skeleton(0)]
        self._simplices[1] = [tuple(simplex) for simplex, _ in simplex_tree.get_skeleton(1) if len(simplex) == 2]
        self._simplices[2] = [tuple(simplex) for simplex, _ in simplex_tree.get_skeleton(2) if len(simplex) == 3]

        graph = nx.Graph()
        graph.add_nodes_from(self._simplices[0])
        graph.add_edges_from(self._simplices[1])

        self._boundary_cycles = CMap(graph, points).get_boundary_cycles()
        self._boundary_cycles.remove(boundary.alpha_cycle)

        self._connected_nodes = nx.node_connected_component(graph, 0)

    ## Check if graph is connected.
    # This is used for flagging when the graph has become disconnected.
    def is_connected(self):
        graph = nx.Graph()
        graph.add_nodes_from(self._simplices[0])
        graph.add_edges_from(self._simplices[1])
        return nx.is_connected(graph)

    ## Check if a boundary cycle is connected to the fence. This is done by and
    # comparing nodes of the boundary cycle to the set of all nodes connected to
    # node #0 (which is guaranteed to be on the fence).
    def is_connected_cycle(self, cycle):
        return not set(cycle2nodes(cycle)).isdisjoint(set(self._connected_nodes))

    ## Check if a simplex is connected to the fence. This is done by and
    # comparing nodes of the boundary cycle to the set of all nodes connected to
    # node #0 (which is guaranteed to be on the fence).
    def is_connected_simplex(self, simplex):
        return not set(simplex).isdisjoint(set(self._connected_nodes))

    ## Access the AlphaComplex's simplices of a given dimension. 0-Simplices will be a list of node numbers, the others
    # will be a list of tuples. The tuples will contain the node numbers of the simplex.
    def simplices(self, dim: int):
        return self._simplices[dim]

    ## Access CombinatorialMap's boundary cycles. Will be returned as a list of boundary cycle with
    # the boundary cycle of the fence removed. See CMap for details on boundary cycle structure (though
    # it really shouldn't matter)
    def boundary_cycles(self):
        return self._boundary_cycles

    ## Find the cycle with the same nodes as a given 2-simplex.
    # WARNING: Your cycle must satisfy the following conditions
    #
    #       1. Must be a cycle of length 3
    #       2. Must be connect to the fence
    #
    # If these conditions are not met, a unique representation cannot be guaranteed and
    # this function will raise an error.
    #
    # Note: This function is necessary because there is no embedding information of the
    # alpha complex into the combinatorial map. For example, if a 2-simplex is added, but
    # no boundary cycles are changed, we have no other was of identifying which boundary
    # cycle label should be updated.
    def simplex2cycle(self, simplex):
        if len(simplex) != 3 or not self.is_connected_simplex(simplex):
            raise ValueError("Invalid simplex, cannot guarantee unique cycle")
        return nodes2cycle(simplex, self._boundary_cycles)


## This class is used to determine and represent the differences between two states.
# This means determining which simplices have been added or removed as well as which
# boundary cycles have been added or removed.
#
# A set of nodes is an added simplex when the nodes
# formed a simplex of the new state but do not form a simplex of the old state Similarly,
# a set of nodes is a removed simplex when the nodes formed a simplex of the old state but
# do not form a simplex of the new state (And similarly for boundary cycles).
#
# This class also has the ability to determine when an atomic state change has occurred.
# A a state change is one of the following
#
#       1. A 1-simplex is added or removed
#       2. A 2-simplex is added or removed
#       3. A free pair consisting of a 2-simplex and a 1-simplex is added
#       4. A delaunay flip occurred
#
# Disconnections and re-connections are simply a special case of removing/adding a 1-Simplex.
#
# The state transitions are identified by simply counting the number of simplices and boundary
# cycles that have been added or removed. With some minimal compatibility checking, this can
# uniquely identify an atomic transition.
class StateChange(object):
    ## Identify Atomic States
    #
    # (#1-simplices added, #1-simpleices removed, #2-simplices added, #2-simplices removed, #boundary cycles added,
    # #boundary cycles removed)
    case2name = {
        (0, 0, 0, 0, 0, 0): "",
        (1, 0, 0, 0, 2, 1): "Add 1-Simplex",
        (1, 0, 0, 0, 1, 0): "Add 1-Simplex",
        (0, 1, 0, 0, 1, 2): "Remove 1-Simplex",
        (0, 1, 0, 0, 0, 1): "Remove 1-Simplex",
        (0, 0, 1, 0, 0, 0): "Add 2-Simplex",
        (0, 0, 0, 1, 0, 0): "Remove 2-Simplex",
        (1, 0, 1, 0, 2, 1): "Add 1-Simplex and 2-Simplex",
        (0, 1, 0, 1, 1, 2): "Remove 1-Simplex and 2-Simplex",
        (1, 1, 2, 2, 2, 2): "Delauney Flip",
        (0, 1, 0, 0, 2, 1): "Disconnect",
        (0, 1, 0, 0, 1, 1): "Disconnect",
        (1, 0, 0, 0, 1, 2): "Reconnect",
        (1, 0, 0, 0, 1, 1): "Reconnect"
    }

    def __init__(self, old_state: TopologicalState, new_state: TopologicalState) -> None:
        self.new_state = new_state
        self.edges_added = set_difference(new_state.simplices(1), old_state.simplices(1))
        self.edges_removed = set_difference(old_state.simplices(1), new_state.simplices(1))

        self.simplices_added = set_difference(new_state.simplices(2), old_state.simplices(2))
        self.simplices_removed = set_difference(old_state.simplices(2), new_state.simplices(2))

        self.cycles_added = set_difference(new_state.boundary_cycles(), old_state.boundary_cycles())
        self.cycles_removed = set_difference(old_state.boundary_cycles(), new_state.boundary_cycles())

        self.case = (len(self.edges_added), len(self.edges_removed), len(self.simplices_added),
                     len(self.simplices_removed), len(self.cycles_added), len(self.cycles_removed))

    ## Determine if the current state transition is atomic.
    # A transition is considered non-atomic if on of the following are true:
    #
    #       1. The case is not found in the list of possible transitions
    #       2. If a 1-simplex and 2-simplex are added/removed simultaniously, the 1-simplex
    #           should be an edge of the 2-simplex
    #       3. If a delaunay flip appears to have occurred, the removed 1-simplex should be
    #           an edge of both removed 2-simplices; similarly the added 1-simplex should be
    #           an edge of the added 2-simplices.
    #       4. The set of vertices of the 1-simplices should contain the vertices of each 2-simplex
    #           that is added or removed.
    #
    def is_atomic(self) -> bool:
        if self.case not in self.case2name.keys():
            return False
        elif self.case == (1, 0, 1, 0, 2, 1):
            simplex = self.simplices_added[0]
            edge = self.edges_added[0]
            if not is_subset(edge, simplex):
                return False
        elif self.case == (0, 1, 0, 1, 1, 2):
            simplex = self.simplices_removed[0]
            edge = self.edges_removed[0]
            if not is_subset(edge, simplex):
                return False
        elif self.case == (1, 1, 2, 2, 2, 2):
            old_edge = self.edges_removed[0]
            new_edge = self.edges_added[0]
            if not all([is_subset(old_edge, s) for s in self.simplices_removed]):
                return False
            elif not all([is_subset(new_edge, s) for s in self.simplices_added]):
                return False

            nodes = set(old_edge).union(set(new_edge))
            if not all([is_subset(s, nodes) for s in self.simplices_removed]):
                return False
            elif not all([is_subset(s, nodes) for s in self.simplices_added]):
                return False
        return True

    ## Get name of transition.
    # If non-atomic, return "Invalid Case"
    def get_name(self) -> str:
        if self.is_atomic():
            return self.case2name[self.case]
        else:
            return "Invalid Case"

    ## Allow class to be printable.
    # Used mostly for debugging
    def __str__(self) -> str:
        return "State Change:" + str(self.case) + "\n" \
            + "New edges:" + str(self.edges_added) + "\n" \
            + "Removed edges:" + str(self.edges_removed) + "\n" \
            + "New Simplices:" + str(self.simplices_added) + "\n" \
            + "Removed Simplices:" + str(self.simplices_removed) + "\n" \
            + "New cycles" + str(self.cycles_added) + "\n" \
            + "Removed Cycles" + str(self.cycles_removed)


## Exception indicating non-atomic state change.
# This exception should be raised when a function that requires an atomic change
# is given a non-atomic change.
class InvalidStateChange(Exception):
    def __init__(self, state_change):
        self.state_change = state_change

    def __str__(self) -> str:
        return "Invalid State Change \n\n" \
               + str(self.state_change)

