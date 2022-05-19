from itertools import product

import gudhi
import networkx as nx

from boundary_cycles import CMap, share_edge
from utilities import *


class TopologicalState(object):
    def __init__(self, sensor_network):
        alpha_complex = gudhi.AlphaComplex(sensor_network.points)
        simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=sensor_network.sensing_radius ** 2)

        self._simplices = dict()
        for dim in range(4):
            self._simplices[dim] = [simplex for simplex, _ in simplex_tree.get_filtration() if len(simplex) == dim + 1]

        self.cmap = CMap(sensor_network.points, self.simplices(1), self.simplices(2))
        self.boundary_cycles = self.cmap.get_boundary_cycles()

    def simplices(self, dim: int):
        return self._simplices[dim]

    def is_face_connected(self):
        return nx.is_connected(self.face_graph)

    @property
    def face_graph(self):
        graph = nx.Graph()
        graph.add_nodes_from(self.simplices(2))

        for simplex1, simplex2 in product(self.simplices(2), self.simplices(2)):
            if share_edge(simplex1, simplex2):
                graph.add_edge(simplex1, simplex2)
        return graph

    def num_connected_components(self):
        return nx.number_connected_components(self.face_graph)

    def __repr__(self):
        return "\n".join(f"{dim}-simplices: {self._simplices[dim]}" for dim in range(1, 4))


class StateChange:
    valid_cases = [(0, 0, 0, 0, 0, 0, 0, 0)]

    def __init__(self, old_state: TopologicalState, new_state: TopologicalState) -> None:
        self.old_state = old_state
        self.new_state = new_state
        self.edges_added = set_difference(new_state.simplices(1), old_state.simplices(1))
        self.edges_removed = set_difference(old_state.simplices(1), new_state.simplices(1))

        self.simplices_added = set_difference(new_state.simplices(2), old_state.simplices(2))
        self.simplices_removed = set_difference(old_state.simplices(2), new_state.simplices(2))

        self.volumns_added = set_difference(new_state.simplices(3), old_state.simplices(3))
        self.volumns_removed = set_difference(old_state.simplices(3), new_state.simplices(3))

        self.cycles_added = set_difference(new_state.boundary_cycles, old_state.boundary_cycles)
        self.cycles_removed = set_difference(old_state.boundary_cycles, new_state.boundary_cycles)

        self.case = (len(self.edges_added), len(self.edges_removed),
                     len(self.simplices_added), len(self.simplices_removed),
                     len(self.volumns_added), len(self.volumns_removed),
                     len(self.cycles_added), len(self.cycles_removed))

    def is_atomic(self) -> bool:
        return self.case in self.valid_cases

    def is_disconnection(self):
        return self.old_state.num_connected_components() < self.new_state.num_connected_components()

    def is_reconnection(self):
        return self.old_state.num_connected_components() > self.new_state.num_connected_components()

    def __str__(self) -> str:
        disconnect = ""
        # if self.is_disconnection():
        #     disconnect = "DISCONNECTION"
        # elif self.is_reconnection():
        #     disconnect = "RECONNECTION"
        return (
            f"State Change: {self.case}\n"
            f"New edges: {self.edges_added}\n"
            f"Removed edges: {self.edges_removed}\n"
            f"New 2-Simplices: {self.simplices_added}\n"
            f"Removed 2-Simplices: {self.simplices_removed}\n"
            f"New 3-Simplices: {self.volumns_added}\n"
            f"Removed 3-Simplices: {self.volumns_removed}\n"
            f"New cycles {self.cycles_added}\n"
            f"Removed Cycles {self.cycles_removed}\n"
            f"{disconnect}\n"
        )
