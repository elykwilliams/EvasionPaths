import networkx as nx

case_name = {
    (0, 0): "Trivial",
    (1, 0): "Birth",
    (0, 1): "Death",
    (1, 1): "No Change",
    (2, 1): "Split",
    (1, 2): "Merge"
}


def get_sym_diff(set1: set, set2: set):
    """Returns symmetric difference between two sets"""
    return set1.difference(set2), set2.difference(set1)


def get_case(added: set, removed: set) -> str:
    """Returns the case name based on the size of the added and removed sets"""
    return case_name[(len(added), len(removed))]


class ReebGraph:
    """ReebGraph class to handle operations on graph"""

    def __init__(self, holes: dict):
        self.graph = nx.Graph()
        self.stack = dict()
        self.finalized = []

        # Initial update
        for hole in holes:
            height = self._get_height(set())
            event_id = self._insert_new_node(0, height, "Init")
            self.stack[hole] = (event_id, height)

    def _insert_new_node(self, time: float, height: float, name: str):
        event_id = len(self.graph)
        self.graph.add_node(event_id, name=name, pos=(time, height))
        return event_id

    def _insert_new_edge(self, oldId: int, newId: int, val: bool) -> None:
        self.graph.add_edge(oldId, newId, val=val)

    def update(self, time, new_holes_dict, old_holes_dict):
        added, removed = get_sym_diff(set(new_holes_dict.keys()), set(old_holes_dict.keys()))
        name = get_case(added, removed)
        height = self._get_height(removed)
        node_id = self._insert_new_node(time, height, name)

        # Connect removed boundary cycles
        for hole in removed:
            event_id, _ = self.stack[hole]
            self._insert_new_edge(event_id, node_id, old_holes_dict[hole])
            del self.stack[hole]

        # Push new boundary cycles onto the stack
        for hole in sorted(added):
            self.stack[hole] = (node_id, height)
            height = len(self.stack)

    def _get_height(self, removed: set):
        return min(self.stack[hole][1] for hole in removed) if removed else len(self.stack)

    def finalize(self, time, holes):
        if self.finalized:
            return

        for hole in holes:
            old_id, height = self.stack[hole]
            new_id = self._insert_new_node(time, height, "final")
            self._insert_new_edge(old_id, new_id, holes[hole])
            self.finalized.append(new_id)

    def resume(self):
        self.graph.remove_nodes_from(self.finalized)
        self.finalized = []
