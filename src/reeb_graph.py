import networkx as nx


case_name = {
    (0, 0): "Trivial",
    (1, 0): "Birth",
    (0, 1): "Death",
    (1, 1): "No Change",
    (2, 1): "Split",
    (1, 2): "Merge"
}


def get_sym_diff(holes2: set, holes1: set):
    """(added, removed)"""
    return holes2.difference(holes1), holes1.difference(holes2)


def get_case(added: set, removed: set) -> str:
    return case_name[(len(added), len(removed))]


class ReebGraph:
    def __init__(self, holes: dict):
        self.graph = nx.Graph()
        self.stack = dict()
        self.finalized = []

        # Initial update
        for h in holes:
            height = self.get_height(set())
            event_id = self.insert_new_node(0, height, "Init")
            self.stack[h] = (event_id, height)

    def insert_new_node(self, time: float, height: float, name: str):
        event_id = len(self.graph)
        self.graph.add_node(event_id, name=name, pos=(time, height))
        return event_id

    def insert_new_edge(self, oldId: int, newId: int, val: bool) -> None:
        color = 'r' if val else 'g'
        self.graph.add_edge(oldId, newId, color=color)

    def update(self, time, holes_dict2, holes_dict1):
        # Determine event type
        added, removed = get_sym_diff(set(holes_dict2.keys()), set(holes_dict1.keys()))
        name = get_case(added, removed)

        # determine height
        height = self.get_height(removed)

        # insert event node
        nodeId = self.insert_new_node(time, height, name)

        # connect removed boundary cycles
        for h in removed:
            eventId, _ = self.stack[h]
            self.insert_new_edge(eventId, nodeId, holes_dict1[h])
            del self.stack[h]

        # Push new boundary cycles onto the stack
        for hole in sorted(added):
            self.stack[hole] = (nodeId, height)
            height = len(self.stack)

    def get_height(self, removed: set):
        if removed:
            return min(self.stack[hole][1] for hole in removed)
        return len(self.stack)

    def finalize(self, time, holes):
        if self.finalized:
            return

        for h in holes:
            oldId, height = self.stack[h]
            newId = self.insert_new_node(time, height, "final")
            self.insert_new_edge(oldId, newId, holes[h])
            self.finalized.append(newId)

    def resume(self):
        self.graph.remove_nodes_from(self.finalized)
        self.finalized = []
