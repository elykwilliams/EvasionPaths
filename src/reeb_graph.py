import networkx as nx

def get_case(holes2: set, holes1: set):
    return len(holes2.difference(holes1)), len(holes1.difference(holes2))


class ReebGraph:
    def __init__(self, holes):
        self.graph = nx.Graph()
        self.edge_stack = {hole: (n, hole) for n, hole in enumerate(holes)}
        for n, holes in enumerate(holes):
            self.graph.add_node(n, pos=(0, n), name="Birth", height=n)

    def insert_new_node(self, time, height, name):
        self.graph.add_node(len(self.graph), pos=(time, height), data=name)
        return len(self.graph)-1

    def insert_new_edge(self, oldID, newId, val):
        color = 'r' if val else 'g'
        self.graph.add_edge(oldID, newId, color=color)

    def update(self, holes):
        pass
