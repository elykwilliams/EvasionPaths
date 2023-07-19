# ******************************************************************************
#  Copyright (c) 2023, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************

import networkx as nx


def get_sym_diff(set1: set, set2: set):
    """Returns symmetric difference between two sets"""
    return set1.difference(set2), set2.difference(set1)


class ReebGraph:
    """ReebGraph class to handle operations on graph"""

    def __init__(self, holes: dict):
        self.graph = nx.DiGraph()
        self.stack = {}
        self.finalized = []

        self.used_heights = set()

        # Initial update
        for hole in holes:
            height = self._generate_height()
            self.used_heights.add(height)
            event_id = self._insert_new_node(0, height)
            self.stack[hole] = (event_id, height)

    def _insert_new_node(self, time: float, height: float):
        event_id = len(self.graph)
        self.graph.add_node(event_id, pos=(time, height))
        return event_id

    def _insert_new_edge(self, oldId: int, newId: int, val: bool) -> None:
        self.graph.add_edge(oldId, newId, val=val)

    def update(self, time, new_holes_dict, old_holes_dict):
        added, removed = get_sym_diff(set(new_holes_dict.keys()), set(old_holes_dict.keys()))

        if not added and not removed:
            return

        if removed:
            height = min(self.stack[hole][1] for hole in removed)
        else:
            height = self._generate_height()  # This is just the case of a birth

        node_id = self._insert_new_node(time, height)

        # Connect removed boundary cycles
        for hole in removed:
            event_id, _ = self.stack[hole]
            self._insert_new_edge(event_id, node_id, old_holes_dict[hole])
            self.used_heights.remove(self.stack[hole][1])
            del self.stack[hole]

        # Push new boundary cycles onto the stack
        for hole in sorted(added):
            self.stack[hole] = (node_id, height)
            self.used_heights.add(height)
            height = self._generate_height()

    def _generate_height(self):
        h = 0
        while h in self.used_heights:
            h += 1
        return h

    def finalize(self, time, holes):
        if self.finalized:
            return

        for hole in holes:
            old_id, height = self.stack[hole]
            new_id = self._insert_new_node(time, height)
            self._insert_new_edge(old_id, new_id, holes[hole])
            self.finalized.append(new_id)

    def resume(self):
        self.graph.remove_nodes_from(self.finalized)
        self.finalized = []
