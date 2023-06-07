import unittest

from reeb_graph import *


class TestReebGraph(unittest.TestCase):

    def test_determine_case_death(self):
        hist1 = {"A", "B", "C", "D"}
        hist2 = {"A", "B", "D"}
        assert (0, 1) == get_case(hist2, hist1)

    def test_determine_case_split(self):
        hist1 = {"A", "B", "C", "D"}
        hist2 = {"A", "B", "C", "E", "F"}
        assert (2, 1) == get_case(hist2, hist1)

    def test_init_stack_size(self):
        holes = {"A": True, "B": True}
        assert len(ReebGraph(holes).edge_stack) == 2

    def test_init_graph(self):
        holes = {"A": True, "B": True}
        rg = ReebGraph(holes)
        assert len(rg.graph) == len(holes)

    def test_init_node_data(self):
        holes = {"A": True, "B": True}
        rg = ReebGraph(holes)
        print(rg.graph.nodes.keys())
        assert rg.graph.nodes[0]["name"] == "Birth"
        assert rg.graph.nodes[1]["height"] == 1

    def test_add_new_node(self):
        holes = {"A": True, "B": True}
        rg = ReebGraph(holes)
        nodeId = rg.insert_new_node(0.5, 1, "NoChange")
        assert nodeId == 2
        assert len(rg.graph) == 3

    def test_add_edge(self):
        holes = {"A": True, "B": True}
        rg = ReebGraph(holes)
        oldnodeId, bc_val = rg.edge_stack["B"]
        nodeId = rg.insert_new_node(0.5, rg.graph.nodes[oldnodeId]["pos"][1], "NoChange")
        rg.insert_new_edge(oldnodeId, nodeId, True)
        assert rg.graph.size() == 1

    def test_add_edge_color(self):
        holes = {"A": True, "B": True}
        rg = ReebGraph(holes)
        oldnodeId, bc_val = rg.edge_stack["B"]
        nodeId = rg.insert_new_node(0.5, rg.graph.nodes[oldnodeId]["pos"][1], "NoChange")
        rg.insert_new_edge(oldnodeId, nodeId, True)
        assert rg.graph[nodeId][oldnodeId]['color'] == 'r'




if __name__ == '__main__':
    unittest.main()
