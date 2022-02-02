from unittest import mock

import networkx as nx

from topology2 import ConnectedTopology


class TestTopology:

    def test_init(self):
        alpha_complex = mock.Mock()
        cmap = mock.Mock()
        topology = ConnectedTopology(alpha_complex, cmap)
        assert topology.alpha_complex is not None and topology.cmap is not None

    def test_2simplices(self):
        alpha_complex = mock.Mock()
        alpha_complex.simplices.side_effect = lambda dim: ["A", "B", "C"]
        cmap = mock.Mock()
        topology = ConnectedTopology(alpha_complex, cmap)
        assert topology.simplices(2) == ["A", "B", "C"]

    def test_1simplices(self):
        alpha_complex = mock.Mock()
        alpha_complex.simplices.side_effect = lambda dim: ["A", "B", "C"]
        cmap = mock.Mock()
        topology = ConnectedTopology(alpha_complex, cmap)
        assert topology.simplices(1) == ["A", "B", "C"]

    def test_boundary_cycles(self):
        alpha_complex = mock.Mock()
        cmap = mock.Mock()
        cmap.boundary_cycles = ["A", "B", "C"]
        topology = ConnectedTopology(alpha_complex, cmap)
        assert topology.boundary_cycles == ["A", "B", "C"]

    def test_graph_nodes(self):
        graph = nx.Graph()
        graph.add_nodes_from(["A", "B", "C"])
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        alpha_complex = mock.Mock()
        cmap = mock.Mock()
        cmap.boundary_cycles.return_value = ["A", "B", "C"]
        topology = ConnectedTopology(alpha_complex, cmap)
        assert topology._graph.nodes == graph.nodes
