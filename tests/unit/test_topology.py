from unittest import mock

import pytest

from topology2 import ConnectedTopology


@pytest.mark.fixture
def BoundaryCycle(name, darts):
    bc = mock.MagicMock()
    bc.id = name
    bc.darts = set(darts)
    return bc


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

    def test_graph_adds_nodes(self):
        alpha_complex = mock.Mock()
        cmap = mock.Mock()

        cycleA = BoundaryCycle("A", darts=["ab"])
        cycleB = BoundaryCycle("B", darts=["ba", "bc"])
        cycleC = BoundaryCycle("C", darts=["cb"])
        cmap.boundary_cycles = [cycleA, cycleB, cycleC]
        cmap.alpha.side_effect = lambda dart: dart[-1::-1]
        cmap.get_cycle.side_effect = [cycleA, cycleB, cycleC] * 2
        topology = ConnectedTopology(alpha_complex, cmap)
        assert topology._graph.order() == 3

    def test_graph_has_correct_nodes(self):
        alpha_complex = mock.Mock()
        cmap = mock.Mock()

        cycleA = BoundaryCycle("A", darts=["ab"])
        cycleB = BoundaryCycle("B", darts=["ba", "bc"])
        cycleC = BoundaryCycle("C", darts=["cb"])

        cmap.boundary_cycles = [cycleA, cycleB, cycleC]
        cmap.alpha.side_effect = lambda dart: dart[-1::-1]
        cmap.get_cycle.side_effect = [cycleA, cycleB, cycleC] * 2

        topology = ConnectedTopology(alpha_complex, cmap)
        assert set(topology._graph.nodes) == {"A", "B", "C"}

    def test_has_edges(self):
        alpha_complex = mock.Mock()
        cmap = mock.Mock()

        cycleA = BoundaryCycle("A", darts=["ab"])
        cycleB = BoundaryCycle("B", darts=["ba", "bc"])
        cycleC = BoundaryCycle("C", darts=["cb"])

        cmap.boundary_cycles = [cycleA, cycleB, cycleC]
        cmap.alpha.side_effect = lambda dart: dart[-1::-1]
        cmap.get_cycle.side_effect = [cycleA, cycleB, cycleC] * 2

        topology = ConnectedTopology(alpha_complex, cmap)
        assert len(topology._graph.edges) == 2

    def test_has_correct_edges(self):
        alpha_complex = mock.Mock()
        cmap = mock.Mock()

        cycleA = BoundaryCycle("A", darts=["ab"])
        cycleB = BoundaryCycle("B", darts=["ba", "bc"])
        cycleC = BoundaryCycle("C", darts=["cb"])

        cmap.boundary_cycles = [cycleA, cycleB, cycleC]
        cmap.alpha.side_effect = lambda dart: dart[-1::-1]

        def get_cycle(dart):
            if dart in cycleA.darts:
                return cycleA
            elif dart in cycleB.darts:
                return cycleB
            else:
                return cycleC

        cmap.get_cycle.side_effect = get_cycle

        topology = ConnectedTopology(alpha_complex, cmap)
        assert all(edge in topology._graph.edges for edge in {(cycleA, cycleB), (cycleB, cycleC)})
