from unittest import mock

import pytest

from topology import Topology


@pytest.mark.fixture
def BoundaryCycle(name, darts=()):
    bc = mock.MagicMock()
    bc.id = name
    bc.darts = set(darts)
    return bc


class TestTopology:
    alpha_complex = mock.Mock()

    cycleA = BoundaryCycle("A", darts=["ab"])
    cycleB = BoundaryCycle("B", darts=["ba", "bc"])
    cycleC = BoundaryCycle("C", darts=["cb"])

    @pytest.fixture
    def cmap(self):
        map = mock.Mock()

        def get_cycle(dart):
            if dart in self.cycleA.darts:
                return self.cycleA
            elif dart in self.cycleB.darts:
                return self.cycleB
            else:
                return self.cycleC

        map.boundary_cycles = [self.cycleA, self.cycleB, self.cycleC]
        map.alpha.side_effect = lambda dart: dart[-1::-1]
        map.get_cycle.side_effect = get_cycle
        return map

    def test_init(self, cmap):
        topology = Topology(self.alpha_complex, cmap)
        assert topology.alpha_complex is not None and topology.cmap is not None

    def test_2simplices(self, cmap):
        self.alpha_complex.simplices.side_effect = lambda dim: ["A", "B", "C"]
        topology = Topology(self.alpha_complex, cmap)
        assert topology.simplices(2) == ["A", "B", "C"]

    def test_1simplices(self, cmap):
        self.alpha_complex.simplices.side_effect = lambda dim: ["A", "B", "C"]
        topology = Topology(self.alpha_complex, self.cmap)
        assert topology.simplices(1) == ["A", "B", "C"]

    def test_boundary_cycles(self, cmap):
        cmap.boundary_cycles = ["A", "B", "C"]
        topology = Topology(self.alpha_complex, cmap)
        assert topology.boundary_cycles == ["A", "B", "C"]

    # def test_graph_adds_nodes(self, cmap):
    #     topology = ConnectedTopology(self.alpha_complex, cmap)
    #     assert topology._graph.order() == 3
    #
    # def test_graph_has_correct_nodes(self, cmap):
    #     topology = ConnectedTopology(self.alpha_complex, cmap)
    #     assert set(topology._graph.nodes) == {self.cycleA, self.cycleB, self.cycleC}
    #
    # def test_has_edges(self, cmap):
    #     topology = ConnectedTopology(self.alpha_complex, cmap)
    #     assert len(topology._graph.edges) == 2
    #
    # def test_has_correct_edges(self, cmap):
    #     topology = ConnectedTopology(self.alpha_complex, cmap)
    #     assert all(edge in topology._graph.edges for edge
    #                in {(self.cycleA, self.cycleB), (self.cycleB, self.cycleC)})
    #
    # def test_is_not_connected_graph(self, cmap):
    #     cycleD = BoundaryCycle("D")
    #     cmap.boundary_cycles.append(cycleD)
    #     topology = ConnectedTopology(self.alpha_complex, cmap)
    #     assert not topology.is_connected()
