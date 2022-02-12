from alpha_complex import AlphaComplex, Simplex
from combinatorial_map import RotationInfo2D, CombinatorialMap2D, BoundaryCycle2D
from cycle_labelling import CycleLabellingDict
from state_change import StateChange2D
from topology import ConnectedTopology
from update_data import LabelUpdateFactory, Add2SimplicesUpdate2D


class TestIntegrateAlphaComplex:
    points = [(1.5, 0.5), (1, 1), (0.5, 1.6), (0, 1), (0, 0), (1, 0)]
    radius = 0.5

    def test_init_simplices(self):
        ac = AlphaComplex(self.points, self.radius)
        assert set(ac.simplices(2)) == {Simplex(frozenset({0, 1, 5}))}

    def test_init_edges(self):
        ac = AlphaComplex(self.points, self.radius)
        assert len(ac.simplices(1)) == 8

    def test_integrate_rotinfo(self):
        ac = AlphaComplex(self.points, self.radius)
        ri = RotationInfo2D(self.points, ac)
        adj = {0: [1, 5],
               1: [2, 3, 5, 0],
               2: [3, 1],
               3: [4, 1, 2],
               4: [5, 3],
               5: [4, 0, 1]}
        assert all(set(ri.adj[n]) == set(adj[n]) for n in adj)

    def test_integrate_cmap(self):
        ac = AlphaComplex(self.points, self.radius)
        ri = RotationInfo2D(self.points, ac)
        cmap = CombinatorialMap2D(ri)
        c1 = frozenset({(0, 5), (5, 1), (1, 0)})
        c2 = frozenset({(5, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)})
        c3 = frozenset({(2, 1), (1, 3), (3, 2)})
        c4 = frozenset({(3, 1), (1, 5), (5, 4), (4, 3)})
        assert set(cycle.darts for cycle in cmap.boundary_cycles) == {c1, c2, c3, c4}

    def test_integrate_topology(self):
        ac = AlphaComplex(self.points, self.radius)
        ri = RotationInfo2D(self.points, ac)
        cmap = CombinatorialMap2D(ri)
        topology = ConnectedTopology(ac, cmap)
        assert topology.simplices(2) == {Simplex(frozenset({1, 5, 0}))}

    def test_integrate_labelling(self):
        ac = AlphaComplex(self.points, self.radius)
        ri = RotationInfo2D(self.points, ac)
        cmap = CombinatorialMap2D(ri)
        topology = ConnectedTopology(ac, cmap)
        labelling = CycleLabellingDict(topology)

        c1 = frozenset({(0, 5), (5, 1), (1, 0)})
        c2 = frozenset({(5, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)})
        c3 = frozenset({(2, 1), (1, 3), (3, 2)})
        c4 = frozenset({(3, 1), (1, 5), (5, 4), (4, 3)})
        assert labelling.dict == {BoundaryCycle2D(c1): False,
                                  BoundaryCycle2D(c2): True,
                                  BoundaryCycle2D(c3): True,
                                  BoundaryCycle2D(c4): True}

    def test_integrate_statechange(self):
        ac = AlphaComplex(self.points, self.radius)
        ri = RotationInfo2D(self.points, ac)
        cmap = CombinatorialMap2D(ri)
        topology = ConnectedTopology(ac, cmap)

        points = [(1.5, 0.5), (1, 1), (0.5, 1.5), (0, 1), (0, 0), (1, 0)]
        ac2 = AlphaComplex(points, self.radius)
        ri2 = RotationInfo2D(points, ac2)
        cmap2 = CombinatorialMap2D(ri2)
        topology2 = ConnectedTopology(ac2, cmap2)

        sc = StateChange2D(topology2, topology)
        assert sc.case == (0, 0, 1, 0, 0, 0)

    def test_integrate_lableupdate(self):
        ac = AlphaComplex(self.points, self.radius)
        ri = RotationInfo2D(self.points, ac)
        cmap = CombinatorialMap2D(ri)
        topology = ConnectedTopology(ac, cmap)
        labelling = CycleLabellingDict(topology)

        points = [(1.5, 0.5), (1, 1), (0.5, 1.5), (0, 1), (0, 0), (1, 0)]
        ac2 = AlphaComplex(points, self.radius)
        ri2 = RotationInfo2D(points, ac2)
        cmap2 = CombinatorialMap2D(ri2)
        topology2 = ConnectedTopology(ac2, cmap2)

        sc = StateChange2D(topology2, topology)
        label_update = LabelUpdateFactory().get_update(sc, labelling)
        assert type(label_update) == Add2SimplicesUpdate2D

    def test_integrate_do_update(self):
        ac = AlphaComplex(self.points, self.radius)
        ri = RotationInfo2D(self.points, ac)
        cmap = CombinatorialMap2D(ri)
        topology = ConnectedTopology(ac, cmap)
        labelling = CycleLabellingDict(topology)

        points = [(1.5, 0.5), (1, 1), (0.5, 1.5), (0, 1), (0, 0), (1, 0)]
        ac2 = AlphaComplex(points, self.radius)
        ri2 = RotationInfo2D(points, ac2)
        cmap2 = CombinatorialMap2D(ri2)
        topology2 = ConnectedTopology(ac2, cmap2)

        sc = StateChange2D(topology2, topology)
        label_update = LabelUpdateFactory().get_update(sc, labelling)
        labelling.update(label_update)
        assert not labelling[BoundaryCycle2D(frozenset({(2, 1), (3, 2), (1, 3)}))]
