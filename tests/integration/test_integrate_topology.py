from typing import Hashable
from unittest import mock

from update_data import LabelUpdateFactory, RemoveSimplexPairUpdate2D

from state_change import StateChange
from topology import Topology, generate_topology

# import pytest

BoundaryCycle = frozenset


def CombinatorialMap(cycle_dict):
    m = mock.Mock()
    m.dict = cycle_dict
    m.boundary_cycles = m.dict.keys()
    m.alpha.side_effect = lambda dart: dart[-1::-1]
    m.get_cycle.side_effect = lambda dart: next(key for key, value in m.dict.items() if dart in value)
    return m


def Simplex(name, edges=(), nodes=()):
    s = mock.Mock()
    s.to_cycle.return_value = name
    s.is_subface.side_effect = lambda e: True if e in edges or e[-1::-1] in edges else False
    s.nodes = nodes
    return s


def AlphaComplex(simplices, edges):
    ac = mock.Mock()
    ac.simplices.side_effect = lambda dim: simplices if dim == 2 else edges
    ac.dim = 2
    return ac


def test_create_cmap():
    d = {"A": frozenset({"AE", "AB", "AC"}),
         "B": frozenset({"BD", "BA", "BC", "BE"}),
         "C": frozenset({"CB", "CD", "CE"})}
    cmap = CombinatorialMap(d)
    assert isinstance(list(cmap.boundary_cycles)[0], Hashable)


def test_create_alpha_complex():
    simplexC = Simplex("C", edges=("CB", "CD", "CE"))
    edges = ("AD", "AB", "AE", "BD", "BE", "BC", "CE", "CD")
    alpha_complex = AlphaComplex((simplexC,), edges)
    assert alpha_complex.simplices(2)[0].is_subsimplex("BC")


class TestIntegrateTopology:

    def test_init_simplices(self):
        simplexC = Simplex("C", edges=("CB", "CD", "CE"))
        edges = ("AD", "AB", "AE", "BD", "BE", "BC", "CE", "CD")
        alpha_complex = AlphaComplex({simplexC}, edges)

        d = {"A": frozenset({"AE", "AB", "AC"}),
             "B": frozenset({"BD", "BA", "BC", "BE"}),
             "C": frozenset({"CB", "CD", "CE"})}
        cmap = CombinatorialMap(d)
        top = Topology(alpha_complex, cmap)

        assert top.simplices(2) == {simplexC}

    def test_init_cycles(self):
        simplexC = Simplex("C", edges=("CB", "CD", "CE"))
        edges = ("AD", "AB", "AE", "BD", "BE", "BC", "CE", "CD")
        alpha_complex = AlphaComplex({simplexC}, edges)

        d = {"A": frozenset({"AE", "AB", "AC"}),
             "B": frozenset({"BD", "BA", "BC", "BE"}),
             "C": frozenset({"CB", "CD", "CE"})}
        cmap = CombinatorialMap(d)
        top = Topology(alpha_complex, cmap)

        assert cmap.get_cycle("AE") in top.boundary_cycles

    def test_integrate_statechange(self):
        simplexC = Simplex("C", edges=("CB", "CD", "CE"))
        edges = ("AD", "AB", "AE", "BD", "BE", "BC", "CE", "CD")
        alpha_complex = AlphaComplex({simplexC}, edges)

        d = {"A": frozenset({"AE", "AB", "AC"}),
             "B": frozenset({"BD", "BA", "BC", "BE"}),
             "C": frozenset({"CB", "CD", "CE"})}
        cmap = CombinatorialMap(d)
        top1 = Topology(alpha_complex, cmap)

        edges = ("AD", "AB", "AE", "BD", "BE", "CE", "CD")
        alpha_complex = AlphaComplex({}, edges)

        d = {"A": frozenset({"AE", "AB", "AC"}),
             "D": frozenset({"BD", "BA", "BE", "CD", "CE"})}
        cmap = CombinatorialMap(d)
        top2 = Topology(alpha_complex, cmap)
        sc = StateChange(top2, top1)

        assert sc.case == (0, 1, 0, 1, 1, 2)

    def test_integrate_UpdateFactory(self):
        simplexC = Simplex("C", edges=("CB", "CD", "CE"))
        edges = ("AD", "AB", "AE", "BD", "BE", "BC", "CE", "CD")
        alpha_complex = AlphaComplex({simplexC}, edges)

        d = {"A": frozenset({"AE", "AB", "AC"}),
             "B": frozenset({"BD", "BA", "BC", "BE"}),
             "C": frozenset({"CB", "CD", "CE"})}
        cmap = CombinatorialMap(d)
        top1 = Topology(alpha_complex, cmap)

        edges = ("AD", "AB", "AE", "BD", "BE", "CE", "CD")
        alpha_complex = AlphaComplex({}, edges)

        d = {"A": frozenset({"AE", "AB", "AC"}),
             "D": frozenset({"BD", "BA", "BE", "CD", "CE"})}
        cmap = CombinatorialMap(d)
        top2 = Topology(alpha_complex, cmap)

        labelling = {"A": True,
                     "B": True,
                     "C": False}
        assert type(LabelUpdateFactory().get_update(top2, top1, labelling)) == RemoveSimplexPairUpdate2D


def test_generate_topology():
    from alpha_complex import Simplex
    points = [(1.5, 0.5), (1, 1), (0.5, 1.6), (0, 1), (0, 0), (1, 0)]
    radius = 0.5

    topology = generate_topology(points, radius)
    assert topology.simplices_difference(2) == {Simplex({1, 5, 0})}
