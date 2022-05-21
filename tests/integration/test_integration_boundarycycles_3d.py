from unittest import mock

import numpy as np

from alpha_complex import Simplex
from combinatorial_map import CombinatorialMap3D, OrientedSimplex, BoundaryCycle
from topology import Topology


def test_3dcmap_bcycle_lists():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)]
    triangles = [(0, 1, 2), (0, 2, 3), (0, 1, 4), (0, 3, 4), (1, 2, 4), (2, 3, 4)]
    cmap = CombinatorialMap3D(points, edges, triangles)

    assert len(cmap.boundary_cycles) == 2

def test_3dcmap_lookup_bcycle():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)]
    triangles = [(0, 1, 2), (0, 2, 3), (0, 1, 4), (0, 3, 4), (1, 2, 4), (2, 3, 4), (0, 2, 4)]
    cmap = CombinatorialMap3D(points, edges, triangles)

    oriented_simplex = OrientedSimplex((0, 2, 1))
    assert cmap.get_cycle(oriented_simplex).nodes == {0, 1, 2, 4}


def mock_alphacomplex(edges, faces, tets):
    simplices_dict = {1: edges, 2:faces, 3:tets}
    ac = mock.Mock()
    ac.simplices.side_effect = \
        lambda dim: [Simplex(frozenset(item)) for item in simplices_dict[dim]]
    ac.nodes = {0, 1, 2, 3, 4, 5}
    ac.dim = 3
    return ac

def test_3d_topology_cmap():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)]
    triangles = [(0, 1, 2), (0, 2, 3), (0, 1, 4), (0, 3, 4), (1, 2, 4), (2, 3, 4), (0, 2, 4)]
    tetrahedrons = [(0, 1, 2, 4)]

    ac = mock_alphacomplex(edges, triangles, tetrahedrons)
    cmap = CombinatorialMap3D(points, ac.simplices(1), ac.simplices(2))
    topology = Topology(ac, cmap)

    bcycle1 = cmap.get_cycle(OrientedSimplex((0, 1, 2)))
    bcycle2 = cmap.get_cycle(OrientedSimplex((0, 2, 1)))
    bcycle3 = cmap.get_cycle(OrientedSimplex((0, 3, 2)))

    assert topology.boundary_cycles == {bcycle1, bcycle2, bcycle3}


def test_3d_topology_cmap_alphacycle():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)]
    triangles = [(0, 1, 2), (0, 2, 3), (0, 1, 4), (0, 3, 4), (1, 2, 4), (2, 3, 4), (0, 2, 4)]
    tetrahedrons = [(0, 1, 2, 4)]

    ac = mock_alphacomplex(edges, triangles, tetrahedrons)
    cmap = CombinatorialMap3D(points, ac.simplices(1), ac.simplices(2))
    topology = Topology(ac, cmap)

    oriented_faces = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1), (1, 4, 2), (3, 2, 4)]
    result = frozenset(OrientedSimplex(face) for face in oriented_faces)
    assert topology.alpha_cycle == BoundaryCycle(result)
