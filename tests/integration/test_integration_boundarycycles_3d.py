from unittest import mock

import numpy as np

from alpha_complex import Simplex
from combinatorial_map import CombinatorialMap3D, OrientedSimplex, BoundaryCycle, RotationInfo3D
from topology import Topology, generate_topology
from cycle_labelling import CycleLabellingDict
from boundary_geometry import UnitCube
from motion_model import BilliardMotion
from sensor_network import SensorNetwork
from state_change import StateChange
from alpha_complex import AlphaComplex
from time_stepping import EvasionPathSimulation

def mock_alphacomplex(edges, faces, tets):
    simplices_dict = {1: edges, 2: faces, 3: tets}
    ac = mock.Mock()
    ac.simplices.side_effect = \
        lambda dim: [Simplex(frozenset(item)) for item in simplices_dict[dim]]
    ac.nodes = {0, 1, 2, 3, 4, 5}
    ac.dim = 3
    return ac


def test_3dcmap_bcycle_lists():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)]
    triangles = [(0, 1, 2), (0, 2, 3), (0, 1, 4), (0, 3, 4), (1, 2, 4), (2, 3, 4)]
    ac = mock_alphacomplex(edges, triangles, [])
    rotinfo = RotationInfo3D(points, ac)
    cmap = CombinatorialMap3D(rotinfo)

    assert len(cmap.boundary_cycles) == 2

def test_3dcmap_lookup_bcycle():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)]
    triangles = [(0, 1, 2), (0, 2, 3), (0, 1, 4), (0, 3, 4), (1, 2, 4), (2, 3, 4), (0, 2, 4)]
    ac = mock_alphacomplex(edges, triangles, [])
    rotinfo = RotationInfo3D(points, ac)
    cmap = CombinatorialMap3D(rotinfo)

    oriented_simplex = OrientedSimplex((0, 2, 1))
    assert cmap.get_cycle(oriented_simplex).nodes == {0, 1, 2, 4}




def test_3d_topology_cmap():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)]
    triangles = [(0, 1, 2), (0, 2, 3), (0, 1, 4), (0, 3, 4), (1, 2, 4), (2, 3, 4), (0, 2, 4)]
    tetrahedrons = [(0, 1, 2, 4)]

    ac = mock_alphacomplex(edges, triangles, tetrahedrons)
    rotinfo = RotationInfo3D(points, ac)
    cmap = CombinatorialMap3D(rotinfo)
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
    rotinfo = RotationInfo3D(points, ac)
    cmap = CombinatorialMap3D(rotinfo)
    topology = Topology(ac, cmap)

    oriented_faces = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1), (1, 4, 2), (3, 2, 4)]
    result = frozenset(OrientedSimplex(face) for face in oriented_faces)
    assert topology.alpha_cycle == BoundaryCycle(result)



def test_simplex_find_boundary_cycles():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)]
    triangles = [(0, 1, 2), (0, 2, 3), (0, 1, 4), (0, 3, 4), (1, 2, 4), (2, 3, 4), (0, 2, 4)]
    tetrahedrons = [(0, 1, 2, 4), (0, 2, 3, 4)]

    ac = mock_alphacomplex(edges, triangles, tetrahedrons)
    rotinfo = RotationInfo3D(points, ac)
    cmap = CombinatorialMap3D(rotinfo)
    topology = Topology(ac, cmap)

    my_tetrahedron = Simplex(frozenset({0, 1, 2, 4}))

    assert my_tetrahedron.to_cycle(topology.boundary_cycles) != None

def test_simplex_find_boundary_cycles():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)]
    triangles = [(0, 1, 2), (0, 2, 3), (0, 1, 4), (0, 3, 4), (1, 2, 4), (2, 3, 4), (0, 2, 4)]
    tetrahedrons = [(0, 1, 2, 4), (0, 2, 3, 4)]

    ac = mock_alphacomplex(edges, triangles, tetrahedrons)
    rotinfo = RotationInfo3D(points, ac)
    cmap = CombinatorialMap3D(rotinfo)
    topology = Topology(ac, cmap)

    my_tetrahedron = Simplex(frozenset({0, 1, 2, 4}))
    boundary_cycles = my_tetrahedron.to_cycle(topology.boundary_cycles)

    triangle_nodes = {(0,1,4), (0,2,1), (2,4,1), (0,4,2)}
    simplices = { OrientedSimplex(nodes) for nodes in triangle_nodes}

# Struggling to create the expected boundary cycles, will come back to this later.
    assert boundary_cycles == BoundaryCycle(frozenset(simplices))

# Write a function that uses the mock alpha complex to construct a combinatorial map,
# topology, and use the topology to initialize a Cycle Labelling object.
# Check that each boundary cycle has the correct labelling (tetrahedrons are false, all others are true).
# There should be a bug here that needs fixing (bug = 2d specific code).
#
# Question, what is the expected labelling for the fence?

def test_check_cycle_labeling():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)]
    triangles = [(0, 1, 2), (0, 2, 3), (0, 1, 4), (0, 3, 4), (1, 2, 4), (2, 3, 4), (0, 2, 4)]
    tetrahedrons = [(0, 1, 2, 4), (0, 2, 3, 4)]

    ac = mock_alphacomplex(edges, triangles, tetrahedrons)
    rotinfo = RotationInfo3D(points, ac)
    cmap = CombinatorialMap3D(rotinfo)
    topology = Topology(ac, cmap)
    cycle_labeling = CycleLabellingDict(topology)

    check_for_invader = []

    t_1_triangle_nodes = {(0,1,4), (0,2,1), (2,4,1), (0,4,2)}
    simplices = { OrientedSimplex(nodes) for nodes in t_1_triangle_nodes}
    t_1_bc = BoundaryCycle(frozenset(simplices))
    check_for_invader.append(cycle_labeling.dict[t_1_bc])

    t_2_triangle_nodes = {(0,2,4), (0,3,2), (2,3,4), (0,4,3)}
    simplices = { OrientedSimplex(nodes) for nodes in t_2_triangle_nodes}
    t_2_bc = BoundaryCycle(frozenset(simplices))
    check_for_invader.append(cycle_labeling.dict[t_2_bc])


    outer_bc__triangle_nodes = {(0,3,4), (0,4,1), (1,4,2), (2,4,3), (0,2,3), (0,1,2)}
    simplices = { OrientedSimplex(nodes) for nodes in outer_bc__triangle_nodes}
    outer_bc = BoundaryCycle(frozenset(simplices))
    check_for_invader.append(cycle_labeling.dict[outer_bc])

    assert check_for_invader == [False, False, True]


# Write a test function that creates a UnitCube, BilliardMotion, and a sensor network.
# Use it to create a topology and initialize a cycle labelling.
# assert len(cycle_labelling) > 1
#
# You may also want to test that sensors are actually 3-dimensional.
# assert len(sensor.pos) == 3

def test_create_sensor_network():
    # points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
    # edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)]
    # triangles = [(0, 1, 2), (0, 2, 3), (0, 1, 4), (0, 3, 4), (1, 2, 4), (2, 3, 4), (0, 2, 4)]
    # tetrahedrons = [(0, 1, 2, 4), (0, 2, 3, 4)]
    #
    # ac = mock_alphacomplex(edges, triangles, tetrahedrons)
    # cmap = CombinatorialMap3D(points, ac.simplices(1), ac.simplices(2))
    # topology = Topology(ac, cmap)
    # cycle_labeling = CycleLabellingDict(topology)

    num_sensors: int = 20
    sensing_radius: float = 0.2
    timestep_size: float = 0.01

    unit_square = UnitCube(spacing=sensing_radius)

    billiard = BilliardMotion(domain=unit_square)

    sensor_network = SensorNetwork(motion_model=billiard,
                                   domain=unit_square,
                                   sensing_radius=sensing_radius,
                                   n_sensors=num_sensors,
                                   vel_mag=1)
    topology = generate_topology(sensor_network.points, sensor_network.sensing_radius)
    cycle_labelling = CycleLabellingDict(topology)
    sensor = sensor_network.mobile_sensors[0]
    assert len(cycle_labelling.dict) > 1
    assert len(sensor.pos) == 3



# Finally, Write a test function that creates two topologies that are only slightly different using two mock alpha complexes.
# Use the two topologies to create a state change object. Verify that the state change has the correct case.
# assert state_change.case == expected_result

def test_state_change():
    points_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
    edges_1 = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)]
    triangles_1 = [(0, 1, 2), (0, 2, 3), (0, 1, 4), (0, 3, 4), (1, 2, 4), (2, 3, 4), (0, 2, 4)]
    tetrahedrons_1 = [(0, 1, 2, 4), (0, 2, 3, 4)]

    ac_1 = mock_alphacomplex(edges_1, triangles_1, tetrahedrons_1)
    rotinfo = RotationInfo3D(points_1, ac_1)
    cmap = CombinatorialMap3D(rotinfo)
    topology_1 = Topology(ac_1, cmap)

    points_2 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
    edges_2 = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)]
    triangles_2 = [(0, 1, 2), (0, 2, 3), (0, 1, 4), (0, 3, 4), (1, 2, 4), (2, 3, 4), (0, 2, 4)]
    tetrahedrons_2 = [(0, 2, 3, 4)]

    ac_2 = mock_alphacomplex(edges_2, triangles_2, tetrahedrons_2)
    rotinfo = RotationInfo3D(points_2, ac_2)
    cmap = CombinatorialMap3D(rotinfo)
    topology_2 = Topology(ac_2, cmap)

    state_change = StateChange(topology_2, topology_1)
    assert state_change.case == (0, 0, 0, 0, 0, 1, 0, 0)
