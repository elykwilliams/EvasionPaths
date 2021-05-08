# ************************************************************
# Copyright (c) 2020, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************

from time_stepping import *
from motion_model import *
import matplotlib.pyplot as plt


def get_graph(sim):
    """ This function is to access the combinatorial map externally primarily
        this function is meant to help with plotting and not to be used internally"""
    from gudhi import AlphaComplex
    alpha_complex = AlphaComplex([s.position for s in sim.sensor_network.sensors])
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=sim.sensing_radius ** 2)

    simplices1 = [tuple(simplex) for simplex, _ in simplex_tree.get_skeleton(1) if len(simplex) == 2]

    graph = nx.Graph()
    graph.add_nodes_from(range(len([s.position for s in sim.sensor_network.sensors])))
    graph.add_edges_from(simplices1)

    return graph


def show_boundary_points(sim):
    axis = plt.gca()
    xpts = [s.position[0] for s in sim.sensor_network.sensors if s.boundary_flag]
    ypts = [s.position[1] for s in sim.sensor_network.sensors if s.boundary_flag]
    axis.plot(xpts, ypts, "k*")


def show_domain_boundary(sim):
    axis = plt.gca()
    b = sim.boundary
    xpts, ypts = b.domain_boundary_points()
    axis.plot(xpts, ypts)


def show_labelled_graph(sim):
    graph = get_graph(sim)
    points = [s.position for s in sim.sensor_network.sensors]
    nx.draw(graph, points)
    nx.draw_networkx_labels(graph, dict(enumerate(points)))


def show_sensor_points(sim):
    axis = plt.gca()
    xpts = [s.position[0] for s in sim.sensor_network.sensors]
    ypts = [s.position[1] for s in sim.sensor_network.sensors]
    axis.plot(xpts, ypts, "k*")
    return


def show_sensor_radius(sim):
    axis = plt.gca()
    for pt in [s.position for s in sim.sensor_network.sensors]:
        axis.add_artist(plt.Circle(pt, sim.sensing_radius, color='b', alpha=0.1, clip_on=False))


def show_possible_intruder(sim):
    axis = plt.gca()
    graph = get_graph(sim)
    points = [s.position for s in sim.sensor_network.sensors]
    cmap = CMap(graph, points)

    for cycle_nodes in cmap.boundary_cycle_nodes_ordered():
        xpts = [points[n][0] for n in cycle_nodes]
        ypts = [points[n][1] for n in cycle_nodes]
        if set(cycle_nodes) == set(cycle2nodes(sim.boundary.alpha_cycle)):
            continue

        cycle = nodes2cycle(cycle_nodes, sim.state.boundary_cycles())
        if cycle == sim.boundary.alpha_cycle:
            axis.fill(xpts, ypts, color='k', alpha=0.2)

        if cycle not in sim.cycle_label:
            continue

        if sim.cycle_label[nodes2cycle(cycle_nodes, sim.state.boundary_cycles())]:
            axis.fill(xpts, ypts, color='k', alpha=0.2)
        else:
            pass
    show_sensor_points(sim)


def show_alpha_complex(sim):

    axis = plt.gca()
    points = [s.position for s in sim.sensor_network.sensors]
    for simplex in sim.state.simplices(2):
        xpts = [points[n][0] for n in simplex]
        ypts = [points[n][1] for n in simplex]
        if nodes2cycle(simplex, sim.state.boundary_cycles()) in sim.cycle_label:
            axis.fill(xpts, ypts, color='r', alpha=0.1)

    for edge in sim.state.simplices(1):
        xpts = [points[n][0] for n in edge]
        ypts = [points[n][1] for n in edge]
        axis.plot(xpts, ypts, color='r', alpha=0.15)

    show_sensor_points(sim)


def show_state(sim):
    show_possible_intruder(sim)
    show_sensor_radius(sim)
    show_alpha_complex(sim)


def show_combinatorial_map(sim):
    graph = get_graph(sim)
    temp_dict = {edge: edge2dart(edge) for edge in graph.edges}
    temp_dict.update({reversed(edge): edge2dart(tuple(reversed(edge))) for edge in graph.edges})
    points = [s.position for s in sim.sensor_network.sensors]
    nx.draw(graph, points)
    nx.draw_networkx_labels(graph, dict(enumerate(points)))
    nx.draw_networkx_edge_labels(graph, dict(enumerate(points)), temp_dict, label_pos=0.2)
    pass


if __name__ == "__main__":
    num_sensors = 10
    sensing_radius = 0.2
    timestep_size = 0.1

    unit_square = RectangularDomain(spacing=sensing_radius)

    brownian_motion = BilliardMotion(dt=timestep_size,
                                     vel=0.1,
                                     boundary=unit_square,
                                     n_int_sensors=num_sensors)

    simulation = EvasionPathSimulation(boundary=unit_square,
                                       motion_model=brownian_motion,
                                       n_int_sensors=num_sensors,
                                       sensing_radius=sensing_radius,
                                       dt=timestep_size)

    plt.figure(1)
    show_boundary_points(simulation)
    show_domain_boundary(simulation)

    plt.figure(2)
    show_labelled_graph(simulation)

    plt.figure(3)
    show_sensor_points(simulation)
    show_sensor_radius(simulation)

    plt.figure(4)
    show_alpha_complex(simulation)

    plt.figure(5)
    show_possible_intruder(simulation)

    plt.figure(6)
    show_state(simulation)

    plt.figure(7)
    show_combinatorial_map(simulation)

    plt.show()
