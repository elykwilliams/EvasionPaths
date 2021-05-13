# ************************************************************
# Copyright (c) 2021, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************

from combinatorial_map import *

import networkx as nx
import matplotlib.pyplot as plt


## Show fence sensors.
# Note that fence sensors are places slightly outside
# of the actual domain.
# TODO accept SensorNetwork
def show_boundary_points(sim):
    axis = plt.gca()
    xpts = [s.position[0] for s in sim.sensor_network.fence_sensors]
    ypts = [s.position[1] for s in sim.sensor_network.fence_sensors]
    axis.plot(xpts, ypts, 'k*')


## Outline domain.
# Note that fence sensors are places slightly outside
# of the actual domain. The points used to plt the domain
# are not use in simulation.
# TODO accept Domain
def show_domain_boundary(sim):
    axis = plt.gca()
    b = sim.sensor_network.motion_model.domain
    xpts, ypts = b.domain_boundary_points()
    axis.plot(xpts, ypts)


## Show Sensor Network graph.
# TODO accept simulation or SensorNetwork
def show_labelled_graph(sim):
    graph = sim.state.graph
    points = [s.position for s in sim.sensor_network]
    nx.draw(graph, points)
    nx.draw_networkx_labels(graph, dict(enumerate(points)))


## Display sensors.
# TODO accept SensorNetwork
def show_sensor_points(sim):
    axis = plt.gca()
    xpts = [s.position[0] for s in sim.sensor_network]
    ypts = [s.position[1] for s in sim.sensor_network]
    axis.plot(xpts, ypts, 'k*')
    return


## Display sensing disks.
# TODO accept SensorNetwork
def show_sensor_radius(sim):
    axis = plt.gca()
    for pt in [s.position for s in sim.sensor_network]:
        axis.add_artist(plt.Circle(pt, sim.sensor_network.sensing_radius, color='b', alpha=0.1, clip_on=False))


## Shade holes in AlphaComplex with possible intruder.
def show_possible_intruder(sim):
    axis = plt.gca()
    graph = sim.state.graph
    points = [s.position for s in sim.sensor_network]
    cmap = CMap(graph, points)

    # get boundary cycles with nodes in correct order
    for cycle_nodes in cmap.boundary_cycle_nodes_ordered():

        xpts = [points[n][0] for n in cycle_nodes]
        ypts = [points[n][1] for n in cycle_nodes]

        # Exclude alpha-cycle
        cycle = nodes2cycle(cycle_nodes, sim.state.boundary_cycles())
        if cycle == alpha_cycle(sim.sensor_network.fence_sensors):
            axis.fill(xpts, ypts, color='k', alpha=0.2)

        # skip disconnected cycles
        if cycle not in sim.cycle_label:
            continue

        # fill contaminated cycles transparent black
        if sim.cycle_label[nodes2cycle(cycle_nodes, sim.state.boundary_cycles())]:
            axis.fill(xpts, ypts, color='k', alpha=0.2)
        else:
            pass
    show_sensor_points(sim)


## Display AlphaComplex.
# Shows 0, 1, and 2 simplices.
# TODO accept SensorNetwork or simulation
def show_alpha_complex(sim):

    axis = plt.gca()
    points = [s.position for s in sim.sensor_network]
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


## Full display.
# Shades region with intruder, sensing balls, and alpha complex
# all on the same figure.
def show_state(sim):
    show_possible_intruder(sim)
    show_sensor_radius(sim)
    show_alpha_complex(sim)


## Display Fat Graph/Combinatorial Map.
def show_combinatorial_map(sim):
    graph = sim.state.graph
    temp_dict = {edge: edge2dart(edge) for edge in graph.edges}
    temp_dict.update({reversed(edge): edge2dart(tuple(reversed(edge))) for edge in graph.edges})
    points = [s.position for s in sim.sensor_network]
    nx.draw(graph, points)
    nx.draw_networkx_labels(graph, dict(enumerate(points)))
    nx.draw_networkx_edge_labels(graph, dict(enumerate(points)), temp_dict, label_pos=0.2)
    pass


if __name__ == '__main__':
    from boundary_geometry import RectangularDomain
    from motion_model import BilliardMotion
    from time_stepping import EvasionPathSimulation
    from sensor_network import SensorNetwork

    num_sensors = 10
    sensing_radius = 0.2
    timestep_size = 0.1

    unit_square = RectangularDomain(spacing=sensing_radius)
    brownian_motion = BilliardMotion(domain=unit_square)
    sensor_network = SensorNetwork(brownian_motion, unit_square, sensing_radius, num_sensors)
    simulation = EvasionPathSimulation(sensor_network=sensor_network, dt=timestep_size)

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
