# ************************************************************
# Copyright (c) 2021, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************

import matplotlib.pyplot as plt
import networkx as nx

from boundary_geometry import Domain
from combinatorial_map import OrientedSimplex
from sensor_network import SensorNetwork
from time_stepping import EvasionPathSimulation


## Show fence sensors.
# Note that fence sensors are places slightly outside
# of the actual domain.


def show_fence(sensors: SensorNetwork, ax=None) -> None:
    if not ax:
        ax = plt.gca()
    xpts, ypts = zip(*[s.pos for s in sensors.fence_sensors])
    ax.plot(xpts, ypts, 'k*')


## Outline domain.
# Note that fence sensors are places slightly outside
# of the actual domain. The points used to plt the domain
# are not use in simulation.
def show_domain_boundary(domain: Domain, ax=None) -> None:
    if not ax:
        ax = plt.gca()
    xpts, ypts = domain.domain_boundary_points()
    ax.plot(xpts, ypts)


## Show Sensor Network graph.
def show_labelled_graph(sim: EvasionPathSimulation, ax=None):
    if not ax:
        ax = plt.gca()
    graph = nx.Graph()
    graph.add_nodes_from(sim.topology.alpha_complex.nodes)
    graph.add_edges_from(simplex.nodes for simplex in sim.topology.simplices(1))
    nx.draw(graph, sim.sensor_network.points, ax=ax)
    nx.draw_networkx_labels(graph, dict(enumerate(sim.sensor_network.points)), ax=ax)


## Display sensors.
def show_sensor_points(sensors: SensorNetwork, ax=None) -> None:
    if not ax:
        ax = plt.gca()
    xpts, ypts = zip(*sensors.points)
    ax.plot(xpts, ypts, 'k*')


## Display sensing disks.
def show_sensor_radius(sensors: SensorNetwork, ax=None) -> None:
    if not ax:
        ax = plt.gca()
    for pt in sensors.points:
        ax.add_artist(plt.Circle(pt, sensors.sensing_radius, color='b', alpha=1, clip_on=False))


## Shade holes in AlphaComplex with possible intruder.
def show_possible_intruder(sim: EvasionPathSimulation, ax=None) -> None:
    if not ax:
        ax = plt.gca()
    graph = nx.Graph()
    graph.add_nodes_from(sim.topology.alpha_complex.nodes)
    graph.add_edges_from(simplex.nodes for simplex in sim.topology.simplices(1))
    cmap = sim.topology.cmap

    # get boundary cycles with nodes in correct order
    for cycle in sim.cycle_label.label:
        cycle_nodes = cmap.get_cycle_nodes(next(iter(cycle)))
        xpts, ypts = zip(*[sim.sensor_network.points[n] for n in cycle_nodes])

        # Exclude alpha-cycle
        if cycle == sim.topology.alpha_cycle:
            continue

        # fill contaminated cycles transparent black
        if sim.cycle_label.label[cycle]:
            ax.fill(xpts, ypts, color='k', alpha=0.5)

    show_sensor_points(sim.sensor_network, ax)


## Display AlphaComplex.
# Shows 0, 1, and 2 simplices.
def show_alpha_complex(sim: EvasionPathSimulation, ax=None) -> None:
    if not ax:
        ax = plt.gca()
    points = sim.sensor_network.points
    for simplex in sim.topology.simplices(2):
        xpts, ypts = zip(*[points[node] for node in simplex.nodes])
        # if simplex.to_cycle(sim.topology.boundary_cycles) in sim.cycle_label:
        ax.fill(xpts, ypts, color='r', alpha=0.5)

    for edge in sim.topology.simplices(1):
        xpts, ypts = zip(*[points[node] for node in edge.nodes])
        ax.plot(xpts, ypts, color='r', alpha=1)

    show_sensor_points(sim.sensor_network, ax)


## Full display.
# Shades region with intruder, sensing balls, and alpha complex
# all on the same figure.
def show_state(sim: EvasionPathSimulation, ax=None) -> None:
    if not ax:
        ax = plt.gca()
    show_possible_intruder(sim, ax)
    show_sensor_radius(sim.sensor_network, ax)
    show_alpha_complex(sim, ax)


## Display Fat Graph/Combinatorial Map.
def show_combinatorial_map(sim: EvasionPathSimulation, ax=None) -> None:
    if not ax:
        ax = plt.gca()
    graph = nx.Graph()
    graph.add_nodes_from(sim.topology.alpha_complex.nodes)
    graph.add_edges_from(simplex.nodes for simplex in sim.topology.simplices(1))

    temp_dict = {edge: OrientedSimplex(edge) for edge in graph.edges}
    temp_dict.update({reversed(edge): sim.topology.cmap.alpha(OrientedSimplex(edge)) for edge in graph.edges})

    nx.draw(graph, sim.sensor_network.points, ax=ax)
    nx.draw_networkx_labels(graph, dict(enumerate(sim.sensor_network.points)), ax=ax)
    nx.draw_networkx_edge_labels(graph, dict(enumerate(sim.sensor_network.points)), temp_dict, label_pos=0.2, ax=ax)


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
    sensor_network = SensorNetwork(brownian_motion, unit_square, sensing_radius, num_sensors, 0.1)
    simulation = EvasionPathSimulation(sensor_network=sensor_network, dt=timestep_size)

    plt.figure(1)
    show_fence(simulation.sensor_network)
    show_domain_boundary(simulation.sensor_network.motion_model.domain)

    plt.figure(2)
    show_labelled_graph(simulation)

    plt.figure(3)
    show_sensor_points(simulation.sensor_network)
    show_sensor_radius(simulation.sensor_network)

    plt.figure(4)
    show_alpha_complex(simulation)

    plt.figure(5)
    show_possible_intruder(simulation)

    plt.figure(6)
    show_state(simulation)

    plt.figure(7)
    show_combinatorial_map(simulation)

    plt.show()
