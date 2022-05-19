import matplotlib.pyplot as plt
import networkx as nx

Point = tuple
Node = int
Edge = tuple
Face = tuple
Volume = tuple


def plot_sensors(num, sensor_network):
    fig = plt.gcf()
    ax = plt.gca()

    ax.cla()

    # variable txt created to display the frame of the animation
    txt = fig.suptitle('')
    txt.set_text(f'num={num}')  # to keep track of the frames

    # clear axes and fix axis min,max values
    ax.set_xlim3d(-0.25, 1.25)
    ax.set_ylim3d(-0.25, 1.25)
    ax.set_zlim3d(-0.25, 1.25)

    # plot sensors
    ax.scatter(*sensor_network.mobile_sensors.T)
    ax.scatter(*sensor_network.fence_sensors.T)

    # draw arrows indicating magnitude and direction of movement
    ax.quiver(*sensor_network.mobile_sensors.T, *sensor_network.velocities.T, length=0.05)


def update_anim(num, simulation):
    simulation.do_timestep()
    simulation.time += simulation.dt
    plot_sensors(num, simulation.sensor_network)
    print(simulation.state)


def plot_edge(e: Edge, sensor_network):
    fig = plt.gcf()
    ax = plt.gca()

def plot_reeb(ts, cycle_labelling: dict):
    cycles = cycle_labelling.keys()
    cycles_graph = nx.Graph()

    print(len(cycles))
    for cycle1 in cycles:
        for face in cycle1:
            for cycle2 in cycles:
                if face.alpha() in cycle2:
                    cycles_graph.add_edge(cycle1, cycle2)
    color_map = []
    for node in cycles_graph:
        if cycle_labelling[node]:
            color_map.append('blue')
        else:
            color_map.append('red')
    nx.draw_kamada_kawai(cycles_graph, node_size=100, node_color=color_map, with_labels=False)
    plt.show()

if __name__ == "__main__":
    from sensor_network import SensorNetwork
    from topological_state import TopologicalState
    import numpy as np
    import random

    n_sensors = 15
    sensing_radius = 0.2

    mobile_sensors = np.random.rand(n_sensors, 3)
    velocity = np.random.uniform(-0.1, 0.1, (n_sensors, 3))

    sensor_network = SensorNetwork(mobile_sensors, velocity, sensing_radius)
    ts = TopologicalState(sensor_network)

    labels = dict()
    for cycle in ts.boundary_cycles:
        if not all(set(face.nodes).issubset(range(len(sensor_network.fence_sensors))) for face in cycle):
            labels[cycle] = any(set(face.nodes).issubset(range(len(sensor_network.fence_sensors))) for face in cycle)
    print(f"Number of fence sensors = {len(sensor_network.fence_sensors)}")
    for cycle in labels.keys():
        nodes = set()
        for face in cycle:
            nodes.update(set(face.nodes))
        print(nodes)
    plot_reeb(ts, labels)
