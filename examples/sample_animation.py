# Kyle Williams 3/4/20
from motion_model import *
from evasion_path import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from gudhi import AlphaComplex

num_sensors = 10
sensing_radius = 0.1
timestep_size = 0.1

run_number = 5
output_dir = './'
filename_base = "N" + str(num_sensors) + "R" + "".join(str(sensing_radius).split("."))\
                + "dt" + "".join(str(timestep_size).split(".")) + "-" + str(run_number)
filename_base = "Sample3"

unit_square = RectangularDomain(spacing=sensing_radius)

brownian_motion = BilliardMotion(dt=timestep_size,
                                 vel=0.1,
                                 boundary=unit_square,
                                 n_sensors=num_sensors+len(unit_square))

#
# brownian_motion = BrownianMotion(dt=timestep_size,
#                                  sigma=0.01,
#                                  boundary=unit_square)


simulation = EvasionPathSimulation(boundary=unit_square,
                                   motion_model=brownian_motion,
                                   n_int_sensors=num_sensors,
                                   sensing_radius=sensing_radius,
                                   dt=timestep_size)


class SimulationOver(Exception):
    pass


def get_cmap(sim):
    """ This function is to access the combinatorial map externally primarily
        this function is meant to help with plotting and not to be used internally"""
    alpha_complex = AlphaComplex(sim.points)
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=sim.sensing_radius ** 2)

    simplices1 = [tuple(simplex) for simplex, _ in simplex_tree.get_skeleton(1) if len(simplex) == 2]

    graph = nx.Graph()
    graph.add_nodes_from(range(sim.n_total_sensors))
    graph.add_edges_from(simplices1)

    return CMap(graph, sim.points)


def plot_balls(sim):
    fig = plt.gcf()
    ax = fig.gca()
    for point in sim.points:
        ax.add_artist(plt.Circle(point, sim.sensing_radius, color='b', alpha=0.1, clip_on=False))

    xpts = [x for (x, y) in sim.points]
    ypts = [y for (x, y) in sim.points]
    ax.scatter(xpts, ypts)


def plot_alpha_complex(sim):
    fig = plt.gcf()
    ax = fig.gca()

    for simplex in sim.state.simplices2:
        xpts = [sim.points[n][0] for n in simplex]
        ypts = [sim.points[n][1] for n in simplex]
        if simplex2cycle(simplex, sim.state.boundary_cycles) in sim.cell_label:
            ax.fill(xpts, ypts, color='r', alpha=0.1)

    for edge in sim.state.simplices1:
        xpts = [sim.points[n][0] for n in edge]
        ypts = [sim.points[n][1] for n in edge]
        ax.plot(xpts, ypts, color='r', alpha=0.15)


def plot_no_intruder(sim):
    fig = plt.gcf()
    ax = fig.gca()

    for cycle_nodes in get_cmap(sim).boundary_cycle_nodes_ordered():
        x_pts = [sim.points[n][0] for n in cycle_nodes]
        y_pts = [sim.points[n][1] for n in cycle_nodes]
        if set(cycle_nodes) == set(cycle2nodes(sim.boundary.alpha_cycle)):
            continue

        cycle = simplex2cycle(cycle_nodes, sim.state.boundary_cycles)
        if cycle not in sim.cell_label:
            continue

        if sim.cell_label[simplex2cycle(cycle_nodes, sim.state.boundary_cycles)]:
            ax.fill(x_pts, y_pts, color='k', alpha=0.2)
        else:
            pass


def update(timestep):
    global simulation
    if not simulation.cell_label.has_intruder():
        raise SimulationOver
    simulation.evasion_paths = ""
    simulation.do_timestep()
    simulation.time += simulation.dt
    print(simulation.evasion_paths)
    fig = plt.figure(1)
    ax = plt.gca()
    ax.cla()
    ax.axis("off")
    ax.axis("equal")
    title_str = "T = " + "{:5.2f}:\n".format(simulation.time)
    ax.set_title(title_str, loc="left")
    fig.add_axes(ax)

    plot_no_intruder(simulation)
    plot_balls(simulation)
    plot_alpha_complex(simulation)

    with open(filename_base+".log", "a+") as file:
        file.write("{0:5.2f}: {1}\n".format(simulation.time, simulation.evasion_paths))


def animate():
    n_steps = 250
    ms_per_frame = 1000*timestep_size
    fig = plt.figure(1)
    try:
        ani = FuncAnimation(fig, update, interval=ms_per_frame, frames=n_steps)
    except SimulationOver as s:
        pass
    finally:
        plt.show()
        #ani.save(filename_base+'.mp4')


if __name__ == "__main__":
    animate()
