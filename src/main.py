# Kyle Williams 2/25/20

from evasion_path import *
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from joblib import Parallel, delayed


def fun(i):
    simplex = EvasionPathSimulation(0.1, 1000)
    try:
        time = simplex.run()
    except GraphNotConnected:
        print("Graph not connected")
        return -simplex.n_steps
    except Exception as e:
        fig = plt.figure(i)
        ax1 = plt.subplot(1, 2, 1)
        fig.add_axes(ax1)
        plot(simplex.G, simplex.points, fig, ax1)

        ax2 = plt.subplot(1, 2, 2)
        fig.add_axes(ax2)
        G = nx.Graph()
        G.add_nodes_from(range(simplex.n_sensors))  # nodes numbered 0 though N points -1
        G.add_edges_from(simplex.edges)
        plot(G, simplex.old_points, fig, ax2)

        print(e)
        print(i, simplex.evasion_paths)
        plt.show()
        return -simplex.n_steps
    else:
        return simplex.n_steps, time

if __name__=="__main__":
#print(Parallel(n_jobs=-1)(delayed(fun)(i) for i in range(1, 11)))
# print(Parallel(n_jobs=-1)(delayed(fun)() for _ in range(4)))
    print(fun(1))

# def update(timestep):
#     global simulation
#     for _ in range(10):
#         simulation.do_timestep()
#
#     fig = plt.figure(1)
#     ax = plt.gca()
#     fig.add_axes(ax)
#     simulation.plot(fig, ax)
#
#     fig.suptitle("Timestep = " + str(timestep*10))
#
#     print(timestep)
#
#
# simulation = EvasionPathSimulation(0.1, 500)
# # repeat process  and animate
# n_steps = 500
# fig = plt.figure(1)
# ani = FuncAnimation(fig, update, interval=1000, frames=n_steps)
# ani.save('animation_working.gif', writer='imagemagick')



