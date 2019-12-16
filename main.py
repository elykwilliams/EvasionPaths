from evasion_path import *
from matplotlib.animation import FuncAnimation
from matplotlib.pyplot import fill
import networkx as nx

simulation = EvasionPathSimulation()


def update(timestep):
    global simulation
    simulation.do_timestep()

    ax.clear()
    nx.draw(simulation.G, simulation.points)
    for hole in simulation.holes:
        nx.draw(hole, simulation.points, node_color='red', edge_color="red", ax=ax)


# Animate
ax = plt.gca()
fig = plt.figure(1)

# repeat process  and animate
ani = FuncAnimation(fig, update, interval=500, frames=1)
ani.save('animation_working.gif', writer='imagemagick')

plt.show()

