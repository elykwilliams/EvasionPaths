# Kyle Williams 12/16/19

from evasion_path import *
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import networkx as nx


def update(timestep):
    global simulation
    simulation.do_timestep()

    ax.clear()
    nx.draw(simulation.G, simulation.points)
    for hole in simulation.holes:
        nx.draw(hole, simulation.points, node_color='red', edge_color="red")


simulation = EvasionPathSimulation(dt=0.0001, end_time=0.01)
ax = plt.gca()
fig = plt.figure(1)
update(0)


fig2 = plt.figure(2)
ax = plt.gca()
simulation.run()
update(0)
plt.show()

# Animate
# ax = plt.gca()
# fig = plt.figure(1)
# update(0)
# # repeat process  and animate
# ani = FuncAnimation(fig, update, interval=500, frames=1)
# ani.save('animation_working.gif', writer='imagemagick')
#


