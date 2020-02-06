# Kyle Williams 12/16/19

from evasion_path import *
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


def update(timestep):
    global simulation
    simulation.do_timestep()

    fig = plt.figure(1)
    ax = plt.gca()
    fig.add_axes(ax)
    simulation.plot(fig, ax)

    fig.suptitle("Timestep = " + str(timestep))

    print(timestep)


simulation = EvasionPathSimulation(0.1, 100)
# repeat process  and animate
n_steps = 50
fig = plt.figure(1)
ani = FuncAnimation(fig, update, interval=500, frames=n_steps)
ani.save('animation_working.gif', writer='imagemagick')



