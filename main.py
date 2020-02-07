# Kyle Williams 12/16/19

from evasion_path import *
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from joblib import Parallel, delayed


def fun():
    simplex = EvasionPathSimulation(0.1, 0)
    try:
        time = simplex.run()
    except Exception:
        return 0
    else:
        return time


print(Parallel(n_jobs=-1)(delayed(fun)() for _ in range(4)))


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



