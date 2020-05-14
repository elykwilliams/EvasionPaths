# ************************************************************
# Copyright (c) 2020, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************

from matplotlib.animation import FuncAnimation
from plotting_tools import *
from motion_model import *
from time_stepping import *

num_sensors = 20
sensing_radius = 0.095
timestep_size = 0.1

filename_base = "SampleAnimation"

unit_square = RectangularDomain(spacing=sensing_radius)

brownian_motion = BilliardMotion(dt=timestep_size,
                                 vel=0.1,
                                 boundary=unit_square,
                                 n_total_sensors=num_sensors+len(unit_square))

simulation = EvasionPathSimulation(boundary=unit_square,
                                   motion_model=brownian_motion,
                                   n_int_sensors=num_sensors,
                                   sensing_radius=sensing_radius,
                                   dt=timestep_size)


class SimulationOver(Exception):
    pass


def update(_):
    if not simulation.cycle_label.has_intruder():
        raise SimulationOver

    simulation.do_timestep()
    simulation.time += simulation.dt

    axis = plt.gca()
    axis.cla()
    axis.axis("off")
    axis.axis("equal")
    title_str = "T = " + "{:5.2f}:\n".format(simulation.time)
    axis.set_title(title_str, loc="left")

    show_state(simulation)

    with open(filename_base+".log", "a+") as file:
        file.write("{0:5.2f}: {1}\n".format(simulation.time, simulation.evasion_paths))


def animate():
    n_steps = 250
    ms_per_frame = 1000*timestep_size
    fig = plt.figure(1)

    ani = None
    try:
        ani = FuncAnimation(fig, update, interval=ms_per_frame, frames=n_steps)
    except SimulationOver:
        print("Simulation Complete")
    finally:
        plt.show()
        ani.save(filename_base+'.mp4')


if __name__ == "__main__":
    animate()
