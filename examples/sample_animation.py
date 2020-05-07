# Kyle Williams 3/4/20
from motion_model import *
from time_stepping import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from plotting_tools import *

num_sensors = 20
sensing_radius = 0.095
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

simulation = EvasionPathSimulation(boundary=unit_square,
                                   motion_model=brownian_motion,
                                   n_int_sensors=num_sensors,
                                   sensing_radius=sensing_radius,
                                   dt=timestep_size)


class SimulationOver(Exception):
    pass


def update(timestep):
    global simulation

    if not simulation.cycle_label.has_intruder():
        raise SimulationOver
    simulation.do_timestep()
    simulation.time += simulation.dt

    ax = plt.gca()
    ax.cla()
    ax.axis("off")
    ax.axis("equal")
    title_str = "T = " + "{:5.2f}:\n".format(simulation.time)
    ax.set_title(title_str, loc="left")

    show_state(simulation)

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
