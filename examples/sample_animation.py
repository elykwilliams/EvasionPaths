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

## This is a sample script to show how to create animations using matplotlib.
# In creating an animaiton, the timestepping must be done manually, and plotted
# after each time step. This is done in the update function. It should be noted
# that the simulation object should be in the global namespace so that it saves
# its state (i.e. not passed by value into the update function).

num_sensors = 20
sensing_radius = 0.2
timestep_size = 0.01

filename_base = "SampleAnimation"

unit_square = RectangularDomain(spacing=sensing_radius)

billiard = BilliardMotion(domain=unit_square)

sensor_network = SensorNetwork(motion_model=billiard,
                               sensing_radius=sensing_radius,
                               vel_mag=1,
                               n_sensors=num_sensors)

simulation = EvasionPathSimulation(sensor_network=sensor_network,
                                   dt=timestep_size)


# raise exception if simulation is over to kill animation.
class SimulationOver(Exception):
    pass


# Update takes the frame number as an argument by default, other arguments
# can be added by specifying fargs= ... in the FuncAnimation parameters
def update(_):
    # Check is simulation is over
    if not simulation.cycle_label.has_intruder():
        raise SimulationOver

    # Update simulation
    simulation.do_timestep()
    simulation.time += simulation.dt

    # Setup figure
    axis = plt.gca()
    axis.cla()
    axis.axis("off")
    axis.axis("equal")
    title_str = "T = " + "{:5.2f}:\n".format(simulation.time)
    axis.set_title(title_str, loc="left")

    # plot
    show_state(simulation)

    # log the steps that were taken
    with open(filename_base + ".log", "a+") as file:
        file.write("{0:5.2f} \n".format(simulation.time))


# Animation driver
def animate():
    # Number of time steps
    n_steps = 250

    # milliseconds per frame in resulting mp4 file
    ms_per_frame = 5000 * timestep_size

    fig = plt.figure(1)
    try:
        ani = FuncAnimation(fig, update, interval=ms_per_frame, frames=n_steps)
    except SimulationOver:
        print("Simulation Complete")
    finally:
        # uncomment below to show plot while computing
        plt.show()
        # ani.save(filename_base+'.mp4')


if __name__ == "__main__":
    animate()
