# ************************************************************
# Copyright (c) 2020, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************

import sys, os
# If you're having difficulty importing modules from the src directory, uncomment below
# current_dir = os.path.dirname(os.path.realpath(__file__))
# src_dir = os.path.join(current_dir, "..", "src")
# sys.path.append(src_dir)

from matplotlib.animation import FuncAnimation
from plotting_tools import *
from motion_model import *
from time_stepping import *


############################################################
## This is a sample script to show how to create animations using matplotlib.
# In creating an animaiton, the timestepping must be done manually, and plotted
# after each time step. This is done in the update function. It should be noted
# that the simulation object should be in the global namespace so that it saves
# its state (i.e. not passed by value into the update function).



############################################################
# Define the parameters of the simulation
num_sensors: int = 20
sensing_radius: float = 0.2
timestep_size: float = 0.01

unit_square = RectangularDomain(spacing=sensing_radius)

my_boundary = unit_square



############################################################
# Define the motion model -- see src/motion_model.py for more details
billiard = BilliardMotion(dt=timestep_size, 
                          vel=1, 
                          boundary=my_boundary, 
                          n_int_sensors=num_sensors)

# See the paper for the Dorsogna model to better understand the parameters
dorsogna_coeff = {"Ca": 0.45, "la": 1, "Cr": 0.5, "lr": 0.1}
dorsogna: MotionModel = Dorsogna(dt=timestep_size, 
                                 boundary=my_boundary, 
                                 max_vel=1, 
                                 n_int_sensors=num_sensors, 
                                 sensing_radius=sensing_radius, 
                                 DO_coeff=dorsogna_coeff)

brownian: MotionModel = BrownianMotion(dt=timestep_size,
                                        boundary=my_boundary,
                                        sigma=0.5)

##############################
# ASSIGN MOTION MODEL HERE
my_motion_model = billiard



############################################################
# Run the simulation
simulation = EvasionPathSimulation(boundary=my_boundary,
                                   motion_model=my_motion_model,
                                   n_int_sensors=num_sensors,
                                   sensing_radius=sensing_radius,
                                   dt=timestep_size)

# Raise exception when the number of tme steps is reached to stop the simulation. 
class SimulationOver(Exception):
    pass

# File name for the resulting animation
output_dir: str = "./output"
filename_base: str = "SampleAnimation"

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
    with open(output_dir + "/" + filename_base+".log", "a+") as file:
        file.write("{0:5.2f} \n".format(simulation.time))


# Animation driver
def animate():

    # Number of time steps
    n_steps = 250

    # milliseconds per frame in resulting mp4 file
    ms_per_frame = 5000*timestep_size

    fig = plt.figure(1)
    try:
        ani = FuncAnimation(fig, update, interval=ms_per_frame, frames=n_steps)
    except SimulationOver:
        print("Simulation Complete")
    finally:
        # uncomment below to show plot while computing
        # plt.show()
        ani.save(output_dir + "/" + filename_base+'.mp4')



############################################################
if __name__ == "__main__":
    animate()
