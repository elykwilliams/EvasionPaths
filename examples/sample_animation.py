# ******************************************************************************
#  Copyright (c) 2023, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from boundary_geometry import RectangularDomain
from motion_model import BilliardMotion
from plotting_tools import show_state
from sensor_network import SensorNetwork, generate_mobile_sensors, generate_fence_sensors
from time_stepping import EvasionPathSimulation

## This is a sample script to show how to create animations using matplotlib.
# In creating an animation, the time-stepping must be done manually, and plotted
# after each time step. This is done in the update function. It should be noted
# that the simulation object should be in the global namespace so that it saves
# its state (i.e. not passed by value into the update function).

num_sensors = 20
sensing_radius = 0.2
timestep_size = 0.01
sensor_velocity = 1

filename_base = "SampleAnimation"

unit_square = RectangularDomain()

billiard = BilliardMotion(unit_square)

fence = generate_fence_sensors(unit_square, sensing_radius)
mobile_sensors = generate_mobile_sensors(unit_square, num_sensors, sensing_radius, sensor_velocity)

sensor_network = SensorNetwork(mobile_sensors, billiard, fence, sensing_radius)

simulation = EvasionPathSimulation(sensor_network, timestep_size)


# Update takes the frame number as an argument by default, other arguments
# can be added by specifying fargs= ... in the FuncAnimation parameters
def update(_):
    # Check is simulation is over
    if not simulation.cycle_label.has_intruder():
        return

    # Update simulation
    simulation.do_timestep()

    # Setup figure
    axis = plt.gca()
    axis.cla()
    axis.axis("off")
    axis.axis("equal")
    title_str = "T = " + "{:5.2f}:\n".format(simulation.time)
    axis.set_title(title_str, loc="left")

    # plot
    show_state(simulation, ax=axis)


# Animation driver
if __name__ == "__main__":
    # Number of time steps
    n_steps = 250

    # milliseconds per frame in resulting mp4 file
    ms_per_frame = 5000 * timestep_size

    fig = plt.figure(1)
    ani = FuncAnimation(fig, update, interval=ms_per_frame, frames=n_steps)

    # uncomment below to show plot while computing
    plt.show()
    # ani.save(filename_base + '.mp4')
