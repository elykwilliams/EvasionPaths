# ************************************************************
# Copyright (c) 2021, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************

import numpy as np
from matplotlib.animation import FuncAnimation

from boundary_geometry import RectangularDomain
from motion_model import ODEMotion
from plotting_tools import *
from sensor_network import generate_fence_sensors, generate_mobile_sensors
from time_stepping import EvasionPathSimulation


## This is a sample script to show how to create animations using matplotlib.
# In creating an animaiton, the timestepping must be done manually, and plotted
# after each time step. This is done in the update function. It should be noted
# that the simulation object should be in the global namespace so that it saves
# its state (i.e. not passed by value into the update function).
class GravityMotion(ODEMotion):
    def __init__(self, domain, gravity):
        super().__init__(domain)
        self.G = gravity

    def gradient(self, *_):
        return np.zeros(self.n_sensors), self.G*np.ones(self.n_sensors)

    def time_derivative(self, _, state):
        # ode solver gives us np array in the form [xvals | yvals | vxvals | vyvals]
        # split into individual np array
        split_state = [state[i:i + self.n_sensors] for i in range(0, len(state), self.n_sensors)]

        # Need to compute time derivative of each,
        # I just have d(x, y)/dt = (vx, vy), d(vx, vy)/dt = (0, G)
        dxdt, dydt = split_state[2], split_state[3]
        dvxdt, dvydt = self.gradient(split_state[0], split_state[1])
        return np.concatenate([dxdt, dydt, -dvxdt, -dvydt])


num_sensors = 20
sensing_radius = 0.2
timestep_size = 0.01
sensor_velocity = 1

filename_base = "SampleAnimation"

unit_square = RectangularDomain()

billiard = GravityMotion(unit_square, 10)

fence = generate_fence_sensors(unit_square, sensing_radius)
mobile_sensors = generate_mobile_sensors(unit_square, num_sensors, sensing_radius, sensor_velocity)

sensor_network = SensorNetwork(mobile_sensors, billiard, fence, sensing_radius)

sim = EvasionPathSimulation(sensor_network, timestep_size)


# Update takes the frame number as an argument by default, other arguments
# can be added by specifying fargs= ... in the FuncAnimation parameters
def update(_):
    # Update simulation
    sim.sensor_network.move(sim.dt)
    sim.time += sim.dt

    # Setup figure
    axis = plt.gca()
    axis.cla()
    axis.axis("off")
    axis.axis("equal")
    title_str = "T = " + "{:5.2f}:\n".format(sim.time)
    axis.set_title(title_str, loc="left")

    # plot
    show_sensor_radius(sim.sensor_network)
    show_sensor_points(sim.sensor_network)

    sim.sensor_network.update()


# Animation driver
def animate():

    # Number of time steps
    n_steps = 250

    # milliseconds per frame in resulting mp4 file
    ms_per_frame = 2000*timestep_size

    fig = plt.figure(1)
    _ = FuncAnimation(fig, update, interval=ms_per_frame, frames=n_steps)
    plt.show()

    # See sample_animation.py for how to save animation


if __name__ == "__main__":
    animate()
