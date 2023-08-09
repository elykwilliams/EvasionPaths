# ******************************************************************************
#  Copyright (c) 2023, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************

import numpy as np
from matplotlib.animation import FuncAnimation

from motion_model import ODEMotion
from plotting_tools import *
from sensor_network import generate_mobile_sensors


## This is a sample script to show how to create animations using matplotlib.
# In creating an animaiton, the timestepping must be done manually, and plotted
# after each time step. This is done in the update function. It should be noted
# that the simulation object should be in the global namespace so that it saves
# its state (i.e. not passed by value into the update function).
class GravityMotion(ODEMotion):
    def __init__(self, gravity):
        super().__init__()
        self.G = gravity

    def gradient(self, *_):
        return np.zeros(self.n_sensors), -self.G*np.ones(self.n_sensors)

    def time_derivative(self, _, state):
        # ode solver gives us np array in the form [xvals | yvals | vxvals | vyvals]
        xs, ys, vxs, vys = np.array_split(state, 4)

        # Need to compute time derivative of each,
        # I just have d(x, y)/dt = (vx, vy), d(vx, vy)/dt = (0, -G)
        dxdt, dydt = vxs, vys
        dvxdt, dvydt = self.gradient([xs, ys])
        return np.concatenate([dxdt, dydt, dvxdt, dvydt])


num_sensors = 20
sensing_radius = 0.2
timestep_size = 0.01
sensor_velocity = 1

filename_base = "SampleAnimation"

domain = RectangularDomain()

motion_model = GravityMotion(9.81)

mobile_sensors = generate_mobile_sensors(domain, num_sensors, sensing_radius, sensor_velocity)

sensor_network = SensorNetwork(mobile_sensors, motion_model, [], sensing_radius, domain)


# Update takes the frame number as an argument by default, other arguments
# can be added by specifying fargs= ... in the FuncAnimation parameters
def update(frame):
    # Update simulation
    sensor_network.move(timestep_size)
    time = frame*timestep_size

    # Setup figure
    axis = plt.gca()
    axis.cla()
    axis.axis("off")
    axis.axis("equal")
    title_str = "T = " + "{:5.2f}:\n".format(time)
    axis.set_title(title_str, loc="left")

    # plot
    show_sensor_points(sensor_network, axis)
    show_domain_boundary(domain, axis)

    sensor_network.update()


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
