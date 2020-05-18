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
from numpy import arctan2

## This is a sample script to show how to create animations using matplotlib.
# In creating an animaiton, the timestepping must be done manually, and plotted
# after each time step. This is done in the update function. It should be noted
# that the simulation object should be in the global namespace so that it saves
# its state (i.e. not passed by value into the update function).


class myBoundary(Boundary):
    def __init__(self, spacing, radius):
        self.radius = radius
        self.spacing = spacing
        self.dx = self.spacing
        self.vradius = self.radius+self.dx
        super().__init__()

    def in_domain(self, point: tuple) -> bool:
        return point[0]**2 + point[1]**2 < self.radius**2

    def generate_boundary_points(self):
        points = []
        for theta in np.arange(0, 2*pi, self.spacing/self.vradius):
            points.append((self.vradius*cos(theta), self.vradius*sin(theta)))
        return points

    def generate_interior_points(self, n_int_sensors: int) -> list:
        points = []
        for _ in range(n_int_sensors):
            rand_theta = np.random.uniform(0, 2*pi)
            rand_r = np.random.uniform(0, self.radius)
            points.append((rand_r*cos(rand_theta), rand_r*sin(rand_theta)))
        return points


class myMotionModel(MotionModel):

    ## Initialize boundary with typical velocity.
    def __init__(self, dt: float, boundary: myBoundary, sigma: float) -> None:
        super().__init__(dt, boundary)
        self.sigma = sigma
        self.boundary = boundary

    ## Random function.
    def epsilon(self) -> float:
        return self.sigma * sqrt(self.dt) * random.normal(0, 1)

    ## Update each coordinate with brownian model.
    def update_point(self, pt: tuple, index=0) -> tuple:
        return pt[0] + self.epsilon(), pt[1] + self.epsilon()

    def reflect(self, pt: tuple, index) -> tuple:
        x, y = pt
        theta = arctan2(y, x)
        r = sqrt(x ** 2 + y ** 2)
        dr = self.epsilon()
        while not self.boundary.in_domain(pt):
            pt = (r - dr) * cos(theta), (r - dr) * sin(theta)
        return pt


num_sensors = 20
sensing_radius = 0.3
timestep_size = 0.1

filename_base = "SampleAnimation"

unit_square = myBoundary(spacing=sensing_radius, radius=1)

brownian_motion = myMotionModel(dt=timestep_size,
                                sigma=0.1,
                                boundary=unit_square)

simulation = EvasionPathSimulation(boundary=unit_square,
                                   motion_model=brownian_motion,
                                   n_int_sensors=num_sensors,
                                   sensing_radius=sensing_radius,
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
    with open(filename_base+".log", "a+") as file:
        file.write("{0:5.2f}: {1}\n".format(simulation.time, simulation.evasion_paths))


# Animation driver
def animate():

    # Number of time steps
    n_steps = 250

    # milliseconds per frame in resulting mp4 file
    ms_per_frame = 1000*timestep_size

    fig = plt.figure(1)
    try:
        ani = FuncAnimation(fig, update, interval=ms_per_frame, frames=n_steps)
    except SimulationOver:
        print("Simulation Complete")
    finally:
        plt.show()  # show plot while computing
        ani.save(filename_base+'.mp4')


if __name__ == "__main__":
    animate()
