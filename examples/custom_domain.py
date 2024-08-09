# ************************************************************
# Copyright (c) 2020, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************

from matplotlib.animation import FuncAnimation
from evasionpaths.plotting_tools import *
from evasionpaths.motion_model import *
from evasionpaths.time_stepping import *
from numpy import arctan2

## This is a sample script to show how to create animations using matplotlib.
# In creating an animaiton, the timestepping must be done manually, and plotted
# after each time step. This is done in the update function. It should be noted
# that the simulation object should be in the global namespace so that it saves
# its state (i.e. not passed by value into the update function).


class myBoundary(Boundary):
    ## Initialize with dimension of desired boundary.
    # Sensor positions will be reflected so that interior sensors stay in the
    # specified domain. Default to the unit square with spacing of 0.2. Spacing
    # should be less that 2*sensing_radius.
    def __init__(self, spacing: float = 0.2,  # default of 0.2, with unit square
                 x_min: float = 0, x_max: float = 1,
                 y_min: float = 0, y_max: float = 1) -> None:
        self.x_max, self.y_max = x_max, y_max
        self.x_min, self.y_min = x_min, y_min
        self.spacing = spacing

        # Initialize fence boundary
        self.dx = self.spacing * np.sin(np.pi / 6)  # virtual boundary width
        self.vx_min, self.vx_max = self.x_min - self.dx, self.x_max + self.dx
        self.vy_min, self.vy_max = self.y_min - self.dx, self.y_max + self.dx

        super().__init__()

    ## Check if point is in domain.
    def in_domain(self, point: tuple) -> bool:
        return self.x_min <= point[0] <= self.x_max \
               and self.y_min <= point[1] <= self.y_max

    ## Generate points in counter-clockwise order.
    def generate_boundary_points(self) -> list:
        points = []
        points.extend([(x, self.vy_min) for x in np.arange(self.vx_min, 0.999*self.vx_max, self.spacing)])  # bottom
        points.extend([(self.vx_max, y) for y in np.arange(self.vy_min, 0.999*self.vx_max, self.spacing)])  # right
        points.extend([(self.x_max - x, self.vy_max) for x in np.arange(self.vx_min, 0.999*self.vx_max, self.spacing)])  # top
        points.extend([(self.vx_min, self.y_max - y) for y in np.arange(self.vy_min, 0.999*self.vy_max, self.spacing)])  # left
        return points

    ## Generate points distributed randomly (uniformly) in the interior.
    def generate_interior_points(self, n_int_sensors: int) -> list:
        rand_x = np.random.uniform(self.x_min, self.x_max, size=n_int_sensors)
        rand_y = np.random.uniform(self.y_min, self.y_max, size=n_int_sensors)
        return list(zip(rand_x, rand_y))

    ## Generate Points to plot domain boundary.
    def domain_boundary_points(self):
        x_pts = [self.x_min, self.x_min, self.x_max, self.x_max, self.x_min]
        y_pts = [self.y_min, self.y_max, self.y_max, self.y_min, self.y_min]
        return x_pts, y_pts

    ## reflect position if outside of domain.
    def reflect_point(self, old_pt, new_pt):
        pt = new_pt
        if new_pt[0] <= self.x_min:
            pt = (self.x_min + abs(self.x_min - new_pt[0]), new_pt[1])
        elif new_pt[0] >= self.x_max:
            pt = (self.x_max - abs(self.x_max - new_pt[0]), new_pt[1])

        new_pt = pt
        if new_pt[1] <= self.y_min:
            pt = (new_pt[0], self.y_min + abs(self.y_min - new_pt[1]))
        elif new_pt[1] >= self.y_max:
            pt = (new_pt[0], self.y_max - abs(self.y_max - new_pt[1]))

        return pt

    ## Reflect velocity angle to keep velocity consistent.
    def reflect_velocity(self, old_pt, new_pt):
        vel_angle = np.arctan2(new_pt[1] - old_pt[1], new_pt[0] - old_pt[0])
        if new_pt[0] <= self.x_min or new_pt[0] >= self.x_max:
            vel_angle = np.pi - vel_angle
        if new_pt[1] <= self.y_min or new_pt[1] >= self.y_max:
            vel_angle = - vel_angle
        return vel_angle % (2 * np.pi)







class myMotionModel(MotionModel):
    def __init__(self, dt, boundary, max_vel, n_int_sensors, sensing_radius, G):
        super().__init__(dt, boundary)
        self.velocities = np.random.uniform(-max_vel, max_vel, (n_int_sensors, 2))
        self.n_sensors = n_int_sensors
        self.boundary = boundary
        self.sensing_radius = sensing_radius
        self.G = G

    def update_point(self, pt: tuple, index: int) -> tuple:
        #Not used
        return pt


    def reflect(self, old_pt, new_pt, index) -> tuple:
        norm_v = norm(self.velocities[index])
        theta = self.boundary.reflect_velocity(old_pt, new_pt)
        self.velocities[index] = (norm_v*cos(theta), norm_v*sin(theta))
        return self.boundary.reflect_point(old_pt, new_pt)

    def gradient(self, xs, ys):
        gradUx, gradUy = [0] * self.n_sensors, [0] * self.n_sensors
        for i in range(0, self.n_sensors):
            pairwise_dist = [norm((xs[i] - xs[j], ys[i] - ys[j])) for j in range(0, self.n_sensors)]
            gradUx[i] = self.G * sum((xs[i] - xs[j]) / pairwise_dist[j] ** 3
                                     for j in range(0, self.n_sensors) if pairwise_dist[j] < 2*sensing_radius and pairwise_dist[j]!=0)
            gradUy[i] = self.G * sum((ys[i] - ys[j]) / pairwise_dist[j] ** 3
                                     for j in range(0, self.n_sensors) if pairwise_dist[j] < 2*sensing_radius and pairwise_dist[j]!=0)
        return array(gradUx), array(gradUy)

    def time_derivative(self, _, state):
        # ode solver gives us np array in the form [xvals | yvals | vxvals | vyvals]
        # split into individual np array
        split_state = [state[i:i + self.n_sensors] for i in range(0, len(state), self.n_sensors)]
        gradU = self.gradient(split_state[0], split_state[1])

        # Need to compute time derivative of each,
        # I just have d(x, y)/dt = (vx, vy), d(vx, vy)/dt = (1, -1)
        dxdt = split_state[2]
        dydt = split_state[3]
        dvxdt = -gradU[0]
        dvydt = -gradU[1]
        return np.concatenate([dxdt, dydt, dvxdt, dvydt])

    def update_points(self, old_points, dt) -> list:
        self.dt = dt
        # Remove boundary points, and put into form ode solver wants
        xs = [old_points[i][0] for i in range(len(self.boundary), len(old_points))]
        ys = [old_points[i][1] for i in range(len(self.boundary), len(old_points))]
        vxs = [self.velocities[i][0] for i in range(self.n_sensors)]
        vys = [self.velocities[i][1] for i in range(self.n_sensors)]
        init_val = xs + ys + vxs + vys

        # Solve with init_val as t=0, solve for values at t+dt
        new_val = solve_ivp(self.time_derivative, [0, self.dt], init_val, t_eval=[self.dt], rtol=1e-8)

        # Convert back from np array
        new_val = new_val.y.tolist()

        # split state back into position and velocity,
        split_state = [[y[0] for y in new_val[i:i + self.n_sensors]]
                       for i in range(0, len(new_val), self.n_sensors)]

        # zip into list of tuples
        self.velocities = list(zip(split_state[2], split_state[3]))

        # Reflect any points outside boundary
        points = list(zip(split_state[0], split_state[1]))
        for n, pt in enumerate(points):
            if not self.boundary.in_domain(pt):
                points[n] = self.reflect(old_points[n+len(self.boundary)], pt, n)

        return old_points[0:len(self.boundary)] + points




num_sensors = 20
sensing_radius = 0.2
timestep_size = 0.01

filename_base = "SampleAnimation"

unit_square = myBoundary(spacing=sensing_radius)

n_body_model = myMotionModel(dt=timestep_size, boundary=unit_square, max_vel=1, n_int_sensors=num_sensors, sensing_radius=sensing_radius, G=1)

simulation = EvasionPathSimulation(boundary=unit_square,
                                   motion_model=n_body_model,
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

"""
    # log the steps that were taken
    with open(filename_base+".log", "a+") as file:
        file.write("{0:5.2f}\n".format(simulation.time))
"""

# Animation driver
def animate():

    # Number of time steps
    n_steps = 250

    # milliseconds per frame in resulting mp4 file
    ms_per_frame = 2000*timestep_size

    fig = plt.figure(1)
    ani = FuncAnimation(fig, update, interval=ms_per_frame, frames=n_steps)
    plt.show()

    # Uncomment below to save animations
    """
    try:
        ani = FuncAnimation(fig, update, interval=ms_per_frame, frames=n_steps)
    except SimulationOver:
        print("Simulation Complete")
    finally:
        plt.show()  # show plot while computing
        ani.save(filename_base+'.mp4')
    """


if __name__ == "__main__":
    animate()
