# ************************************************************
# Copyright (c) 2020, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************

from boundary_geometry import *
from numpy import sqrt, random, sin, cos, pi, sum
from abc import ABC, abstractmethod
from numpy.linalg import norm
from numpy import array
from scipy.integrate import solve_ivp


## This class provides the basic interface for a model of motion.
# it should do two things, update the point positions, and reflect
# points off of the boundary. Because the reflection depends on the
# boundary, motion models and boundaries must be compatible.
class MotionModel(ABC):

    ## Initialize motion model with boundary and time-scale
    def __init__(self, dt: float, boundary: Boundary) -> None:
        self.dt = dt
        self.boundary = boundary

    ## Update an individual point.
    # This function will be called on all points.
    # reflection should be separate. The index is the position
    # in the set of ALL points, and can be useful in looking up
    # sensor specific data. Will return new position or sensor.
    @abstractmethod
    def update_point(self, pt: tuple, index: int) -> tuple:
        return pt

    @abstractmethod
    def reflect(self, old_pt, new_pt, index):
        self.boundary.reflect_velocity(old_pt, new_pt)
        return self.boundary.reflect_point(old_pt, new_pt)

    ## Update all non-fence points.
    # If a point is not in the domain, reflect. It is sometimes
    # necessary to override this class method since this method is
    # called only once per time-step.
    def update_points(self, old_points: list, dt: float) -> list:
        self.dt = dt
        return self.boundary.points \
            + [self.update_point(pt, n) for n, pt in enumerate(old_points) if n >= len(self.boundary)]


## Provide random motion for rectangular domain.
# Will move a point randomly with an average step
# size of sigma*dt
class BrownianMotion(MotionModel):

    ## Initialize boundary with typical velocity.
    def __init__(self, dt: float, boundary: Boundary, sigma: float) -> None:
        super().__init__(dt, boundary)
        self.sigma = sigma

    ## Random function.
    def epsilon(self) -> float:
        return self.sigma * sqrt(self.dt) * random.normal(0, 1)

    ## Update each coordinate with brownian model.
    def update_point(self, old_pt: tuple, index) -> tuple:
        new_pt = old_pt[0] + self.epsilon(), old_pt[1] + self.epsilon()
        return new_pt if self.boundary.in_domain(new_pt) else self.reflect(old_pt, new_pt, index)

    def reflect(self, old_pt, new_pt, index):
        return self.boundary.reflect_point(old_pt, new_pt)


## Implement Billiard Motion for Rectangular Domain.
# All sensors will have same velocity bit will have random angles.
# Points will move a distance of vel*dt each update.
class BilliardMotion(MotionModel):

    ## Initialize Boundary with additional velocity and number of sensors.
    # The number of sensors is required to know how to initialize the velocity
    # angles.
    def __init__(self, dt: float, boundary: Boundary, vel: float, n_int_sensors: int) -> None:
        super().__init__(dt, boundary)
        self.vel = vel
        self.vel_angle = random.uniform(0, 2 * pi, n_int_sensors + len(boundary))

    ## Update point using x = x + v*dt.
    def update_point(self, pt: tuple, index: int) -> tuple:
        theta = self.vel_angle[index]
        new_pt = pt[0] + self.dt * self.vel * cos(theta), pt[1] + self.dt * self.vel * sin(theta)
        return new_pt if self.boundary.in_domain(new_pt) else self.reflect(pt, new_pt, index)

    def reflect(self, old_pt, new_pt, index):
        self.vel_angle[index] = self.boundary.reflect_velocity(old_pt, new_pt)
        return self.boundary.reflect_point(old_pt, new_pt)


## Implement randomized variant of Billiard motion.
# Each update, a sensor has a chance of randomly changing direction.
class RunAndTumble(BilliardMotion):

    ## Update angles before updating points.
    # Each update every point has a 1 : 5 chance of having its velocity
    # angle changed. Then update as normal.
    def update_point(self, pt: tuple, index: int) -> tuple:
        if random.randint(0, 5) == 4:
            self.vel_angle[index] = random.uniform(0, 2 * pi)
        return super().update_point(pt, index)




class CollectiveMotion(MotionModel):
    G = 1

    def __init__(self, dt, boundary, max_vel, n_int_sensors, sensing_radius, eta_scale_factor):
        super().__init__(dt, boundary)
        self.velocities = np.random.uniform(-max_vel, max_vel, (n_int_sensors, 2))
        self.n_sensors = n_int_sensors
        self.boundary = boundary  # just to make type checker happy
        self.sensing_radius = sensing_radius
        self.eta = eta_scale_factor * sensing_radius


    def update_point(self, pt: tuple, index: int) -> tuple:
        # Not used
        return pt


    def reflect(self, pt: tuple, index: int) -> tuple:
        v = self.velocities[index]
        theta = np.arctan2(v[1], v[0])
        norm_v = sqrt(v[0]**2 + v[1]**2)
        if pt[0] <= self.boundary.x_min or pt[0] >= self.boundary.x_max:
            theta = pi - theta
        if pt[1] <= self.boundary.y_min or pt[1] >= self.boundary.y_max:
            theta = - theta
        theta %= 2 * pi
        self.velocities[index] = (norm_v*cos(theta), norm_v*sin(theta))
        return pt


    def interaction_strength(self, pw_dist):
        def exp_seed(dist):
            if dist > 0:
                seed = np.exp(-1/(dist**2))
            else:
                seed = 0
            return seed
        bump_function = [0] * self.n_sensors
        for i in range(0, self.n_sensors):
            eta_term = exp_seed(self.eta + self.sensing_radius - pw_dist[i])
            dist_term = exp_seed(pw_dist[i] - self.sensing_radius)
            if eta_term + dist_term == 0:
                bump_function[i] = 0
            else:
                bump_function[i] = eta_term/(eta_term + dist_term)
        return bump_function


    def gradient(self, xs, ys):
        gradUx, gradUy = [0] * self.n_sensors, [0] * self.n_sensors
        for i in range(0, self.n_sensors):
            pairwise_dist = [norm((xs[i] - xs[j], ys[i] - ys[j])) for j in range(0, self.n_sensors)]
            gradUx[i] = self.G * sum((self.interaction_strength(pairwise_dist)[j]*(xs[i] - xs[j]))/pairwise_dist[j]**3
                                     for j in range(0, self.n_sensors) if pairwise_dist[j] != 0)
            gradUy[i] = self.G * sum((self.interaction_strength(pairwise_dist)[j]*(ys[i] - ys[j]))/pairwise_dist[j]**3
                                     for j in range(0, self.n_sensors) if pairwise_dist[j] != 0)
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


    def update_points(self, old_points: list) -> list:
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
                points[n] = self.reflect(pt, n)

        return old_points[0:len(self.boundary)] + points



class DorsognaModel(MotionModel):
    def __init__(self, dt, boundary, max_vel, n_int_sensors, sensing_radius, eta_scale_factor, DO_coeff):
        super().__init__(dt, boundary)
        self.velocities = np.random.uniform(-max_vel, max_vel, (n_int_sensors, 2))
        self.n_sensors = n_int_sensors
        self.boundary = boundary
        self.sensing_radius = sensing_radius
        self.eta = eta_scale_factor * sensing_radius
        self.DO_coeff = DO_coeff

    def update_point(self, pt: tuple, index: int) -> tuple:
        #Not used
        return pt

    def reflect(self, pt: tuple, index: int) -> tuple:
        v = self.velocities[index]
        theta = np.arctan2(v[1], v[0])
        norm_v = sqrt(v[0]**2 + v[1]**2)
        if pt[0] <= self.boundary.x_min or pt[0] >= self.boundary.x_max:
            theta = pi - theta
        if pt[1] <= self.boundary.y_min or pt[1] >= self.boundary.y_max:
            theta = - theta
        theta %= 2 * pi
        self.velocities[index] = (norm_v*cos(theta), norm_v*sin(theta))
        return pt



    def gradient(self, xs, ys):
        gradUx, gradUy = [0] * self.n_sensors, [0] * self.n_sensors

        for i in range(0, self.n_sensors):
            pairwise_dist = [0] * self.n_sensors
            attract_term, repel_term = [0] * self.n_sensors, [0] * self.n_sensors

            for j in range(0, self.n_sensors):
                if norm((xs[i] - xs[j], ys[i] - ys[j])) > 2 * self.sensing_radius:
                    pairwise_dist[j] = 0.0
                else:
                    pairwise_dist[j] = norm((xs[i] - xs[j], ys[i] - ys[j]))

            for j in range(0, self.n_sensors):
                if pairwise_dist[j] == 0.0:
                    attract_term[j] = 0
                    repel_term[j] = 0
                else:
                    attract_term[j] = self.DO_coeff[0]*np.exp(-pairwise_dist[j]/self.DO_coeff[1])/(self.DO_coeff[1]*pairwise_dist[j])
                    repel_term[j] = self.DO_coeff[2]*np.exp(-pairwise_dist[j]/self.DO_coeff[3])/(self.DO_coeff[3] * pairwise_dist[j])


            gradUx[i] = sum(((xs[i]-xs[j]) * attract_term[j]) - ((xs[i]-xs[j]) * repel_term[j])
                               for j in range(0, self.n_sensors) if pairwise_dist[j] != 0.0)
            gradUy[i] = sum(((ys[i]-ys[j]) * attract_term[j]) - ((ys[i]-ys[j]) * repel_term[j])
                               for j in range(0, self.n_sensors) if pairwise_dist[j] != 0.0)

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
        dvxdt = (1.5 - (0.5 * norm((dxdt, dydt))**2)) * dxdt - gradU[0]
        dvydt = (1.5 - (0.5 * norm((dxdt, dydt))**2)) * dydt - gradU[1]
        return np.concatenate([dxdt, dydt, dvxdt, dvydt])


    def update_points(self, old_points: list) -> list:
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
                points[n] = self.reflect(pt, n)

        return old_points[0:len(self.boundary)] + points

# Coefficients are: [C_attraction, L_attraction, C_repulsion, L_repulsion]
# DO_coeff = [0.5, 2, 0.5, 0.5]
# tuned_coeff_2 = [0.95, 1, 1, 0.1]