# ************************************************************
# Copyright (c) 2021, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************

from utilities import cart2pol, pol2cart
from boundary_geometry import Domain

import numpy as np
from numpy import array, random
from numpy.linalg import norm
from scipy.integrate import solve_ivp
from abc import ABC, abstractmethod


## Compute distance between two sensors.
# TODO move to sensor tools
def dist(s1, s2):
    return norm(array(s1.old_pos) - array(s2.old_pos))


## This class provides the basic interface for a model of motion.
# it should do two things, update the point positions, and reflect
# points off of the boundary. Because the reflection depends on the
# boundary, motion models and boundaries must be compatible.
class MotionModel(ABC):

    ## Initialize motion model with boundary.
    def __init__(self, domain: Domain) -> None:
        self.domain = domain

    ## Update position individual point.
    # This function will be called on all points.
    # Implementations should do the following:
    #   1. move sensor
    #   2. update velocity
    #   3. check if reflection is needed
    @abstractmethod
    def update_position(self, sensor, dt):
        if sensor.position not in self.domain:
            self.reflect(sensor)

    ## Elastic reflection off boundary wall.
    # This function should not need to be over written unless
    # an inelastic collision is needed. This function adjusts
    # the sensor position to where it should be, and the new
    # velocity angle.
    def reflect(self, sensor):
        sensor.pvel[1] = self.domain.reflect_velocity(sensor.old_pos, sensor.position)
        sensor.position = self.domain.reflect_point(sensor.old_pos, sensor.position)

    ## Compute any nonlocal updates.
    # This function should be called before update_position().
    # It should compute any updates needed so that update_position()
    # is a strictly local function. For example: If there any equations
    # to be solved, points and velocities should be pre-computed here,
    # then update the sensors when update_position is called.
    def compute_update(self, all_sensors, dt):
        pass

    ## Initialize sensor velocity.
    # Velocity is stored in polar form. vel_mag is a parameter that
    # can be used to indicate a scale of magnitude.
    @staticmethod
    @abstractmethod
    def initial_pvel(vel_mag):
        return None


## Provide random motion for rectangular domain.
# Will move a point randomly with an average step
# size of sigma*dt
class BrownianMotion(MotionModel):

    ## Initialize boundary with typical velocity.
    def __init__(self, domain: Domain, sigma: float) -> None:
        super().__init__(domain)
        self.sigma = sigma

    # TODO document
    @staticmethod
    def initial_pvel(vel_mag):
        return None

    ## Random function.
    def epsilon(self, dt) -> float:
        return self.sigma * np.sqrt(dt) * random.normal(0, 1, 2)

    # TODO Document
    def update_position(self, sensor, dt):
        sensor.position = array(sensor.old_pos) + self.epsilon(dt)
        if sensor.position not in self.domain:
            self.reflect(sensor)


## Implement Billiard Motion for Rectangular Domain.
# All sensors will have same velocity bit will have random angles.
# Points will move a distance of vel*dt each update.
class BilliardMotion(MotionModel):

    # TODO Document
    @staticmethod
    def initial_pvel(vel_mag):
        return array([vel_mag, random.uniform(0, 2*np.pi)])

    ## Update point using x = x + v*dt.
    def update_position(self, sensor, dt):
        vel = array(pol2cart(sensor.pvel))
        sensor.position = array(sensor.old_pos) + dt*vel
        if sensor.position not in self.domain:
            self.reflect(sensor)


## Implement randomized variant of Billiard motion.
# Each update, a sensor has a chance of randomly changing direction.
class RunAndTumble(BilliardMotion):
    def __init__(self, domain: Domain, dt: float):
        super().__init__(domain)
        self.large_dt = dt

    ## Update angles before updating points.
    # Each update every point has a 1 : 5 chance of having its velocity
    # angle changed. Then update as normal.
    def compute_update(self, all_sensors, dt):
        if self.large_dt == dt:
            for sensor in all_sensors.mobile_sensors:
                if random.randint(0, 5) == 4:
                    sensor.old_pvel[1] = random.uniform(0, 2 * np.pi)


# TODO Document
# TODO use sensors.fence sensors not domain
class Viscek(BilliardMotion):

    def __init__(self, domain: Domain, dt: float, sensing_radius: float):
        super().__init__(domain)
        self.large_dt = dt
        self.radius = sensing_radius

    ## Noise function.
    # Velocity can vary by as much as pi/12 in either direction.
    @staticmethod
    def eta():
        return (np.pi / 12) * random.uniform(-1, 1)

    ## Average velocity angles at each timestep.
    # This function will set the velocity angle to be the average 
    # of the velocity angle of sensors nearby, plus some noise.
    def compute_update(self, sensors, dt):
        if dt == self.large_dt:
            for s1 in sensors.mobile_sensors:
                sensor_angles = [s2.old_vel_angle for s2 in sensors.mobile_sensors if dist(s1, s2) < self.radius]
                s1.old_pvel[1] = (np.mean(sensor_angles) + self.eta()) % (2 * np.pi)


# TODO Document
# TODO use sensors.fence sensors not domain
class ODEMotion(MotionModel, ABC):
    def __init__(self, domain):
        super().__init__(domain)
        self.n_sensors = 0
        self.points = dict()
        self.velocities = dict()

    @abstractmethod
    def time_derivative(self, t, state):
        pass

    def compute_update(self, sensors, dt):
        self.n_sensors = len(sensors.mobile_sensors)
        # Put into form ode solver wants [ xs | ys | vxs | vys ]
        xs = [s.old_pos[0] for s in sensors.mobile_sensors]
        ys = [s.old_pos[1] for s in sensors.mobile_sensors]
        vxs = [pol2cart(s.old_pvel)[0] for s in sensors.mobile_sensors]
        vys = [pol2cart(s.old_pvel)[1] for s in sensors.mobile_sensors]
        init_val = xs + ys + vxs + vys

        # Solve with init_val as t=0, solve for values at t+dt
        new_val = solve_ivp(self.time_derivative, [0, dt], init_val, t_eval=[dt], rtol=1e-8)

        # Convert back from np array
        new_val = new_val.y

        # split state back into position and velocity,
        split_state = [[y[0] for y in new_val[i * self.n_sensors:(i + 1) * self.n_sensors]] for i in range(4)]

        # zip into list of tuples
        self.velocities = dict(zip(sensors.mobile_sensors, zip(split_state[2], split_state[3])))

        # Reflect any points outside boundary
        self.points = dict(zip(sensors.mobile_sensors, zip(split_state[0], split_state[1])))

    def update_position(self, sensor, dt):
        sensor.pvel = cart2pol(self.velocities[sensor])
        sensor.position = self.points[sensor]
        if sensor.position not in self.domain:
            self.reflect(sensor)


# TODO Document
class Dorsogna(ODEMotion):
    def __init__(self, domain, sensing_radius, eta_scale_factor, coeff):
        assert len(coeff) != 4, 'Not enough parameters in DO_coeff'
        super().__init__(domain)
        self.sensing_radius = sensing_radius
        self.eta = eta_scale_factor * sensing_radius
        self.DO_coeff = coeff

    @staticmethod
    def initial_pvel(vel_mag):
        return cart2pol(np.random.uniform(-vel_mag, vel_mag, 2))

    def gradient(self, xs, ys):
        grad_x, grad_y = np.zeros(self.n_sensors), np.zeros(self.n_sensors)

        for i in range(self.n_sensors):
            for j in range(self.n_sensors):
                r = norm((xs[i] - xs[j], ys[i] - ys[j]))
                if r < 2 * self.sensing_radius and i != j:
                    attract_term = (self.DO_coeff[0] * np.exp(-r / self.DO_coeff[1]) / (self.DO_coeff[1] * r))
                    repel_term = (self.DO_coeff[2] * np.exp(-r / self.DO_coeff[3]) / (self.DO_coeff[3] * r))

                    grad_x[i] += (xs[i] - xs[j]) * attract_term - (xs[i] - xs[j]) * repel_term
                    grad_y[i] += (ys[i] - ys[j]) * attract_term - (ys[i] - ys[j]) * repel_term

        return {'x': array(grad_x), 'y': array(grad_y)}

    def time_derivative(self, _, state):
        # ode solver gives us np array in the form [xvals | yvals | vxvals | vyvals]
        # split into individual np array
        n = len(state) // 4
        xs, ys, *v = (state[i*n:(i+1)*n] for i in range(4))
        grad = self.gradient(xs, ys)

        # Need to compute time derivative of each, we obviously have dxdt, dydt = vx, vy
        # use following for dvdt, coor is 'x' or 'y'
        v = dict(zip('xy', v))

        def accelerate(sensor, dim):
            return (1.5 - (0.5 * norm((v['x'][sensor], v['y'][sensor])) ** 2)) * v[dim][sensor] - grad[dim][sensor]

        dvdt = {dim: [accelerate(sensor, dim) for sensor in range(self.n_sensors)] for dim in 'xy'}
        return np.concatenate([v['x'], v['y'], dvdt['x'], dvdt['y']])
