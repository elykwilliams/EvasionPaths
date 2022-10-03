# ************************************************************
# Copyright (c) 2021, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy import array, random
from numpy.linalg import norm
from scipy.integrate import solve_ivp

from boundary_geometry import Domain


## This class provides the basic interface for a model of motion.
# it should do two things, update the point positions, and reflect
# points off of the boundary. Because the reflection depends on the
# boundary, motion models and boundaries must be compatible.
@dataclass
class MotionModel(ABC):
    domain: Domain

    def local_update(self, sensor, dt):
        if sensor.boundary_flag:
            return
        self.update_velocity(sensor, dt)
        self.update_position(sensor, dt)
        if sensor.pos not in self.domain:
            self.domain.reflect(sensor)

    @abstractmethod
    def update_position(self, sensor, dt):
        ...

    def update_velocity(self, sensor, dt):
        pass

    ## Compute any nonlocal updates.
    # This function should be called before update_position().
    # It should compute any updates needed so that update_position()
    # is a strictly local function. For example: If there any equations
    # to be solved, points and velocities should be pre-computed here,
    # then update the sensors when update_position is called.
    def nonlocal_update(self, all_sensors, dt):
        pass


# Velocity is constant
class BilliardMotion(MotionModel):
    def update_position(self, sensor, dt):
        sensor.pos = sensor.old_pos + dt * sensor.vel


## Provide random motion.
# Will move a point randomly with an average step size of sigma*dt
@dataclass
class BrownianMotion(BilliardMotion):
    sigma: float
    large_dt: float

    def epsilon(self) -> float:
        return self.sigma * random.normal(0, 1, 2)

    def update_velocity(self, sensor, dt):
        if self.large_dt == dt:  # only sample in intervals of dt
            sensor.vel = self.epsilon() / np.sqrt(dt)


## Implement randomized variant of Billiard motion.
# Each update, a sensor has a chance of randomly changing direction.
@dataclass
class RunAndTumble(BilliardMotion):
    large_dt: float

    def update_velocity(self, sensor, dt):
        if self.large_dt != dt:
            return
        if random.randint(0, 5) == 4:
            vel = norm(sensor.vel)
            sensor.vel = self.initial_vel(vel)
            sensor.old_vel = sensor.vel


## The Viscek Model of Motion.
# This model is a variation of billiard motion where sensor's
# velocity is averaged with the neighboring sensor's velocity.
# Optional noise can be added to the system.
# The averaging is only done on the top level time-step,
# substeps the motion is simply billiard motion.
# TODO find reference
class Viscek(BilliardMotion):
    large_dt: float  # averaging will be done in increments of large_dt
    radius: float  # the radius of sensors that will be averaged.

    ## Noise function.
    # Velocity can vary by as much as pi/12 in either direction.
    @staticmethod
    def eta():
        return (np.pi / 12) * random.uniform(-1, 1)

    ## Average velocity angles at each timestep.
    # This function will set the velocity angle to be the average 
    # of the velocity angle of sensors nearby, plus some noise.
    def nonlocal_update(self, sensors, dt):
        if dt != self.large_dt:
            return
        for s1 in sensors.mobile_sensors:
            sensor_angles = [np.arctan2(s2.old_vel[1], s2.old_vel[0]) for s2 in sensors.mobile_sensors
                             if s1.dist(s2) < self.radius]
            theta = (np.mean(sensor_angles) + self.eta()) % (2 * np.pi)
            s1.vel = norm(s1.vel) * np.array([np.cos(theta), np.sin(theta)])


## Differential equation based motion model.
# Update sensor positions according to differential equation.
# Uses scipy solve_ivp to solve possibly nonlinear differential equations.
# This will solve the equation dx/dt = f(t, x) on the interval [0, dt] and return the
# solution at time dt with a tolerance to 1e-8.
# This class is setup to compute and store positions/velocities for all sensors
# in compute_update() and then return those values when requested in update().
class ODEMotion(MotionModel, ABC):

    def __init__(self, domain):
        super().__init__(domain)
        self.n_sensors = 0
        self.points = dict()
        self.velocities = dict()

    ## The right hand side function f(t, x).
    @abstractmethod
    def time_derivative(self, t, state):
        pass

    ## Solve dx/dt = f(t, x) with x(t0) = old_pos.
    # Compute value x(t0 + dt) at each mobile sensor, store for later.
    # differential equation does not take fence sensors into account.
    def nonlocal_update(self, sensors, dt):
        self.n_sensors = len(sensors.mobile_sensors)
        # Put into form ode solver wants [ xs | ys | vxs | vys ]
        xs = [s.old_pos[0] for s in sensors.mobile_sensors]
        ys = [s.old_pos[1] for s in sensors.mobile_sensors]
        vxs = [s.old_vel[0] for s in sensors.mobile_sensors]
        vys = [s.old_vel[1] for s in sensors.mobile_sensors]
        init_val = xs + ys + vxs + vys

        # Solve with init_val as t=0, solve for values at t+dt
        solution = solve_ivp(self.time_derivative, [0, dt], init_val, t_eval=[dt], rtol=1e-8)

        # Convert back from np array
        solution = solution.y[:, 0]

        # split state back into position and velocity,
        xs, ys, *v = [[val for val in solution[i * self.n_sensors:(i + 1) * self.n_sensors]] for i in range(4)]

        # zip into list of tuples
        self.velocities = dict(zip(sensors.mobile_sensors, zip(*v)))
        self.points = dict(zip(sensors.mobile_sensors, zip(xs, ys)))

    ## Retrieve precomputed position and velocity for given sensor.
    # This function is called *after* compute_update() has been called.
    # it will simply retrieve the precomputed position and velocities.
    def update_position(self, sensor, _):
        sensor.pos = np.array(self.points[sensor])

    def update_velocity(self, sensor, _):
        sensor.vel = np.array(self.velocities[sensor])


## Motion using the D'Orsogna model of motion.
# TODO find reference
# The idea behind this model is for sensors to have some
# sort of potential so that when sensors are far away from each
# other they attract, but when they are close together they repel.
class Dorsogna(ODEMotion):

    ## Initialize parameters.
    # eta_scale_factor is not used, it was previously used to implement a partition of
    # unity so that the gradient was Cinfinty but still went to 0 outside a given radius.
    # TODO rename and document coeff parameters, give sample values
    def __init__(self, domain, sensing_radius, eta_scale_factor, coeff):
        assert len(coeff) != 4, 'Not enough parameters in coeff'
        super().__init__(domain)
        self.sensing_radius = sensing_radius
        self.eta = eta_scale_factor * sensing_radius
        self.DO_coeff = coeff

    ## Sensor "potential" function.
    # Designed to attract at large scales and repel at small scales.
    # only interacts with neighbor sensors.
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

    ## The right hand side function f(t, x).
    # we have the system
    #       dx_i/dt = v_i, dv_i/dt = (1.5 - (1/2 || v_i ||**2 v_i) ) - grad(i)
    # ode solver gives us np array in the form [xvals | yvals | vxvals | vyvals]
    # return to solver in the same format.
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

@dataclass
class TimeseriesData(MotionModel):
    filename: str
    large_dt: float
    step_no: int = 1
    df: pd.DataFrame = pd.DataFrame(())
    next_time_positions: np.array = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        self.df = pd.read_csv(self.filename)
        self.current_row = self.from_dataframe(0)

    def update_position(self, sensor_i, _):
        return

    def nonlocal_update(self, sensor_network, dt):
        if dt == self.large_dt:
            self.next_time_positions = self.from_dataframe(self.step_no)
            self.step_no += 1

        for i, sensor in enumerate(sensor_network.mobile_sensors):
            old_position = np.array(sensor.pos)
            new_position = np.array(self.next_time_positions[i])
            sensor.pos = new_position * dt + (1-dt) * old_position

    def from_dataframe(self, row):
        sensors = []
        # For the data we can use something like route: "../setup_data/fence.csv"
        for col in range(len(self.df.columns)):
            entry = self.df.iloc[row, col][1:-1].split()
            coordinates = list(map(float, entry))
            sensors.append(coordinates)
        return sensors


