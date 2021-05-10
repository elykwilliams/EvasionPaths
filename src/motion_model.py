# ************************************************************
# Copyright (c) 2020, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************
import math

from boundary_geometry import *
from numpy import sqrt, random, sin, cos, pi, mean
from numpy.linalg import norm
from numpy import array
from scipy.integrate import solve_ivp
from abc import ABC, abstractmethod


def cart2pol(points):
    return array([(norm(p), math.atan2(p[1], p[0])) for p in points])


## Compute distance between two sensors.
def dist(s1, s2):
    return norm(array(s1.old_pos) - array(s2.old_pos))


## This class provides the basic interface for a model of motion.
# it should do two things, update the point positions, and reflect
# points off of the boundary. Because the reflection depends on the
# boundary, motion models and boundaries must be compatible.
class MotionModel(ABC):

    ## Initialize motion model with boundary and time-scale
    def __init__(self, boundary: Boundary) -> None:
        self.boundary = boundary

    ## Update an individual point.
    # This function will be called on all points.
    # reflection should be separate. The index is the position
    # in the set of ALL points, and can be useful in looking up
    # sensor specific data. Will return new position or sensor.
    @abstractmethod
    def update_position(self, sensor, dt):
        pass

    def reflect(self, sensor):
        sensor.pvel[1] = self.boundary.reflect_velocity(sensor.old_pos, sensor.position)
        sensor.position = self.boundary.reflect_point(sensor.old_pos, sensor.position)

    def compute_update(self, all_sensors, dt):
        pass

    @staticmethod
    def initial_pvel(vel_mag):
        return None


## Provide random motion for rectangular domain.
# Will move a point randomly with an average step
# size of sigma*dt
class BrownianMotion(MotionModel):

    ## Initialize boundary with typical velocity.
    def __init__(self, boundary: Boundary, sigma: float) -> None:
        super().__init__(boundary)
        self.sigma = sigma

    ## Random function.
    def epsilon(self, dt) -> float:
        return self.sigma * sqrt(dt) * random.normal(0, 1, 2)

    def update_position(self, sensor, dt):
        sensor.position = array(sensor.old_pos) + self.epsilon(dt)
        if not self.boundary.in_domain(sensor.position):
            self.reflect(sensor)


## Implement Billiard Motion for Rectangular Domain.
# All sensors will have same velocity bit will have random angles.
# Points will move a distance of vel*dt each update.
class BilliardMotion(MotionModel):

    @staticmethod
    def initial_pvel(vel_mag):
        return array([vel_mag, random.uniform(0, 2*np.pi)])

    ## Update point using x = x + v*dt.
    def update_position(self, sensor, dt):
        vel = sensor.old_pvel[0] * array([cos(sensor.old_pvel[1]), sin(sensor.old_pvel[1])])
        sensor.position = array(sensor.old_pos) + dt*vel
        if not self.boundary.in_domain(sensor.position):
            self.reflect(sensor)


## Implement randomized variant of Billiard motion.
# Each update, a sensor has a chance of randomly changing direction.
class RunAndTumble(BilliardMotion):
    def __init__(self,  boundary: Boundary, dt: float):
        super().__init__(boundary)
        self.large_dt = dt

    ## Update angles before updating points.
    # Each update every point has a 1 : 5 chance of having its velocity
    # angle changed. Then update as normal.
    def compute_update(self, all_sensors, dt):
        if self.large_dt == dt:
            for sensor in all_sensors.mobile_sensors:
                if random.randint(0, 5) == 4:
                    sensor.old_pvel[1] = random.uniform(0, 2 * pi)


class Viscek(BilliardMotion):

    def __init__(self, boundary: Boundary, dt: float, sensing_radius: float):
        super().__init__(boundary)
        self.large_dt = dt
        self.radius = sensing_radius

    ## Noise function.
    # Velocity can vary by as much as pi/12 in either direction.
    @staticmethod
    def eta():
        return (pi / 12) * random.uniform(-1, 1)

    ## Average velocity angles at each timestep.
    # This function will set the velocity angle to be the average 
    # of the velocity angle of sensors nearby, plus some noise.
    def compute_update(self, sensors, dt):
        if dt == self.large_dt:
            for s1 in sensors.mobile_sensors:
                sensor_angles = [s2.old_vel_angle for s2 in sensors.mobile_sensors if dist(s1, s2) < self.radius]
                s1.old_pvel[1] = (mean(sensor_angles) + self.eta()) % (2 * pi)


class ODEMotion(MotionModel, ABC):
    def __init__(self, boundary):
        super().__init__(boundary)
        self.n_sensors = 0
        self.points = dict()
        self.velocities = dict()

    @abstractmethod
    def time_derivative(self, _, state):
        pass

    def compute_update(self, sensors, dt):
        self.n_sensors = len(sensors.mobile_sensors)
        # Put into form ode solver wants [ xs | ys | vxs | vys ]
        xs = [s.old_pos[0] for s in sensors.mobile_sensors]
        ys = [s.old_pos[1] for s in sensors.mobile_sensors]
        vxs = [s.old_pvel[0]*cos(s.old_pvel[1]) for s in sensors.mobile_sensors]
        vys = [s.old_pvel[0]*sin(s.old_pvel[1]) for s in sensors.mobile_sensors]
        init_val = xs + ys + vxs + vys

        # Solve with init_val as t=0, solve for values at t+dt
        new_val = solve_ivp(self.time_derivative, [0, dt], init_val, t_eval=[dt], rtol=1e-8)

        # Convert back from np array
        new_val = new_val.y.tolist()

        # split state back into position and velocity,
        split_state = [[y[0] for y in new_val[i:i + self.n_sensors]] for i in range(0, len(new_val), self.n_sensors)]

        # zip into list of tuples
        self.velocities = dict(zip(sensors, zip(split_state[2], split_state[3])))

        # Reflect any points outside boundary
        self.points = dict(zip(sensors, zip(split_state[0], split_state[1])))

    def update_position(self, sensor, dt):
        sensor.pvel = cart2pol(self.velocities[sensor])
        sensor.position = self.points[sensor]
        if not self.boundary.in_domain(sensor.position):
            self.reflect(sensor)


class Dorsogna(ODEMotion):
    def __init__(self, boundary, sensing_radius, eta_scale_factor, DO_coeff):
        assert len(DO_coeff) != 4, "Not enough parameters in DO_coeff"
        super().__init__(boundary)
        self.sensing_radius = sensing_radius
        self.eta = eta_scale_factor * sensing_radius
        self.DO_coeff = DO_coeff

    @staticmethod
    def initial_pvel(vel_mag):
        return cart2pol(np.random.uniform(-vel_mag, vel_mag, 2))

    def gradient(self, xs, ys):
        gradUx, gradUy = [0.0] * self.n_sensors, [0.0] * self.n_sensors

        for i in range(0, self.n_sensors):
            for j in range(self.n_sensors):
                r = norm((xs[i] - xs[j], ys[i] - ys[j]))
                if 0.0 < r < 2 * self.sensing_radius:
                    attract_term = (self.DO_coeff[0] * np.exp(-r / self.DO_coeff[1]) / (self.DO_coeff[1] * r))
                    repel_term = (self.DO_coeff[2] * np.exp(-r / self.DO_coeff[3]) / (self.DO_coeff[3] * r))

                    gradUx[i] += (xs[i] - xs[j]) * attract_term - (xs[i] - xs[j]) * repel_term
                    gradUy[i] += (ys[i] - ys[j]) * attract_term - (ys[i] - ys[j]) * repel_term

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
        dvxdt = array(self.n_sensors * [0])
        dvydt = array(self.n_sensors * [0])
        for i in range(self.n_sensors):
            dvxdt[i] = (1.5 - (0.5 * norm((dxdt[i], dydt[i])) ** 2)) * dxdt[i] - gradU[0][i]
            dvydt[i] = (1.5 - (0.5 * norm((dxdt[i], dydt[i])) ** 2)) * dydt[i] - gradU[1][i]
        return np.concatenate([dxdt, dydt, dvxdt, dvydt])