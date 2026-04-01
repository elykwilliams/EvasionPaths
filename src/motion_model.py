# ******************************************************************************
#  Copyright (c) 2023, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from numpy import array, random
from numpy.linalg import norm
from scipy.integrate import solve_ivp

from boundary_geometry import Domain
from sensor_network import Sensor


## This class provides the basic interface for a model of motion.
# it should do two things, update the point positions, and reflect
# points off of the boundary. Because the reflection depends on the
# boundary, motion models and boundaries must be compatible.
@dataclass
class MotionModel(ABC):
    def local_update(self, sensor, dt):
        if sensor.boundary_flag:
            return
        self.update_velocity(sensor, dt)
        self.update_position(sensor, dt)

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

    @staticmethod
    def reflect(domain: Domain, sensor: Sensor):
        old = sensor.old_pos
        while sensor.pos not in domain:
            intersect = domain.get_intersection_point(old, sensor.pos)
            if intersect is None:
                raise ValueError("Intersection point is None. Check the domain's intersection function.")

            unit_normal = domain.normal(intersect)
            if unit_normal is None:
                raise ValueError("Normal vector is None. Check the domain's normal function.")
            disp = sensor.pos - intersect
            sensor.pos -= 2 * np.dot(disp, unit_normal) * unit_normal

            V = sensor.pos - intersect
            unit_v = V / np.linalg.norm(V)
            sensor.vel = unit_v * np.linalg.norm(sensor.vel)

            # Keep the next segment anchored infinitesimally inside the domain.
            old = intersect - 1.0e-8 * unit_normal


# Velocity is constant
class BilliardMotion(MotionModel):
    def update_position(self, sensor, dt):
        sensor.pos = sensor.old_pos + dt * sensor.old_vel


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
@dataclass
class Viscek(BilliardMotion):
    large_dt: float  # averaging will be done in increments of large_dt
    radius: float  # the radius over which neighbors are averaged
    noise_scale: float = np.pi / 12

    def update_position(self, sensor, dt):
        # On the top-level timestep, nonlocal_update() has already written the
        # synchronized Vicsek heading into sensor.vel, so advance with that
        # direction instead of the previous step's velocity.
        if np.isclose(dt, self.large_dt):
            sensor.pos = sensor.old_pos + dt * sensor.vel
            return
        sensor.pos = sensor.old_pos + dt * sensor.old_vel

    ## Noise function.
    # Velocity can vary by as much as pi/12 in either direction.
    def eta(self):
        return self.noise_scale * random.uniform(-1, 1)

    ## Average velocity angles at each timestep.
    # This function will set the velocity angle to be the average 
    # of the velocity angle of sensors nearby, plus some noise.
    def nonlocal_update(self, sensors, dt):
        if not np.isclose(dt, self.large_dt):
            return
        if self.radius <= 0:
            return

        for s1 in sensors.mobile_sensors:
            neighbor_vels = [np.asarray(s2.old_vel, dtype=float) for s2 in sensors.mobile_sensors if s1.dist(s2) < self.radius]
            if not neighbor_vels:
                continue

            # Circular mean via unit vectors avoids branch cut issues near +/- pi.
            headings = np.asarray([v / (norm(v) + 1e-12) for v in neighbor_vels], dtype=float)
            mean_heading = np.mean(headings, axis=0)
            heading_norm = norm(mean_heading)
            if heading_norm < 1e-12:
                base_heading = np.asarray(s1.old_vel, dtype=float)
                heading_norm = norm(base_heading)
                if heading_norm < 1e-12:
                    continue
                mean_heading = base_heading / heading_norm
            else:
                mean_heading = mean_heading / heading_norm

            theta = (np.arctan2(mean_heading[1], mean_heading[0]) + self.eta()) % (2 * np.pi)
            speed = norm(s1.old_vel)
            s1.vel = speed * np.array([np.cos(theta), np.sin(theta)])


## Differential equation based motion model.
# Update sensor positions according to differential equation.
# Uses scipy solve_ivp to solve possibly nonlinear differential equations.
# This will solve the equation dx/dt = f(t, x) on the interval [0, dt] and return the
# solution at time dt with a tolerance to 1e-8.
# This class is setup to compute and store positions/velocities for all sensors
# in compute_update() and then return those values when requested in update().
class ODEMotion(MotionModel, ABC):

    def __init__(self):
        super().__init__()
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
    def __init__(self, sensing_radius, eta_scale_factor, coeff):
        super().__init__()
        assert len(coeff) != 4, 'Not enough parameters in coeff'
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
    df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(()))
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


@dataclass
class HomologicalDynamicsMotion(BilliardMotion):
    sensing_radius: float
    max_speed: float = 1.0
    lambda_shrink: float = 1.0
    mu_curvature: float = 0.5
    eta_cohesion: float = 0.2
    repulsion_strength: float = 0.1
    repulsion_power: float = 2.0
    d_safe_manual: float = 0.2
    auto_d_safe: bool = True

    _topology: Optional[object] = field(default=None, init=False, repr=False)
    _labels: Optional[Dict] = field(default=None, init=False, repr=False)
    _time: float = field(default=0.0, init=False, repr=False)
    _velocities: Dict[Sensor, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    @staticmethod
    def _rotate_clockwise(vec):
        return np.array([vec[1], -vec[0]], dtype=float)

    def set_homology_context(self, topology, labels, sim_time):
        self._topology = topology
        self._labels = dict(labels) if labels is not None else {}
        self._time = float(sim_time)

    @staticmethod
    def _cycle_area(cycle_nodes, points):
        poly = points[np.asarray(cycle_nodes, dtype=int)]
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

    @staticmethod
    def _cycle_perimeter(cycle_nodes, points):
        if len(cycle_nodes) < 2:
            return 0.0
        perimeter = 0.0
        m = len(cycle_nodes)
        for i in range(m):
            a = points[cycle_nodes[i]]
            b = points[cycle_nodes[(i + 1) % m]]
            perimeter += float(np.linalg.norm(b - a))
        return perimeter

    def _extract_uncovered_cycles(self):
        if self._topology is None or self._labels is None:
            return []

        topology = self._topology
        if getattr(topology, "dim", None) != 2:
            raise NotImplementedError("HomologicalDynamicsMotion currently supports 2D topologies only.")

        try:
            is_excluded = topology.is_excluded_cycle
        except Exception:
            is_excluded = lambda _cycle: False

        uncovered = []
        seen = set()
        for cycle, is_true in self._labels.items():
            if not bool(is_true):
                continue
            if is_excluded(cycle):
                continue
            try:
                if not topology.is_connected_cycle(cycle):
                    continue
            except Exception:
                continue

            try:
                simplex = next(iter(cycle))
                nodes = topology.cmap.get_cycle_nodes(simplex)
            except Exception:
                nodes = list(cycle.nodes)

            nodes = [int(v) for v in nodes]
            if len(nodes) > 1 and nodes[0] == nodes[-1]:
                nodes = nodes[:-1]
            if len(nodes) < 3:
                continue

            key = tuple(nodes)
            if key in seen:
                continue
            seen.add(key)
            uncovered.append(nodes)
        return uncovered

    def _normalize_cycle_nodes(self, cycle_nodes, points):
        nodes = [int(v) for v in cycle_nodes]
        if len(nodes) > 1 and nodes[0] == nodes[-1]:
            nodes = nodes[:-1]
        if len(nodes) < 3:
            return None
        if self._cycle_area(nodes, points) < 0.0:
            nodes = list(reversed(nodes))
        return nodes

    def _extract_uncovered_cycle_metadata(self, points):
        metadata = []
        seen = set()
        for cycle in self._extract_uncovered_cycles():
            nodes = self._normalize_cycle_nodes(cycle, points)
            if nodes is None:
                continue
            key = tuple(nodes)
            if key in seen:
                continue
            seen.add(key)
            metadata.append(
                {
                    "nodes": nodes,
                    "node_set": frozenset(nodes),
                    "size": len(nodes),
                    "area": abs(self._cycle_area(nodes, points)),
                    "perimeter": self._cycle_perimeter(nodes, points),
                    "force_weight": 1.0,
                }
            )
        return metadata

    def _selected_cycle_metadata(self, points):
        return self._extract_uncovered_cycle_metadata(points)

    def _effective_d_safe(self, n_mobile, domain_area):
        if not self.auto_d_safe:
            return max(0.0, float(self.d_safe_manual))
        if domain_area <= 0.0:
            return max(0.0, float(self.d_safe_manual))
        rho = min(1.0, (float(n_mobile) * np.pi * self.sensing_radius * self.sensing_radius) / float(domain_area))
        return float(np.sqrt(3.0) * self.sensing_radius * np.sqrt(max(0.0, rho)))

    def nonlocal_update(self, sensor_network, _):
        self._velocities = {}
        if self._topology is None or self._labels is None:
            for sensor in sensor_network.mobile_sensors:
                self._velocities[sensor] = np.array(sensor.old_vel, dtype=float)
            return

        if self._topology.dim != 2:
            raise NotImplementedError("HomologicalDynamicsMotion currently supports 2D topologies only.")

        points = np.asarray(sensor_network.points, dtype=float)
        n_fence = len(sensor_network.fence_sensors)
        n_total = points.shape[0]
        n_mobile = len(sensor_network.mobile_sensors)
        domain = sensor_network.domain
        domain_area = float((domain.max[0] - domain.min[0]) * (domain.max[1] - domain.min[1])) if hasattr(domain, "min") and hasattr(domain, "max") else 1.0
        d_safe = self._effective_d_safe(n_mobile=n_mobile, domain_area=domain_area)
        eps = 1e-9

        edges = [tuple(sorted(int(v) for v in edge)) for edge in self._topology.simplices(1)]
        adjacency = [set() for _ in range(n_total)]
        for i, j in edges:
            if 0 <= i < n_total and 0 <= j < n_total and i != j:
                adjacency[i].add(j)
                adjacency[j].add(i)

        cycle_metadata = self._selected_cycle_metadata(points)
        shrink_force = np.zeros_like(points)
        curvature_force = np.zeros_like(points)
        cohesion_force = np.zeros_like(points)
        repulsion_force = np.zeros_like(points)

        cycle_nodes_for_repulsion = []
        cycle_adjacency_pairs = set()
        active_cycle_nodes = set()

        for meta in cycle_metadata:
            c = list(meta["nodes"])
            cycle_weight = float(meta.get("force_weight", 1.0))
            cycle_nodes_for_repulsion.append(c)
            m = len(c)

            for idx, node in enumerate(c):
                prev_node = c[(idx - 1) % m]
                next_node = c[(idx + 1) % m]
                active_cycle_nodes.add(node)
                cycle_adjacency_pairs.add(tuple(sorted((node, prev_node))))
                cycle_adjacency_pairs.add(tuple(sorted((node, next_node))))

                x_i = points[node]
                x_p = points[prev_node]
                x_q = points[next_node]
                edge_prev = x_i - x_p
                edge_next = x_q - x_i
                l_prev = float(np.linalg.norm(edge_prev))
                l_next = float(np.linalg.norm(edge_next))

                grad_area = 0.5 * self._rotate_clockwise(x_q - x_p)
                shrink_force[node] += -cycle_weight * self.lambda_shrink * grad_area

                if l_prev >= eps and l_next >= eps:
                    grad_perimeter = (x_i - x_p) / l_prev - (x_q - x_i) / l_next
                    curvature_force[node] += -cycle_weight * self.mu_curvature * grad_perimeter

        for i in range(n_total):
            if i < n_fence and i not in active_cycle_nodes:
                continue
            for j in adjacency[i]:
                if j <= i:
                    continue
                delta = points[j] - points[i]
                cohesion_force[i] += self.eta_cohesion * delta
                cohesion_force[j] -= self.eta_cohesion * delta

        if d_safe > eps:
            for meta, cycle in zip(cycle_metadata, cycle_nodes_for_repulsion):
                cycle_weight = float(meta.get("force_weight", 1.0))
                m = len(cycle)
                for i in range(m):
                    ii = cycle[i]
                    for j in range(i + 1, m):
                        jj = cycle[j]
                        if tuple(sorted((ii, jj))) in cycle_adjacency_pairs:
                            continue
                        delta = points[ii] - points[jj]
                        dist = float(np.linalg.norm(delta))
                        if dist < eps or dist >= d_safe:
                            continue
                        denom = (dist * dist + eps * eps) ** (0.5 * self.repulsion_power + 1.0)
                        grad = (
                            cycle_weight
                            *
                            self.repulsion_power
                            * self.repulsion_strength
                            * (points[ii] - points[jj])
                            / denom
                        )
                        repulsion_force[ii] += grad
                        repulsion_force[jj] -= grad

        velocity = shrink_force + curvature_force + cohesion_force + repulsion_force

        velocity[:n_fence] = 0.0

        speeds = np.linalg.norm(velocity, axis=1)
        too_fast = speeds > self.max_speed
        velocity[too_fast] = velocity[too_fast] / speeds[too_fast, None] * self.max_speed

        for mobile_i, sensor in enumerate(sensor_network.mobile_sensors):
            global_i = n_fence + mobile_i
            self._velocities[sensor] = np.array(velocity[global_i], dtype=float)

    def update_velocity(self, sensor, _):
        sensor.vel = np.array(self._velocities.get(sensor, sensor.old_vel), dtype=float)


@dataclass
class SequentialHomologicalMotion(HomologicalDynamicsMotion):
    overlap_threshold: float = 0.3

    _target_cycle_nodes: Optional[tuple] = field(default=None, init=False, repr=False)
    _target_cycle_nodeset: Optional[frozenset] = field(default=None, init=False, repr=False)
    _active_target_nodesets: tuple = field(default_factory=tuple, init=False, repr=False)
    _phase_index: int = field(default=0, init=False, repr=False)

    @staticmethod
    def _jaccard_overlap(left, right):
        if not left or not right:
            return 0.0
        intersection = len(left & right)
        union = len(left | right)
        if union == 0:
            return 0.0
        return float(intersection) / float(union)

    def _select_new_target(self, cycle_metadata):
        if not cycle_metadata:
            self._target_cycle_nodes = None
            self._target_cycle_nodeset = None
            self._active_target_nodesets = ()
            return None

        target = min(
            cycle_metadata,
            key=lambda meta: (meta["size"], meta["area"], meta["perimeter"]),
        )
        self._target_cycle_nodes = tuple(target["nodes"])
        self._target_cycle_nodeset = target["node_set"]
        self._active_target_nodesets = (target["node_set"],)
        self._phase_index += 1
        return target

    def _refresh_target(self, cycle_metadata):
        if not cycle_metadata:
            self._target_cycle_nodes = None
            self._target_cycle_nodeset = None
            self._active_target_nodesets = ()
            return []

        if self._target_cycle_nodeset is None:
            target = self._select_new_target(cycle_metadata)
            return [target] if target is not None else []

        old_target_nodeset = self._target_cycle_nodeset

        best = max(
            cycle_metadata,
            key=lambda meta: (
                self._jaccard_overlap(self._target_cycle_nodeset, meta["node_set"]),
                meta["size"],
                meta["area"],
            ),
        )
        overlap = self._jaccard_overlap(self._target_cycle_nodeset, best["node_set"])
        if overlap >= float(self.overlap_threshold):
            self._target_cycle_nodes = tuple(best["nodes"])
            self._target_cycle_nodeset = best["node_set"]
            primary = best
        else:
            primary = self._select_new_target(cycle_metadata)
            return [primary] if primary is not None else []

        descendants = []
        for meta in cycle_metadata:
            node_set = meta["node_set"]
            if node_set == primary["node_set"]:
                continue
            if node_set < old_target_nodeset:
                descendants.append(meta)

        descendants.sort(key=lambda meta: (meta["size"], meta["area"], meta["perimeter"]))
        active = [primary] + descendants
        primary_size = max(1, int(primary["size"]))
        for meta in active:
            meta["force_weight"] = float(primary_size) / float(max(1, int(meta["size"])))
        self._active_target_nodesets = tuple(meta["node_set"] for meta in active)
        return active

    def _selected_cycle_metadata(self, points):
        cycle_metadata = self._extract_uncovered_cycle_metadata(points)
        return self._refresh_target(cycle_metadata)
