# ******************************************************************************
#  Copyright (c) 2024, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy import arange, pi
from numpy.linalg import norm

from utilities import pol2cart
from alpha_complex import AlphaComplex
from sensor_network import Sensor

## Specify domain geometry.
# The domain class handles all issues relating to the
# domain geometry. It should be unaware of sensors and
# operate with strictly geometric objects. A user defined
# domain should
# - Determine if a point is in the domain
# - provide the outward UNIT normal for any point on the boundary
# - given a point inside and a point outside, determine the intersection
#   point of the segment connecting these two points with the domain.
# - generate random points inside the domain
# - optionally provide fence sensor positions
# - optionally provide boundary points for plotting
@dataclass
class Domain(ABC):

    @abstractmethod
    def __contains__(self, point: tuple) -> bool:
        ...

    @abstractmethod
    def point_generator(self, n_sensors: int) -> list:
        ...

    ## Return points along the boundary for plotting (not the fence).
    # Move to plotting tools
    @abstractmethod
    def domain_boundary_points(self):
        ...

    @abstractmethod
    def normal(self, boundary_point):
        pass

    @abstractmethod
    def get_intersection_point(self, old, new):
        pass


## A rectangular domain.
# This domain implements a physical boundary separate from the fence so that
# sensors don't get too close and mess up the associated boundary cycle.
# The input parameters specify the dimension of the desired virtual boundary
# and the physical locations of the sensors are places slightly outside of
# this boundary in a way that still allows interior sensors to form simplices
# with fence sensors.
class RectangularDomain(Domain):

    ## Initialize with dimension of desired boundary.
    # Sensor positions will be reflected so that interior sensors stay in the
    # specified domain. Defaults to unit square
    def __init__(self,
                 x_min: float = 0, x_max: float = 1,
                 y_min: float = 0, y_max: float = 1) -> None:
        self.max = (x_max, y_max)
        self.min = (x_min, y_min)
        self.dim = 2

    ## Check if point is in domain.
    def __contains__(self, point: tuple) -> bool:
        return all(self.min[i] <= point[i] <= self.max[i] for i in range(self.dim))

    @staticmethod
    def eps():
        epsilon = pow(10, -4)
        return random.uniform(-epsilon, epsilon) * 0

    ## Generate fence in counter-clockwise order.
    # spacing should be less than 2*sensing_radius.
    # offset_distance optionally overrides the historical sqrt(3)/2 spacing rule.
    def fence(self, spacing, offset_distance=None) -> list:
        # Initialize fence position
        dx = spacing * np.sin(np.pi / 3) if offset_distance is None else float(offset_distance)
        vx_min, vx_max = self.min[0] - dx, self.max[0] + dx
        vy_min, vy_max = self.min[1] - dx, self.max[1] + dx

        points = []
        points.extend(
            [(x + self.eps(), vy_min + self.eps()) for x in np.arange(vx_min, 0.999 * vx_max, spacing)])  # bottom
        points.extend(
            [(vx_max + self.eps(), y + self.eps()) for y in np.arange(vy_min, 0.999 * vx_max, spacing)])  # right
        points.extend([(self.max[0] - x + self.eps(), vy_max + self.eps())
                       for x in np.arange(vx_min, 0.999 * vx_max, spacing)])  # top
        points.extend([(vx_min + self.eps(), self.max[1] - y + self.eps())
                       for y in np.arange(vy_min, 0.999 * vy_max, spacing)])  # left
        return points

    ## Generate points distributed (uniformly) randomly in the interior.
    def point_generator(self, n_sensors: int) -> list:
        rand_x = np.random.uniform(self.min[0], self.max[0], size=n_sensors)
        rand_y = np.random.uniform(self.min[1], self.max[1], size=n_sensors)
        return list(zip(rand_x, rand_y))

    ## Generate points to plot domain boundary.
    def domain_boundary_points(self):
        x_pts = [self.min[0], self.min[0], self.max[0], self.max[0], self.min[0]]
        y_pts = [self.min[1], self.max[1], self.max[1], self.min[1], self.min[1]]
        return x_pts, y_pts

    def normal(self, boundary_point):
        center = 0.5 * np.array([self.max[0] + self.min[0], self.max[1] + self.min[1]])
        offset_point = boundary_point - center  # center domain at (0, 0)
        if self.min[0] - center[0] < offset_point[0] < self.max[0] - center[0]:
            normal = np.array([0, offset_point[1]])
        else:
            normal = np.array([offset_point[0], 0])
        return normal / np.linalg.norm(normal)

    def get_intersection_point(self, old, new):
        assert (old in self and new not in self)
        tvals = []
        for dim in range(self.dim):
            delta = new[dim] - old[dim]
            if abs(delta) <= 1e-12:
                continue
            tvals.append((self.max[dim] - old[dim]) / delta)
            tvals.append((self.min[dim] - old[dim]) / delta)
        candidates = [t for t in tvals if 0 < t < 1]
        if not candidates:
            raise ValueError("No valid boundary intersection found for reflected step.")
        t = min(candidates)
        return old + t * (new - old)


class CircularDomain(Domain):
    def __init__(self, radius) -> None:
        self.radius = radius
        self.dim = 2

    ## Check if point is in domain.
    def __contains__(self, point: tuple) -> bool:
        return norm(point) < self.radius

    ## Generate points in counter-clockwise order.
    def fence(self, spacing) -> list:
        # Initialize fence
        dx = spacing
        v_rad = self.radius + dx
        return [pol2cart((v_rad, t)) for t in arange(0, 2 * pi, spacing)]

    ## Generate points distributed randomly (uniformly) in the interior.
    def point_generator(self, n_sensors):
        theta = np.random.uniform(0, 2 * pi, size=n_sensors)
        # Sample radius via sqrt(U) so the induced planar density is uniform.
        radius = self.radius * np.sqrt(np.random.uniform(0.0, 1.0, size=n_sensors))
        return [pol2cart(p) for p in zip(radius, theta)]

    ## Generate Points to plot domain boundary.
    def domain_boundary_points(self):
        x_pts = [self.radius * np.cos(t) for t in arange(0, 2 * pi, 0.01)]
        y_pts = [self.radius * np.sin(t) for t in arange(0, 2 * pi, 0.01)]
        return x_pts, y_pts

    def normal(self, boundary_point):
        return boundary_point / np.linalg.norm(boundary_point)

    def get_intersection_point(self, old, new):
        disp = new - old
        tvals = np.roots([norm(disp) ** 2, 2 * np.dot(disp, old), norm(old) ** 2 - self.radius ** 2])
        t = min(filter(lambda x: 0 < x < 1, tvals))
        return old + t * disp


class SquareAnnulusDomain(Domain):
    def __init__(self, sensor_radius: float) -> None:
        self.sensor_radius = float(sensor_radius)
        self.outer_half_side = 5.0 * self.sensor_radius
        self.inner_half_side = 2.0 * self.sensor_radius
        self.dim = 2

    @staticmethod
    def _tol() -> float:
        return 1.0e-9

    def __contains__(self, point: tuple) -> bool:
        x, y = map(float, point)
        tol = self._tol()
        inside_outer = abs(x) <= self.outer_half_side + tol and abs(y) <= self.outer_half_side + tol
        inside_hole = abs(x) < self.inner_half_side - tol and abs(y) < self.inner_half_side - tol
        return inside_outer and not inside_hole

    def point_generator(self, n_sensors: int) -> list:
        points = []
        while len(points) < n_sensors:
            p = np.array(
                [
                    np.random.uniform(-self.outer_half_side, self.outer_half_side),
                    np.random.uniform(-self.outer_half_side, self.outer_half_side),
                ],
                dtype=float,
            )
            if p in self:
                points.append(p)
        return points

    def domain_boundary_points(self):
        outer = [
            (-self.outer_half_side, -self.outer_half_side),
            (self.outer_half_side, -self.outer_half_side),
            (self.outer_half_side, self.outer_half_side),
            (-self.outer_half_side, self.outer_half_side),
            (-self.outer_half_side, -self.outer_half_side),
        ]
        inner = [
            (-self.inner_half_side, -self.inner_half_side),
            (-self.inner_half_side, self.inner_half_side),
            (self.inner_half_side, self.inner_half_side),
            (self.inner_half_side, -self.inner_half_side),
            (-self.inner_half_side, -self.inner_half_side),
        ]
        xpts = [pt[0] for pt in outer] + [np.nan] + [pt[0] for pt in inner]
        ypts = [pt[1] for pt in outer] + [np.nan] + [pt[1] for pt in inner]
        return xpts, ypts

    def _face_distance_candidates(self, pt):
        x, y = map(float, pt)
        return [
            (abs(x - self.outer_half_side), np.array([1.0, 0.0], dtype=float)),
            (abs(x + self.outer_half_side), np.array([-1.0, 0.0], dtype=float)),
            (abs(y - self.outer_half_side), np.array([0.0, 1.0], dtype=float)),
            (abs(y + self.outer_half_side), np.array([0.0, -1.0], dtype=float)),
            (abs(x - self.inner_half_side), np.array([-1.0, 0.0], dtype=float)),
            (abs(x + self.inner_half_side), np.array([1.0, 0.0], dtype=float)),
            (abs(y - self.inner_half_side), np.array([0.0, -1.0], dtype=float)),
            (abs(y + self.inner_half_side), np.array([0.0, 1.0], dtype=float)),
        ]

    def normal(self, boundary_point):
        tol = 1.0e-7
        candidates = [(dist, normal) for dist, normal in self._face_distance_candidates(boundary_point) if dist <= tol]
        if not candidates:
            candidates = self._face_distance_candidates(boundary_point)
        _, normal = min(candidates, key=lambda item: item[0])
        return normal

    def _line_intersections(self, old, new):
        old = np.asarray(old, dtype=float)
        new = np.asarray(new, dtype=float)
        disp = new - old
        tol = self._tol()
        candidates = []

        def add_candidate(axis: int, boundary: float, other_limit: float):
            delta = disp[axis]
            if abs(delta) <= tol:
                return
            t = (boundary - old[axis]) / delta
            if not (-tol <= t <= 1.0 + tol):
                return
            t = float(np.clip(t, 0.0, 1.0))
            point = old + t * disp
            other = point[1 - axis]
            if abs(other) <= other_limit + tol:
                candidates.append((t, point))

        for boundary in (-self.outer_half_side, self.outer_half_side):
            add_candidate(axis=0, boundary=boundary, other_limit=self.outer_half_side)
            add_candidate(axis=1, boundary=boundary, other_limit=self.outer_half_side)

        for boundary in (-self.inner_half_side, self.inner_half_side):
            add_candidate(axis=0, boundary=boundary, other_limit=self.inner_half_side)
            add_candidate(axis=1, boundary=boundary, other_limit=self.inner_half_side)

        return candidates

    def _boundary_anchor(self, point):
        tol = 1.0e-7
        x, y = map(float, point)

        if abs(abs(x) - self.outer_half_side) <= tol and abs(y) <= self.outer_half_side + tol:
            return np.array([np.sign(x) * self.outer_half_side if x != 0 else self.outer_half_side, np.clip(y, -self.outer_half_side, self.outer_half_side)], dtype=float)
        if abs(abs(y) - self.outer_half_side) <= tol and abs(x) <= self.outer_half_side + tol:
            return np.array([np.clip(x, -self.outer_half_side, self.outer_half_side), np.sign(y) * self.outer_half_side if y != 0 else self.outer_half_side], dtype=float)
        if abs(abs(x) - self.inner_half_side) <= tol and abs(y) <= self.inner_half_side + tol:
            return np.array([np.sign(x) * self.inner_half_side if x != 0 else self.inner_half_side, np.clip(y, -self.inner_half_side, self.inner_half_side)], dtype=float)
        if abs(abs(y) - self.inner_half_side) <= tol and abs(x) <= self.inner_half_side + tol:
            return np.array([np.clip(x, -self.inner_half_side, self.inner_half_side), np.sign(y) * self.inner_half_side if y != 0 else self.inner_half_side], dtype=float)
        return None

    def get_intersection_point(self, old, new):
        old = np.asarray(old, dtype=float)
        new = np.asarray(new, dtype=float)
        candidates = self._line_intersections(old, new)
        if not candidates:
            anchor = self._boundary_anchor(old)
            if anchor is not None:
                return anchor
            raise ValueError("No valid square-annulus boundary intersection found for reflected step.")
        _, point = min(candidates, key=lambda item: item[0])
        return point


class UnitCube(Domain):

    def __init__(self) -> None:
        self.min = (0, 0, 0)
        self.max = (1, 1, 1)
        self.dim = 3

    ## Determine if given point it in domain or not.
    def __contains__(self, point) -> bool:
        return all(0 <= px <= 1 for px in point)

    ## Generate n_int sensors randomly inside the domain.
    def point_generator(self, n_sensors: int):
        return np.random.rand(n_sensors, 3)

    def domain_boundary_points(self):
        return []

    def normal(self, boundary_point):
        """
        TODO: Fix this!!!!!!
        """
        # center = 0.5 * np.array([self.max[d] + self.min[d] for d in range(self.dim)])
        # offset_point = boundary_point - center  # center domain at (0, 0, 0)
        # print(f"Boundary Point: {boundary_point}, Center: {center}, Offset: {offset_point}")
        #
        # if self.min[0] < boundary_point[0] < self.max[0]:
        #     normal = np.array([0, offset_point[1]])
        # else:
        #     normal = np.array([offset_point[0], 0])
        # return normal / np.linalg.norm(normal)

        center = 0.5 * np.array([self.max[d] + self.min[d] for d in range(self.dim)])
        offset_point = boundary_point - center  # Center the domain at (0, 0, 0)

        # Determine which face the boundary point is on by checking x, y, or z coordinates
        if np.isclose(boundary_point[0], self.min[0]) or np.isclose(boundary_point[0], self.max[0]):
            normal = np.array([offset_point[0], 0, 0])  # Normal along x-axis
        elif np.isclose(boundary_point[1], self.min[1]) or np.isclose(boundary_point[1], self.max[1]):
            normal = np.array([0, offset_point[1], 0])  # Normal along y-axis
        elif np.isclose(boundary_point[2], self.min[2]) or np.isclose(boundary_point[2], self.max[2]):
            normal = np.array([0, 0, offset_point[2]])  # Normal along z-axis
        else:
            raise ValueError(f"Point {boundary_point} is not on the boundary of the domain.")

        return normal / np.linalg.norm(normal)


    def get_intersection_point(self, old, new):
        assert (old in self and new not in self)
        tvals = []
        for dim in range(len(old)):
            tvals.append((self.max[dim] - old[dim]) / (new[dim] - old[dim]))
            tvals.append((self.min[dim] - old[dim]) / (new[dim] - old[dim]))
        t = min(filter(lambda x: 0 < x < 1, tvals))
        return old + t * (new - old)


## Generate boundary points in counterclockwise order.
# WARNING: Points must be generated in counterclockwise order so that the
# alpha_cycle can be easily computed.
# Note: this function will not produce a fence suitable for computation and needs to be preprocessed, see warning
def UnitCubeFence(spacing):
    # TODO check that domain is contained in the fence covered region
    dx = np.sqrt(3) * spacing / 2

    points = np.arange(-dx, 1.001 + dx, spacing)
    grid = list(product(points, points))
    epsilon = pow(10, -5)
    x0_face = [(-dx + random.uniform(-epsilon, epsilon),
                y + random.uniform(-epsilon, epsilon),
                z + random.uniform(-epsilon, epsilon)) for y, z in grid]
    y0_face = [(x + random.uniform(-epsilon, epsilon),
                -dx + random.uniform(-epsilon, epsilon),
                z + random.uniform(-epsilon, epsilon)) for x, z in grid]
    z0_face = [(x + random.uniform(-epsilon, epsilon),
                y + random.uniform(-epsilon, epsilon),
                -dx + random.uniform(-epsilon, epsilon)) for x, y in grid]
    x1_face = [(1 + dx + random.uniform(-epsilon, epsilon),
                y + random.uniform(-epsilon, epsilon),
                z + random.uniform(-epsilon, epsilon)) for y, z in grid]
    y1_face = [(x + random.uniform(-epsilon, epsilon),
                1 + dx + random.uniform(-epsilon, epsilon),
                z + random.uniform(-epsilon, epsilon)) for x, z in grid]
    z1_face = [(x + random.uniform(-epsilon, epsilon),
                y + random.uniform(-epsilon, epsilon),
                1 + dx + random.uniform(-epsilon, epsilon)) for x, y in grid]
    fence_sensors = np.concatenate((x0_face, y0_face, z0_face, x1_face, y1_face, z1_face))
    return np.unique(fence_sensors, axis=0)


def reindex_for_simplex(alpha_complex, fence_points):
    # Traverse the 2-simplices in the alpha complex
    for simplex, _ in alpha_complex.simplex_tree.get_filtration():
        if len(simplex) == 3 and 0 in simplex:  # Find the first 2-simplice with vertex 0
            # Reorder rows so simplex vertices become indices {0, 1, 2}.
            # This avoids overlapping swaps that can duplicate points.
            idx1, idx2 = [idx for idx in simplex if idx != 0]
            n = len(fence_points)
            front = [0, idx1, idx2]
            seen = set(front)
            perm = np.array(front + [i for i in range(n) if i not in seen], dtype=int)
            return True, np.asarray(fence_points)[perm]
    return False, fence_points


def get_unitcube_fence(spacing):
    epsilon = 1e-5  # Perturbation factor
    dx = np.sqrt(3) * spacing / 2

    # Create a grid of points along x, y, and z coordinates
    points = np.arange(-dx, 1.001 + dx, spacing)
    grid = list(product(points, points))

    # Generate perturbed grid points for each face of the unit cube
    x0_face = [(-dx + random.uniform(-epsilon, epsilon), y + random.uniform(-epsilon, epsilon),
                z + random.uniform(-epsilon, epsilon)) for y, z in grid]
    y0_face = [(x + random.uniform(-epsilon, epsilon), -dx + random.uniform(-epsilon, epsilon),
                z + random.uniform(-epsilon, epsilon)) for x, z in grid]
    z0_face = [(x + random.uniform(-epsilon, epsilon), y + random.uniform(-epsilon, epsilon),
                -dx + random.uniform(-epsilon, epsilon)) for x, y in grid]
    x1_face = [(1 + dx + random.uniform(-epsilon, epsilon), y + random.uniform(-epsilon, epsilon),
                z + random.uniform(-epsilon, epsilon)) for y, z in grid]
    y1_face = [(x + random.uniform(-epsilon, epsilon), 1 + dx + random.uniform(-epsilon, epsilon),
                z + random.uniform(-epsilon, epsilon)) for x, z in grid]
    z1_face = [(x + random.uniform(-epsilon, epsilon), y + random.uniform(-epsilon, epsilon),
                1 + dx + random.uniform(-epsilon, epsilon)) for x, y in grid]

    fence_sensors = np.concatenate((x0_face, y0_face, z0_face, x1_face, y1_face, z1_face))
    fence_sensors = np.unique(fence_sensors, axis=0)
    # CHECK IF WE NEED THE SQRT!!!!!!!
    alpha_complex = AlphaComplex(fence_sensors, spacing)
    _, reordered_sensors = reindex_for_simplex(alpha_complex, fence_sensors)
    fence = []
    for sensor in reordered_sensors:
        s = Sensor(sensor, (0, 0, 0), spacing, True)
        fence.append(s)
    return fence

class BunimovichStadium(Domain):

    def __init__(self, w, r, L: float, square_init: bool = False, square_init_length: float | None = None):
        self.w = w
        self.r = r
        self.length = L
        self.dim = 2
        self.square_init = bool(square_init)
        self.square_init_length = float(square_init_length) if square_init_length is not None else float(min(w, r))

    @staticmethod
    def _tol():
        return 1.0e-9

    def __contains__(self, point):
        x, y = point
        tol = self._tol()
        if -self.w - tol <= x <= self.w + tol and -self.r - tol <= y <= self.r + tol:
            return True
        if (x + self.w) ** 2 + y ** 2 <= self.r ** 2 + tol:
            return True
        if (x - self.w) ** 2 + y ** 2 <= self.r ** 2 + tol:
            return True
        return False

    def normal(self, pt):
        if pt[0] < -self.w:
            center_cir = np.array([-self.w, 0])
            normal = pt - center_cir
        elif pt[0] > self.w:
            center_cir = np.array([self.w, 0])
            normal = pt - center_cir
        else:
            normal = np.array([0, pt[1]])
        return normal / np.linalg.norm(normal)

    @staticmethod
    def _is_real_root(root, tol=1e-10):
        return abs(np.imag(root)) <= tol

    def _flat_intersection_candidates(self, old_pos, new_pos):
        disp = new_pos - old_pos
        tol = self._tol()
        if abs(disp[1]) <= tol:
            return []

        candidates = []
        for y_boundary in (self.r, -self.r):
            t = (y_boundary - old_pos[1]) / disp[1]
            if not (-tol <= t <= 1.0 + tol):
                continue

            t = float(np.clip(t, 0.0, 1.0))
            point = old_pos + t * disp
            if -self.w - tol <= point[0] <= self.w + tol:
                candidates.append((t, point))
        return candidates

    def _circle_intersection_candidates(self, old_pos, new_pos):
        disp = new_pos - old_pos
        tol = self._tol()
        a = float(np.dot(disp, disp))
        if a <= tol ** 2:
            return []

        candidates = []
        for center_x, side in ((-self.w, "left"), (self.w, "right")):
            center = np.array([center_x, 0.0], dtype=float)
            rel = old_pos - center
            b = 2.0 * float(np.dot(rel, disp))
            c = float(np.dot(rel, rel) - self.r ** 2)

            for root in np.roots([a, b, c]):
                if not self._is_real_root(root, tol=1e-8):
                    continue

                t = float(np.real(root))
                if not (-tol <= t <= 1.0 + tol):
                    continue

                t = float(np.clip(t, 0.0, 1.0))
                point = old_pos + t * disp
                if side == "left" and point[0] <= -self.w + tol:
                    candidates.append((t, point))
                elif side == "right" and point[0] >= self.w - tol:
                    candidates.append((t, point))
        return candidates

    def _boundary_anchor(self, point):
        tol = 1.0e-7
        x, y = map(float, point)

        if -self.w - tol <= x <= self.w + tol and abs(abs(y) - self.r) <= tol:
            return np.array([np.clip(x, -self.w, self.w), self.r if y >= 0.0 else -self.r], dtype=float)

        for center_x, side in ((-self.w, "left"), (self.w, "right")):
            rel = np.array([x - center_x, y], dtype=float)
            rel_norm = float(np.linalg.norm(rel))
            if rel_norm <= tol:
                continue
            if abs(rel_norm - self.r) <= tol:
                if side == "left" and x <= -self.w + tol:
                    return np.array([center_x, 0.0], dtype=float) + (self.r / rel_norm) * rel
                if side == "right" and x >= self.w - tol:
                    return np.array([center_x, 0.0], dtype=float) + (self.r / rel_norm) * rel
        return None

    def get_intersection_point(self, old_pos, new_pos):
        candidates = self._flat_intersection_candidates(old_pos, new_pos)
        candidates.extend(self._circle_intersection_candidates(old_pos, new_pos))
        if not candidates:
            anchor = self._boundary_anchor(old_pos)
            if anchor is not None:
                return anchor
            raise ValueError("No valid Bunimovich stadium boundary intersection found for reflected step.")

        _, point = min(candidates, key=lambda item: item[0])
        return point

    def point_generator(self, N):
        points = []
        if self.square_init:
            if self.square_init_length > 2 * self.w or self.square_init_length > 2 * self.r:
                raise ValueError("Square initialization box does not fit in the stadium.")
            while len(points) < N:
                p = np.random.uniform(-self.square_init_length / 2.0, self.square_init_length / 2.0, 2)
                if p in self:
                    points.append(p)
            return points

        x_extent = self.w + self.r
        while len(points) < N:
            # Rejection sample from the full stadium bounding box so the accepted
            # points are uniform over the stadium interior.
            p = np.array([
                np.random.uniform(-x_extent, x_extent),
                np.random.uniform(-self.r, self.r),
            ])
            if p in self:
                points.append(p)

        return points

    def domain_boundary_points(self):
        theta1 = np.linspace(np.pi / 2, 3 * np.pi / 2, 1000)
        x1 = -self.w + self.r * np.cos(theta1)
        y1 = self.r * np.sin(theta1)

        x3 = np.array([-self.w, self.w])  # bottom length
        y3 = np.array([-self.r, -self.r])

        theta2 = np.linspace(-1 * np.pi / 2, np.pi / 2, 1000)
        x2 = self.w + self.r * np.cos(theta2)
        y2 = self.r * np.sin(theta2)

        x4 = np.array([-self.w, self.w])  # top length
        y4 = np.array([self.r, self.r])

        x = np.concatenate((x1, x3, x2, x4))
        y = np.concatenate((y1, y3, y2, y4))
        return x, y

    def fence(self, spacing):

        fence = []  # List to store the fence points

        theta = np.linspace(np.pi / 2, 0.999 * 3 * np.pi / 2,
                            int(np.ceil(2 * np.pi * self.r / spacing)))  # Angles spaced apart by 'spacing'
        x_pts = (self.r + (np.sqrt(3) / 2) * spacing) * np.cos(theta) + -self.w  # x-coordinates of the fence points
        y_pts = (self.r + (np.sqrt(3) / 2) * spacing) * np.sin(theta)  # y-coordinates of the fence points

        for x, y in zip(x_pts, y_pts):
            fence.append((x, y))

        for i in range(int(2 * self.w // spacing) + 1):
            x1 = -self.w + i * spacing
            y1 = -self.r - (np.sqrt(3) / 2) * spacing

            if x1 > self.w:
                break

            fence.append((x1, y1))

        theta2 = np.linspace(-1 * np.pi / 2, 0.999 * np.pi / 2, int(np.ceil(2 * np.pi * self.r / spacing)))
        x2 = (self.r + (np.sqrt(3) / 2) * spacing) * np.cos(theta2) + self.w  # x-coordinates of the fence points
        y2 = (self.r + (np.sqrt(3) / 2) * spacing) * np.sin(theta2)  # y-coordinates of the fence points

        for x, y in zip(x2, y2):
            fence.append((x, y))

        for i in range(int(2 * self.w // spacing) + 1):
            x3 = self.w - i * spacing
            y3 = self.r + (np.sqrt(3) / 2) * spacing

            if x3 < -self.w:
                break

            fence.append((x3, y3))

        return fence
