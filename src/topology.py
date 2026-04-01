# ******************************************************************************
#  Copyright (c) 2023, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np

from alpha_complex import AlphaComplex, Simplex
from combinatorial_map import BoundaryCycle, RotationInfo2D, CombinatorialMap2D, CombinatorialMap, CombinatorialMap3D, \
    OrientedSimplex, RotationInfo3D


@dataclass
class Topology:
    alpha_complex: AlphaComplex
    cmap: CombinatorialMap
    _graph: nx.Graph = None
    fence_node_count: Optional[int] = None
    fence_node_groups: Tuple[Tuple[int, ...], ...] = ()
    excluded_fence_groups: Tuple[int, ...] = ()
    interior_point: Optional[np.ndarray] = None
    outer_winding_sign: int = -1
    _outer_cycle_cache: Optional[BoundaryCycle] = None
    _excluded_cycles_cache: Optional[Tuple[BoundaryCycle, ...]] = None
    _component_id_by_face: Optional[Dict[Simplex, int]] = None
    _outer_anchor_component_id: Optional[int] = None

    def simplices(self, dim):
        return self.alpha_complex.simplices(dim)

    @property
    def boundary_cycles(self):
        """All cycles in teh partition, even corresponding to simplices"""
        return self.cmap.boundary_cycles

    @property
    def homology_generators(self):
        """Cycles that are not the boundary of a simplex"""
        return {cycle for cycle in self.cmap.boundary_cycles if not self.is_boundary(cycle)}

    @property
    def alpha_cycle(self) -> BoundaryCycle:
        face = OrientedSimplex(tuple(range(self.dim)))
        try:
            cycle = self.cmap.get_cycle(face)
        except KeyError:
            raise KeyError("The simplex (0, 1, ... dim-1), is not a face of the external boundary.")
        return cycle

    @property
    def dim(self) -> int:
        return self.alpha_complex.dim

    def is_boundary(self, cycle):
        # Return True if the cycle is the boundary of a dim-simplex
        return Simplex(cycle.nodes) in self.simplices(self.dim)

    @property
    def face_connectivity_graph(self):
        if self._graph:
            return self._graph
        self._graph = nx.Graph()
        self._graph.add_nodes_from(self.simplices(self.dim - 1))
        for face_list in self.cmap.rotation_info.incident_simplices.values():
            # Connectivity only needs one spanning set of edges per incident bucket.
            # Star edges preserve connected components while avoiding O(k^2) cliques.
            unique_faces = []
            seen = set()
            for nodes in face_list:
                face = Simplex(nodes)
                if face in seen:
                    continue
                seen.add(face)
                unique_faces.append(face)

            if len(unique_faces) < 2:
                continue

            root = unique_faces[0]
            self._graph.add_edges_from((root, other) for other in unique_faces[1:])
        return self._graph

    def is_connected_cycle(self, cycle):
        if not cycle:
            return False

        if self.dim < 3:
            graph = self.face_connectivity_graph
            source = Simplex(next(iter(cycle)).nodes)
            target = Simplex(range(self.dim))
            if source not in graph or target not in graph:
                return False
            return nx.has_path(graph, source, target)

        source = Simplex(next(iter(cycle)).nodes)
        component_id_by_face = self._face_component_map()
        source_component = component_id_by_face.get(source)
        target_component = self._outer_anchor_component()

        if source_component is None or target_component is None:
            return False

        return source_component == target_component

    def _face_component_map(self) -> Dict[Simplex, int]:
        if self._component_id_by_face is not None:
            return self._component_id_by_face

        component_id_by_face: Dict[Simplex, int] = {}
        for component_id, component in enumerate(nx.connected_components(self.face_connectivity_graph)):
            for face in component:
                component_id_by_face[face] = component_id
        self._component_id_by_face = component_id_by_face
        return self._component_id_by_face

    def _outer_anchor_face(self):
        """
        Pick an anchor face from the detected outer cycle.
        Fall back to historical {0,1,...,dim-1} only if needed.
        """
        graph = self.face_connectivity_graph

        try:
            for face in self.outer_cycle:
                candidate = Simplex(face.nodes)
                if candidate in graph:
                    return candidate
        except Exception:
            pass

        preferred = Simplex(range(self.dim))
        if preferred in graph:
            return preferred
        return None

    def _outer_anchor_component(self) -> Optional[int]:
        if self._outer_anchor_component_id is not None:
            return self._outer_anchor_component_id

        face = self._outer_anchor_face()
        if face is None:
            return None

        self._outer_anchor_component_id = self._face_component_map().get(face)
        return self._outer_anchor_component_id

    def _fence_face_count(self, cycle: BoundaryCycle) -> int:
        if self.fence_node_count is None:
            return 0
        return sum(1 for face in cycle if all(v < self.fence_node_count for v in face.nodes))

    def _fence_node_overlap(self, cycle: BoundaryCycle) -> int:
        if self.fence_node_count is None:
            return 0
        return sum(1 for v in cycle.nodes if v < self.fence_node_count)

    @staticmethod
    def _grouped_face_count(cycle: BoundaryCycle, node_group: Tuple[int, ...]) -> int:
        group = set(node_group)
        return sum(1 for face in cycle if all(v in group for v in face.nodes))

    @staticmethod
    def _grouped_node_overlap(cycle: BoundaryCycle, node_group: Tuple[int, ...]) -> int:
        group = set(node_group)
        return sum(1 for v in cycle.nodes if v in group)

    def _cycle_for_fence_group(
        self,
        node_group: Tuple[int, ...],
        *,
        excluded: Tuple[BoundaryCycle, ...] = (),
    ) -> Optional[BoundaryCycle]:
        candidates = [cycle for cycle in self.homology_generators if cycle not in excluded]
        supported = [
            cycle
            for cycle in candidates
            if self._grouped_face_count(cycle, node_group) > 0 or self._grouped_node_overlap(cycle, node_group) > 0
        ]
        if not supported:
            return None
        return max(
            supported,
            key=lambda cycle: (
                self._grouped_face_count(cycle, node_group),
                self._grouped_node_overlap(cycle, node_group),
                len(cycle),
            ),
        )

    def _abs_oriented_volume(self, cycle: BoundaryCycle) -> float:
        if self.dim != 3:
            return 0.0
        points = np.asarray(self.cmap.rotation_info.points, dtype=float)
        volume = 0.0
        for face in cycle:
            if len(face.nodes) != 3:
                continue
            i, j, k = face.nodes
            a = points[i]
            b = points[j]
            c = points[k]
            volume += float(np.dot(a, np.cross(b, c))) / 6.0
        return abs(volume)

    def _signed_winding_number(self, cycle: BoundaryCycle, point: Optional[np.ndarray]) -> float:
        if self.dim != 3 or point is None:
            return 0.0

        points = np.asarray(self.cmap.rotation_info.points, dtype=float)
        p = np.asarray(point, dtype=float)
        total_solid_angle = 0.0
        eps = 1e-12

        for face in cycle:
            if len(face.nodes) != 3:
                continue

            i, j, k = face.nodes
            a = points[i] - p
            b = points[j] - p
            c = points[k] - p

            la = np.linalg.norm(a)
            lb = np.linalg.norm(b)
            lc = np.linalg.norm(c)
            if min(la, lb, lc) < eps:
                continue

            numerator = float(np.dot(a, np.cross(b, c)))
            denominator = (
                la * lb * lc
                + float(np.dot(a, b)) * lc
                + float(np.dot(b, c)) * la
                + float(np.dot(c, a)) * lb
            )
            total_solid_angle += 2.0 * float(np.arctan2(numerator, denominator))

        return total_solid_angle / (4.0 * float(np.pi))

    def _winding_sign_preference(self) -> int:
        return -1 if self.outer_winding_sign < 0 else 1

    @property
    def outer_cycle(self) -> BoundaryCycle:
        if self._outer_cycle_cache is not None:
            return self._outer_cycle_cache

        # Keep 2D behavior aligned with historical alpha-cycle handling.
        if self.dim < 3:
            try:
                self._outer_cycle_cache = self.alpha_cycle
            except KeyError:
                self._outer_cycle_cache = max(self.boundary_cycles, key=len)
            return self._outer_cycle_cache

        homology = list(self.homology_generators)
        if not homology:
            self._outer_cycle_cache = max(self.boundary_cycles, key=len)
            return self._outer_cycle_cache

        fence_supported = [cycle for cycle in homology if self._fence_face_count(cycle) > 0]
        candidates = fence_supported if fence_supported else homology

        winding_by_cycle: Dict[BoundaryCycle, float] = {}
        if self.interior_point is not None:
            winding_by_cycle = {
                cycle: self._signed_winding_number(cycle, self.interior_point)
                for cycle in candidates
            }
            enclosing = [cycle for cycle in candidates if abs(winding_by_cycle[cycle]) > 0.5]
            if enclosing:
                candidates = enclosing

        def winding(cycle: BoundaryCycle) -> float:
            return winding_by_cycle.get(cycle, 0.0)

        self._outer_cycle_cache = max(
            candidates,
            key=lambda cycle: (
                self._fence_face_count(cycle),
                self._fence_node_overlap(cycle),
                abs(winding(cycle)),
                self._winding_sign_preference() * winding(cycle),
                self._abs_oriented_volume(cycle),
                len(cycle),
            ),
        )
        return self._outer_cycle_cache

    @property
    def excluded_cycles(self) -> Tuple[BoundaryCycle, ...]:
        if self._excluded_cycles_cache is not None:
            return self._excluded_cycles_cache

        selected = []
        try:
            outer = self.outer_cycle
        except Exception:
            outer = None
        if outer is not None:
            selected.append(outer)

        if self.fence_node_groups and self.excluded_fence_groups:
            for group_id in self.excluded_fence_groups:
                if not (0 <= group_id < len(self.fence_node_groups)):
                    continue
                cycle = self._cycle_for_fence_group(self.fence_node_groups[group_id], excluded=tuple(selected))
                if cycle is not None and cycle not in selected:
                    selected.append(cycle)

        self._excluded_cycles_cache = tuple(selected)
        return self._excluded_cycles_cache

    def is_excluded_cycle(self, cycle) -> bool:
        return cycle in set(self.excluded_cycles)

    def is_face_connected(self):
        return nx.is_connected(self.face_connectivity_graph)


def generate_topology(
    points,
    radius,
    point_radii=None,
    fence_node_count=None,
    fence_node_groups=None,
    excluded_fence_groups=None,
    interior_point=None,
    outer_winding_sign=-1,
):
    ac = AlphaComplex(points, radius, point_radii=point_radii)
    if ac.dim == 2:
        rot_info = RotationInfo2D(points, ac)
        cmap = CombinatorialMap2D(rot_info)
    elif ac.dim == 3:
        rot_info = RotationInfo3D(points, ac)
        cmap = CombinatorialMap3D(rot_info)
    else:
        raise NotImplementedError(f"No Implementation for CombinatorialMap for dimension {ac.dim}")
    interior = None if interior_point is None else np.asarray(interior_point, dtype=float)
    winding_sign = -1 if outer_winding_sign < 0 else 1
    return Topology(
        ac,
        cmap,
        fence_node_count=fence_node_count,
        fence_node_groups=tuple(tuple(int(v) for v in group) for group in (fence_node_groups or ())),
        excluded_fence_groups=tuple(int(v) for v in (excluded_fence_groups or ())),
        interior_point=interior,
        outer_winding_sign=winding_sign,
    )
