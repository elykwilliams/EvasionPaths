# ******************************************************************************
#  Copyright (c) 2023, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************
"""Visualization helpers focused on 3D UnitCube state rendering."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Iterable, List, Tuple

import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

if TYPE_CHECKING:
    from time_stepping import EvasionPathSimulation


def _cycle_triangles(cycle) -> List[Tuple[int, int, int]]:
    """Return unique oriented-agnostic triangular faces from a boundary cycle."""
    seen = set()
    triangles: List[Tuple[int, int, int]] = []
    for face in cycle:
        nodes = tuple(face.nodes)
        if len(nodes) != 3:
            continue
        key = tuple(sorted(nodes))
        if key in seen:
            continue
        seen.add(key)
        triangles.append(nodes)
    return triangles


def _cycle_color(cycle, *, cmap_name: str = "tab20") -> Tuple[float, float, float, float]:
    """
    Stable per-cycle color keyed by cycle representation, so colors remain
    consistent across frames for the same cycle identity.
    """
    cmap = cm.get_cmap(cmap_name)
    key = str(cycle).encode("utf-8")
    idx = int(hashlib.md5(key).hexdigest(), 16) % cmap.N
    return cmap(idx)


def show_unitcube_void_state(
    sim: "EvasionPathSimulation",
    *,
    ax,
    elev: float = 22.0,
    azim: float = -55.0,
    surface_alpha: float = 0.35,
    cycle_cmap: str = "tab20",
) -> None:
    """
    Render 3D state for UnitCube simulations:
    - translucent triangular surfaces for currently uncovered boundary cycles
    - mobile sensors and fence sensors as point clouds
    """
    points = np.asarray(sim.sensor_network.points)

    # Draw uncovered-cycle boundary surfaces (exclude alpha cycle / fence boundary).
    poly_faces = []
    poly_colors = []
    for cycle, label in sim.cycle_label.label.items():
        if sim.topology.is_excluded_cycle(cycle) or not bool(label):
            continue
        rgba = _cycle_color(cycle, cmap_name=cycle_cmap)
        for tri in _cycle_triangles(cycle):
            poly_faces.append(points[list(tri)])
            poly_colors.append((rgba[0], rgba[1], rgba[2], surface_alpha))

    if poly_faces:
        surf = Poly3DCollection(
            poly_faces,
            facecolors=poly_colors,
            edgecolors=(0.2, 0.2, 0.2, 0.45),
            linewidths=0.4,
        )
        ax.add_collection3d(surf)

    # Mobile sensors (all non-fence points are included in sensor_network.points)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="tab:red", s=12, depthshade=True)

    # Fence sensors
    fence_pts = np.asarray([s.pos for s in sim.sensor_network.fence_sensors])
    if fence_pts.size > 0:
        ax.scatter(fence_pts[:, 0], fence_pts[:, 1], fence_pts[:, 2], c="black", s=5, alpha=0.35, depthshade=False)

    # Unit cube framing and camera.
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, 1.12)
    ax.set_zlim(-0.12, 1.12)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
