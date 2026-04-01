# ******************************************************************************
#  Copyright (c) 2023, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Set, Tuple

import networkx as nx
import numpy as np

from boundary_geometry import RectangularDomain, UnitCube, get_unitcube_fence
from motion_model import BilliardMotion
from sensor_network import SensorNetwork, generate_fence_sensors, generate_mobile_sensors

if TYPE_CHECKING:
    from time_stepping import EvasionPathSimulation


# Legacy ReebGraph implementation retained for compatibility with CycleLabelling.
def get_sym_diff(new_dict: dict, old_dict: dict):
    """
    Compares two hole dictionaries (BoundaryCycle -> label) using canonical keys.
    Returns the actual BoundaryCycle objects that were added or removed.
    """
    new_canon = {str(c): c for c in new_dict}
    old_canon = {str(c): c for c in old_dict}

    added_keys = new_canon.keys() - old_canon.keys()
    removed_keys = old_canon.keys() - new_canon.keys()

    added = {new_canon[k] for k in added_keys}
    removed = {old_canon[k] for k in removed_keys}

    return added, removed


class ReebGraph:
    """Legacy ReebGraph class used by CycleLabelling."""

    def __init__(self, holes: dict):
        self.graph = nx.DiGraph()
        self.stack = {}
        self.finalized = []

        self.used_heights = set()

        for hole in holes:
            height = self._generate_height()
            self.used_heights.add(height)
            event_id = self._insert_new_node(0, height)
            self.stack[self._canon(hole)] = (event_id, height)

    def _canon(self, cycle):
        return str(cycle)

    def _insert_new_node(self, time: float, height: float):
        event_id = len(self.graph)
        self.graph.add_node(event_id, pos=(time, height))
        return event_id

    def _insert_new_edge(self, oldId: int, newId: int, val: bool) -> None:
        self.graph.add_edge(oldId, newId, val=val)

    def update(self, time, new_holes_dict, old_holes_dict):
        added, removed = get_sym_diff(set(new_holes_dict.keys()), set(old_holes_dict.keys()))

        if removed:
            heights = []
            for hole in removed:
                canon_key = self._canon(hole)
                heights.append(self.stack[canon_key][1])
            height = min(heights)
        else:
            height = self._generate_height()

        node_id = self._insert_new_node(time, height)

        for hole in removed:
            canon_key = self._canon(hole)
            event_id, _ = self.stack[canon_key]
            self._insert_new_edge(event_id, node_id, old_holes_dict[hole])
            self.used_heights.remove(self.stack[canon_key][1])
            del self.stack[canon_key]

        for hole in sorted(added):
            canon_key = self._canon(hole)
            self.stack[canon_key] = (node_id, height)
            self.used_heights.add(height)
            height = self._generate_height()

    def _generate_height(self):
        h = 0
        while h in self.used_heights:
            h += 1
        return h

    def finalize(self, time, holes):
        if self.finalized:
            return
        for hole in holes:
            canon_key = self._canon(hole)
            old_id, height = self.stack[canon_key]
            new_id = self._insert_new_node(time, height)
            self._insert_new_edge(old_id, new_id, holes[hole])
            self.finalized.append(new_id)

    def resume(self):
        self.graph.remove_nodes_from(self.finalized)
        self.finalized = []


# New Reeb event graph implementation for 2D experiment visualization.
def cycle_key(cycle) -> str:
    return str(cycle)


def cycle_nodes(cycle) -> Set[int]:
    return set(cycle.nodes)


def is_true(label_value: bool) -> bool:
    return bool(label_value)


def _normalize_excluded_cycles(excluded_cycles):
    if excluded_cycles is None:
        return ()
    if isinstance(excluded_cycles, (list, tuple, set, frozenset)):
        return tuple(excluded_cycles)
    return (excluded_cycles,)


@dataclass
class StepSummary:
    step: int
    time: float
    n_cycles: int
    n_true: int
    n_false: int
    n_birth: int
    n_death: int
    n_continue: int
    n_split_edges: int
    n_merge_edges: int
    n_transform_edges: int
    n_label_flips: int


class ReebEventGraphBuilder:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.prev_node_id_by_key: Dict[str, int] = {}
        self.prev_label_by_key: Dict[str, bool] = {}
        self.prev_nodeset_by_key: Dict[str, Set[int]] = {}
        self.lane_by_key: Dict[str, int] = {}
        self.next_lane = 0
        self.summaries: List[StepSummary] = []
        self.excluded_cycle_keys: Set[str] = set()
        self.snapshot_debug: Dict[int, Dict[str, object]] = {}

    def _new_lane(self) -> int:
        lane = self.next_lane
        self.next_lane += 1
        return lane

    def _add_cycle_node(self, *, key: str, step: int, time: float, label: bool, lane: int, kind: str) -> int:
        node_id = len(self.graph)
        self.graph.add_node(
            node_id,
            key=key,
            step=step,
            time=time,
            label=label,
            lane=lane,
            kind=kind,
            pos=(time, lane),
        )
        return node_id

    def _edge_event(
        self,
        parent: str,
        child: str,
        children_for_parent: Dict[str, List[str]],
        parents_for_child: Dict[str, List[str]],
    ) -> str:
        if len(parents_for_child[child]) > 1:
            return "merge"
        if len(children_for_parent[parent]) > 1:
            return "split"
        return "transform"

    def add_snapshot(self, *, step: int, time: float, labels: Dict, alpha_cycle=None, excluded_cycles=None) -> None:
        excluded = _normalize_excluded_cycles(excluded_cycles)
        if not excluded and alpha_cycle is not None:
            excluded = (alpha_cycle,)

        excluded_keys = {cycle_key(cycle) for cycle in excluded}
        self.excluded_cycle_keys.update(excluded_keys)
        interior_items = [(c, v) for c, v in labels.items() if cycle_key(c) not in excluded_keys]
        curr_label_by_key = {cycle_key(c): is_true(v) for c, v in interior_items}
        curr_nodeset_by_key = {cycle_key(c): cycle_nodes(c) for c, _ in interior_items}
        curr_keys = set(curr_label_by_key)
        prev_keys = set(self.prev_label_by_key)

        born = curr_keys - prev_keys
        died = prev_keys - curr_keys
        persistent = curr_keys & prev_keys

        parents_for_child: Dict[str, List[str]] = {k: [] for k in born}
        children_for_parent: Dict[str, List[str]] = {k: [] for k in died}

        # Robust lineage matching: avoid treating tiny incidental overlaps as true merges.
        for child in born:
            child_nodes = curr_nodeset_by_key[child]
            overlap_scores: List[Tuple[str, int]] = []
            for parent in died:
                parent_nodes = self.prev_nodeset_by_key[parent]
                score = len(parent_nodes & child_nodes)
                if score > 0:
                    overlap_scores.append((parent, score))
            if not overlap_scores:
                continue

            overlap_scores.sort(key=lambda item: item[1], reverse=True)
            best_parent, best_score = overlap_scores[0]
            selected_parents = [best_parent]

            if best_score >= 2:
                tied = [p for p, s in overlap_scores[1:] if s >= 2 and s >= best_score - 1]
                selected_parents.extend(tied)

            for parent in selected_parents:
                parents_for_child[child].append(parent)
                children_for_parent[parent].append(child)

        born_lane: Dict[str, int] = {}
        split_parents = [p for p, kids in children_for_parent.items() if len(kids) > 1]
        for parent in split_parents:
            kids = sorted(children_for_parent[parent])
            if not kids:
                continue
            parent_lane = self.lane_by_key.get(parent, self._new_lane())
            born_lane[kids[0]] = parent_lane
            for sibling in kids[1:]:
                born_lane[sibling] = self._new_lane()

        for child in sorted(born):
            if child in born_lane:
                continue
            parents = parents_for_child.get(child, [])
            if len(parents) == 1 and parents[0] in self.lane_by_key:
                born_lane[child] = self.lane_by_key[parents[0]]
            elif len(parents) > 1:
                parent_lanes = [self.lane_by_key[p] for p in parents if p in self.lane_by_key]
                if parent_lanes:
                    born_lane[child] = min(parent_lanes)

        curr_node_id_by_key: Dict[str, int] = {}
        for key in sorted(curr_keys):
            if key in self.lane_by_key:
                lane = self.lane_by_key[key]
            elif key in born_lane:
                lane = born_lane[key]
            else:
                lane = self._new_lane()
            self.lane_by_key[key] = lane

            curr_node_id_by_key[key] = self._add_cycle_node(
                key=key,
                step=step,
                time=time,
                label=curr_label_by_key[key],
                lane=lane,
                kind="cycle",
            )

        n_continue = 0
        n_split_edges = 0
        n_merge_edges = 0
        n_transform_edges = 0
        n_label_flips = 0
        born_edge_debug: List[Dict[str, object]] = []

        for key in persistent:
            prev_id = self.prev_node_id_by_key[key]
            curr_id = curr_node_id_by_key[key]
            flip = self.prev_label_by_key[key] != curr_label_by_key[key]
            self.graph.add_edge(prev_id, curr_id, event="continue", label_flip=flip)
            n_continue += 1
            if flip:
                n_label_flips += 1

        for child in born:
            for parent in parents_for_child[child]:
                prev_id = self.prev_node_id_by_key[parent]
                curr_id = curr_node_id_by_key[child]
                event = self._edge_event(parent, child, children_for_parent, parents_for_child)
                flip = self.prev_label_by_key[parent] != curr_label_by_key[child]
                self.graph.add_edge(prev_id, curr_id, event=event, label_flip=flip)
                born_edge_debug.append(
                    {
                        "parent": parent,
                        "child": child,
                        "event": event,
                        "label_flip": flip,
                        "parent_label": self.prev_label_by_key[parent],
                        "child_label": curr_label_by_key[child],
                    }
                )
                if event == "split":
                    n_split_edges += 1
                elif event == "merge":
                    n_merge_edges += 1
                else:
                    n_transform_edges += 1
                if flip:
                    n_label_flips += 1

        for parent in sorted(died):
            if children_for_parent[parent]:
                continue
            prev_id = self.prev_node_id_by_key[parent]
            lane = self.lane_by_key.get(parent, self._new_lane())
            label = self.prev_label_by_key[parent]
            end_id = self._add_cycle_node(
                key=parent,
                step=step,
                time=time,
                label=label,
                lane=lane,
                kind="termination",
            )
            self.graph.add_edge(prev_id, end_id, event="terminate", label_flip=False)

        n_true = sum(1 for v in curr_label_by_key.values() if v)
        n_false = len(curr_label_by_key) - n_true
        self.summaries.append(
            StepSummary(
                step=step,
                time=time,
                n_cycles=len(curr_keys),
                n_true=n_true,
                n_false=n_false,
                n_birth=len(born),
                n_death=len(died),
                n_continue=n_continue,
                n_split_edges=n_split_edges,
                n_merge_edges=n_merge_edges,
                n_transform_edges=n_transform_edges,
                n_label_flips=n_label_flips,
            )
        )
        self.snapshot_debug[step] = {
            "time": time,
            "born_labels": {k: curr_label_by_key[k] for k in sorted(born)},
            "died_labels": {k: self.prev_label_by_key[k] for k in sorted(died)},
            "persistent_labels": {
                k: (self.prev_label_by_key[k], curr_label_by_key[k])
                for k in sorted(persistent)
                if self.prev_label_by_key[k] != curr_label_by_key[k]
            },
            "parents_for_child": {
                k: list(v) for k, v in sorted(parents_for_child.items()) if v
            },
            "children_for_parent": {
                k: list(v) for k, v in sorted(children_for_parent.items()) if v
            },
            "born_edge_debug": born_edge_debug,
        }

        self.lane_by_key = {key: self.lane_by_key[key] for key in curr_keys}
        self.prev_node_id_by_key = curr_node_id_by_key
        self.prev_label_by_key = curr_label_by_key
        self.prev_nodeset_by_key = curr_nodeset_by_key

    def has_excluded_cycles(self) -> bool:
        if not self.excluded_cycle_keys:
            return False
        return any(self.graph.nodes[n].get("key") in self.excluded_cycle_keys for n in self.graph.nodes)

    def get_snapshot_debug(self, step: int) -> Dict[str, object]:
        return self.snapshot_debug.get(step, {})

    def close(self, *, step: int, time: float) -> None:
        for key, prev_id in list(self.prev_node_id_by_key.items()):
            lane = self.lane_by_key[key]
            label = self.prev_label_by_key[key]
            end_id = self._add_cycle_node(key=key, step=step, time=time, label=label, lane=lane, kind="termination")
            self.graph.add_edge(prev_id, end_id, event="terminate", label_flip=False)


def all_interior_cycles_false(labels: Dict, excluded_cycles) -> bool:
    excluded_keys = {cycle_key(cycle) for cycle in _normalize_excluded_cycles(excluded_cycles)}
    interior_truths = [is_true(v) for c, v in labels.items() if cycle_key(c) not in excluded_keys]
    if not interior_truths:
        return True
    return not any(interior_truths)


def build_simulation(
    *,
    num_sensors: int,
    sensing_radius: float,
    timestep_size: float,
    sensor_velocity: float,
    seed: int,
) -> "EvasionPathSimulation":
    from time_stepping import EvasionPathSimulation

    np.random.seed(seed)
    domain = RectangularDomain()
    motion_model = BilliardMotion()
    fence = generate_fence_sensors(domain, sensing_radius)
    mobile_sensors = generate_mobile_sensors(domain, num_sensors, sensing_radius, sensor_velocity)
    sensor_network = SensorNetwork(mobile_sensors, motion_model, fence, sensing_radius, domain)
    return EvasionPathSimulation(sensor_network, timestep_size)


def build_unitcube_simulation(
    *,
    num_sensors: int,
    sensing_radius: float,
    timestep_size: float,
    sensor_velocity: float,
    seed: int,
    end_time: float = 0.0,
) -> "EvasionPathSimulation":
    from time_stepping import EvasionPathSimulation

    np.random.seed(seed)
    domain = UnitCube()
    motion_model = BilliardMotion()
    fence = get_unitcube_fence(spacing=sensing_radius)
    mobile_sensors = generate_mobile_sensors(domain, num_sensors, sensing_radius, sensor_velocity)
    sensor_network = SensorNetwork(mobile_sensors, motion_model, fence, sensing_radius, domain)
    return EvasionPathSimulation(sensor_network, timestep_size, end_time=end_time)


def run_online_reeb_simulation(
    simulation: "EvasionPathSimulation",
    *,
    max_steps: int = 1200,
    clear_streak_needed: int = 8,
) -> Tuple["EvasionPathSimulation", ReebEventGraphBuilder, int, int]:
    def nontrivial_history_entries(history_entries):
        return [
            (labels, alpha_change, boundary_change, time)
            for labels, alpha_change, boundary_change, time in history_entries
            if any(alpha_change) or boundary_change != (0, 0)
        ]

    builder = ReebEventGraphBuilder()
    builder.add_snapshot(
        step=0,
        time=simulation.time,
        labels=simulation.cycle_label.label,
        excluded_cycles=simulation.topology.excluded_cycles,
    )

    clear_streak = 0
    step = 0
    history_cursor = len(simulation.cycle_label.history)
    while step < max_steps:
        simulation.do_timestep()
        step += 1
        new_history_entries = simulation.cycle_label.history[history_cursor:]
        history_cursor = len(simulation.cycle_label.history)
        nontrivial_entries = nontrivial_history_entries(new_history_entries)

        if nontrivial_entries:
            for labels, _alpha_change, _boundary_change, event_time in nontrivial_entries:
                builder.add_snapshot(
                    step=step,
                    time=float(event_time),
                    labels=labels,
                    excluded_cycles=simulation.topology.excluded_cycles,
                )
        else:
            builder.add_snapshot(
                step=step,
                time=simulation.time,
                labels=simulation.cycle_label.label,
                excluded_cycles=simulation.topology.excluded_cycles,
            )

        if all_interior_cycles_false(simulation.cycle_label.label, simulation.topology.excluded_cycles):
            clear_streak += 1
        else:
            clear_streak = 0

        if clear_streak >= clear_streak_needed:
            break

    builder.close(step=step + 1, time=simulation.time)
    return simulation, builder, step, clear_streak


def run_2d_reeb_simulation(
    *,
    num_sensors: int = 12,
    sensing_radius: float = 0.3,
    timestep_size: float = 0.01,
    sensor_velocity: float = 1.0,
    max_steps: int = 1200,
    clear_streak_needed: int = 8,
    seed: int = 6,
) -> Tuple["EvasionPathSimulation", ReebEventGraphBuilder, int, int]:
    simulation = build_simulation(
        num_sensors=num_sensors,
        sensing_radius=sensing_radius,
        timestep_size=timestep_size,
        sensor_velocity=sensor_velocity,
        seed=seed,
    )
    return run_online_reeb_simulation(
        simulation,
        max_steps=max_steps,
        clear_streak_needed=clear_streak_needed,
    )


def run_3d_reeb_simulation_online(
    *,
    num_sensors: int = 20,
    sensing_radius: float = 0.3,
    timestep_size: float = 0.01,
    sensor_velocity: float = 1.0,
    max_steps: int = 1200,
    clear_streak_needed: int = 8,
    seed: int = 6,
    end_time: float = 0.0,
) -> Tuple["EvasionPathSimulation", ReebEventGraphBuilder, int, int]:
    simulation = build_unitcube_simulation(
        num_sensors=num_sensors,
        sensing_radius=sensing_radius,
        timestep_size=timestep_size,
        sensor_velocity=sensor_velocity,
        seed=seed,
        end_time=end_time,
    )
    return run_online_reeb_simulation(
        simulation,
        max_steps=max_steps,
        clear_streak_needed=clear_streak_needed,
    )


def _compact_view_graph(graph: nx.DiGraph) -> Tuple[nx.DiGraph, List[int]]:
    if graph.number_of_nodes() == 0:
        return nx.DiGraph(), []

    event_nodes: Set[int] = set()
    for n in graph.nodes:
        if graph.in_degree(n) == 0 or graph.out_degree(n) == 0:
            event_nodes.add(n)
        if graph.nodes[n].get("kind") == "termination":
            event_nodes.add(n)

    for u, v, d in graph.edges(data=True):
        event = d.get("event")
        if event in {"split", "merge"}:
            event_nodes.add(u)
            event_nodes.add(v)
        elif event == "terminate":
            # Keep only the termination endpoint as an event marker.
            event_nodes.add(v)
        if d.get("label_flip"):
            event_nodes.add(u)
            event_nodes.add(v)

    compact = nx.DiGraph()
    for n in event_nodes:
        compact.add_node(n, **graph.nodes[n])

    def add_or_merge_edge(a: int, b: int, event: str, label_flip: bool) -> None:
        if compact.has_edge(a, b):
            compact[a][b]["label_flip"] = compact[a][b].get("label_flip", False) or label_flip
            if compact[a][b].get("event") not in {"split", "merge", "terminate"}:
                compact[a][b]["event"] = event
            return
        compact.add_edge(a, b, event=event, label_flip=label_flip)

    for u, v, d in graph.edges(data=True):
        event = d.get("event")
        if event in {"split", "merge"} and (u in event_nodes and v in event_nodes):
            add_or_merge_edge(u, v, event=event, label_flip=bool(d.get("label_flip", False)))

    for start in sorted(event_nodes, key=lambda n: (graph.nodes[n]["time"], graph.nodes[n].get("step", 0), n)):
        for succ in graph.successors(start):
            edge_data = graph[start][succ]
            event = edge_data.get("event")
            if event == "terminate":
                if succ in event_nodes:
                    add_or_merge_edge(start, succ, event="terminate", label_flip=bool(edge_data.get("label_flip", False)))
                continue
            if event not in {"continue", "transform"}:
                continue

            accum_flip = bool(edge_data.get("label_flip", False))
            current = succ
            path_event = "continue"
            visited = {start}
            while current not in event_nodes:
                if current in visited:
                    break
                visited.add(current)

                if graph.in_degree(current) != 1 or graph.out_degree(current) != 1:
                    event_nodes.add(current)
                    compact.add_node(current, **graph.nodes[current])
                    break

                nxt = next(iter(graph.successors(current)))
                data = graph[current][nxt]
                next_event = data.get("event")
                if next_event == "terminate":
                    accum_flip = accum_flip or bool(data.get("label_flip", False))
                    current = nxt
                    path_event = "terminate"
                    break
                if data.get("event") not in {"continue", "transform"}:
                    event_nodes.add(current)
                    compact.add_node(current, **graph.nodes[current])
                    break
                accum_flip = accum_flip or bool(data.get("label_flip", False))
                current = nxt

            if current in event_nodes:
                t0 = graph.nodes[start]["time"]
                t1 = graph.nodes[current]["time"]
                if t1 > t0:
                    add_or_merge_edge(start, current, event=path_event, label_flip=accum_flip)

    event_node_list = sorted(event_nodes, key=lambda n: (graph.nodes[n]["time"], graph.nodes[n]["lane"], n))
    return compact, event_node_list


def _optimize_compact_layout(compact: nx.DiGraph, event_nodes: List[int]) -> Dict[int, Tuple[float, float]]:
    if compact.number_of_nodes() == 0:
        return {}

    lanes = sorted({int(compact.nodes[n].get("lane", 0)) for n in event_nodes})
    lane_to_rank = {lane: idx for idx, lane in enumerate(lanes)}
    y_spacing = 1.8
    y_by_node = {
        n: float(lane_to_rank[int(compact.nodes[n].get("lane", 0))]) * y_spacing
        for n in event_nodes
    }

    steps = sorted({int(compact.nodes[n].get("step", 0)) for n in event_nodes})
    layers: Dict[int, List[int]] = {s: [] for s in steps}
    for n in event_nodes:
        layers[int(compact.nodes[n].get("step", 0))].append(n)
    for s in steps:
        layers[s].sort(key=lambda n: (y_by_node[n], n))

    # Keep x as true event time so arrows always read forward in time.
    x_by_node = {n: float(compact.nodes[n]["time"]) for n in event_nodes}

    bucket: Dict[Tuple[int, int], List[int]] = {}
    for n in event_nodes:
        key = (int(compact.nodes[n].get("step", 0)), lane_to_rank[int(compact.nodes[n].get("lane", 0))])
        bucket.setdefault(key, []).append(n)
    for colliders in bucket.values():
        if len(colliders) <= 1:
            continue
        colliders.sort(key=lambda n: (compact.nodes[n].get("kind", ""), n))
        center = 0.5 * (len(colliders) - 1)
        for idx, n in enumerate(colliders):
            y_by_node[n] += (idx - center) * 0.22

    return {n: (x_by_node[n], y_by_node[n]) for n in event_nodes}


def _draw_compact_graph_panel(
    ax,
    graph: nx.DiGraph,
    event_nodes: List[int],
    *,
    highlight_time: float | None = None,
    highlight_half_width: float = 0.0,
) -> None:
    if graph.number_of_nodes() == 0:
        return

    pos = nx.get_node_attributes(graph, "pos")
    if highlight_time is not None and highlight_half_width > 0:
        ax.axvspan(
            highlight_time - highlight_half_width,
            highlight_time + highlight_half_width,
            color="red",
            alpha=0.15,
            zorder=0,
        )

    node_colors = []
    for node in event_nodes:
        kind = graph.nodes[node].get("kind", "cycle")
        label = graph.nodes[node].get("label", True)
        if kind == "termination":
            node_colors.append("lightgray")
        elif label:
            node_colors.append("tab:red")
        else:
            node_colors.append("tab:green")

    def edge_rad(u: int, v: int, event: str) -> float:
        if event not in {"split", "merge"}:
            return 0.0
        y0 = pos[u][1]
        y1 = pos[v][1]
        if y1 > y0:
            return 0.14
        if y1 < y0:
            return -0.14
        return 0.10

    for u, v, d in graph.edges(data=True):
        event = d.get("event", "continue")
        flip = bool(d.get("label_flip", False))
        rad = edge_rad(u, v, event)
        style = "dashed" if flip else "solid"
        color = "black" if flip else "0.35"
        width = 1.2 if flip else 1.0
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edgelist=[(u, v)],
            edge_color=color,
            arrows=True,
            width=width,
            style=style,
            connectionstyle=f"arc3,rad={rad}",
            arrowsize=12,
            min_source_margin=6,
            min_target_margin=6,
        )

    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        nodelist=event_nodes,
        node_size=100,
        node_color=node_colors,
        edgecolors="black",
        linewidths=0.5,
    )


def draw_reeb_graph(graph: nx.DiGraph, *, title: str = "Reeb Event Graph (2D)") -> None:
    import matplotlib.pyplot as plt

    compact, event_nodes = _compact_view_graph(graph)
    nx.set_node_attributes(compact, _optimize_compact_layout(compact, event_nodes), "pos")
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    _draw_compact_graph_panel(ax, compact, event_nodes)

    if title:
        ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("cycle lane")
    plt.tight_layout()
    plt.show()


def draw_live_reeb_panel(
    ax,
    graph: nx.DiGraph,
    *,
    highlight_time: float | None = None,
    highlight_half_width: float = 0.0,
    title: str = "",
) -> None:
    """Draw a compact reeb panel for the provided graph state on an existing axis."""
    ax.clear()
    if title:
        ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("cycle lane")

    if graph.number_of_nodes() == 0:
        return

    compact, event_nodes = _compact_view_graph(graph)
    layout_pos = _optimize_compact_layout(compact, event_nodes)
    nx.set_node_attributes(compact, layout_pos, "pos")
    _draw_compact_graph_panel(
        ax,
        compact,
        event_nodes,
        highlight_time=highlight_time,
        highlight_half_width=highlight_half_width,
    )

    xs = [p[0] for p in layout_pos.values()]
    ys = [p[1] for p in layout_pos.values()]
    if xs and ys:
        x_pad = max(0.01, 0.02 * (max(xs) - min(xs)))
        ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
        ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)


def animate_reeb_graph(
    graph: nx.DiGraph,
    *,
    interval_ms: int = 130,
    save_frames_dir: Path | None = None,
):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    max_step = max(graph.nodes[n]["step"] for n in graph.nodes)
    fig, ax = plt.subplots(figsize=(12, 7))
    saved_steps: Set[int] = set()

    def frame(step: int):
        ax.clear()
        ax.set_title(f"Reeb Event Graph Growth (step={step})")
        ax.set_xlabel("time")
        ax.set_ylabel("cycle lane")

        sub_nodes = [n for n in graph.nodes if graph.nodes[n]["step"] <= step]
        sub = graph.subgraph(sub_nodes).copy()
        compact, event_nodes = _compact_view_graph(sub)
        nx.set_node_attributes(compact, _optimize_compact_layout(compact, event_nodes), "pos")
        _draw_compact_graph_panel(ax, compact, event_nodes)

        if compact.number_of_nodes() > 0:
            ys = [compact.nodes[n]["pos"][1] for n in compact.nodes]
            ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)
        if save_frames_dir is not None and step not in saved_steps:
            fig.savefig(save_frames_dir / f"frame_{step:05d}.png", dpi=130)
            saved_steps.add(step)

    return FuncAnimation(fig, frame, frames=range(max_step + 1), interval=interval_ms, repeat=False)


def animate_sensor_and_reeb(
    builder: ReebEventGraphBuilder,
    *,
    run_steps: int,
    num_sensors: int,
    sensing_radius: float,
    timestep_size: float,
    sensor_velocity: float,
    seed: int,
    interval_ms: int = 130,
    save_frames_dir: Path | None = None,
):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from plotting_tools import show_state

    simulation = build_simulation(
        num_sensors=num_sensors,
        sensing_radius=sensing_radius,
        timestep_size=timestep_size,
        sensor_velocity=sensor_velocity,
        seed=seed,
    )
    if simulation.topology.dim != 2:
        raise ValueError("Coupled animation is currently only supported for 2D simulations.")

    compact, event_nodes = _compact_view_graph(builder.graph)
    layout_pos = _optimize_compact_layout(compact, event_nodes)
    nx.set_node_attributes(compact, layout_pos, "pos")
    xs = [p[0] for p in layout_pos.values()]
    ys = [p[1] for p in layout_pos.values()]
    x_pad = max(0.01, 0.02 * (max(xs) - min(xs) if xs else 1.0))
    y_pad = 1.0
    highlight_half_width = max(0.001, 0.48 * timestep_size)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(12, 10),
        gridspec_kw={"height_ratios": [3, 2]},
    )
    state = {"step": 0, "clear_streak": 0}
    saved_steps: Set[int] = set()

    def draw_graph_panel() -> None:
        ax_bottom.clear()
        ax_bottom.set_xlabel("time")
        ax_bottom.set_ylabel("cycle lane")
        _draw_compact_graph_panel(
            ax_bottom,
            compact,
            event_nodes,
            highlight_time=simulation.time,
            highlight_half_width=highlight_half_width,
        )
        if xs and ys:
            ax_bottom.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
            ax_bottom.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

    def draw_sensor_panel() -> None:
        ax_top.clear()
        ax_top.axis("off")
        ax_top.axis("equal")
        ax_top.set_title(f"T = {simulation.time:6.3f}", loc="left")
        show_state(simulation, ax=ax_top)

    def update(frame_idx: int):
        if frame_idx > 0 and state["step"] < run_steps:
            simulation.do_timestep()
            state["step"] += 1
            if all_interior_cycles_false(simulation.cycle_label.label, simulation.topology.outer_cycle):
                state["clear_streak"] += 1
            else:
                state["clear_streak"] = 0

        draw_sensor_panel()
        draw_graph_panel()
        if save_frames_dir is not None and state["step"] not in saved_steps:
            fig.savefig(save_frames_dir / f"frame_{state['step']:05d}.png", dpi=130)
            saved_steps.add(state["step"])

    draw_sensor_panel()
    draw_graph_panel()
    if save_frames_dir is not None and 0 not in saved_steps:
        fig.savefig(save_frames_dir / "frame_00000.png", dpi=130)
        saved_steps.add(0)
    ani = FuncAnimation(fig, update, interval=interval_ms, frames=range(run_steps + 1), repeat=False)
    ani._reeb_state = state
    return ani


def animate_online_reeb_graph(
    simulation: "EvasionPathSimulation",
    builder: ReebEventGraphBuilder,
    *,
    max_steps: int = 1200,
    clear_streak_needed: int = 8,
    interval_ms: int = 130,
    save_frames_dir: Path | None = None,
):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if builder.graph.number_of_nodes() == 0:
        builder.add_snapshot(
            step=0,
            time=simulation.time,
            labels=simulation.cycle_label.label,
            excluded_cycles=simulation.topology.excluded_cycles,
        )

    fig, ax = plt.subplots(figsize=(12, 7))
    state = {
        "step": 0,
        "clear_streak": 0,
        "finished": False,
        "closed": False,
        "history_cursor": len(simulation.cycle_label.history),
    }
    saved_steps: Set[int] = set()

    def nontrivial_history_entries(history_entries):
        return [
            (labels, alpha_change, boundary_change, time)
            for labels, alpha_change, boundary_change, time in history_entries
            if any(alpha_change) or boundary_change != (0, 0)
        ]

    def draw_panel() -> None:
        ax.clear()
        ax.set_title(f"T = {simulation.time:6.3f}", loc="left")
        ax.set_xlabel("time")
        ax.set_ylabel("cycle lane")

        if builder.graph.number_of_nodes() == 0:
            return

        compact, event_nodes = _compact_view_graph(builder.graph)
        layout_pos = _optimize_compact_layout(compact, event_nodes)
        nx.set_node_attributes(compact, layout_pos, "pos")
        _draw_compact_graph_panel(
            ax,
            compact,
            event_nodes,
            highlight_time=simulation.time,
            highlight_half_width=max(0.001, 0.48 * simulation.dt),
        )
        xs = [p[0] for p in layout_pos.values()]
        ys = [p[1] for p in layout_pos.values()]
        if xs and ys:
            x_pad = max(0.01, 0.02 * (max(xs) - min(xs) if xs else 1.0))
            ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
            ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)

    def update(_frame_idx: int):
        if state["finished"]:
            draw_panel()
            return

        if state["step"] >= max_steps:
            state["finished"] = True
        else:
            simulation.do_timestep()
            state["step"] += 1
            new_history_entries = simulation.cycle_label.history[state["history_cursor"] :]
            state["history_cursor"] = len(simulation.cycle_label.history)
            nontrivial_entries = nontrivial_history_entries(new_history_entries)

            if nontrivial_entries:
                for labels, _alpha_change, _boundary_change, event_time in nontrivial_entries:
                    builder.add_snapshot(
                        step=state["step"],
                        time=float(event_time),
                        labels=labels,
                        excluded_cycles=simulation.topology.excluded_cycles,
                    )
            else:
                builder.add_snapshot(
                    step=state["step"],
                    time=simulation.time,
                    labels=simulation.cycle_label.label,
                    excluded_cycles=simulation.topology.excluded_cycles,
                )
            if all_interior_cycles_false(simulation.cycle_label.label, simulation.topology.excluded_cycles):
                state["clear_streak"] += 1
            else:
                state["clear_streak"] = 0
            if state["clear_streak"] >= clear_streak_needed:
                state["finished"] = True

        if state["finished"] and not state["closed"]:
            builder.close(step=state["step"] + 1, time=simulation.time)
            state["closed"] = True

        draw_panel()
        if save_frames_dir is not None and state["step"] not in saved_steps:
            fig.savefig(save_frames_dir / f"frame_{state['step']:05d}.png", dpi=130)
            saved_steps.add(state["step"])

        if state["finished"]:
            ani.event_source.stop()

    draw_panel()
    if save_frames_dir is not None and 0 not in saved_steps:
        fig.savefig(save_frames_dir / "frame_00000.png", dpi=130)
        saved_steps.add(0)

    ani = FuncAnimation(fig, update, interval=interval_ms, frames=max_steps + 3, repeat=False)
    ani._reeb_state = state
    return ani


def print_atomic_change_report(
    simulation: "EvasionPathSimulation",
    summaries: Iterable[StepSummary],
    *,
    dt: float,
) -> None:
    uncovered_by_step = {s.step: s.n_true for s in summaries}
    history = simulation.cycle_label.history
    for labels, alpha_change, boundary_change, time in history[1:]:
        if tuple(boundary_change) in {(0, 0), (1, 1)}:
            continue
        step = max(1, int(round(time / dt)))
        uncovered = uncovered_by_step.get(step)
        if uncovered is None:
            uncovered = max(0, sum(1 for v in labels.values() if bool(v)) - 1)
        print(f"step={step} t={time:.3f} {alpha_change}{boundary_change} uncovered={uncovered}")


def outer_cycle_exclusion_report(
    simulation: "EvasionPathSimulation",
    builder: ReebEventGraphBuilder,
) -> Dict[str, bool]:
    """
    Check whether the current outer/alpha cycle leaked into the Reeb event graph.
    """
    alpha_key = cycle_key(simulation.topology.outer_cycle)
    alpha_in_graph = any(builder.graph.nodes[n].get("key") == alpha_key for n in builder.graph.nodes)
    return {
        "ok": not alpha_in_graph,
        "alpha_in_reeb_graph": alpha_in_graph,
    }


def make_frames_dir(user_dir: str | None = None) -> Path:
    if user_dir:
        folder = Path(user_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = Path.cwd() / f"reeb_frames_{stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def default_gif_path(prefix: str = "reeb_animation") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / f"{prefix}_{stamp}.gif"


def save_animation_gif(ani, path: Path, *, fps: int = 8) -> None:
    from matplotlib.animation import PillowWriter

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    ani.save(path, writer=writer)
