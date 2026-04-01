# ******************************************************************************
#  Copyright (c) 2023, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************
"""Single-run 3D UnitCube simulation with live top+bottom visualization.

Top panel: 3D UnitCube void/sensor view.
Bottom panel: online Reeb graph growth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from UI import (
    Live3DTimelineSession,
    build_and_open_3d_timeline_ui,
    collect_3d_scene_snapshot,
)
from reeb_graph import (
    ReebEventGraphBuilder,
    all_interior_cycles_false,
    default_gif_path,
    draw_live_reeb_panel,
    build_unitcube_simulation,
    outer_cycle_exclusion_report,
    print_atomic_change_report,
    save_animation_gif,
)
from utilities import MaxRecursionDepthError
from visualization import show_unitcube_void_state


def _short_cycle_key(key: str) -> str:
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:8]


def _print_matching_debug(
    simulation,
    builder: ReebEventGraphBuilder,
    *,
    snapshot_step: int,
    alpha_change: tuple,
    boundary_change: tuple,
    event_time: float,
) -> None:
    if snapshot_step <= 0:
        return
    dbg = builder.get_snapshot_debug(snapshot_step)
    if not dbg:
        return

    born_labels = dbg.get("born_labels", {})
    died_labels = dbg.get("died_labels", {})
    parents_for_child = dbg.get("parents_for_child", {})
    born_edge_debug = dbg.get("born_edge_debug", [])
    persistent_flips = dbg.get("persistent_labels", {})

    if (
        not any(alpha_change)
        and boundary_change == (0, 0)
        and not born_labels
        and not died_labels
        and not persistent_flips
    ):
        return

    born_repr = ", ".join(f"{_short_cycle_key(k)}:{int(v)}" for k, v in born_labels.items()) or "-"
    died_repr = ", ".join(f"{_short_cycle_key(k)}:{int(v)}" for k, v in died_labels.items()) or "-"
    parent_repr = (
        ", ".join(
            f"{_short_cycle_key(child)}<-[{','.join(_short_cycle_key(p) for p in parents)}]"
            for child, parents in parents_for_child.items()
        )
        or "-"
    )
    edge_repr = (
        ", ".join(
            f"{_short_cycle_key(e['parent'])}->{_short_cycle_key(e['child'])}:{e['event']}:flip={int(e['label_flip'])}:"
            f"{int(e['parent_label'])}->{int(e['child_label'])}"
            for e in born_edge_debug
        )
        or "-"
    )
    flip_repr = (
        ", ".join(f"{_short_cycle_key(k)}:{int(v0)}->{int(v1)}" for k, (v0, v1) in persistent_flips.items())
        or "-"
    )
    print(
        f"DEBUG step={snapshot_step} t={float(event_time):.3f} atomic={alpha_change} boundary={boundary_change} "
        f"born=[{born_repr}] died=[{died_repr}] parents=[{parent_repr}] edges=[{edge_repr}] persistent_flips=[{flip_repr}]"
    )


def _print_live_atomic_summary(
    builder: ReebEventGraphBuilder,
    *,
    outer_step: int,
    alpha_change: tuple,
    boundary_change: tuple,
    event_time: float,
) -> None:
    if tuple(boundary_change) in {(0, 0), (1, 1)}:
        return
    if not builder.summaries:
        return

    summary = builder.summaries[-1]

    active_interior = int(summary.n_true)
    total_interior = int(summary.n_cycles)

    print(
        f"LIVE step={outer_step} t={float(event_time):.6f} alpha_change={alpha_change} boundary_change={boundary_change} "
        f"active={active_interior}/{total_interior} births={summary.n_birth} deaths={summary.n_death} "
        f"true={summary.n_true} false={summary.n_false}"
    )


def _state_change_dump(state_change) -> dict:
    result = {
        "dim": int(state_change.dim),
        "alpha_change": list(state_change.alpha_complex_change()),
        "boundary_change": list(state_change.boundary_cycle_change()),
        "is_atomic": bool(state_change.is_atomic_change()),
        "simplices_added_counts": {},
        "simplices_removed_counts": {},
        "simplices_added": {},
        "simplices_removed": {},
    }
    for dim in range(1, state_change.dim + 1):
        diff = state_change.simplices_difference[dim]
        added = diff.added()
        removed = diff.removed()
        result["simplices_added_counts"][str(dim)] = len(added)
        result["simplices_removed_counts"][str(dim)] = len(removed)
        result["simplices_added"][str(dim)] = [str(item) for item in sorted(added, key=str)]
        result["simplices_removed"][str(dim)] = [str(item) for item in sorted(removed, key=str)]
    return result


def _write_recursion_debug_dump(simulation, builder, args, state, exc: MaxRecursionDepthError) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path.cwd() / f"recursion_debug_3d_{stamp}.json"

    labels = simulation.cycle_label.label
    alpha_key = str(simulation.topology.outer_cycle)
    interior = [(cycle, val) for cycle, val in labels.items() if str(cycle) != alpha_key]
    active = sum(1 for _cycle, val in interior if bool(val))

    payload = {
        "exception": {
            "type": type(exc).__name__,
            "message": str(exc),
            "level": exc.level,
            "adaptive_dt": exc.adaptive_dt,
            "sim_time": exc.sim_time,
        },
        "run_config": {
            "num_sensors": args.num_sensors,
            "radius": args.radius,
            "dt": args.dt,
            "velocity": args.velocity,
            "max_steps": args.max_steps,
            "clear_streak": args.clear_streak,
            "seed": args.seed,
            "end_time": args.end_time,
            "ui_only": args.ui_only,
        },
        "runtime_state": {
            "outer_step": state["step"],
            "sim_time": simulation.time,
            "clear_streak": state["clear_streak"],
            "labels_total_interior": len(interior),
            "labels_active_interior": active,
            "sensor_points": [list(map(float, p)) for p in simulation.sensor_network.points],
            "fence_points": [list(map(float, s.pos)) for s in simulation.sensor_network.fence_sensors],
        },
        "reeb_tail_summary": [
            {
                "step": s.step,
                "time": s.time,
                "n_cycles": s.n_cycles,
                "n_true": s.n_true,
                "n_false": s.n_false,
                "n_birth": s.n_birth,
                "n_death": s.n_death,
                "n_continue": s.n_continue,
                "n_split_edges": s.n_split_edges,
                "n_merge_edges": s.n_merge_edges,
                "n_transform_edges": s.n_transform_edges,
                "n_label_flips": s.n_label_flips,
            }
            for s in builder.summaries[-10:]
        ],
        "history_tail": [
            {
                "time": entry[3],
                "alpha_change": list(entry[1]),
                "boundary_change": list(entry[2]),
            }
            for entry in simulation.cycle_label.history[-15:]
        ],
        "state_change": _state_change_dump(exc.state_change),
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _atomic_events_from_history_delta(history_entries) -> list[dict]:
    events = []
    for labels, alpha_change, boundary_change, event_time in history_entries:
        if not any(alpha_change) and boundary_change == (0, 0):
            continue
        events.append(
            {
                "time": float(event_time),
                "alpha_change": list(alpha_change),
                "boundary_change": list(boundary_change),
                "uncovered_cycles": max(0, sum(1 for v in labels.values() if bool(v)) - 1),
            }
        )
    return events


def _nontrivial_history_entries(history_entries):
    return [
        (labels, alpha_change, boundary_change, event_time)
        for labels, alpha_change, boundary_change, event_time in history_entries
        if any(alpha_change) or boundary_change != (0, 0)
    ]


class LiveMatplotlib3DScene:
    def __init__(self, *, timestep_size: float) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        self.timestep_size = timestep_size
        self.plt.ion()
        self.fig = self.plt.figure(figsize=(8.5, 7.2))
        self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")
        self.info = self.fig.text(
            0.02,
            0.01,
            "",
            fontsize=9,
            va="bottom",
            ha="left",
            family="monospace",
        )

    def _alive(self) -> bool:
        return self.fig is not None and self.plt.fignum_exists(self.fig.number)

    def update(self, *, simulation, step: int, atomic_events: list[dict]) -> None:
        if not self._alive():
            return

        scene = collect_3d_scene_snapshot(simulation)
        elev = getattr(self.ax, "elev", 22.0)
        azim = getattr(self.ax, "azim", -55.0)
        self.ax.cla()

        mobile = np.asarray(scene["mobile_points"], dtype=float)
        if mobile.size:
            self.ax.scatter(mobile[:, 0], mobile[:, 1], mobile[:, 2], c="tab:red", s=18, depthshade=True)
        fence = np.asarray(scene["fence_points"], dtype=float)
        if fence.size:
            self.ax.scatter(fence[:, 0], fence[:, 1], fence[:, 2], c="black", s=8, alpha=0.35, depthshade=False)

        edges = np.asarray(scene["alpha_edge_positions"], dtype=float)
        if edges.size:
            edge_arr = edges.reshape(-1, 2, 3)
            for segment in edge_arr:
                self.ax.plot(
                    segment[:, 0],
                    segment[:, 1],
                    segment[:, 2],
                    color="0.45",
                    alpha=0.55,
                    linewidth=0.8,
                )

        for cycle in scene["cycles"]:
            tri = np.asarray(cycle["positions"], dtype=float)
            if tri.size == 0:
                continue
            tri_faces = tri.reshape(-1, 3, 3)
            surf = Poly3DCollection(
                tri_faces,
                facecolors=cycle["color"],
                edgecolors=(0.2, 0.2, 0.2, 0.45),
                linewidths=0.4,
                alpha=0.36,
            )
            self.ax.add_collection3d(surf)

        self.ax.set_xlim(-0.12, 1.12)
        self.ax.set_ylim(-0.12, 1.12)
        self.ax.set_zlim(-0.12, 1.12)
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.view_init(elev=elev, azim=azim)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.set_title(f"3D Sensor Network | step={step} t={simulation.time:.6f}", loc="left")

        if atomic_events:
            msg = [f"atomic events ({len(atomic_events)}):"]
            for event in atomic_events[:5]:
                msg.append(
                    f"t={event['time']:.6f} alpha={event['alpha_change']} boundary={event['boundary_change']}"
                )
            if len(atomic_events) > 5:
                msg.append(f"... +{len(atomic_events)-5} more")
            self.info.set_text("\n".join(msg))
        else:
            self.info.set_text("atomic events: none at this step")

        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def finalize(self) -> None:
        if self._alive():
            self.plt.ioff()
            self.plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one 3D UnitCube sim with live Reeb graph growth.")
    parser.add_argument("--num-sensors", type=int, default=20)
    parser.add_argument("--radius", type=float, default=0.45)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--velocity", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--clear-streak", type=int, default=8)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--end-time", type=float, default=0.0)
    parser.add_argument("--interval-ms", type=int, default=120)
    parser.add_argument(
        "--ui-view",
        choices=["matplotlib", "web"],
        default="web",
        help="Viewer used when opening post-run UI.",
    )
    parser.add_argument("--debug-matching", action="store_true", default=False)
    parser.add_argument(
        "--debug-recursion-dump",
        action="store_true",
        default=True,
        help="Write a JSON crash dump when MaxRecursionDepthError occurs.",
    )
    parser.add_argument(
        "--ui-only",
        action="store_true",
        default=False,
        help="Run simulation headlessly and open the three.js UI (no matplotlib animation or GIF).",
    )
    parser.add_argument(
        "--live-ui",
        action="store_true",
        default=False,
        help="Open a live browser UI and stream simulation + Reeb updates while running (ui-only mode).",
    )
    parser.add_argument(
        "--live-mpl-scene",
        action="store_true",
        default=False,
        help="When using --live-ui, also open a live matplotlib 3D sensor scene window.",
    )
    parser.add_argument(
        "--live-mpl-update-every-steps",
        type=int,
        default=1,
        help="Refresh the live matplotlib 3D scene every N outer steps.",
    )
    parser.add_argument(
        "--live-batch-atomic-changes",
        type=int,
        default=20,
        help="Flush live UI updates every N non-trivial atomic events.",
    )
    parser.add_argument("--save-gif", action="store_true", default=False)
    parser.add_argument("--gif-path", type=str, default="")
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument(
        "--strict-outer-cycle-check",
        action="store_true",
        default=False,
        help="Raise if outer/alpha cycle is detected in Reeb graph.",
    )
    parser.add_argument(
        "--graph-out",
        type=str,
        default="",
        help="Optional pickle output path for the built Reeb graph (networkx.DiGraph).",
    )
    args = parser.parse_args()

    if not args.ui_only:
        try:
            args.ui_only = input("Run in UI-only mode (no matplotlib/gif)? [y/N]: ").strip().lower() == "y"
        except EOFError:
            args.ui_only = False
    if args.ui_only and not args.live_ui:
        try:
            args.live_ui = input("Open live UI while simulation runs? [y/N]: ").strip().lower() == "y"
        except EOFError:
            args.live_ui = False
    if args.ui_only:
        # UI-only mode is browser Reeb timeline only (no standalone matplotlib windows).
        args.ui_view = "web"
        args.live_mpl_scene = False

    print(f"Number of sensors: {args.num_sensors}, Radius: {args.radius}")

    simulation = build_unitcube_simulation(
        num_sensors=args.num_sensors,
        sensing_radius=args.radius,
        timestep_size=args.dt,
        sensor_velocity=args.velocity,
        seed=args.seed,
        end_time=args.end_time,
    )

    builder = ReebEventGraphBuilder()
    builder.add_snapshot(
        step=0,
        time=simulation.time,
        labels=simulation.cycle_label.label,
        alpha_cycle=simulation.topology.outer_cycle,
    )
    if builder.has_excluded_cycles():
        raise RuntimeError("Alpha/fence cycle leaked into Reeb graph at initialization.")

    state = {"step": 0, "clear_streak": 0, "finished": False, "closed": False}
    live_session = None
    live_mpl_scene = None
    atomic_since_flush = 0
    history_cursor = len(simulation.cycle_label.history)
    if args.ui_only and args.live_ui:
        batch_atomic = max(1, int(args.live_batch_atomic_changes))
        live_session = Live3DTimelineSession(
            timestep_size=args.dt,
            interval_ms=args.interval_ms,
        )
        live_session.append_snapshot(
            simulation=simulation,
            builder=builder,
            step=0,
            atomic_events=[],
            flush=True,
        )
        opened = live_session.open()
        if opened:
            print(f"Opened live UI: {live_session.url} (flush every {batch_atomic} atomic changes)")
        else:
            print(f"Live UI server ready at: {live_session.url} (auto-open failed, open manually)")
        if args.live_mpl_scene:
            live_mpl_scene = LiveMatplotlib3DScene(timestep_size=args.dt)
            live_mpl_scene.update(simulation=simulation, step=0, atomic_events=[])
            print("Opened live matplotlib 3D sensor scene window.")
    else:
        batch_atomic = max(1, int(args.live_batch_atomic_changes))

    def advance_one_step() -> None:
        nonlocal history_cursor, atomic_since_flush
        if not state["finished"]:
            if state["step"] >= args.max_steps:
                state["finished"] = True
            else:
                try:
                    simulation.do_timestep()
                except MaxRecursionDepthError as exc:
                    print("\n[ERROR] MaxRecursionDepthError encountered during simulation.do_timestep().")
                    print(f"[ERROR] {exc}")
                    if args.debug_recursion_dump:
                        dump_path = _write_recursion_debug_dump(simulation, builder, args, state, exc)
                        print(f"[ERROR] Wrote recursion debug dump: {dump_path}")
                    raise
                state["step"] += 1

                new_history_entries = simulation.cycle_label.history[history_cursor:]
                history_cursor = len(simulation.cycle_label.history)
                nontrivial_entries = _nontrivial_history_entries(new_history_entries)

                if nontrivial_entries:
                    for labels, alpha_change, boundary_change, event_time in nontrivial_entries:
                        builder.add_snapshot(
                            step=state["step"],
                            time=float(event_time),
                            labels=labels,
                            alpha_cycle=simulation.topology.outer_cycle,
                        )
                        _print_live_atomic_summary(
                            builder,
                            outer_step=state["step"],
                            alpha_change=alpha_change,
                            boundary_change=boundary_change,
                            event_time=float(event_time),
                        )
                        if args.debug_matching:
                            _print_matching_debug(
                                simulation,
                                builder,
                                snapshot_step=state["step"],
                                alpha_change=alpha_change,
                                boundary_change=boundary_change,
                                event_time=float(event_time),
                            )
                else:
                    builder.add_snapshot(
                        step=state["step"],
                        time=simulation.time,
                        labels=simulation.cycle_label.label,
                        alpha_cycle=simulation.topology.outer_cycle,
                    )

                if builder.has_excluded_cycles():
                    raise RuntimeError("Alpha/fence cycle leaked into Reeb graph during animation.")

                current_events = _atomic_events_from_history_delta(nontrivial_entries)
                if live_session is not None:
                    atomic_since_flush += len(current_events)
                    flush_now = atomic_since_flush >= batch_atomic
                    live_session.append_snapshot(
                        simulation=simulation,
                        builder=builder,
                        step=state["step"],
                        atomic_events=current_events,
                        flush=flush_now,
                    )
                    if flush_now:
                        atomic_since_flush = 0
                if live_mpl_scene is not None:
                    update_every = max(1, int(args.live_mpl_update_every_steps))
                    should_update = (state["step"] % update_every == 0) or bool(current_events)
                    if should_update:
                        live_mpl_scene.update(
                            simulation=simulation,
                            step=state["step"],
                            atomic_events=current_events,
                        )

                if all_interior_cycles_false(simulation.cycle_label.label, simulation.topology.outer_cycle):
                    state["clear_streak"] += 1
                else:
                    state["clear_streak"] = 0

                if state["clear_streak"] >= args.clear_streak:
                    state["finished"] = True

        if state["finished"] and not state["closed"]:
            builder.close(step=state["step"] + 1, time=simulation.time)
            if builder.has_excluded_cycles():
                raise RuntimeError("Alpha/fence cycle leaked into Reeb graph on close.")
            state["closed"] = True
            if live_session is not None:
                live_session.flush()

    if args.ui_only:
        while not state["finished"]:
            advance_one_step()
    else:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        fig = plt.figure(figsize=(12, 10))
        ax_top = fig.add_subplot(2, 1, 1, projection="3d")
        ax_bottom = fig.add_subplot(2, 1, 2)

        def draw() -> None:
            ax_top.clear()
            ax_top.set_title(f"T = {simulation.time:6.3f}", loc="left")
            show_unitcube_void_state(simulation, ax=ax_top)

            draw_live_reeb_panel(
                ax_bottom,
                builder.graph,
                highlight_time=simulation.time,
                highlight_half_width=max(0.001, 0.48 * args.dt),
                title="",
            )

        def update(_):
            advance_one_step()
            draw()

            if state["finished"]:
                ani.event_source.stop()

        draw()
        ani = FuncAnimation(fig, update, interval=args.interval_ms, frames=args.max_steps + 3, repeat=False)

        if args.save_gif:
            gif_path = Path(args.gif_path) if args.gif_path else default_gif_path("sample_reeb_3d_online")
            save_animation_gif(ani, gif_path, fps=args.gif_fps)

        plt.tight_layout()
        plt.show()

    history = simulation.cycle_label.history
    nontrivial_atomic_count = 0
    for _labels, alpha_change, boundary_change, _time in history[1:]:
        if any(alpha_change) or boundary_change != (0, 0):
            nontrivial_atomic_count += 1

    print("----- Simulation Summary -----")
    print(f"mode={'ui-only' if args.ui_only else 'matplotlib'}")
    print(
        f"3D UnitCube online Reeb build complete: "
        f"t={simulation.time:.4f}, steps={state['step']}, clear_streak={state['clear_streak']}, "
        f"nodes={builder.graph.number_of_nodes()}, edges={builder.graph.number_of_edges()}"
    )
    print(f"nontrivial_atomic_events={nontrivial_atomic_count}")
    if nontrivial_atomic_count > 0:
        print_atomic_change_report(simulation, builder.summaries, dt=args.dt)
    else:
        print("No non-trivial atomic changes recorded.")
    check = outer_cycle_exclusion_report(simulation, builder)
    print(f"outer_cycle_check: ok={check['ok']} alpha_in_reeb_graph={check['alpha_in_reeb_graph']}")
    if args.strict_outer_cycle_check and not check["ok"]:
        raise RuntimeError("Outer/alpha cycle detected in Reeb graph.")

    if live_session is not None:
        live_session.close(finished=True)
    if live_mpl_scene is not None:
        print("Simulation finished. Matplotlib 3D window left open for inspection.")
        live_mpl_scene.finalize()

    launch_ui = args.ui_only
    if live_session is not None:
        launch_ui = False
    if not launch_ui:
        try:
            launch_ui = input("Simulation finished. Open 3D timeline UI? [y/N]: ").strip().lower() == "y"
        except EOFError:
            launch_ui = False

    if launch_ui:
        html_path = build_and_open_3d_timeline_ui(
            simulation=simulation,
            builder=builder,
            run_steps=state["step"],
            num_sensors=args.num_sensors,
            sensing_radius=args.radius,
            timestep_size=args.dt,
            sensor_velocity=args.velocity,
            seed=args.seed,
            end_time=args.end_time,
            interval_ms=args.interval_ms,
        )
        print(f"Opened 3D timeline UI: {html_path}")

    if args.graph_out:
        out_path = Path(args.graph_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            pickle.dump(builder.graph, f)
        print(f"Saved Reeb graph to: {out_path}")


if __name__ == "__main__":
    main()
