#!/usr/bin/env python3
"""Run motion-model simulations and export a tabbed Reeb/timeline UI for inspection."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from UI import (
    _build_reeb_graph_plot_data,
    _collect_atomic_events_by_step,
    _open_via_local_viewer_server,
    _render_2d_frame,
    _write_timeline_html,
)
from benchmark_common import (
    DOMAIN_DISPLAY,
    InitialCondition,
    MODEL_DISPLAY,
    build_domain,
    build_motion_model,
    canonical_model_name,
    combo_key,
    default_params_archive_path,
    domain_metadata,
    load_best_params_by_combo,
    parse_csv_strs,
    replicate_seed,
)
from reeb_graph import ReebEventGraphBuilder
from sensor_network import Sensor, SensorNetwork, generate_fence_sensors, generate_mobile_sensors
from time_stepping import EvasionPathSimulation
from topology import generate_topology


@dataclass
class SimArtifacts:
    sim_index: int
    seed_init: int
    seed_run: int
    init_retries_used: int
    steps: int
    final_time: float
    tau: float
    detection_time: float
    cleared: bool
    timed_out: bool
    max_steps_hit: bool
    viewer_relpath: str


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze motion-model simulations in a tabbed timeline UI.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model id, e.g. sequential_homological, sequential_homological_motion, homological, billiard.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="square",
        choices=["square", "circle", "rectangle_2to1_area1", "stadium_w0p6", "stadium_w1p2"],
    )
    parser.add_argument("--n-sensors", type=int, required=True)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--num-sims", type=int, default=6)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--velocity", type=float, default=1.0)
    parser.add_argument("--t-cap", type=float, default=5.0)
    parser.add_argument(
        "--fence-radius",
        type=float,
        default=0.0,
        help="Fence sensing radius. Default 0 uses the mobile radius.",
    )
    parser.add_argument(
        "--fence-offset-ratio",
        type=float,
        default=float("nan"),
        help="Optional offset ratio multiplied by fence radius when placing the fence.",
    )
    parser.add_argument("--use-weighted-alpha", action="store_true", default=False)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional hard step cap per sim. Default 0 means no explicit cap beyond t-cap.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed used to derive benchmark-style init/run seeds.")
    parser.add_argument("--failure-penalty", type=float, default=10.0)
    parser.add_argument("--worst-case-weight", type=float, default=0.25)
    parser.add_argument(
        "--params-json",
        type=str,
        default="",
        help="Optional tuned-params archive. If omitted for tuned models, defaults to output/params.",
    )
    parser.add_argument(
        "--models-requiring-params",
        type=str,
        default="homological,sequential_homological",
        help="Comma-separated model ids that should load tuned params.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="",
        help="Output dir (default: experiments/output/analyze_motion_model_sims/<run-name>).",
    )
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--frame-interval-ms", type=int, default=130)
    parser.add_argument("--open", action="store_true", help="Open the tabbed viewer after completion.")
    return parser


def _effective_fence_radius(radius: float, fence_radius: float) -> float:
    return float(radius if fence_radius <= 0.0 else fence_radius)


def _effective_fence_offset_ratio(raw_value: float) -> float | None:
    if math.isnan(float(raw_value)):
        return None
    return float(raw_value)


def _write_tabbed_gallery_html(
    output_html: Path,
    *,
    title: str,
    subtitle: str,
    summaries: List[Dict],
) -> None:
    tabs_payload = json.dumps(summaries)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Motion Model Analyze Sims</title>
  <style>
    :root {{
      --bg: #eef2f7;
      --panel: #ffffff;
      --ink: #102a43;
      --muted: #486581;
      --border: #d9e2ec;
      --blue: #1565c0;
      --red: #c62828;
      --amber: #b26a00;
      --tabtext: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 16px;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    .shell {{
      max-width: 1480px;
      margin: 0 auto;
      display: grid;
      gap: 12px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      box-shadow: 0 8px 18px rgba(16, 42, 67, 0.08);
      overflow: hidden;
    }}
    .header {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .title {{
      font-size: 16px;
      font-weight: 700;
    }}
    .meta {{
      font-size: 13px;
      color: var(--muted);
    }}
    .tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      background: #f8fbff;
    }}
    .tab {{
      border: 0;
      border-radius: 999px;
      padding: 9px 14px;
      font-weight: 700;
      cursor: pointer;
      color: var(--tabtext);
      opacity: 0.76;
      transition: transform 0.12s ease, opacity 0.12s ease;
    }}
    .tab:hover {{ transform: translateY(-1px); opacity: 0.9; }}
    .tab.active {{ box-shadow: 0 0 0 3px rgba(16, 42, 67, 0.12); opacity: 1; }}
    .tab.clear {{ background: var(--blue); }}
    .tab.fail {{ background: var(--red); }}
    .tab.capped {{ background: var(--amber); }}
    .sim-meta {{
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
      color: var(--muted);
      font-size: 13px;
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
    }}
    iframe {{
      width: 100%;
      height: 1260px;
      border: 0;
      display: block;
      background: #fff;
    }}
    @media (max-width: 900px) {{
      iframe {{ height: 1600px; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="card">
      <div class="header">
        <div class="title">{title}</div>
        <div class="meta">{subtitle}</div>
      </div>
      <div id="tabs" class="tabs"></div>
      <div id="simMeta" class="sim-meta"></div>
      <iframe id="viewer" title="Simulation viewer"></iframe>
    </div>
  </div>
  <script>
    const sims = {tabs_payload};
    const tabsEl = document.getElementById("tabs");
    const viewerEl = document.getElementById("viewer");
    const metaEl = document.getElementById("simMeta");

    function renderMeta(sim) {{
      const status = sim.cleared ? "cleared" : (sim.timed_out ? "timed out" : (sim.max_steps_hit ? "max steps" : "ended"));
      metaEl.innerHTML =
        '<span><strong>sim</strong> ' + sim.sim_index + '</span>' +
        '<span><strong>status</strong> ' + status + '</span>' +
        '<span><strong>steps</strong> ' + sim.steps + '</span>' +
        '<span><strong>final_time</strong> ' + sim.final_time.toFixed(3) + '</span>' +
        '<span><strong>tau</strong> ' + sim.tau.toFixed(3) + '</span>' +
        '<span><strong>seed_init</strong> ' + sim.seed_init + '</span>' +
        '<span><strong>seed_run</strong> ' + sim.seed_run + '</span>';
    }}

    function activate(index) {{
      const sim = sims[index];
      viewerEl.src = sim.viewer;
      renderMeta(sim);
      for (const button of tabsEl.querySelectorAll(".tab")) {{
        button.classList.toggle("active", Number(button.dataset.index) === index);
      }}
    }}

    sims.forEach((sim, index) => {{
      const button = document.createElement("button");
      let klass = 'fail';
      if (sim.cleared) klass = 'clear';
      else if (sim.max_steps_hit || sim.timed_out) klass = 'capped';
      button.className = 'tab ' + klass;
      button.dataset.index = String(index);
      button.textContent = 'sim_' + String(sim.sim_index).padStart(3, '0');
      button.addEventListener("click", () => activate(index));
      tabsEl.appendChild(button);
    }});

    if (sims.length > 0) {{
      activate(0);
    }}
  </script>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")


def _load_tuned_params_for_model(
    *,
    model_name: str,
    domain_name: str,
    dt: float,
    sensor_velocity: float,
    t_cap: float,
    params_json: str,
    models_requiring_params: List[str],
    failure_penalty: float,
    worst_case_weight: float,
) -> tuple[Dict[str, Dict], Path | None]:
    if model_name not in models_requiring_params:
        return {}, None
    path = Path(params_json).resolve() if params_json else default_params_archive_path(
        REPO_ROOT,
        model_name=model_name,
        domain_name=domain_name,
        dt=dt,
        sensor_velocity=sensor_velocity,
        t_cap=t_cap,
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Missing tuned params for {model_name}. Expected archive at: {path}"
        )
    return load_best_params_by_combo(
        path,
        failure_penalty=float(failure_penalty),
        worst_case_weight=float(worst_case_weight),
    ), path


def _build_sensor_network_for_run(
    *,
    model_name: str,
    tuned_params_by_combo: Dict[str, Dict],
    domain_name: str,
    n_sensors: int,
    radius: float,
    fence_radius: float,
    fence_offset_ratio: float | None,
    dt: float,
    sensor_velocity: float,
    init_condition,
    seed_run: int,
    use_weighted_alpha: bool,
) -> SensorNetwork:
    np.random.seed(seed_run)
    domain = build_domain(domain_name)
    motion_model = build_motion_model(
        model_name,
        domain_name=domain_name,
        n_sensors=n_sensors,
        radius=radius,
        dt=dt,
        sensor_velocity=sensor_velocity,
        tuned_params_by_combo=tuned_params_by_combo,
    )
    fence = [
        Sensor(np.array(pos, dtype=float), np.zeros(2, dtype=float), float(fence_radius), boundary_sensor=True)
        for pos in init_condition.fence_positions
    ]
    mobile = [
        Sensor(np.array(pos, dtype=float), np.array(vel, dtype=float), float(radius), boundary_sensor=False)
        for pos, vel in zip(init_condition.mobile_positions, init_condition.mobile_velocities)
    ]
    return SensorNetwork(
        mobile,
        motion_model,
        fence,
        radius,
        domain,
        use_weighted_alpha=bool(use_weighted_alpha),
    )


def _is_face_connected_initial_condition(
    *,
    init: InitialCondition,
    radius: float,
    domain,
    use_weighted_alpha: bool,
    fence_radius: float,
) -> bool:
    del domain
    fence_positions = np.asarray(init.fence_positions, dtype=float)
    mobile_positions = np.asarray(init.mobile_positions, dtype=float)
    points = np.vstack([fence_positions, mobile_positions])
    point_radii = None
    if use_weighted_alpha:
        point_radii = np.concatenate(
            [
                np.full(len(fence_positions), float(fence_radius), dtype=float),
                np.full(len(mobile_positions), float(radius), dtype=float),
            ]
        )
    interior_point = np.mean(fence_positions, axis=0) if fence_positions.size else None
    topology = generate_topology(
        points,
        radius,
        point_radii=point_radii,
        fence_node_count=int(len(fence_positions)),
        interior_point=interior_point,
    )
    return bool(topology.is_face_connected())


def _generate_connected_initial_condition(
    *,
    domain,
    n_sensors: int,
    radius: float,
    fence_radius: float,
    fence_offset_ratio: float | None,
    sensor_velocity: float,
    seed: int,
    max_retries: int,
    use_weighted_alpha: bool,
) -> tuple[InitialCondition | None, int | None, int, bool]:
    offset_distance = None
    if fence_offset_ratio is not None:
        offset_distance = float(fence_offset_ratio) * float(fence_radius)
    for retry in range(max_retries + 1):
        retry_seed = int((seed + retry * 104_729) % (2**32 - 1))
        np.random.seed(retry_seed)
        fence = generate_fence_sensors(domain, float(fence_radius), offset_distance=offset_distance)
        mobile = generate_mobile_sensors(domain, int(n_sensors), float(radius), float(sensor_velocity))
        init = InitialCondition(
            fence_positions=np.asarray([np.array(sensor.pos, dtype=float) for sensor in fence], dtype=float),
            mobile_positions=np.asarray([np.array(sensor.pos, dtype=float) for sensor in mobile], dtype=float),
            mobile_velocities=np.asarray([np.array(sensor.vel, dtype=float) for sensor in mobile], dtype=float),
        )
        if _is_face_connected_initial_condition(
            init=init,
            radius=float(radius),
            domain=domain,
            use_weighted_alpha=bool(use_weighted_alpha),
            fence_radius=float(fence_radius),
        ):
            return init, retry_seed, retry, True
    return None, None, max_retries, False


def _run_motion_simulation(
    *,
    sensor_network: SensorNetwork,
    dt: float,
    t_cap: float,
    max_steps: int,
) -> tuple[EvasionPathSimulation, ReebEventGraphBuilder, int, bool, bool, bool]:
    simulation = EvasionPathSimulation(sensor_network, dt, end_time=t_cap)
    builder = ReebEventGraphBuilder()
    builder.add_snapshot(
        step=0,
        time=float(simulation.time),
        labels=simulation.cycle_label.label,
        excluded_cycles=simulation.topology.excluded_cycles,
    )

    step = 0
    timed_out = False
    max_steps_hit = False
    history_cursor = len(simulation.cycle_label.history)
    while simulation.cycle_label.has_intruder():
        if max_steps > 0 and step >= max_steps:
            max_steps_hit = True
            break
        simulation.do_timestep()
        step += 1

        new_history_entries = simulation.cycle_label.history[history_cursor:]
        history_cursor = len(simulation.cycle_label.history)
        nontrivial_entries = [
            (labels, alpha_change, boundary_change, event_time)
            for labels, alpha_change, boundary_change, event_time in new_history_entries
            if any(alpha_change) or tuple(boundary_change) != (0, 0)
        ]
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
                time=float(simulation.time),
                labels=simulation.cycle_label.label,
                excluded_cycles=simulation.topology.excluded_cycles,
            )
        if 0 < simulation.Tend < simulation.time:
            timed_out = True
            break

    builder.close(step=step + 1, time=float(simulation.time))
    cleared = not simulation.cycle_label.has_intruder()
    return simulation, builder, step, cleared, timed_out, max_steps_hit


def _write_sim_timeline_bundle(
    *,
    sim_dir: Path,
    sensor_network_builder,
    seed_run: int,
    builder: ReebEventGraphBuilder,
    original_simulation: EvasionPathSimulation,
    run_steps: int,
    dt: float,
    frame_interval_ms: int,
    export_config: Dict,
) -> None:
    frames_dir = sim_dir / "frames"
    sim_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    replay_network = sensor_network_builder(seed_run)
    replay = EvasionPathSimulation(replay_network, dt, end_time=original_simulation.Tend)

    frame_records = []
    step_to_summary = {summary.step: summary for summary in builder.summaries}
    atomic_events_by_step = _collect_atomic_events_by_step(
        history=original_simulation.cycle_label.history,
        summaries=builder.summaries,
    )
    for step in range(run_steps + 1):
        if step > 0:
            replay.do_timestep()
        frame_name = f"frame_{step:05d}.png"
        _render_2d_frame(replay, frames_dir / frame_name)
        summary = step_to_summary.get(step)
        frame_records.append(
            {
                "step": step,
                "time": float(replay.time),
                "phase": "motion_model",
                "image": str(Path("frames") / frame_name),
                "summary": {
                    "n_cycles": int(summary.n_cycles) if summary else 0,
                    "n_true": int(summary.n_true) if summary else 0,
                    "n_false": int(summary.n_false) if summary else 0,
                    "n_birth": int(summary.n_birth) if summary else 0,
                    "n_death": int(summary.n_death) if summary else 0,
                    "n_continue": int(summary.n_continue) if summary else 0,
                    "n_split_edges": int(summary.n_split_edges) if summary else 0,
                    "n_merge_edges": int(summary.n_merge_edges) if summary else 0,
                    "n_transform_edges": int(summary.n_transform_edges) if summary else 0,
                    "n_label_flips": int(summary.n_label_flips) if summary else 0,
                },
                "atomic_events": atomic_events_by_step.get(step, []),
                "snapshot_debug": builder.get_snapshot_debug(step),
                "policy_terms": {"raw": {}, "weighted": {}},
            }
        )

    timeline = {
        "interval_ms": int(max(1, frame_interval_ms)),
        "highlight_half_width": float(max(0.001, 0.48 * dt)),
        "reeb_graph": _build_reeb_graph_plot_data(builder),
        "active_term_names": [],
        "frames": frame_records,
        "export_config": dict(export_config),
    }
    (sim_dir / "timeline.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    _write_timeline_html(sim_dir / "index.html", timeline)


def main() -> None:
    args = _make_parser().parse_args()
    model_name = canonical_model_name(args.model)
    if model_name not in MODEL_DISPLAY:
        raise ValueError(f"Unknown model: {args.model}. Allowed: {sorted(MODEL_DISPLAY)}")

    models_requiring_params = [canonical_model_name(name) for name in parse_csv_strs(args.models_requiring_params)]
    fence_radius = _effective_fence_radius(float(args.radius), float(args.fence_radius))
    fence_offset_ratio = _effective_fence_offset_ratio(float(args.fence_offset_ratio))
    tuned_params_by_combo, params_path = _load_tuned_params_for_model(
        model_name=model_name,
        domain_name=args.domain,
        dt=float(args.dt),
        sensor_velocity=float(args.velocity),
        t_cap=float(args.t_cap),
        params_json=args.params_json,
        models_requiring_params=models_requiring_params,
        failure_penalty=float(args.failure_penalty),
        worst_case_weight=float(args.worst_case_weight),
    )
    selected_combo = combo_key(args.n_sensors, args.radius)
    selected_params = dict(tuned_params_by_combo.get(selected_combo, {}).get("params", {}))
    if model_name in models_requiring_params and not selected_params:
        raise ValueError(
            f"No tuned params found for {model_name} at {selected_combo} in {params_path}"
        )

    run_name = args.run_name or (
        f"{model_name}_{args.domain}_n{int(args.n_sensors)}_r{float(args.radius):.3f}_inspect"
    )
    outdir = Path(args.outdir).resolve() if args.outdir else (
        REPO_ROOT / "experiments" / "output" / "analyze_motion_model_sims" / run_name
    )
    outdir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict] = []
    sim_artifacts: List[SimArtifacts] = []
    total = max(1, int(args.num_sims))
    for sim_idx in range(total):
        print(f"[analyze_motion_model_sims] running sim {sim_idx + 1}/{total}")
        seed_init = replicate_seed(int(args.seed), args.domain, int(args.n_sensors), float(args.radius), sim_idx)
        seed_run = replicate_seed(int(args.seed) + 101, args.domain, int(args.n_sensors), float(args.radius), sim_idx)
        domain = build_domain(args.domain)
        init_condition, init_seed_used, init_retries_used, feasible = _generate_connected_initial_condition(
            domain=domain,
            n_sensors=int(args.n_sensors),
            radius=float(args.radius),
            fence_radius=float(fence_radius),
            fence_offset_ratio=fence_offset_ratio,
            sensor_velocity=float(args.velocity),
            seed=seed_init,
            max_retries=200,
            use_weighted_alpha=bool(args.use_weighted_alpha),
        )
        if not feasible or init_condition is None:
            raise RuntimeError(
                f"Unable to sample a connected initialization for sim {sim_idx} "
                f"(seed_init={seed_init}, n={args.n_sensors}, r={args.radius})."
            )

        def _sensor_network_builder(run_seed: int) -> SensorNetwork:
            return _build_sensor_network_for_run(
                model_name=model_name,
                tuned_params_by_combo=tuned_params_by_combo,
                domain_name=args.domain,
                n_sensors=int(args.n_sensors),
                radius=float(args.radius),
                fence_radius=float(fence_radius),
                fence_offset_ratio=fence_offset_ratio,
                dt=float(args.dt),
                sensor_velocity=float(args.velocity),
                init_condition=init_condition,
                seed_run=run_seed,
                use_weighted_alpha=bool(args.use_weighted_alpha),
            )

        sensor_network = _sensor_network_builder(seed_run)
        simulation, builder, steps, cleared, timed_out, max_steps_hit = _run_motion_simulation(
            sensor_network=sensor_network,
            dt=float(args.dt),
            t_cap=float(args.t_cap),
            max_steps=int(args.max_steps),
        )

        final_time = float(simulation.time)
        failed = (not cleared) or timed_out or max_steps_hit
        tau = float(args.t_cap if failed else min(final_time, float(args.t_cap)))
        detection_time = float("inf") if failed else final_time

        sim_dir = outdir / f"sim_{sim_idx:03d}"
        _write_sim_timeline_bundle(
            sim_dir=sim_dir,
            sensor_network_builder=_sensor_network_builder,
            seed_run=seed_run,
            builder=builder,
            original_simulation=simulation,
            run_steps=steps,
            dt=float(args.dt),
            frame_interval_ms=int(args.frame_interval_ms),
            export_config={
                "source_dir": str(sim_dir.resolve()),
                "category": "motion_model",
                "motion_model": MODEL_DISPLAY[model_name],
                "display_name": run_name,
                "source_run": run_name,
                "checkpoint": "",
            },
        )

        sim_summary = {
            "sim_index": sim_idx,
            "viewer": str((sim_dir / "index.html").relative_to(outdir)),
            "seed_init": int(init_seed_used if init_seed_used is not None else seed_init),
            "seed_run": int(seed_run),
            "init_retries_used": int(init_retries_used),
            "steps": int(steps),
            "final_time": final_time,
            "tau": float(tau),
            "detection_time": (None if not np.isfinite(detection_time) else float(detection_time)),
            "cleared": bool(cleared),
            "timed_out": bool(timed_out),
            "max_steps_hit": bool(max_steps_hit),
        }
        (sim_dir / "summary.json").write_text(json.dumps(sim_summary, indent=2, sort_keys=True), encoding="utf-8")
        summaries.append(sim_summary)
        sim_artifacts.append(
            SimArtifacts(
                sim_index=int(sim_idx),
                seed_init=int(sim_summary["seed_init"]),
                seed_run=int(seed_run),
                init_retries_used=int(init_retries_used),
                steps=int(steps),
                final_time=float(final_time),
                tau=float(tau),
                detection_time=float(detection_time),
                cleared=bool(cleared),
                timed_out=bool(timed_out),
                max_steps_hit=bool(max_steps_hit),
                viewer_relpath=sim_summary["viewer"],
            )
        )
        status = "cleared" if cleared else ("timed_out" if timed_out else ("max_steps" if max_steps_hit else "ended"))
        print(
            f"[analyze_motion_model_sims] sim {sim_idx + 1}/{total} "
            f"status={status} steps={steps} final_time={final_time:.3f}"
        )

    clear_rate = float(np.mean([1.0 if item.cleared else 0.0 for item in sim_artifacts])) if sim_artifacts else 0.0
    mean_tau = float(np.mean([item.tau for item in sim_artifacts])) if sim_artifacts else math.nan
    mean_detection = float(
        np.mean([item.detection_time for item in sim_artifacts if np.isfinite(item.detection_time)])
    ) if any(np.isfinite(item.detection_time) for item in sim_artifacts) else math.nan

    manifest = {
        "script": "experiments/analyze_motion_model_sims.py",
        "model": model_name,
        "model_display": MODEL_DISPLAY[model_name],
        "domain": args.domain,
        "domain_display": DOMAIN_DISPLAY[args.domain],
        "domain_metadata": domain_metadata(args.domain),
        "n_sensors": int(args.n_sensors),
        "radius": float(args.radius),
        "dt": float(args.dt),
        "velocity": float(args.velocity),
        "t_cap": float(args.t_cap),
        "fence_radius": float(fence_radius),
        "fence_offset_ratio": fence_offset_ratio,
        "use_weighted_alpha": bool(args.use_weighted_alpha),
        "max_steps": int(args.max_steps),
        "base_seed": int(args.seed),
        "failure_penalty": float(args.failure_penalty),
        "worst_case_weight": float(args.worst_case_weight),
        "num_sims": int(total),
        "params_json": str(params_path) if params_path is not None else "",
        "selected_combo_key": selected_combo,
        "selected_params": selected_params,
        "summary": {
            "clear_rate": clear_rate,
            "mean_tau": mean_tau,
            "mean_detection_time": mean_detection,
        },
        "sim_summaries": summaries,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_tabbed_gallery_html(
        outdir / "index.html",
        title=f"{MODEL_DISPLAY[model_name]} Analyze Sims",
        subtitle=(
            f"{DOMAIN_DISPLAY[args.domain]} | n={int(args.n_sensors)} | r={float(args.radius):.3f} | "
            f"dt={float(args.dt):.3f} | velocity={float(args.velocity):.3f}"
        ),
        summaries=summaries,
    )

    print(f"Saved motion-model analysis UI to: {outdir}")
    if args.open:
        import webbrowser

        webbrowser.open(_open_via_local_viewer_server(outdir))


if __name__ == "__main__":
    main()
