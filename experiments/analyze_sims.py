#!/usr/bin/env python3
"""Run multiple post-training RL simulations and inspect them in a tabbed viewer."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from UI import _build_reeb_graph_plot_data, _open_via_local_viewer_server, _write_timeline_html
from reeb_graph import ReebEventGraphBuilder
from rl_reeb_from_model import (
    _load_config,
    _policy_term_maps,
    _render_2d_frame,
    _resolve_checkpoint,
)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze many RL rollouts in a tabbed UI.")
    parser.add_argument("--run-dir", type=str, required=True, help="Training run directory.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        help="Checkpoint selector: best|final|init|step_2000 (or explicit .zip path).",
    )
    parser.add_argument("--num-sims", type=int, default=12, help="How many simulations to run and save.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional per-sim step cap (0 = natural termination).")
    parser.add_argument("--frame-interval-ms", type=int, default=130)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions instead of deterministic policy.")
    parser.add_argument(
        "--outdir",
        type=str,
        default="",
        help="Output dir (default: <run-dir>/analyze_sims_<checkpoint>).",
    )
    parser.add_argument("--open", action="store_true", help="Open the tabbed viewer after completion.")
    return parser


def _write_tabbed_gallery_html(output_html: Path, *, checkpoint_tag: str, summaries: List[Dict]) -> None:
    tabs_payload = json.dumps(summaries)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Analyze Sims</title>
  <style>
    :root {{
      --bg: #eef2f7;
      --panel: #ffffff;
      --ink: #102a43;
      --muted: #486581;
      --border: #d9e2ec;
      --blue: #1565c0;
      --red: #c62828;
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
      opacity: 0.74;
      transition: transform 0.12s ease, opacity 0.12s ease;
    }}
    .tab:hover {{ transform: translateY(-1px); opacity: 0.88; }}
    .tab.active {{ box-shadow: 0 0 0 3px rgba(16, 42, 67, 0.12); opacity: 1.0; }}
    .tab.clear {{ background: var(--blue); }}
    .tab.fail {{ background: var(--red); }}
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
        <div class="title">Analyze Sims</div>
        <div class="meta">checkpoint = {checkpoint_tag}</div>
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
      const status = sim.cleared ? "cleared" : (sim.timed_out ? "timed out" : "ended");
      metaEl.innerHTML =
        '<span><strong>sim</strong> ' + sim.sim_index + '</span>' +
        '<span><strong>status</strong> ' + status + '</span>' +
        '<span><strong>return</strong> ' + sim.return.toFixed(3) + '</span>' +
        '<span><strong>steps</strong> ' + sim.steps + '</span>' +
        '<span><strong>final_time</strong> ' + sim.final_time.toFixed(3) + '</span>';
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
      button.className = 'tab ' + (sim.cleared ? 'clear' : 'fail');
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


def main() -> None:
    args = _make_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    config = _load_config(run_dir)
    ckpt_tag, ckpt_path = _resolve_checkpoint(run_dir, args.checkpoint)

    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise SystemExit("stable-baselines3 is required. Install with: pip install stable-baselines3") from exc

    # Import custom classes for checkpoint deserialization.
    from rl_boundary_cycle_policy import BoundaryCycleActorCriticPolicy, BoundaryCycleStructuredExtractor  # noqa: F401
    from rl_gat_baseline import BaselineGraphAttentionExtractor  # noqa: F401
    from rl_gat_policy import DartAwareActorCriticPolicy, GraphAttentionExtractor  # noqa: F401
    from rl_homological_gat_policy import (  # noqa: F401
        HomologicalGATActorCriticPolicy,
        HomologicalGATLegacyActorCriticPolicy,
        HomologicalGraphAttentionExtractor,
        load_homological_gat_ppo,
    )
    from rl_structured_velocity_policy import (  # noqa: F401
        StructuredVelocityActorCriticPolicy,
        StructuredVelocityCycleExtractor,
        StructuredVelocityGraphExtractor,
    )
    from rl_unit_square_experiment import make_eval_env

    outdir = Path(args.outdir).resolve() if args.outdir else (run_dir / f"analyze_sims_{ckpt_tag}")
    outdir.mkdir(parents=True, exist_ok=True)

    model = load_homological_gat_ppo(ckpt_path)
    env = make_eval_env(
        config,
        enable_event_logging=False,
        event_log_path=None,
        max_steps_override=(None if int(args.max_steps) <= 0 else int(args.max_steps)),
    )

    num_sims = max(1, int(args.num_sims))
    deterministic = not bool(args.stochastic)
    summaries: List[Dict] = []

    for sim_idx in range(num_sims):
        print(f"[analyze_sims] running sim {sim_idx + 1}/{num_sims}")
        sim_dir = outdir / f"sim_{sim_idx:03d}"
        frames_dir = sim_dir / "frames"
        sim_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)

        obs, reset_info = env.reset(seed=int(config.eval_seeds[sim_idx % len(config.eval_seeds)]))
        sim = env.simulation
        done = False
        truncated = False
        ep_return = 0.0
        step = 0

        builder = ReebEventGraphBuilder()
        builder.add_snapshot(
            step=0,
            time=float(sim.time),
            labels=sim.cycle_label.label,
            alpha_cycle=sim.topology.outer_cycle,
        )
        history_cursor = len(sim.cycle_label.history)
        frame_records: List[Dict] = []

        frame_name = "frame_00000.png"
        _render_2d_frame(sim, frames_dir / frame_name)
        frame_records.append(
            {
                "step": 0,
                "time": float(sim.time),
                "phase": str(reset_info.get("phase", "simplify")),
                "image": str(Path("frames") / frame_name),
                "atomic_events": [],
                "snapshot_debug": builder.get_snapshot_debug(0),
                "policy_terms": _policy_term_maps(
                    reward_config=config.reward_config,
                    true_cycles_closed=0.0,
                    true_cycles_added=0.0,
                    merge_hazard_count=0.0,
                    elapsed=0.0,
                    effort=0.0,
                    neighbor_close_violation=0.0,
                    neighbor_far_violation=0.0,
                    fence_close_violation=0.0,
                    fence_far_violation=0.0,
                    area_progress=0.0,
                    perimeter_progress=0.0,
                    largest_area_progress=0.0,
                    largest_perimeter_progress=0.0,
                    area_regress=0.0,
                    perimeter_regress=0.0,
                    largest_area_regress=0.0,
                    largest_perimeter_regress=0.0,
                    area_residual=0.0,
                    perimeter_residual=0.0,
                    largest_area_residual=0.0,
                    largest_perimeter_residual=0.0,
                    one_hole_linger_penalty=0.0,
                    one_hole_area_scale=1.0,
                    one_hole_perimeter_scale=1.0,
                    clear_indicator=0.0,
                    success_time_bonus=0.0,
                    timeout_indicator=0.0,
                ),
            }
        )

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            action_arr = np.asarray(action, dtype=float)
            effort = float(np.mean(np.sum(action_arr * action_arr, axis=1))) if action_arr.size else 0.0
            obs, reward, done, truncated, info = env.step(action)
            ep_return += float(reward)
            step += 1
            sim = env.simulation

            new_history = sim.cycle_label.history[history_cursor:]
            history_cursor = len(sim.cycle_label.history)
            nontrivial = [
                (labels, alpha_change, boundary_change, event_time)
                for labels, alpha_change, boundary_change, event_time in new_history
                if any(alpha_change) or tuple(boundary_change) != (0, 0)
            ]

            atomic_events = []
            if nontrivial:
                for labels, alpha_change, boundary_change, event_time in nontrivial:
                    builder.add_snapshot(
                        step=step,
                        time=float(event_time),
                        labels=labels,
                        alpha_cycle=sim.topology.outer_cycle,
                    )
                    atomic_events.append(
                        {
                            "time": float(event_time),
                            "alpha_change": list(alpha_change),
                            "boundary_change": list(boundary_change),
                            "uncovered_cycles": max(0, sum(1 for v in labels.values() if bool(v)) - 1),
                        }
                    )
            else:
                builder.add_snapshot(
                    step=step,
                    time=float(sim.time),
                    labels=sim.cycle_label.label,
                    alpha_cycle=sim.topology.outer_cycle,
                )

            policy_terms = _policy_term_maps(
                reward_config=config.reward_config,
                true_cycles_closed=float(info.get("true_cycles_closed", 0.0)),
                true_cycles_added=float(info.get("true_cycles_added", 0.0)),
                merge_hazard_count=float(info.get("merge_hazard_count", 0.0)),
                elapsed=float(info.get("elapsed", 0.0)),
                effort=effort,
                neighbor_close_violation=float(info.get("neighbor_close_violation", 0.0)),
                neighbor_far_violation=float(info.get("neighbor_far_violation", 0.0)),
                fence_close_violation=float(info.get("fence_close_violation", 0.0)),
                fence_far_violation=float(info.get("fence_far_violation", 0.0)),
                area_progress=float(info.get("true_cycle_area_delta_norm", 0.0)),
                perimeter_progress=float(info.get("true_cycle_perimeter_delta_norm", 0.0)),
                largest_area_progress=float(info.get("largest_true_cycle_area_delta_norm", 0.0)),
                largest_perimeter_progress=float(info.get("largest_true_cycle_perimeter_delta_norm", 0.0)),
                area_regress=float(max(0.0, -float(info.get("true_cycle_area_delta_norm", 0.0)))),
                perimeter_regress=float(max(0.0, -float(info.get("true_cycle_perimeter_delta_norm", 0.0)))),
                largest_area_regress=float(max(0.0, -float(info.get("largest_true_cycle_area_delta_norm", 0.0)))),
                largest_perimeter_regress=float(max(0.0, -float(info.get("largest_true_cycle_perimeter_delta_norm", 0.0)))),
                area_residual=float(info.get("true_cycle_area_norm", 0.0)),
                perimeter_residual=float(info.get("true_cycle_perimeter_norm", 0.0)),
                largest_area_residual=float(info.get("largest_true_cycle_area_norm", 0.0)),
                largest_perimeter_residual=float(info.get("largest_true_cycle_perimeter_norm", 0.0)),
                one_hole_linger_penalty=float(info.get("reward_terms", {}).get("one_hole_linger_penalty", 0.0)),
                one_hole_area_scale=float(info.get("reward_terms", {}).get("one_hole_area_scale", 1.0)),
                one_hole_perimeter_scale=float(info.get("reward_terms", {}).get("one_hole_perimeter_scale", 1.0)),
                clear_indicator=float(1.0 if done else 0.0),
                success_time_bonus=(
                    1.0 / max(float(info.get("time", 0.0)), 1e-8)
                    if done
                    else 0.0
                ),
                timeout_indicator=float(1.0 if (truncated and not done) else 0.0),
            )
            frame_name = f"frame_{step:05d}.png"
            _render_2d_frame(sim, frames_dir / frame_name)
            frame_records.append(
                {
                    "step": step,
                    "time": float(sim.time),
                    "phase": str(info.get("phase", "simplify")),
                    "image": str(Path("frames") / frame_name),
                    "atomic_events": atomic_events,
                    "snapshot_debug": builder.get_snapshot_debug(step),
                    "policy_terms": policy_terms,
                }
            )

        builder.close(step=step + 1, time=float(sim.time))
        cleared = bool(done)
        timed_out = bool(truncated and not done)

        timeline = {
            "interval_ms": int(max(1, args.frame_interval_ms)),
            "highlight_half_width": float(max(0.001, 0.48 * config.dt)),
            "reeb_graph": _build_reeb_graph_plot_data(builder),
            "frames": frame_records,
            "export_config": {
                "source_dir": str(sim_dir.resolve()),
                "category": "rl",
                "motion_model": f"RL ({getattr(config, 'model_kind', 'policy')})",
                "display_name": run_dir.name,
                "source_run": run_dir.name,
                "checkpoint": ckpt_tag,
            },
        }
        (sim_dir / "timeline.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")
        _write_timeline_html(sim_dir / "index.html", timeline)

        sim_summary = {
            "sim_index": sim_idx,
            "viewer": str((sim_dir / "index.html").relative_to(outdir)),
            "return": float(ep_return),
            "steps": int(step),
            "final_time": float(sim.time),
            "cleared": cleared,
            "timed_out": timed_out,
        }
        (sim_dir / "summary.json").write_text(json.dumps(sim_summary, indent=2, sort_keys=True), encoding="utf-8")
        summaries.append(sim_summary)
        status = "cleared" if cleared else ("timed_out" if timed_out else "ended")
        print(
            f"[analyze_sims] sim {sim_idx + 1}/{num_sims} "
            f"status={status} steps={step} final_time={float(sim.time):.3f}"
        )

    env.close()

    manifest = {
        "run_dir": str(run_dir),
        "checkpoint_tag": ckpt_tag,
        "checkpoint_path": str(ckpt_path),
        "config": asdict(config),
        "num_sims": num_sims,
        "sim_summaries": summaries,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_tabbed_gallery_html(outdir / "index.html", checkpoint_tag=ckpt_tag, summaries=summaries)

    print(
        f"Finished {num_sims} simulations from checkpoint '{ckpt_tag}'. "
        f"Saved tabbed analysis UI to: {outdir}"
    )

    if args.open:
        import webbrowser

        webbrowser.open(_open_via_local_viewer_server(outdir))


if __name__ == "__main__":
    main()
