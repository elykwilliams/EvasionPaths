from __future__ import annotations

import json
import re
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)


def launch_motion_reeb_setup_ui(
    *,
    title: str,
    model_key: str,
    model_label: str,
    motion_factory: Callable[..., object],
    num_mobile: int = 12,
    sensing_radius: float = 0.22,
    dt: float = 0.01,
    sensor_velocity: float = 1.0,
    seed: int = 7,
    max_steps: int = 1200,
    clear_streak: int = 8,
    host: str = "127.0.0.1",
    port: int = 0,
    auto_open: bool = True,
    motion_kwargs: dict | None = None,
) -> str:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from UI import (
        _build_reeb_graph_plot_data,
        _collect_atomic_events_by_step,
        _render_2d_frame,
        _write_timeline_html,
    )
    from boundary_geometry import RectangularDomain
    from motion_model import HomologicalDynamicsMotion, SequentialHomologicalMotion
    from reeb_graph import print_atomic_change_report, run_online_reeb_simulation
    from sensor_network import Sensor, SensorNetwork, generate_fence_sensors
    from time_stepping import EvasionPathSimulation
    from topology import generate_topology

    del HomologicalDynamicsMotion, SequentialHomologicalMotion

    rng = np.random.default_rng(seed)
    domain = RectangularDomain(0.0, 1.0, 0.0, 1.0)
    domain_area = float((domain.max[0] - domain.min[0]) * (domain.max[1] - domain.min[1]))
    radius = float(sensing_radius)
    motion_kwargs = dict(motion_kwargs or {})
    output_root = Path(__file__).resolve().parents[1] / "examples" / "output" / f"{model_key}_reeb_ui"
    output_root.mkdir(parents=True, exist_ok=True)

    def _initial_mobile_points():
        angles = np.linspace(0.0, 2.0 * np.pi, num_mobile, endpoint=False)
        a = 0.32
        b = 0.18
        jitter = 0.015
        pts = np.column_stack(
            [
                0.5 + a * np.cos(angles) + rng.normal(0.0, jitter, size=num_mobile),
                0.5 + b * np.sin(angles) + rng.normal(0.0, jitter, size=num_mobile),
            ]
        )
        return np.clip(pts, 0.05, 0.95)

    initial_mobile_points = _initial_mobile_points()

    def _fence_points(current_radius):
        return np.asarray(
            [np.asarray(sensor.pos, dtype=float) for sensor in generate_fence_sensors(domain, float(current_radius))],
            dtype=float,
        )

    def _safe_points(raw_points):
        arr = np.asarray(raw_points, dtype=float)
        if arr.size == 0:
            return np.zeros((0, 2), dtype=float)
        arr = arr.reshape((-1, 2))
        return np.clip(arr, 0.0, 1.0)

    def _all_points(mobile_points, current_radius):
        mobile = _safe_points(mobile_points)
        fence_points = _fence_points(current_radius)
        if mobile.size == 0:
            return fence_points.copy()
        return np.vstack([fence_points, mobile])

    def _cycle_nodes_from_topology_cycle(topology_obj, cycle):
        try:
            simplex = next(iter(cycle))
            nodes = topology_obj.cmap.get_cycle_nodes(simplex)
        except Exception:
            nodes = list(cycle.nodes)
        nodes = [int(v) for v in nodes]
        if len(nodes) > 1 and nodes[0] == nodes[-1]:
            nodes = nodes[:-1]
        return nodes

    def _topology_data(mobile_points, current_radius):
        current_radius = float(current_radius)
        mobile_points = _safe_points(mobile_points)
        fence_points = _fence_points(current_radius)
        all_points = _all_points(mobile_points, current_radius)
        n_mobile = int(mobile_points.shape[0])
        n_fence = int(len(fence_points))
        mobile_area_ratio = float(n_mobile * np.pi * current_radius * current_radius / domain_area)
        total_area_ratio = float((n_mobile + n_fence) * np.pi * current_radius * current_radius / domain_area)
        triangular_kappa_total = float((3.0 * np.sqrt(3.0) / 2.0) * (n_mobile + n_fence) * current_radius * current_radius / domain_area)

        # Estimate the actual covered fraction of the unit square by the union of mobile sensing disks.
        union_grid = 240
        xs = np.linspace(domain.min[0], domain.max[0], union_grid)
        ys = np.linspace(domain.min[1], domain.max[1], union_grid)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        covered = np.zeros(xx.shape, dtype=bool)
        for px, py in mobile_points:
            covered |= (xx - px) ** 2 + (yy - py) ** 2 <= current_radius * current_radius
        mobile_union_fraction = float(np.mean(covered))
        try:
            topo = generate_topology(
                all_points.tolist(),
                current_radius,
                fence_node_count=int(len(fence_points)),
                interior_point=np.array([0.5, 0.5], dtype=float),
            )
        except Exception:
            return {
                "radius": float(current_radius),
                "fence_points": fence_points.tolist(),
                "domain_area": domain_area,
                "mobile_count": n_mobile,
                "fence_count": n_fence,
                "mobile_area_ratio": mobile_area_ratio,
                "mobile_union_fraction": mobile_union_fraction,
                "total_area_ratio": total_area_ratio,
                "triangular_kappa_total": triangular_kappa_total,
                "triangular_cover_satisfied": bool(triangular_kappa_total >= 1.0),
                "edges": [],
                "triangles": [],
                "uncovered_cycles": [],
                "boundary_edges": [],
            }

        edges = sorted({tuple(sorted(int(v) for v in simplex)) for simplex in topo.simplices(1)})
        triangles = sorted({tuple(sorted(int(v) for v in simplex)) for simplex in topo.simplices(2)})

        boundary_edges = set()
        uncovered_cycles = []
        seen_cycles = set()
        for cycle in topo.homology_generators:
            for half_edge in cycle:
                a, b = half_edge.nodes
                boundary_edges.add(tuple(sorted((int(a), int(b)))))

            if topo.is_excluded_cycle(cycle):
                continue
            try:
                if not topo.is_connected_cycle(cycle):
                    continue
            except Exception:
                continue

            nodes = _cycle_nodes_from_topology_cycle(topo, cycle)
            if len(nodes) < 3:
                continue
            key = tuple(nodes)
            if key in seen_cycles:
                continue
            seen_cycles.add(key)
            uncovered_cycles.append(nodes)

        return {
            "radius": float(current_radius),
            "fence_points": fence_points.tolist(),
            "domain_area": domain_area,
            "mobile_count": n_mobile,
            "fence_count": n_fence,
            "mobile_area_ratio": mobile_area_ratio,
            "mobile_union_fraction": mobile_union_fraction,
            "total_area_ratio": total_area_ratio,
            "triangular_kappa_total": triangular_kappa_total,
            "triangular_cover_satisfied": bool(triangular_kappa_total >= 1.0),
            "edges": edges,
            "triangles": triangles,
            "uncovered_cycles": uncovered_cycles,
            "boundary_edges": sorted(boundary_edges),
        }

    def _initial_velocities(n_mobile_points):
        local_rng = np.random.default_rng(seed)
        velocities = []
        for _ in range(n_mobile_points):
            vec = local_rng.normal(0.0, 1.0, size=2)
            norm = float(np.linalg.norm(vec))
            if norm < 1e-12:
                vec = np.array([1.0, 0.0], dtype=float)
                norm = 1.0
            velocities.append(sensor_velocity * vec / norm)
        return np.asarray(velocities, dtype=float)

    def _build_simulation(mobile_points, velocities, current_radius):
        mobile_points = _safe_points(mobile_points)
        velocities = np.asarray(velocities, dtype=float).reshape((-1, 2))
        mobile_sensors = [
            Sensor(np.array(pos, dtype=float), np.array(vel, dtype=float), float(current_radius), False)
            for pos, vel in zip(mobile_points, velocities)
        ]
        fence = generate_fence_sensors(domain, float(current_radius))
        motion_model = motion_factory(
            sensing_radius=float(current_radius),
            **motion_kwargs,
        )
        sensor_network = SensorNetwork(mobile_sensors, motion_model, fence, float(current_radius), domain)
        return EvasionPathSimulation(sensor_network, dt)

    def _build_timeline_for_simulation(*, simulation, builder, run_steps, mobile_points, velocities, current_radius):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base_dir = output_root / f"timeline_{stamp}"
        frames_dir = base_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        replay = _build_simulation(mobile_points, velocities, current_radius)
        frame_records = []
        step_to_summary = {s.step: s for s in builder.summaries}
        atomic_events_by_step = _collect_atomic_events_by_step(
            history=simulation.cycle_label.history,
            summaries=builder.summaries,
        )

        for step in range(run_steps + 1):
            if step > 0:
                replay.do_timestep()
            frame_name = f"frame_{step:05d}.png"
            frame_path = frames_dir / frame_name
            _render_2d_frame(replay, frame_path)

            summary = step_to_summary.get(step)
            frame_records.append(
                {
                    "step": step,
                    "time": float(replay.time),
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
                }
            )

        timeline = {
            "interval_ms": 130,
            "highlight_half_width": float(max(0.001, 0.48 * dt)),
            "reeb_graph": _build_reeb_graph_plot_data(builder),
            "frames": frame_records,
        }
        output_html = base_dir / "index.html"
        (base_dir / "timeline.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")
        _write_timeline_html(output_html, timeline)
        html_text = output_html.read_text(encoding="utf-8")
        html_text = re.sub(
            r"\s*<div class=\"term-grid\">.*?</div>\s*</div>\s*<div class=\"controls\">",
            "\n      <div class=\"controls\">",
            html_text,
            flags=re.S,
        )
        html_text = re.sub(
            r"\s*const rawTermsBox = document\.getElementById\(\"rawTerms\"\);\s*const weightedTermsBox = document\.getElementById\(\"weightedTerms\"\);\s*",
            "\n",
            html_text,
        )
        html_text = re.sub(
            r"\s*function renderTermBars\(container, termMap\) \{.*?\n    \}\n",
            "\n",
            html_text,
            flags=re.S,
        )
        html_text = re.sub(
            r"\s*const terms = item\.policy_terms \|\| \{\};\s*renderTermBars\(rawTermsBox, terms\.raw \|\| null\);\s*renderTermBars\(weightedTermsBox, terms\.weighted \|\| null\);\s*",
            "\n",
            html_text,
        )
        output_html.write_text(html_text, encoding="utf-8")
        return base_dir

    bootstrap = {
        "title": title,
        "model_label": model_label,
        "sensing_radius": radius,
        "dt": float(dt),
        "sensor_velocity": float(sensor_velocity),
        "max_steps": int(max_steps),
        "clear_streak": int(clear_streak),
        "mobile_points": initial_mobile_points.tolist(),
        "topology": _topology_data(initial_mobile_points, radius),
    }

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f3f4f6;
      --panel: #ffffff;
      --ink: #12263a;
      --muted: #5c6f82;
      --line: #d7e2eb;
      --accent: #0f766e;
      --danger: #c2410c;
      --blue: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 8% 10%, #dbeafe 0%, transparent 42%),
        radial-gradient(circle at 85% 5%, #dcfce7 0%, transparent 35%),
        var(--bg);
      min-height: 100vh;
      padding: 18px;
    }}
    .stage {{
      max-width: 1720px;
      margin: 0 auto;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
      overflow: hidden;
    }}
    .center-card {{
      max-width: 860px;
      margin: 18px auto;
    }}
    .head {{
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }}
    .body {{
      padding: 14px 16px 16px;
      display: grid;
      gap: 12px;
    }}
    .toolbar, .inline-row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .inline-row {{
      gap: 12px;
      justify-content: space-between;
    }}
    .radius-box {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #f8fafc;
    }}
    input[type="number"] {{
      width: 90px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 6px 8px;
      font: inherit;
      color: var(--ink);
      background: #fff;
    }}
    button {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 10px;
      padding: 8px 12px;
      font-weight: 600;
      cursor: pointer;
    }}
    button.primary {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
    button.warn {{ background: #fff7ed; color: var(--danger); border-color: #fdba74; }}
    button.active {{ background: var(--blue); color: #fff; border-color: var(--blue); }}
    button:disabled {{ opacity: 0.6; cursor: wait; }}
    .meta {{
      display: grid;
      gap: 4px;
      font-size: 13px;
      color: var(--muted);
    }}
    svg {{
      width: 100%;
      aspect-ratio: 1 / 1;
      display: block;
      background:
        linear-gradient(0deg, rgba(15, 23, 42, 0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(15, 23, 42, 0.05) 1px, transparent 1px),
        #fbfdff;
      background-size: 10% 10%;
      border: 1px solid var(--line);
      border-radius: 14px;
      touch-action: none;
    }}
    .status {{
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      background: #f8fafc;
      font-size: 13px;
      color: var(--muted);
      white-space: pre-wrap;
    }}
    iframe {{
      width: 100%;
      min-height: 1020px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
    }}
    .hidden {{ display: none !important; }}
  </style>
</head>
<body>
  <div class="stage">
    <div id="setupCard" class="card center-card">
      <div class="head">
        <div>
          <div style="font-size:18px;font-weight:700;">{title}</div>
          <div style="font-size:13px;color:var(--muted);">{model_label} setup before running the simulation</div>
        </div>
      </div>
      <div class="body">
        <div class="toolbar">
          <button id="moveBtn" class="active">Move</button>
          <button id="addBtn">Add</button>
          <button id="removeBtn" class="warn">Remove</button>
          <button id="resetBtn">Reset</button>
        </div>
        <div class="inline-row">
          <div class="radius-box">
            <label for="radiusInput"><strong>Sensor radius</strong></label>
            <input id="radiusInput" type="number" min="0.05" max="0.60" step="0.01" />
            <button id="applyRadiusBtn">Apply</button>
          </div>
          <button id="runBtn" class="primary">Run Sim</button>
        </div>
        <div class="meta" id="meta"></div>
        <svg id="networkView" viewBox="0 0 1000 1000"></svg>
        <div class="status" id="status">Arrange the mobile sensors, adjust the sensing radius if needed, then click Run Sim.</div>
      </div>
    </div>

    <div id="playbackCard" class="card hidden">
      <div class="head">
        <div>
          <div style="font-size:18px;font-weight:700;">Playback</div>
          <div style="font-size:13px;color:var(--muted);">Post-run sensor timeline and Reeb graph</div>
        </div>
        <div class="toolbar">
          <button id="anotherBtn">Run Another Sim</button>
        </div>
      </div>
      <div class="body">
        <div class="status" id="playbackStatus">No run yet.</div>
        <iframe id="playbackFrame"></iframe>
      </div>
    </div>
  </div>
  <script>
    const BOOTSTRAP = {json.dumps(bootstrap)};
    const svg = document.getElementById("networkView");
    const setupCard = document.getElementById("setupCard");
    const playbackCard = document.getElementById("playbackCard");
    const statusBox = document.getElementById("status");
    const playbackStatus = document.getElementById("playbackStatus");
    const playbackFrame = document.getElementById("playbackFrame");
    const metaBox = document.getElementById("meta");
    const moveBtn = document.getElementById("moveBtn");
    const addBtn = document.getElementById("addBtn");
    const removeBtn = document.getElementById("removeBtn");
    const resetBtn = document.getElementById("resetBtn");
    const runBtn = document.getElementById("runBtn");
    const anotherBtn = document.getElementById("anotherBtn");
    const radiusInput = document.getElementById("radiusInput");
    const applyRadiusBtn = document.getElementById("applyRadiusBtn");

    let mode = "move";
    let mobilePoints = BOOTSTRAP.mobile_points.map((p) => [Number(p[0]), Number(p[1])]);
    const initialMobilePoints = BOOTSTRAP.mobile_points.map((p) => [Number(p[0]), Number(p[1])]);
    let currentRadius = Number(BOOTSTRAP.sensing_radius);
    let topology = BOOTSTRAP.topology;
    let dragIndex = -1;
    let running = false;
    const SCALE = 1000;
    radiusInput.value = currentRadius.toFixed(2);

    function setMode(nextMode) {{
      mode = nextMode;
      [moveBtn, addBtn, removeBtn].forEach((btn) => btn.classList.remove("active"));
      if (mode === "move") moveBtn.classList.add("active");
      if (mode === "add") addBtn.classList.add("active");
      if (mode === "remove") removeBtn.classList.add("active");
    }}

    function enterSetupMode() {{
      setupCard.classList.remove("hidden");
      playbackCard.classList.add("hidden");
      playbackFrame.removeAttribute("src");
    }}

    function enterPlaybackMode() {{
      setupCard.classList.add("hidden");
      playbackCard.classList.remove("hidden");
      window.scrollTo({{ top: 0, behavior: "smooth" }});
    }}

    function clamp01(v) {{
      return Math.max(0, Math.min(1, v));
    }}

    function worldBounds() {{
      const fence = (topology.fence_points || []).map((p) => [Number(p[0]), Number(p[1])]);
      const all = fence.concat(mobilePoints);
      let minX = Math.min(...all.map((p) => p[0]));
      let maxX = Math.max(...all.map((p) => p[0]));
      let minY = Math.min(...all.map((p) => p[1]));
      let maxY = Math.max(...all.map((p) => p[1]));
      const pad = Math.max(0.06, 0.45 * currentRadius);
      minX -= pad; maxX += pad; minY -= pad; maxY += pad;
      return {{ minX, maxX, minY, maxY }};
    }}

    function worldToCanvas(point) {{
      const b = worldBounds();
      const spanX = Math.max(1e-6, b.maxX - b.minX);
      const spanY = Math.max(1e-6, b.maxY - b.minY);
      const scale = Math.min(SCALE / spanX, SCALE / spanY);
      const usedW = spanX * scale;
      const usedH = spanY * scale;
      const offsetX = 0.5 * (SCALE - usedW);
      const offsetY = 0.5 * (SCALE - usedH);
      return [
        offsetX + (point[0] - b.minX) * scale,
        offsetY + (point[1] - b.minY) * scale,
        scale,
      ];
    }}

    function canvasToWorld(event) {{
      const rect = svg.getBoundingClientRect();
      const px = ((event.clientX - rect.left) / rect.width) * SCALE;
      const py = SCALE - (((event.clientY - rect.top) / rect.height) * SCALE);
      const b = worldBounds();
      const spanX = Math.max(1e-6, b.maxX - b.minX);
      const spanY = Math.max(1e-6, b.maxY - b.minY);
      const scale = Math.min(SCALE / spanX, SCALE / spanY);
      const usedW = spanX * scale;
      const usedH = spanY * scale;
      const offsetX = 0.5 * (SCALE - usedW);
      const offsetY = 0.5 * (SCALE - usedH);
      const x = b.minX + (px - offsetX) / scale;
      const y = b.minY + (py - offsetY) / scale;
      return [clamp01(x), clamp01(y)];
    }}

    function dist2(a, b) {{
      const dx = a[0] - b[0];
      const dy = a[1] - b[1];
      return dx * dx + dy * dy;
    }}

    function nearestMobile(point, threshold=0.06) {{
      let best = -1;
      let bestVal = threshold * threshold;
      mobilePoints.forEach((p, idx) => {{
        const d = dist2(p, point);
        if (d <= bestVal) {{
          best = idx;
          bestVal = d;
        }}
      }});
      return best;
    }}

    async function refreshTopology() {{
      const response = await fetch("/topology", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ mobile_points: mobilePoints, radius: currentRadius }}),
      }});
      topology = await response.json();
      render();
    }}

    function line(a, b, stroke, width, opacity=1.0, dash="") {{
      const p1 = worldToCanvas(a);
      const p2 = worldToCanvas(b);
      const el = document.createElementNS("http://www.w3.org/2000/svg", "line");
      el.setAttribute("x1", String(p1[0]));
      el.setAttribute("y1", String(SCALE - p1[1]));
      el.setAttribute("x2", String(p2[0]));
      el.setAttribute("y2", String(SCALE - p2[1]));
      el.setAttribute("stroke", stroke);
      el.setAttribute("stroke-width", String(width));
      el.setAttribute("opacity", String(opacity));
      if (dash) el.setAttribute("stroke-dasharray", dash);
      return el;
    }}

    function circle(point, rWorld, fill, stroke, width, opacity=1.0) {{
      const [cx, cy, scale] = worldToCanvas(point);
      const el = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      el.setAttribute("cx", String(cx));
      el.setAttribute("cy", String(SCALE - cy));
      el.setAttribute("r", String(Math.max(4, rWorld * scale)));
      el.setAttribute("fill", fill);
      el.setAttribute("stroke", stroke);
      el.setAttribute("stroke-width", String(width));
      el.setAttribute("opacity", String(opacity));
      return el;
    }}

    function polygon(points, fill, stroke, width, opacity=1.0) {{
      const el = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      const pts = points.map((p) => {{
        const [x, y] = worldToCanvas(p);
        return `${{x}},${{SCALE - y}}`;
      }}).join(" ");
      el.setAttribute("points", pts);
      el.setAttribute("fill", fill);
      el.setAttribute("stroke", stroke);
      el.setAttribute("stroke-width", String(width));
      el.setAttribute("opacity", String(opacity));
      return el;
    }}

    function render() {{
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const fencePoints = (topology.fence_points || []).map((p) => [Number(p[0]), Number(p[1])]);
      const allPoints = fencePoints.concat(mobilePoints);

      (topology.triangles || []).forEach((tri) => {{
        const pts = tri.map((idx) => allPoints[idx]);
        svg.appendChild(polygon(pts, "#f4e6b2", "#d6b85a", 1, 0.35));
      }});

      (topology.uncovered_cycles || []).forEach((cycle) => {{
        const pts = cycle.map((idx) => allPoints[idx]);
        svg.appendChild(polygon(pts, "#fef3c7", "#f59e0b", 2, 0.45));
      }});

      (topology.edges || []).forEach(([i, j]) => {{
        svg.appendChild(line(allPoints[i], allPoints[j], "#2f4f72", 2.5, 0.7));
      }});

      fencePoints.forEach((p) => {{
        svg.appendChild(circle(p, currentRadius, "#e5e7eb", "#9ca3af", 2, 0.42));
      }});
      mobilePoints.forEach((p) => {{
        svg.appendChild(circle(p, currentRadius, "#93c5fd", "#4f7cff", 2, 0.28));
      }});

      fencePoints.forEach((p) => {{
        const [x, y] = worldToCanvas(p);
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", String(x - 7));
        rect.setAttribute("y", String(SCALE - y - 7));
        rect.setAttribute("width", "14");
        rect.setAttribute("height", "14");
        rect.setAttribute("fill", "#111827");
        svg.appendChild(rect);
      }});

      mobilePoints.forEach((p, idx) => {{
        const node = circle(p, 0.012, "#1d4ed8", "#93c5fd", 2, 0.95);
        node.dataset.mobileIndex = String(idx);
        svg.appendChild(node);
      }});

      metaBox.innerHTML = `
        <div><strong>model:</strong> ${{BOOTSTRAP.model_label}}</div>
        <div><strong>radius:</strong> ${{currentRadius.toFixed(3)}} | <strong>dt:</strong> ${{BOOTSTRAP.dt.toFixed(3)}} | <strong>max steps:</strong> ${{BOOTSTRAP.max_steps}}</div>
        <div><strong>mobile sensors:</strong> ${{mobilePoints.length}} | <strong>fence sensors:</strong> ${{fencePoints.length}} | <strong>active uncovered cycles:</strong> ${{(topology.uncovered_cycles || []).length}}</div>
        <div><strong>estimated mobile covered fraction:</strong> ${{Number(topology.mobile_union_fraction || 0).toFixed(3)}} | <strong>raw mobile disk-area ratio:</strong> ${{Number(topology.mobile_area_ratio || 0).toFixed(3)}}</div>
        <div><strong>total raw disk-area ratio:</strong> ${{Number(topology.total_area_ratio || 0).toFixed(3)}}</div>
        <div><strong>triangular coverage proxy:</strong> κ = ${{Number(topology.triangular_kappa_total || 0).toFixed(3)}} (${{topology.triangular_cover_satisfied ? "satisfied" : "not satisfied"}}; total sensors includes fence)</div>
      `;
    }}

    svg.addEventListener("pointerdown", async (event) => {{
      if (running) return;
      const p = canvasToWorld(event);
      if (mode === "move") {{
        dragIndex = nearestMobile(p);
      }} else if (mode === "add") {{
        mobilePoints.push(p);
        await refreshTopology();
      }} else if (mode === "remove") {{
        const idx = nearestMobile(p);
        if (idx >= 0) {{
          mobilePoints.splice(idx, 1);
          await refreshTopology();
        }}
      }}
    }});

    svg.addEventListener("pointermove", (event) => {{
      if (running) return;
      if (mode !== "move" || dragIndex < 0) return;
      mobilePoints[dragIndex] = canvasToWorld(event);
      render();
    }});

    async function finishDrag() {{
      if (dragIndex >= 0) {{
        dragIndex = -1;
        await refreshTopology();
      }}
    }}

    svg.addEventListener("pointerup", finishDrag);
    svg.addEventListener("pointerleave", finishDrag);

    moveBtn.addEventListener("click", () => setMode("move"));
    addBtn.addEventListener("click", () => setMode("add"));
    removeBtn.addEventListener("click", () => setMode("remove"));

    applyRadiusBtn.addEventListener("click", async () => {{
      if (running) return;
      const next = Number(radiusInput.value);
      if (!Number.isFinite(next) || next <= 0) return;
      currentRadius = next;
      await refreshTopology();
    }});

    resetBtn.addEventListener("click", async () => {{
      if (running) return;
      mobilePoints = initialMobilePoints.map((p) => [p[0], p[1]]);
      currentRadius = Number(BOOTSTRAP.sensing_radius);
      radiusInput.value = currentRadius.toFixed(2);
      await refreshTopology();
      playbackStatus.textContent = "No run yet.";
    }});

    anotherBtn.addEventListener("click", () => {{
      playbackStatus.textContent = "Ready for another setup.";
      enterSetupMode();
    }});

    runBtn.addEventListener("click", async () => {{
      if (running) return;
      running = true;
      runBtn.disabled = true;
      statusBox.textContent = "Running simulation. This waits for the full run, then builds the replay UI.";
      try {{
        const response = await fetch("/run", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ mobile_points: mobilePoints, radius: currentRadius }}),
        }});
        const payload = await response.json();
        if (!response.ok) {{
          throw new Error(payload.error || "run failed");
        }}
        playbackStatus.textContent = payload.summary || "Playback ready.";
        playbackFrame.src = payload.url + "?t=" + Date.now();
        enterPlaybackMode();
      }} catch (err) {{
        statusBox.textContent = "Run failed: " + err.message;
      }} finally {{
        running = false;
        runBtn.disabled = false;
      }}
    }});

    render();
  </script>
</body>
</html>
"""

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload, status=200):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_html(self, content, status=200):
            data = content.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send_html(html)
                return
            if parsed.path.startswith("/artifacts/"):
                rel = parsed.path[len("/artifacts/") :].lstrip("/")
                file_path = output_root / rel
                if not file_path.exists() or not file_path.is_file():
                    self.send_error(404, "Artifact not found")
                    return
                content_type = "application/octet-stream"
                suffix = file_path.suffix.lower()
                if suffix == ".html":
                    content_type = "text/html; charset=utf-8"
                elif suffix == ".json":
                    content_type = "application/json; charset=utf-8"
                elif suffix == ".png":
                    content_type = "image/png"
                data = file_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(404)

        def do_POST(self):
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                payload = {}

            if parsed.path == "/topology":
                mobile_points = _safe_points(payload.get("mobile_points", initial_mobile_points))
                current_radius = float(payload.get("radius", radius))
                self._send_json(_topology_data(mobile_points, current_radius))
                return

            if parsed.path == "/run":
                try:
                    mobile_points = _safe_points(payload.get("mobile_points", initial_mobile_points))
                    current_radius = float(payload.get("radius", radius))
                    velocities = _initial_velocities(len(mobile_points))
                    simulation = _build_simulation(mobile_points, velocities, current_radius)
                    simulation, builder, steps, clear_value = run_online_reeb_simulation(
                        simulation,
                        max_steps=max_steps,
                        clear_streak_needed=clear_streak,
                    )
                    print_atomic_change_report(simulation, builder.summaries, dt=dt)
                    artifact_dir = _build_timeline_for_simulation(
                        simulation=simulation,
                        builder=builder,
                        run_steps=steps,
                        mobile_points=mobile_points,
                        velocities=velocities,
                        current_radius=current_radius,
                    )
                    rel = artifact_dir.relative_to(output_root)
                    self._send_json(
                        {
                            "url": "/artifacts/" + rel.as_posix() + "/index.html",
                            "summary": f"Finished: steps={steps}, t={simulation.time:.4f}, clear_streak={clear_value}, radius={current_radius:.3f}",
                            "message": "Simulation complete. Playback ready below.",
                        }
                    )
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=500)
                return

            self.send_error(404)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://{server.server_address[0]}:{server.server_address[1]}/"
    if auto_open:
        webbrowser.open(url)
    print(f"{model_label} setup UI running at {url}")
    return url
