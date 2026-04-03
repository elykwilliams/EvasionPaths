from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.parse import urlsplit
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from plotting_tools import show_domain_boundary, show_state
from reeb_graph import (
    StepSummary,
    _compact_view_graph,
    _optimize_compact_layout,
    build_simulation,
    build_unitcube_simulation,
    draw_live_reeb_panel,
)


def _render_2d_frame(simulation, frame_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.clear()
    ax.axis("off")
    ax.axis("equal")
    ax.set_title(f"T = {simulation.time:6.3f}", loc="left")
    show_state(simulation, ax=ax)
    show_domain_boundary(
        simulation.sensor_network.domain,
        ax=ax,
        color="#111111",
        linewidth=2.0,
        linestyle="--",
        zorder=10,
    )
    fig.tight_layout()
    fig.savefig(frame_path, dpi=120)
    plt.close(fig)


def _collect_atomic_events_by_step(
    *,
    history: List[Tuple[dict, tuple, tuple, float]],
    summaries: Iterable[StepSummary],
) -> Dict[int, List[dict]]:
    step_to_time = {s.step: float(s.time) for s in summaries}
    if not step_to_time:
        return {}

    events_by_step: Dict[int, List[dict]] = {}
    for labels, alpha_change, boundary_change, time in history[1:]:
        if not any(alpha_change) and boundary_change == (0, 0):
            continue
        nearest_step = min(step_to_time, key=lambda s: abs(step_to_time[s] - float(time)))
        events_by_step.setdefault(nearest_step, []).append(
            {
                "time": float(time),
                "alpha_change": list(alpha_change),
                "boundary_change": list(boundary_change),
                "uncovered_cycles": max(0, sum(1 for v in labels.values() if bool(v)) - 1),
            }
        )

    for step_events in events_by_step.values():
        step_events.sort(key=lambda item: item["time"])
    return events_by_step


def _build_reeb_graph_plot_data(builder) -> dict:
    compact, event_nodes = _compact_view_graph(builder.graph)
    layout = _optimize_compact_layout(compact, event_nodes)

    nodes = []
    for node_id in event_nodes:
        x, y = layout[node_id]
        node_data = compact.nodes[node_id]
        nodes.append(
            {
                "id": int(node_id),
                "x": float(x),
                "y": float(y),
                "kind": str(node_data.get("kind", "cycle")),
                "label": bool(node_data.get("label", True)),
                "step": int(node_data.get("step", 0)),
                "time": float(node_data.get("time", 0.0)),
            }
        )

    edges = []
    for source, target, data in compact.edges(data=True):
        edges.append(
            {
                "source": int(source),
                "target": int(target),
                "event": str(data.get("event", "continue")),
                "label_flip": bool(data.get("label_flip", False)),
            }
        )

    x_values = [item["x"] for item in nodes]
    y_values = [item["y"] for item in nodes]
    return {
        "nodes": nodes,
        "edges": edges,
        "x_min": float(min(x_values)) if x_values else 0.0,
        "x_max": float(max(x_values)) if x_values else 1.0,
        "y_min": float(min(y_values)) if y_values else 0.0,
        "y_max": float(max(y_values)) if y_values else 1.0,
    }


def _open_via_local_viewer_server(base_dir: Path, *, index_name: str = "index.html") -> str:
    repo_root = Path(__file__).resolve().parents[1]
    server_script = repo_root / "experiments" / "serve_local_experiment_viewer.py"
    info_path = base_dir / ".viewer_server_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            url = str(info.get("url", "")).strip()
            if url:
                try:
                    with urlopen(url, timeout=0.4):
                        return url
                except Exception:
                    pass
        except Exception:
            pass

    subprocess.Popen(
        [
            sys.executable,
            str(server_script),
            "--root-dir",
            str(base_dir),
            "--info-file",
            str(info_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    deadline = time.time() + 5.0
    while time.time() < deadline:
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            url = str(info.get("url", "")).strip()
            if url:
                return url
        time.sleep(0.1)

    return (base_dir / index_name).resolve().as_uri()


def _write_timeline_html(output_html: Path, timeline_data: dict) -> None:
    payload = json.dumps(timeline_data)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EvasionPaths 2D Timeline</title>
  <style>
    :root {{
      --bg: #f2f4f7;
      --panel: #ffffff;
      --ink: #102a43;
      --accent: #0b7285;
      --muted: #486581;
      --border: #d9e2ec;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 10% 10%, #d9f0ff 0%, transparent 45%),
        radial-gradient(circle at 80% 0%, #d5f5e3 0%, transparent 40%),
        var(--bg);
      min-height: 100vh;
      padding: 20px;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(420px, 2fr) minmax(320px, 1fr);
      gap: 14px;
      max-width: 1400px;
      margin: 0 auto;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      box-shadow: 0 8px 18px rgba(16, 42, 67, 0.08);
      overflow: hidden;
    }}
    .titlebar {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      font-size: 14px;
      letter-spacing: 0.02em;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
    }}
    .viewer {{
      width: 100%;
      aspect-ratio: 9 / 7;
      object-fit: contain;
      display: block;
      background: #fff;
    }}
    .reeb-wrap {{
      border-top: 1px solid var(--border);
      padding: 10px 12px 12px;
      display: grid;
      gap: 8px;
      background: #fbfdff;
    }}
    .reeb-toolbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      font-size: 13px;
      color: var(--muted);
    }}
    .reeb-svg {{
      width: 100%;
      height: 230px;
      display: block;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fff;
    }}
    .term-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    .term-card {{
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fff;
      padding: 8px;
      min-height: 150px;
    }}
    .term-title {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .term-rows {{
      display: grid;
      gap: 4px;
      max-height: 220px;
      overflow: auto;
      padding-right: 2px;
    }}
    .term-row {{
      display: grid;
      grid-template-columns: 140px 1fr 64px;
      gap: 6px;
      align-items: center;
      font-size: 11px;
      color: #334e68;
    }}
    .term-bar-wrap {{
      height: 8px;
      border-radius: 4px;
      background: #e9eff5;
      overflow: hidden;
    }}
    .term-bar-fill {{
      height: 100%;
      border-radius: 4px;
      background: #0b7285;
    }}
    .term-bar-fill.neg {{
      background: #c92a2a;
    }}
    .controls {{
      display: grid;
      gap: 10px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      background: #fbfdff;
      position: sticky;
      top: 0;
      z-index: 5;
    }}
    .button-row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    button {{
      border: 1px solid var(--border);
      background: #fff;
      border-radius: 8px;
      padding: 8px 12px;
      cursor: pointer;
      color: var(--ink);
      font-weight: 600;
    }}
    button.primary {{
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }}
    button.secondary {{
      background: #f8fbff;
    }}
    button:disabled {{
      opacity: 0.55;
      cursor: default;
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: var(--accent);
    }}
    .meta {{
      display: grid;
      gap: 8px;
      padding: 12px 14px;
    }}
    .section {{
      border-top: 1px solid var(--border);
      padding-top: 10px;
      margin-top: 2px;
    }}
    .titlebar-actions {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    .status-chip {{
      font-size: 12px;
      color: var(--muted);
    }}
    .mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .events {{
      display: grid;
      gap: 8px;
      max-height: 52vh;
      overflow: auto;
      padding-right: 4px;
    }}
    .event {{
      background: #f8fbff;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      padding: 8px;
    }}
    @media (max-width: 960px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .events {{ max-height: 32vh; }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <div class="card">
      <div class="titlebar">
        <span>2D Simulation Timeline</span>
        <span id="stepLabel">Step 0</span>
      </div>
      <div class="controls">
        <div class="button-row">
          <button id="prevBtn">Prev</button>
          <button class="primary" id="playBtn">Play</button>
          <button id="pauseBtn">Pause</button>
          <button id="nextBtn">Next</button>
          <button id="replayBtn">Replay</button>
        </div>
        <input id="timelineSlider" type="range" min="0" max="0" step="1" value="0"/>
      </div>
      <img id="frameImage" class="viewer" alt="Simulation frame"/>
      <div class="reeb-wrap">
        <div class="reeb-toolbar">
          <span><strong>Reeb Graph (time window)</strong> <span id="reebWindowLabel"></span></span>
          <div class="button-row">
            <button id="zoomInBtn">Zoom In</button>
            <button id="zoomOutBtn">Zoom Out</button>
            <button id="zoomResetBtn">Reset</button>
          </div>
        </div>
        <svg id="reebSvg" class="reeb-svg" viewBox="0 0 1000 230" preserveAspectRatio="none"></svg>
        <div class="term-grid">
          <div class="term-card">
            <div class="term-title"><strong>Policy Terms (Raw)</strong></div>
            <div id="rawTerms" class="term-rows"></div>
          </div>
          <div class="term-card">
            <div class="term-title"><strong>Policy Terms (Weighted)</strong></div>
            <div id="weightedTerms" class="term-rows"></div>
          </div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="titlebar">
        <span>Atomic Change Window</span>
        <div class="titlebar-actions">
          <span id="exportStatus" class="status-chip"></span>
          <button id="exportBtn" class="secondary" type="button" hidden>Export Sim</button>
        </div>
      </div>
      <div class="meta">
        <div id="timeLine"></div>
        <div id="summaryLine"></div>
        <div class="section">
          <div><strong>Atomic events at this step</strong></div>
          <div id="events" class="events"></div>
        </div>
        <div class="section">
          <div><strong>Reeb debug</strong></div>
          <div id="debug" class="mono"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const TIMELINE = {payload};
    const ASSET_VERSION = Date.now();
    const frames = TIMELINE.frames || [];
    const PHASE_COLORS = {{
      simplify: "#f4a261",
      consolidate: "#2a9d8f",
      compress: "#457b9d",
    }};

    const frameImage = document.getElementById("frameImage");
    const slider = document.getElementById("timelineSlider");
    const stepLabel = document.getElementById("stepLabel");
    const timeLine = document.getElementById("timeLine");
    const summaryLine = document.getElementById("summaryLine");
    const eventsBox = document.getElementById("events");
    const debugBox = document.getElementById("debug");
    const reebSvg = document.getElementById("reebSvg");
    const reebWindowLabel = document.getElementById("reebWindowLabel");
    const rawTermsBox = document.getElementById("rawTerms");
    const weightedTermsBox = document.getElementById("weightedTerms");
    const exportBtn = document.getElementById("exportBtn");
    const exportStatus = document.getElementById("exportStatus");
    const reebData = TIMELINE.reeb_graph || {{ nodes: [], edges: [] }};
    const exportConfig = TIMELINE.export_config || null;

    let idx = 0;
    let playbackRun = 0;
    const frameSrcs = frames.map((item) => `${{item.image}}?v=${{ASSET_VERSION}}&step=${{item.step}}`);
    const framePreloads = frameSrcs.map((src) => {{
      const img = new Image();
      img.decoding = "async";
      img.src = src;
      return img;
    }});
    const activeTermNames = Array.isArray(TIMELINE.active_term_names) ? TIMELINE.active_term_names : null;
    const fullMin = Number(reebData.x_min ?? 0);
    const fullMax = Number(reebData.x_max ?? 1);
    const fullSpan = Math.max(1e-6, fullMax - fullMin);
    let windowMin = fullMin;
    let windowMax = fullMax;

    function stopPlayback() {{
      playbackRun += 1;
    }}

    function frameSrcAt(index) {{
      return frameSrcs[index] || "";
    }}

    function waitForFrame(index) {{
      const img = framePreloads[index];
      if (!img) {{
        return Promise.resolve();
      }}
      if (img.complete && img.naturalWidth > 0) {{
        return Promise.resolve();
      }}
      return new Promise((resolve) => {{
        const done = () => resolve();
        img.addEventListener("load", done, {{ once: true }});
        img.addEventListener("error", done, {{ once: true }});
      }});
    }}

    async function startPlayback({{ restart = false }} = {{}}) {{
      stopPlayback();
      const runId = playbackRun;

      if (restart) {{
        idx = 0;
        render();
      }}

      while (runId === playbackRun && idx < frames.length - 1) {{
        await waitForFrame(idx + 1);
        if (runId !== playbackRun) return;

        await new Promise((resolve) => setTimeout(resolve, TIMELINE.interval_ms || 130));
        if (runId !== playbackRun) return;

        idx += 1;
        render();
      }}
    }}

    function svgEl(tag, attrs = {{}}) {{
      const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, String(value)));
      return el;
    }}

    function setWindow(minX, maxX) {{
      const minBound = fullMin;
      const maxBound = fullMax;
      const spanFloor = Math.max(fullSpan / 250, 1e-5);
      let left = Math.max(minBound, minX);
      let right = Math.min(maxBound, maxX);
      if (right - left < spanFloor) {{
        const center = 0.5 * (left + right);
        left = Math.max(minBound, center - 0.5 * spanFloor);
        right = Math.min(maxBound, center + 0.5 * spanFloor);
      }}
      windowMin = left;
      windowMax = right;
    }}

    function zoomBy(factor) {{
      const center = frames[idx]?.time ?? (0.5 * (windowMin + windowMax));
      const span = Math.max(1e-8, windowMax - windowMin);
      const newSpan = span * factor;
      setWindow(center - 0.5 * newSpan, center + 0.5 * newSpan);
      render();
    }}

    function drawReeb(highlightTime, highlightPhase) {{
      while (reebSvg.firstChild) reebSvg.removeChild(reebSvg.firstChild);
      if (!reebData.nodes || !reebData.nodes.length) {{
        const text = svgEl("text", {{ x: 16, y: 26, fill: "#486581", "font-size": 14 }});
        text.textContent = "No Reeb graph data.";
        reebSvg.appendChild(text);
        return;
      }}

      const w = 1000;
      const h = 230;
      const pad = {{ l: 34, r: 14, t: 12, b: 16 }};
      const span = Math.max(1e-8, windowMax - windowMin);
      const xToPx = (x) => pad.l + ((x - windowMin) / span) * (w - pad.l - pad.r);
      const yMin = Number(reebData.y_min ?? 0);
      const yMax = Number(reebData.y_max ?? 1);
      const ySpan = Math.max(1e-8, yMax - yMin);
      const yToPx = (y) => h - pad.b - ((y - yMin) / ySpan) * (h - pad.t - pad.b);
      const isVisibleEdge = (x0, x1) => Math.max(x0, x1) >= windowMin && Math.min(x0, x1) <= windowMax;
      const phaseColor = PHASE_COLORS[highlightPhase] || "#ff6b6b";

      const halfWidth = Number(TIMELINE.highlight_half_width ?? 0.001);
      const bandLeft = xToPx(highlightTime - halfWidth);
      const bandRight = xToPx(highlightTime + halfWidth);
      reebSvg.appendChild(
        svgEl("rect", {{
          x: bandLeft,
          y: 0,
          width: Math.max(1, bandRight - bandLeft),
          height: h,
          fill: phaseColor,
          "fill-opacity": 0.17,
        }})
      );

      const nodeById = new Map((reebData.nodes || []).map((n) => [n.id, n]));
      (reebData.edges || []).forEach((edge) => {{
        const a = nodeById.get(edge.source);
        const b = nodeById.get(edge.target);
        if (!a || !b || !isVisibleEdge(a.x, b.x)) return;
        reebSvg.appendChild(
          svgEl("line", {{
            x1: xToPx(a.x),
            y1: yToPx(a.y),
            x2: xToPx(b.x),
            y2: yToPx(b.y),
            stroke: edge.label_flip ? "#000" : "#666",
            "stroke-width": edge.label_flip ? 1.8 : 1.2,
            "stroke-dasharray": edge.label_flip ? "5,4" : "",
          }})
        );
      }});

      (reebData.nodes || []).forEach((node) => {{
        if (node.x < windowMin || node.x > windowMax) return;
        let fill = node.label ? "#d94841" : "#2f9e44";
        if (node.kind === "termination") fill = "#c2c7cf";
        reebSvg.appendChild(
          svgEl("circle", {{
            cx: xToPx(node.x),
            cy: yToPx(node.y),
            r: 3.2,
            fill,
            stroke: "#1f2933",
            "stroke-width": 0.8,
          }})
        );
      }});

      reebSvg.appendChild(
        svgEl("line", {{
          x1: xToPx(highlightTime),
          y1: 0,
          x2: xToPx(highlightTime),
          y2: h,
          stroke: phaseColor,
          "stroke-width": 1.2,
        }})
      );

      reebWindowLabel.textContent = `[${{windowMin.toFixed(4)}}, ${{windowMax.toFixed(4)}}]`;
    }}

    function renderTermBars(container, termMap) {{
      container.innerHTML = "";
      if (!termMap || Object.keys(termMap).length === 0) {{
        const empty = document.createElement("div");
        empty.className = "event";
        empty.textContent = "No policy term data.";
        container.appendChild(empty);
        return;
      }}

      let entries = Object.entries(termMap);
      if (activeTermNames && activeTermNames.length) {{
        const allowed = new Set(activeTermNames);
        entries = entries.filter(([name]) => allowed.has(name));
      }}
      if (!entries.length) {{
        const empty = document.createElement("div");
        empty.className = "event";
        empty.textContent = "No active policy terms for this experiment.";
        container.appendChild(empty);
        return;
      }}
      const maxAbs = Math.max(1e-9, ...entries.map(([_, v]) => Math.abs(Number(v))));
      entries.forEach(([name, rawVal]) => {{
        const value = Number(rawVal);
        const widthPct = Math.max(2, Math.round((Math.abs(value) / maxAbs) * 100));
        const row = document.createElement("div");
        row.className = "term-row";

        const label = document.createElement("div");
        label.textContent = name;

        const wrap = document.createElement("div");
        wrap.className = "term-bar-wrap";
        const fill = document.createElement("div");
        fill.className = `term-bar-fill ${{value < 0 ? "neg" : ""}}`;
        fill.style.width = `${{widthPct}}%`;
        wrap.appendChild(fill);

        const val = document.createElement("div");
        val.style.textAlign = "right";
        val.style.fontFamily = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
        val.textContent = value.toFixed(4);

        row.appendChild(label);
        row.appendChild(wrap);
        row.appendChild(val);
        container.appendChild(row);
      }});
    }}

    function render() {{
      if (!frames.length) {{
        return;
      }}
      const item = frames[idx];
      const phase = item.phase || "simplify";
      frameImage.src = frameSrcAt(idx);
      stepLabel.textContent = `Step ${{item.step}} / ${{frames.length - 1}}`;
      slider.value = String(idx);
      timeLine.textContent = `time = ${{item.time.toFixed(6)}} | phase = ${{phase}}`;

      const s = item.summary || {{}};
      summaryLine.textContent =
        `cycles=${{s.n_cycles ?? 0}} true=${{s.n_true ?? 0}} false=${{s.n_false ?? 0}} births=${{s.n_birth ?? 0}} deaths=${{s.n_death ?? 0}}`;

      eventsBox.innerHTML = "";
      const events = item.atomic_events || [];
      if (!events.length) {{
        const empty = document.createElement("div");
        empty.className = "event";
        empty.textContent = "No atomic events at this highlighted step.";
        eventsBox.appendChild(empty);
      }} else {{
        events.forEach((ev) => {{
          const div = document.createElement("div");
          div.className = "event";
          div.innerHTML =
            `<div><strong>t=${{ev.time.toFixed(6)}}</strong></div>` +
            `<div>alpha_change=${{JSON.stringify(ev.alpha_change)}}</div>` +
            `<div>boundary_change=${{JSON.stringify(ev.boundary_change)}}</div>` +
            `<div>uncovered_cycles=${{ev.uncovered_cycles}}</div>`;
          eventsBox.appendChild(div);
        }});
      }}

      debugBox.textContent = JSON.stringify(item.snapshot_debug || {{}}, null, 2);
      drawReeb(item.time, phase);
      const terms = item.policy_terms || {{}};
      renderTermBars(rawTermsBox, terms.raw || null);
      renderTermBars(weightedTermsBox, terms.weighted || null);
    }}

    async function exportCurrentSimulation() {{
      if (!exportConfig) {{
        exportStatus.textContent = "Export unavailable";
        return;
      }}
      exportBtn.disabled = true;
      exportStatus.textContent = "Exporting...";
      try {{
        const response = await fetch("/__export_sim", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(exportConfig),
        }});
        const payload = await response.json();
        if (!response.ok || !payload.ok) {{
          throw new Error(payload.error || `Export failed (${{response.status}})`);
        }}
        exportStatus.textContent = "Exported to web";
      }} catch (error) {{
        exportStatus.textContent = "Export failed";
        window.alert(String(error));
      }} finally {{
        exportBtn.disabled = false;
      }}
    }}

    document.getElementById("prevBtn").addEventListener("click", () => {{
      stopPlayback();
      idx = Math.max(0, idx - 1);
      render();
    }});

    document.getElementById("nextBtn").addEventListener("click", () => {{
      stopPlayback();
      idx = Math.min(frames.length - 1, idx + 1);
      render();
    }});

    document.getElementById("playBtn").addEventListener("click", () => {{
      void startPlayback();
    }});

    document.getElementById("replayBtn").addEventListener("click", () => {{
      void startPlayback({{ restart: true }});
    }});

    document.getElementById("pauseBtn").addEventListener("click", () => {{
      stopPlayback();
    }});

    slider.addEventListener("input", (event) => {{
      stopPlayback();
      idx = Number(event.target.value);
      render();
    }});

    document.getElementById("zoomInBtn").addEventListener("click", () => {{
      zoomBy(0.5);
    }});
    document.getElementById("zoomOutBtn").addEventListener("click", () => {{
      zoomBy(2.0);
    }});
    document.getElementById("zoomResetBtn").addEventListener("click", () => {{
      setWindow(fullMin, fullMax);
      render();
    }});

    const isLocalViewerHost = ["127.0.0.1", "localhost"].includes(window.location.hostname);
    if (exportConfig && window.location.protocol.startsWith("http") && isLocalViewerHost) {{
      exportBtn.hidden = false;
      exportBtn.addEventListener("click", exportCurrentSimulation);
    }} else if (exportConfig && window.location.protocol === "file:") {{
      exportBtn.hidden = false;
      exportBtn.disabled = true;
      exportStatus.textContent = "Open via local viewer to export";
    }}

    slider.max = String(Math.max(0, frames.length - 1));
    setWindow(fullMin, fullMax);
    render();
  </script>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")


def build_and_open_2d_timeline_ui(
    *,
    simulation,
    builder,
    run_steps: int,
    num_sensors: int,
    sensing_radius: float,
    timestep_size: float,
    sensor_velocity: float,
    seed: int,
    interval_ms: int = 130,
    output_dir: str | None = None,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(output_dir) if output_dir else Path.cwd() / f"ui_timeline_{stamp}"
    frames_dir = base_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    replay = build_simulation(
        num_sensors=num_sensors,
        sensing_radius=sensing_radius,
        timestep_size=timestep_size,
        sensor_velocity=sensor_velocity,
        seed=seed,
    )

    frame_records: List[dict] = []
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
        "interval_ms": int(interval_ms),
        "highlight_half_width": float(max(0.001, 0.48 * timestep_size)),
        "reeb_graph": _build_reeb_graph_plot_data(builder),
        "frames": frame_records,
    }
    output_html = base_dir / "index.html"
    (base_dir / "timeline.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    _write_timeline_html(output_html, timeline)

    webbrowser.open(_open_via_local_viewer_server(base_dir))
    return output_html


def _cycle_hex_color(cycle) -> str:
    digest = hashlib.md5(str(cycle).encode("utf-8")).hexdigest()
    # Constrain to a mid-range so colors are visible on white background.
    r = 80 + (int(digest[0:2], 16) % 140)
    g = 80 + (int(digest[2:4], 16) % 140)
    b = 80 + (int(digest[4:6], 16) % 140)
    return f"#{r:02x}{g:02x}{b:02x}"


def _cycle_triangles_3d(cycle) -> List[Tuple[int, int, int]]:
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


def _collect_3d_frame_data(simulation) -> dict:
    points = np.asarray(simulation.sensor_network.points, dtype=float)
    fence_points = np.asarray([s.pos for s in simulation.sensor_network.fence_sensors], dtype=float)
    alpha_edge_positions: List[float] = []

    for edge in simulation.topology.simplices(1):
        nodes = tuple(edge.nodes) if hasattr(edge, "nodes") else tuple(edge)
        if len(nodes) != 2:
            continue
        p0 = points[int(nodes[0])]
        p1 = points[int(nodes[1])]
        alpha_edge_positions.extend(p0.tolist())
        alpha_edge_positions.extend(p1.tolist())

    cycles = []
    for cycle, label in simulation.cycle_label.label.items():
        if simulation.topology.is_excluded_cycle(cycle) or not bool(label):
            continue
        tri_nodes = _cycle_triangles_3d(cycle)
        if not tri_nodes:
            continue
        positions: List[float] = []
        for tri in tri_nodes:
            for node_idx in tri:
                positions.extend(points[int(node_idx)].tolist())
        cycles.append(
            {
                "key_hash": hashlib.md5(str(cycle).encode("utf-8")).hexdigest()[:10],
                "color": _cycle_hex_color(cycle),
                "positions": positions,
            }
        )

    return {
        "mobile_points": points.tolist(),
        "fence_points": fence_points.tolist() if fence_points.size > 0 else [],
        "alpha_edge_positions": alpha_edge_positions,
        "cycles": cycles,
    }


def collect_3d_scene_snapshot(simulation) -> dict:
    """Public wrapper for 3D scene snapshot extraction."""
    return _collect_3d_frame_data(simulation)


def _write_timeline_3d_html(output_html: Path, timeline_data: dict) -> None:
    payload = json.dumps(timeline_data)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EvasionPaths 3D Timeline</title>
  <style>
    :root {{
      --bg: #eff3f7;
      --panel: #ffffff;
      --ink: #1a2a3a;
      --accent: #1b6ca8;
      --muted: #4f6375;
      --border: #d9e2ec;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 15% 0%, #d9f4ff 0%, transparent 40%),
        radial-gradient(circle at 90% 10%, #e8f9e8 0%, transparent 35%),
        var(--bg);
      min-height: 100vh;
      padding: 20px;
    }}
    .layout {{
      max-width: 1500px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: minmax(540px, 2fr) minmax(320px, 1fr);
      gap: 14px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      box-shadow: 0 10px 20px rgba(26, 42, 58, 0.08);
      overflow: hidden;
    }}
    .titlebar {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
    }}
    #scene {{
      width: 100%;
      height: min(72vh, 760px);
      background: #f8fbff;
      display: block;
    }}
    .controls {{
      display: grid;
      gap: 10px;
      padding: 12px 14px 16px;
      border-top: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 5;
      background: #fbfdff;
    }}
    .button-row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    button {{
      border: 1px solid var(--border);
      background: #fff;
      border-radius: 8px;
      padding: 8px 12px;
      cursor: pointer;
      color: var(--ink);
      font-weight: 600;
    }}
    button.primary {{
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: var(--accent);
    }}
    .meta {{
      display: grid;
      gap: 8px;
      padding: 12px 14px;
    }}
    .section {{
      border-top: 1px solid var(--border);
      padding-top: 10px;
      margin-top: 2px;
    }}
    .events {{
      display: grid;
      gap: 8px;
      max-height: 42vh;
      overflow: auto;
      padding-right: 4px;
    }}
    .event {{
      background: #f8fbff;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px;
      font-size: 13px;
    }}
    .mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    @media (max-width: 980px) {{
      .layout {{ grid-template-columns: 1fr; }}
      #scene {{ height: 56vh; }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <div class="card">
      <div class="titlebar">
        <span>3D Simulation Timeline</span>
        <span id="stepLabel">Step 0</span>
      </div>
      <canvas id="scene"></canvas>
      <div class="controls">
        <div class="button-row">
          <button id="prevBtn">Prev</button>
          <button class="primary" id="playBtn">Play</button>
          <button id="pauseBtn">Pause</button>
          <button id="nextBtn">Next</button>
        </div>
        <input id="timelineSlider" type="range" min="0" max="0" step="1" value="0"/>
      </div>
    </div>

    <div class="card">
      <div class="titlebar">
        <span>Atomic Change Window</span>
      </div>
      <div class="meta">
        <div id="timeLine"></div>
        <div id="summaryLine"></div>
        <div class="section">
          <div><strong>Atomic events at this step</strong></div>
          <div id="events" class="events"></div>
        </div>
        <div class="section">
          <div><strong>Reeb debug</strong></div>
          <div id="debug" class="mono"></div>
        </div>
      </div>
    </div>
  </div>

  <script type="module">
    async function loadThreeModules() {{
      const sources = [
        {{
          three: "https://unpkg.com/three@0.160.0/build/three.module.js",
          controls: "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js",
        }},
        {{
          three: "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
          controls: "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js",
        }},
      ];
      let lastErr = null;
      for (const src of sources) {{
        try {{
          const threeMod = await import(src.three);
          const controlsMod = await import(src.controls);
          return {{ THREE: threeMod, OrbitControls: controlsMod.OrbitControls }};
        }} catch (err) {{
          lastErr = err;
        }}
      }}
      throw lastErr || new Error("Unable to load three.js modules.");
    }}

    const {{ THREE, OrbitControls }} = await loadThreeModules();

    const TIMELINE = {payload};
    const frames = TIMELINE.frames || [];

    const canvas = document.getElementById("scene");
    const stepLabel = document.getElementById("stepLabel");
    const slider = document.getElementById("timelineSlider");
    const timeLine = document.getElementById("timeLine");
    const summaryLine = document.getElementById("summaryLine");
    const eventsBox = document.getElementById("events");
    const debugBox = document.getElementById("debug");

    const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    renderer.setClearColor(0xf8fbff, 1);

    const scene = new THREE.Scene();
    scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const dirA = new THREE.DirectionalLight(0xffffff, 0.65);
    dirA.position.set(3, 2, 3);
    scene.add(dirA);
    const dirB = new THREE.DirectionalLight(0xffffff, 0.25);
    dirB.position.set(-2, -1, 2);
    scene.add(dirB);

    const camera = new THREE.PerspectiveCamera(50, canvas.clientWidth / canvas.clientHeight, 0.01, 100);
    camera.position.set(1.9, 1.6, 1.9);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0.5, 0.5, 0.5);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    const cubeEdges = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
      new THREE.LineBasicMaterial({{ color: 0x63788c }})
    );
    cubeEdges.position.set(0.5, 0.5, 0.5);
    scene.add(cubeEdges);

    const dynamicRoot = new THREE.Group();
    scene.add(dynamicRoot);

    function clearDynamic() {{
      while (dynamicRoot.children.length) {{
        const obj = dynamicRoot.children.pop();
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {{
          if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
          else obj.material.dispose();
        }}
      }}
    }}

    function makePointCloud(points, color, size) {{
      if (!points || !points.length) return null;
      const flat = points.flat();
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(flat, 3));
      const material = new THREE.PointsMaterial({{ color, size, sizeAttenuation: true }});
      return new THREE.Points(geometry, material);
    }}

    function renderFrame(index) {{
      const item = frames[index];
      if (!item) return;
      clearDynamic();

      const mobile = makePointCloud(item.scene.mobile_points || [], 0xd9480f, 0.028);
      if (mobile) dynamicRoot.add(mobile);
      const fence = makePointCloud(item.scene.fence_points || [], 0x111111, 0.016);
      if (fence) dynamicRoot.add(fence);
      if (item.scene.alpha_edge_positions && item.scene.alpha_edge_positions.length) {{
        const edgeGeom = new THREE.BufferGeometry();
        edgeGeom.setAttribute('position', new THREE.Float32BufferAttribute(item.scene.alpha_edge_positions, 3));
        dynamicRoot.add(new THREE.LineSegments(edgeGeom, new THREE.LineBasicMaterial({{ color: 0x6f7f8f, transparent: true, opacity: 0.55 }})));
      }}

      (item.scene.cycles || []).forEach((cycle) => {{
        if (!cycle.positions || !cycle.positions.length) return;
        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', new THREE.Float32BufferAttribute(cycle.positions, 3));
        geom.computeVertexNormals();
        const mat = new THREE.MeshStandardMaterial({{
          color: cycle.color,
          transparent: true,
          opacity: 0.36,
          side: THREE.DoubleSide
        }});
        const mesh = new THREE.Mesh(geom, mat);
        dynamicRoot.add(mesh);
      }});

      stepLabel.textContent = `Step ${{item.step}} / ${{frames.length - 1}}`;
      slider.value = String(index);
      timeLine.textContent = `time = ${{item.time.toFixed(6)}}`;
      const s = item.summary || {{}};
      summaryLine.textContent =
        `cycles=${{s.n_cycles ?? 0}} true=${{s.n_true ?? 0}} false=${{s.n_false ?? 0}} births=${{s.n_birth ?? 0}} deaths=${{s.n_death ?? 0}}`;

      eventsBox.innerHTML = "";
      const events = item.atomic_events || [];
      if (!events.length) {{
        const empty = document.createElement("div");
        empty.className = "event";
        empty.textContent = "No atomic events at this highlighted step.";
        eventsBox.appendChild(empty);
      }} else {{
        events.forEach((ev) => {{
          const div = document.createElement("div");
          div.className = "event";
          div.innerHTML =
            `<div><strong>t=${{ev.time.toFixed(6)}}</strong></div>` +
            `<div>alpha_change=${{JSON.stringify(ev.alpha_change)}}</div>` +
            `<div>boundary_change=${{JSON.stringify(ev.boundary_change)}}</div>` +
            `<div>uncovered_cycles=${{ev.uncovered_cycles}}</div>`;
          eventsBox.appendChild(div);
        }});
      }}
      debugBox.textContent = JSON.stringify(item.snapshot_debug || {{}}, null, 2);
    }}

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}
    animate();

    function onResize() {{
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h, false);
    }}
    window.addEventListener("resize", onResize);

    let idx = 0;
    let timer = null;

    function stopPlayback() {{
      if (timer) {{
        clearInterval(timer);
        timer = null;
      }}
    }}

    document.getElementById("prevBtn").addEventListener("click", () => {{
      stopPlayback();
      idx = Math.max(0, idx - 1);
      renderFrame(idx);
    }});

    document.getElementById("nextBtn").addEventListener("click", () => {{
      stopPlayback();
      idx = Math.min(frames.length - 1, idx + 1);
      renderFrame(idx);
    }});

    document.getElementById("playBtn").addEventListener("click", () => {{
      stopPlayback();
      timer = setInterval(() => {{
        if (idx >= frames.length - 1) {{
          stopPlayback();
          return;
        }}
        idx += 1;
        renderFrame(idx);
      }}, TIMELINE.interval_ms || 120);
    }});

    document.getElementById("pauseBtn").addEventListener("click", () => {{
      stopPlayback();
    }});

    slider.addEventListener("input", (event) => {{
      stopPlayback();
      idx = Number(event.target.value);
      renderFrame(idx);
    }});

    slider.max = String(Math.max(0, frames.length - 1));
    renderFrame(0);
  </script>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")


def _build_3d_frame_record(
    *,
    simulation,
    builder,
    step: int,
    atomic_events: List[dict] | None = None,
) -> dict:
    summary = builder.summaries[-1] if builder.summaries else None
    return {
        "step": int(step),
        "time": float(simulation.time),
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
        "atomic_events": atomic_events or [],
        "snapshot_debug": builder.get_snapshot_debug(step),
        "scene": _collect_3d_frame_data(simulation),
    }


def build_and_open_3d_timeline_ui(
    *,
    simulation,
    builder,
    run_steps: int,
    num_sensors: int,
    sensing_radius: float,
    timestep_size: float,
    sensor_velocity: float,
    seed: int,
    end_time: float = 0.0,
    interval_ms: int = 120,
    output_dir: str | None = None,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(output_dir) if output_dir else Path.cwd() / f"ui_timeline_3d_{stamp}"
    base_dir.mkdir(parents=True, exist_ok=True)

    replay = build_unitcube_simulation(
        num_sensors=num_sensors,
        sensing_radius=sensing_radius,
        timestep_size=timestep_size,
        sensor_velocity=sensor_velocity,
        seed=seed,
        end_time=end_time,
    )

    frame_records: List[dict] = []
    atomic_events_by_step = _collect_atomic_events_by_step(
        history=simulation.cycle_label.history,
        summaries=builder.summaries,
    )

    for step in range(run_steps + 1):
        if step > 0:
            replay.do_timestep()
        frame_records.append(
            _build_3d_frame_record(
                simulation=replay,
                builder=builder,
                step=step,
                atomic_events=atomic_events_by_step.get(step, []),
            )
        )

    timeline = {
        "interval_ms": int(interval_ms),
        "frames": frame_records,
    }

    output_html = base_dir / "index.html"
    (base_dir / "timeline.json").write_text(json.dumps(timeline), encoding="utf-8")
    _write_timeline_3d_html(output_html, timeline)

    webbrowser.open(_open_via_local_viewer_server(base_dir))
    return output_html


def open_3d_matplotlib_explorer(
    *,
    simulation,
    builder,
    run_steps: int,
    num_sensors: int,
    sensing_radius: float,
    timestep_size: float,
    sensor_velocity: float,
    seed: int,
    end_time: float = 0.0,
    interval_ms: int = 140,
) -> None:
    from matplotlib.widgets import Button, Slider

    replay = build_unitcube_simulation(
        num_sensors=num_sensors,
        sensing_radius=sensing_radius,
        timestep_size=timestep_size,
        sensor_velocity=sensor_velocity,
        seed=seed,
        end_time=end_time,
    )
    step_to_summary = {s.step: s for s in builder.summaries}
    atomic_events_by_step = _collect_atomic_events_by_step(
        history=simulation.cycle_label.history,
        summaries=builder.summaries,
    )

    frames: List[dict] = []
    for step in range(run_steps + 1):
        if step > 0:
            replay.do_timestep()
        summary = step_to_summary.get(step)
        frames.append(
            {
                "step": step,
                "time": float(replay.time),
                "summary": {
                    "n_cycles": int(summary.n_cycles) if summary else 0,
                    "n_true": int(summary.n_true) if summary else 0,
                    "n_false": int(summary.n_false) if summary else 0,
                    "n_birth": int(summary.n_birth) if summary else 0,
                    "n_death": int(summary.n_death) if summary else 0,
                },
                "atomic_events": atomic_events_by_step.get(step, []),
                "scene": _collect_3d_frame_data(replay),
            }
        )

    fig = plt.figure(figsize=(13, 10))
    grid = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.30, bottom=0.14)
    ax_3d = fig.add_subplot(grid[0], projection="3d")
    ax_reeb = fig.add_subplot(grid[1])

    info_text = fig.text(
        0.67,
        0.02,
        "",
        fontsize=9,
        va="bottom",
        ha="left",
        family="monospace",
        bbox={"facecolor": "white", "edgecolor": "0.85", "boxstyle": "round,pad=0.4"},
    )

    max_idx = max(0, len(frames) - 1)
    slider_ax = fig.add_axes([0.19, 0.07, 0.43, 0.03])
    step_slider = Slider(slider_ax, "Step", 0, max_idx, valinit=0, valstep=1)
    prev_ax = fig.add_axes([0.03, 0.062, 0.08, 0.05])
    play_ax = fig.add_axes([0.115, 0.062, 0.08, 0.05])
    pause_ax = fig.add_axes([0.625, 0.062, 0.08, 0.05])
    next_ax = fig.add_axes([0.71, 0.062, 0.08, 0.05])
    prev_btn = Button(prev_ax, "Prev")
    play_btn = Button(play_ax, "Play")
    pause_btn = Button(pause_ax, "Pause")
    next_btn = Button(next_ax, "Next")

    state = {"idx": 0, "playing": False}
    timer = fig.canvas.new_timer(interval=interval_ms)

    def render(index: int) -> None:
        if not frames:
            return
        index = max(0, min(index, max_idx))
        state["idx"] = index
        item = frames[index]
        scene = item["scene"]
        summary = item["summary"]

        elev = getattr(ax_3d, "elev", 22.0)
        azim = getattr(ax_3d, "azim", -55.0)

        ax_3d.cla()
        mobile = np.asarray(scene["mobile_points"], dtype=float)
        if mobile.size:
            ax_3d.scatter(mobile[:, 0], mobile[:, 1], mobile[:, 2], c="tab:red", s=18, depthshade=True)

        fence = np.asarray(scene["fence_points"], dtype=float)
        if fence.size:
            ax_3d.scatter(fence[:, 0], fence[:, 1], fence[:, 2], c="black", s=8, alpha=0.35, depthshade=False)

        edges = np.asarray(scene["alpha_edge_positions"], dtype=float)
        if edges.size:
            edge_arr = edges.reshape(-1, 2, 3)
            for segment in edge_arr:
                ax_3d.plot(
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
            surface = Poly3DCollection(
                tri_faces,
                facecolors=cycle["color"],
                edgecolors=(0.2, 0.2, 0.2, 0.45),
                linewidths=0.4,
                alpha=0.36,
            )
            ax_3d.add_collection3d(surface)

        ax_3d.set_xlim(-0.12, 1.12)
        ax_3d.set_ylim(-0.12, 1.12)
        ax_3d.set_zlim(-0.12, 1.12)
        ax_3d.set_box_aspect((1, 1, 1))
        ax_3d.view_init(elev=elev, azim=azim)
        ax_3d.set_xlabel("x")
        ax_3d.set_ylabel("y")
        ax_3d.set_zlabel("z")
        ax_3d.set_title(
            f"3D Sensor/Alpha View | step={item['step']} t={item['time']:.6f} "
            f"cycles={summary['n_cycles']} true={summary['n_true']} false={summary['n_false']}",
            loc="left",
        )

        draw_live_reeb_panel(
            ax_reeb,
            builder.graph,
            highlight_time=item["time"],
            highlight_half_width=max(0.001, 0.48 * timestep_size),
            title="Reeb Graph",
        )

        events = item["atomic_events"]
        if not events:
            info_text.set_text("Atomic events: none at this step")
        else:
            lines = ["Atomic events:"]
            for event in events[:6]:
                lines.append(
                    f"t={event['time']:.6f} alpha={event['alpha_change']} boundary={event['boundary_change']}"
                )
            if len(events) > 6:
                lines.append(f"... +{len(events) - 6} more")
            info_text.set_text("\n".join(lines))

        fig.canvas.draw_idle()

    def on_slider(_val) -> None:
        render(int(step_slider.val))

    def on_prev(_event) -> None:
        state["playing"] = False
        step_slider.set_val(max(0, int(step_slider.val) - 1))

    def on_next(_event) -> None:
        state["playing"] = False
        step_slider.set_val(min(max_idx, int(step_slider.val) + 1))

    def on_play(_event) -> None:
        state["playing"] = True
        timer.start()

    def on_pause(_event) -> None:
        state["playing"] = False
        timer.stop()

    def on_timer() -> None:
        if not state["playing"]:
            return
        current = int(step_slider.val)
        if current >= max_idx:
            state["playing"] = False
            timer.stop()
            return
        step_slider.set_val(current + 1)

    step_slider.on_changed(on_slider)
    prev_btn.on_clicked(on_prev)
    next_btn.on_clicked(on_next)
    play_btn.on_clicked(on_play)
    pause_btn.on_clicked(on_pause)
    timer.add_callback(on_timer)

    render(0)
    plt.show()


def _write_live_timeline_3d_html(output_html: Path, *, poll_ms: int = 1200) -> None:
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EvasionPaths 3D Live Timeline (Reeb Only)</title>
  <style>
    :root {{
      --bg: #eff3f7; --panel: #ffffff; --ink: #1a2a3a; --accent: #1b6ca8; --muted: #4f6375; --border: #d9e2ec;
      --flip: #e03131;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 20px; min-height: 100vh; color: var(--ink);
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 15% 0%, #d9f4ff 0%, transparent 40%),
        radial-gradient(circle at 90% 10%, #e8f9e8 0%, transparent 35%),
        var(--bg);
    }}
    .layout {{
      max-width: 1600px; margin: 0 auto; display: grid;
      grid-template-columns: minmax(700px, 2.3fr) minmax(320px, 1fr); gap: 14px;
    }}
    .card {{
      background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
      box-shadow: 0 10px 20px rgba(26, 42, 58, 0.08); overflow: hidden;
    }}
    .titlebar {{
      padding: 12px 14px; border-bottom: 1px solid var(--border); color: var(--muted);
      display: flex; justify-content: space-between; align-items: center; gap: 8px;
    }}
    .reeb-wrap {{ padding: 10px 12px 12px; display: grid; gap: 8px; background: #fbfdff; }}
    .reeb-svg {{ width: 100%; height: min(68vh, 760px); display: block; border: 1px solid var(--border); border-radius: 8px; background: #fff; }}
    .controls {{ display: grid; gap: 10px; padding: 12px 14px 16px; border-top: 1px solid var(--border); }}
    .button-row {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    button {{ border: 1px solid var(--border); background: #fff; border-radius: 8px; padding: 8px 12px; cursor: pointer; color: var(--ink); font-weight: 600; }}
    button.primary {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
    input[type="range"] {{ width: 100%; accent-color: var(--accent); }}
    .atomic-row {{
      display: grid;
      grid-template-columns: auto 1fr auto;
      align-items: center;
      gap: 10px;
      font-size: 13px;
      color: var(--muted);
    }}
    .atomic-toggle {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      user-select: none;
    }}
    .meta {{ display: grid; gap: 8px; padding: 12px 14px; }}
    .section {{ border-top: 1px solid var(--border); padding-top: 10px; margin-top: 2px; }}
    .events {{ display: grid; gap: 8px; max-height: 44vh; overflow: auto; padding-right: 4px; }}
    .event {{ background: #f8fbff; border: 1px solid var(--border); border-radius: 8px; padding: 8px; font-size: 13px; }}
    .event.flip {{ border-color: var(--flip); border-width: 2px; background: #fff4f4; }}
    .flip-note {{ color: var(--flip); font-weight: 700; font-size: 13px; min-height: 18px; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; white-space: pre-wrap; word-break: break-word; }}
    @media (max-width: 980px) {{ .layout {{ grid-template-columns: 1fr; }} .reeb-svg {{ height: 52vh; }} }}
  </style>
</head>
<body>
  <div class="layout">
    <div class="card">
      <div class="titlebar">
        <span>3D Live Timeline (Reeb Only)</span>
        <span id="stepLabel">Step 0</span>
      </div>
      <div class="reeb-wrap">
        <div><strong>Reeb Graph</strong> <span id="reebWindowLabel"></span></div>
        <svg id="reebSvg" class="reeb-svg" viewBox="0 0 1200 760" preserveAspectRatio="none"></svg>
      </div>
      <div class="controls">
        <div class="button-row">
          <button id="prevBtn">Prev</button>
          <button class="primary" id="playBtn">Play</button>
          <button id="pauseBtn">Pause</button>
          <button id="nextBtn">Next</button>
          <button id="latestBtn">Latest</button>
        </div>
        <input id="timelineSlider" type="range" min="0" max="0" step="1" value="0"/>
        <div class="atomic-row">
          <label class="atomic-toggle">
            <input id="atomicModeToggle" type="checkbox" />
            <span>Atomic Scrub</span>
          </label>
          <input id="atomicSlider" type="range" min="0" max="0" step="1" value="0" disabled />
          <span id="atomicLabel">event -/-</span>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="titlebar">
        <span>Atomic Change Window</span>
        <span id="statusLabel">loading...</span>
      </div>
      <div class="meta">
        <div id="timeLine"></div>
        <div id="summaryLine"></div>
        <div id="flipNote" class="flip-note"></div>
        <div class="section">
          <div><strong>Atomic events at this step</strong></div>
          <div id="events" class="events"></div>
        </div>
        <div class="section">
          <div><strong>Reeb debug</strong></div>
          <div id="debug" class="mono"></div>
        </div>
      </div>
    </div>
  </div>
  <script>
    const POLL_MS = {poll_ms};
    let TIMELINE = {{ interval_ms: 120, frames: [], reeb_graph: {{ nodes: [], edges: [] }}, finished: false }};
    let idx = 0;
    let timer = null;
    let autoFollow = true;
    let flipSteps = new Set();
    let atomicMode = false;
    let atomicIndex = 0;

    const slider = document.getElementById("timelineSlider");
    const stepLabel = document.getElementById("stepLabel");
    const timeLine = document.getElementById("timeLine");
    const summaryLine = document.getElementById("summaryLine");
    const statusLabel = document.getElementById("statusLabel");
    const eventsBox = document.getElementById("events");
    const debugBox = document.getElementById("debug");
    const reebSvg = document.getElementById("reebSvg");
    const reebWindowLabel = document.getElementById("reebWindowLabel");
    const flipNote = document.getElementById("flipNote");
    const atomicModeToggle = document.getElementById("atomicModeToggle");
    const atomicSlider = document.getElementById("atomicSlider");
    const atomicLabel = document.getElementById("atomicLabel");

    function stopPlayback() {{ if (timer) {{ clearInterval(timer); timer = null; }} }}
    function svgEl(tag, attrs = {{}}) {{
      const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, String(v)));
      return el;
    }}

    function computeFlipSteps() {{
      const graph = TIMELINE.reeb_graph || {{ nodes: [], edges: [] }};
      const byId = new Map((graph.nodes || []).map((n) => [n.id, n]));
      const steps = new Set();
      (graph.edges || []).forEach((e) => {{
        if (!e.label_flip) return;
        const t = byId.get(e.target);
        if (t && Number.isFinite(t.step)) steps.add(Number(t.step));
      }});
      flipSteps = steps;
    }}

    function drawReeb(highlightTime) {{
      while (reebSvg.firstChild) reebSvg.removeChild(reebSvg.firstChild);
      const graph = TIMELINE.reeb_graph || {{ nodes: [], edges: [] }};
      if (!graph.nodes.length) return;
      const xMin = Number(graph.x_min ?? 0), xMax = Number(graph.x_max ?? 1), yMin = Number(graph.y_min ?? 0), yMax = Number(graph.y_max ?? 1);
      const w = 1200, h = 760, pad = {{ l: 36, r: 16, t: 12, b: 18 }};
      const xSpan = Math.max(1e-8, xMax - xMin), ySpan = Math.max(1e-8, yMax - yMin);
      const x = (v) => pad.l + ((v - xMin) / xSpan) * (w - pad.l - pad.r);
      const y = (v) => h - pad.b - ((v - yMin) / ySpan) * (h - pad.t - pad.b);
      const hw = Number(TIMELINE.highlight_half_width ?? 0.001);
      reebSvg.appendChild(svgEl("rect", {{ x: x(highlightTime - hw), y: 0, width: Math.max(1, x(highlightTime + hw) - x(highlightTime - hw)), height: h, fill: "#ff6b6b", "fill-opacity": 0.15 }}));
      const nodeById = new Map(graph.nodes.map((n) => [n.id, n]));
      graph.edges.forEach((e) => {{
        const a = nodeById.get(e.source), b = nodeById.get(e.target);
        if (!a || !b) return;
        reebSvg.appendChild(svgEl("line", {{
          x1: x(a.x), y1: y(a.y), x2: x(b.x), y2: y(b.y),
          stroke: e.label_flip ? "#e03131" : "#666",
          "stroke-width": e.label_flip ? 2.1 : 1.25,
          "stroke-dasharray": e.label_flip ? "6,4" : ""
        }}));
      }});
      graph.nodes.forEach((n) => {{
        let fill = n.label ? "#d94841" : "#2f9e44";
        if (n.kind === "termination") fill = "#c2c7cf";
        reebSvg.appendChild(svgEl("circle", {{ cx: x(n.x), cy: y(n.y), r: 3.2, fill, stroke: "#1f2933", "stroke-width": 0.8 }}));
      }});
      reebSvg.appendChild(svgEl("line", {{ x1: x(highlightTime), y1: 0, x2: x(highlightTime), y2: h, stroke: "#c92a2a", "stroke-width": 1.2 }}));
      reebWindowLabel.textContent = `[${{xMin.toFixed(4)}}, ${{xMax.toFixed(4)}}]`;
    }}

    function renderFrame() {{
      const frames = TIMELINE.frames || [];
      if (!frames.length) return;
      idx = Math.max(0, Math.min(idx, frames.length - 1));
      const item = frames[idx];

      stepLabel.textContent = `Step ${{item.step}} / ${{Math.max(0, frames.length - 1)}}`;
      slider.max = String(Math.max(0, frames.length - 1));
      slider.value = String(idx);

      const stepEvents = item.atomic_events || [];
      const hasAtomic = stepEvents.length > 0;
      if (atomicMode && hasAtomic) {{
        atomicSlider.disabled = false;
        atomicSlider.max = String(Math.max(0, stepEvents.length - 1));
        atomicIndex = Math.max(0, Math.min(atomicIndex, stepEvents.length - 1));
        atomicSlider.value = String(atomicIndex);
        atomicLabel.textContent = `event ${{atomicIndex + 1}}/${{stepEvents.length}}`;
      }} else {{
        atomicSlider.disabled = true;
        atomicSlider.max = "0";
        atomicSlider.value = "0";
        atomicLabel.textContent = hasAtomic ? `event 1/${{stepEvents.length}}` : "event -/-";
      }}

      const focusTime = (atomicMode && hasAtomic) ? Number(stepEvents[atomicIndex].time) : Number(item.time);
      timeLine.textContent = `time = ${{focusTime.toFixed(6)}}`;
      const s = item.summary || {{}};
      summaryLine.textContent = `cycles=${{s.n_cycles ?? 0}} true=${{s.n_true ?? 0}} false=${{s.n_false ?? 0}} births=${{s.n_birth ?? 0}} deaths=${{s.n_death ?? 0}}`;

      const hasFlip = flipSteps.has(Number(item.step));
      flipNote.textContent = hasFlip ? "Label flip at this step (dotted red edge in Reeb graph)" : "";

      eventsBox.innerHTML = "";
      const events = stepEvents;
      if (!events.length) {{
        const empty = document.createElement("div");
        empty.className = hasFlip ? "event flip" : "event";
        empty.textContent = "No atomic events at this highlighted step.";
        eventsBox.appendChild(empty);
      }} else {{
        events.forEach((ev, evIdx) => {{
          const div = document.createElement("div");
          div.className = hasFlip ? "event flip" : "event";
          if (atomicMode && evIdx === atomicIndex) {{
            div.style.outline = "2px solid #0b7285";
            div.style.outlineOffset = "1px";
          }}
          div.innerHTML = `<div><strong>t=${{ev.time.toFixed(6)}}</strong></div><div>alpha_change=${{JSON.stringify(ev.alpha_change)}}</div><div>boundary_change=${{JSON.stringify(ev.boundary_change)}}</div><div>uncovered_cycles=${{ev.uncovered_cycles}}</div>`;
          eventsBox.appendChild(div);
        }});
      }}

      debugBox.textContent = JSON.stringify(item.snapshot_debug || {{}}, null, 2);
      drawReeb(focusTime);
      statusLabel.textContent = TIMELINE.finished ? "finished" : "running";
    }}

    async function refreshTimeline() {{
      try {{
        const res = await fetch(`timeline.json?ts=${{Date.now()}}`, {{ cache: "no-store" }});
        if (!res.ok) return;
        const next = await res.json();
        const prevLen = (TIMELINE.frames || []).length;
        TIMELINE = next;
        computeFlipSteps();
        const newLen = (TIMELINE.frames || []).length;
        if (autoFollow && newLen > 0 && (idx >= prevLen - 1 || prevLen === 0)) idx = newLen - 1;
        renderFrame();
      }} catch (_err) {{}}
    }}

    document.getElementById("prevBtn").addEventListener("click", () => {{ stopPlayback(); autoFollow = false; idx -= 1; atomicIndex = 0; renderFrame(); }});
    document.getElementById("nextBtn").addEventListener("click", () => {{ stopPlayback(); autoFollow = false; idx += 1; atomicIndex = 0; renderFrame(); }});
    document.getElementById("latestBtn").addEventListener("click", () => {{ stopPlayback(); autoFollow = true; idx = Math.max(0, (TIMELINE.frames || []).length - 1); atomicIndex = 0; renderFrame(); }});
    document.getElementById("playBtn").addEventListener("click", () => {{
      stopPlayback(); autoFollow = false;
      timer = setInterval(() => {{
        if (idx >= (TIMELINE.frames || []).length - 1) return;
        idx += 1; atomicIndex = 0; renderFrame();
      }}, TIMELINE.interval_ms || 120);
    }});
    document.getElementById("pauseBtn").addEventListener("click", () => stopPlayback());
    slider.addEventListener("input", (e) => {{ stopPlayback(); autoFollow = false; idx = Number(e.target.value); atomicIndex = 0; renderFrame(); }});
    atomicModeToggle.addEventListener("change", () => {{ atomicMode = atomicModeToggle.checked; atomicIndex = 0; renderFrame(); }});
    atomicSlider.addEventListener("input", (e) => {{ if (!atomicMode) return; atomicIndex = Number(e.target.value); renderFrame(); }});

    setInterval(refreshTimeline, POLL_MS);
    refreshTimeline();
  </script>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")


class Live3DTimelineSession:
    def __init__(
        self,
        *,
        timestep_size: float,
        interval_ms: int = 120,
        output_dir: str | None = None,
        poll_ms: int = 1200,
    ) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(output_dir) if output_dir else Path.cwd() / f"ui_live_3d_{stamp}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.timeline_path = self.base_dir / "timeline.json"
        self.timeline = {
            "interval_ms": int(interval_ms),
            "highlight_half_width": float(max(0.001, 0.48 * timestep_size)),
            "reeb_graph": {"nodes": [], "edges": [], "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0},
            "frames": [],
            "finished": False,
        }
        _write_live_timeline_3d_html(self.base_dir / "index.html", poll_ms=poll_ms)
        self._write_timeline()

        class _LiveHandler(SimpleHTTPRequestHandler):
            def do_GET(self):
                req_path = urlsplit(self.path).path
                if req_path == "/favicon.ico":
                    self.send_response(204)
                    self.end_headers()
                    return
                return super().do_GET()

            def log_message(self, format, *args):
                msg = format % args
                if "GET /timeline.json" in msg or "GET /favicon.ico" in msg:
                    return
                return super().log_message(format, *args)

        handler = partial(_LiveHandler, directory=str(self.base_dir))
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = int(self._httpd.server_address[1])
        self.url = f"http://127.0.0.1:{self.port}/index.html"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def _write_timeline(self) -> None:
        tmp = self.timeline_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.timeline), encoding="utf-8")
        tmp.replace(self.timeline_path)

    def open(self) -> bool:
        try:
            return bool(webbrowser.open(self.url))
        except Exception:
            return False

    def append_snapshot(
        self,
        *,
        simulation,
        builder,
        step: int,
        atomic_events: List[dict] | None = None,
        flush: bool = True,
    ) -> None:
        self.timeline["frames"].append(
            _build_3d_frame_record(
                simulation=simulation,
                builder=builder,
                step=step,
                atomic_events=atomic_events,
            )
        )
        self.timeline["reeb_graph"] = _build_reeb_graph_plot_data(builder)
        if flush:
            self._write_timeline()

    def flush(self) -> None:
        self._write_timeline()

    def close(self, *, finished: bool = True) -> None:
        self.timeline["finished"] = bool(finished)
        self._write_timeline()
        try:
            self._httpd.shutdown()
            self._httpd.server_close()
        except Exception:
            pass
