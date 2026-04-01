def efficient_motion(
    num_mobile=12,
    sensing_radius=0.2,
    dt=0.04,
    seed=5,
    max_speed=0.22,
    lambda_shrink=1.0,
    mu_curvature=0.5,
    eta_cohesion=0.2,
    repulsion_strength=0.01,
    repulsion_power=2.0,
    repulsion_distance=None,
    force_clip=1.0,
    host="127.0.0.1",
    port=0,
    auto_open=True,
):
    """
    Launch a browser-based setup + simulation UI for the efficient 2D motion model.

    Setup mode:
    - Drag mobile sensors.
    - Add/remove mobile sensors.
    - Fence sensors are fixed and not editable.
    - Alpha complex is shown live.

    Simulation mode:
    - Start/pause/step/reset controls.
    - Boundary-Cycle Curvature Flow (BCCF) update runs on mobile sensors.
    - Fence sensors remain stationary.
    """
    import json
    import sys
    import threading
    import time
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from pathlib import Path
    from urllib.parse import urlparse

    import numpy as np

    src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from boundary_geometry import RectangularDomain
    from sensor_network import generate_fence_sensors
    from topology import generate_topology

    rng = np.random.default_rng(seed)
    domain = RectangularDomain(0.0, 1.0, 0.0, 1.0)
    domain_area = float((domain.max[0] - domain.min[0]) * (domain.max[1] - domain.min[1]))

    def _fence_points_for_radius(radius):
        sensors = generate_fence_sensors(domain, float(radius))
        return np.array([np.asarray(s.pos, dtype=float) for s in sensors], dtype=float)

    def _initial_mobile_points():
        angles = np.linspace(0.0, 2.0 * np.pi, num_mobile, endpoint=False)
        a = 0.36
        b = 0.16
        jitter = 0.012
        pts = np.column_stack(
            [
                0.5 + a * np.cos(angles) + rng.normal(0.0, jitter, size=num_mobile),
                0.5 + b * np.sin(angles) + rng.normal(0.0, jitter, size=num_mobile),
            ]
        )
        return np.clip(pts, 0.04, 0.96)

    def _edge_cycle_area(cycle_nodes, all_points):
        poly = all_points[np.asarray(cycle_nodes, dtype=int)]
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

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

    def _topology_data(all_points, radius, n_fence):
        try:
            topo = generate_topology(
                all_points.tolist(),
                float(radius),
                fence_node_count=int(n_fence),
                interior_point=np.array([0.5, 0.5], dtype=float),
            )
        except Exception:
            return [], [], [], []

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

        return edges, triangles, uncovered_cycles, sorted(boundary_edges)

    def _cycle_perimeter(cycle_nodes, all_points):
        if len(cycle_nodes) < 2:
            return 0.0
        perimeter = 0.0
        m = len(cycle_nodes)
        for i in range(m):
            a = all_points[cycle_nodes[i]]
            b = all_points[cycle_nodes[(i + 1) % m]]
            perimeter += float(np.linalg.norm(b - a))
        return perimeter

    def _ordered_hole_cycle(cycle_nodes, all_points):
        c = list(cycle_nodes)
        if len(c) < 3:
            return c
        # Use clockwise order so R_{-pi/2}(a,b)=(b,-a) points inward.
        if _edge_cycle_area(c, all_points) > 0.0:
            c = list(reversed(c))
        return c

    def _neighbors_from_edges(edges, n_nodes):
        neigh = [set() for _ in range(n_nodes)]
        for i, j in edges:
            if 0 <= i < n_nodes and 0 <= j < n_nodes and i != j:
                neigh[i].add(j)
                neigh[j].add(i)
        return neigh

    def _rotate_minus_pi_over_2(v):
        return np.array([v[1], -v[0]], dtype=float)

    def _bccf_force_model(all_points, edges, uncovered_cycles, n_fence):
        eps = 1e-8
        n_nodes = int(all_points.shape[0])
        neigh = _neighbors_from_edges(edges, n_nodes)

        inward_force = np.zeros_like(all_points)
        curvature_force = np.zeros_like(all_points)
        cohesion_force = np.zeros_like(all_points)
        repulsion_force = np.zeros_like(all_points)
        cycle_membership = np.zeros(n_nodes, dtype=float)

        ordered_cycles = []
        total_hole_area = 0.0
        total_hole_perimeter = 0.0
        boundary_node_set = set()

        for cycle in uncovered_cycles:
            c = _ordered_hole_cycle(cycle, all_points)
            if len(c) < 3:
                continue
            area_abs = abs(_edge_cycle_area(c, all_points))
            perimeter = _cycle_perimeter(c, all_points)
            total_hole_area += area_abs
            total_hole_perimeter += perimeter
            ordered_cycles.append(c)
            for node in c:
                cycle_membership[node] += 1.0
                boundary_node_set.add(int(node))

        lam = float(state["lambda_shrink"])
        mu = float(state["mu_curvature"])
        eta = float(state["eta_cohesion"])
        repulse_strength = float(state["repulsion_strength"])
        repulse_power = float(state["repulsion_power"])
        repulse_dist = float(max(eps, state["repulsion_distance"]))
        force_max = float(max(eps, state["force_clip"]))

        for c in ordered_cycles:
            m = len(c)
            centroid = np.mean(all_points[np.asarray(c, dtype=int)], axis=0)
            for k, node in enumerate(c):
                prev_node = c[(k - 1) % m]
                next_node = c[(k + 1) % m]
                x_p = all_points[prev_node]
                x_i = all_points[node]
                x_q = all_points[next_node]

                e_minus = x_i - x_p
                e_plus = x_q - x_i
                l_minus = float(np.linalg.norm(e_minus))
                l_plus = float(np.linalg.norm(e_plus))
                if l_minus < eps or l_plus < eps:
                    continue

                tau_minus = e_minus / l_minus
                tau_plus = e_plus / l_plus
                n_raw = _rotate_minus_pi_over_2(tau_minus) + _rotate_minus_pi_over_2(tau_plus)
                n_norm = float(np.linalg.norm(n_raw))
                if n_norm < eps:
                    to_centroid = centroid - x_i
                    tc_norm = float(np.linalg.norm(to_centroid))
                    if tc_norm < eps:
                        continue
                    n_hat = to_centroid / tc_norm
                else:
                    n_hat = n_raw / n_norm

                dot_val = float(np.clip(np.dot(tau_minus, tau_plus), -1.0, 1.0))
                theta = float(np.arccos(dot_val))
                kappa = float(np.pi - theta)

                omega = 1.0 / max(1.0, float(cycle_membership[node]))
                inward_force[node] += omega * lam * n_hat
                curvature_force[node] += omega * mu * kappa * n_hat

        for i in range(n_fence, n_nodes):
            if not neigh[i]:
                continue
            x_i = all_points[i]
            for j in neigh[i]:
                cohesion_force[i] += eta * (all_points[j] - x_i)

        repulsion_activations = 0
        for i in range(n_fence, n_nodes):
            x_i = all_points[i]
            for j in neigh[i]:
                if j <= i:
                    continue
                x_j = all_points[j]
                delta = x_i - x_j
                dist = float(np.linalg.norm(delta))
                if dist >= repulse_dist:
                    continue
                d = max(dist, eps)
                direction = delta / d
                taper = (repulse_dist - d) / repulse_dist
                magnitude = repulse_strength * taper / (d ** (repulse_power + 1.0))
                fij = magnitude * direction
                repulsion_force[i] += fij
                if j >= n_fence:
                    repulsion_force[j] -= fij
                repulsion_activations += 1

        total_force = inward_force + curvature_force + cohesion_force + repulsion_force
        total_force[:n_fence] = 0.0

        norms = np.linalg.norm(total_force, axis=1)
        clip_mask = norms > force_max
        if np.any(clip_mask):
            total_force[clip_mask] *= (force_max / (norms[clip_mask] + eps))[:, None]

        mobile_slice = slice(n_fence, n_nodes)
        mobile_force = total_force[mobile_slice]
        mobile_norms = np.linalg.norm(mobile_force, axis=1) if mobile_force.size else np.zeros(0, dtype=float)

        diagnostics = {
            "hole_count": int(len(ordered_cycles)),
            "hole_area_total": float(total_hole_area),
            "hole_perimeter_total": float(total_hole_perimeter),
            "boundary_sensor_count": int(sum(1 for idx in boundary_node_set if idx >= n_fence)),
            "mean_force_norm": float(np.mean(mobile_norms)) if mobile_norms.size else 0.0,
            "max_force_norm": float(np.max(mobile_norms)) if mobile_norms.size else 0.0,
            "force_clipped_count": int(np.sum(clip_mask[n_fence:])),
            "repulsion_activations": int(repulsion_activations),
        }
        return total_force, diagnostics

    initial_radius = float(sensing_radius)
    initial_fence_points = _fence_points_for_radius(initial_radius)
    initial_mobile_points = _initial_mobile_points()
    initial_repulsion_distance = (
        float(repulsion_distance) if repulsion_distance is not None else float(2.0 * initial_radius)
    )

    state = {
        "fence_points": initial_fence_points.copy(),
        "initial_fence_points": initial_fence_points.copy(),
        "mobile_points": initial_mobile_points.copy(),
        "initial_mobile_points": initial_mobile_points.copy(),
        "sensing_radius": initial_radius,
        "initial_sensing_radius": initial_radius,
        "lambda_shrink": float(lambda_shrink),
        "initial_lambda_shrink": float(lambda_shrink),
        "mu_curvature": float(mu_curvature),
        "initial_mu_curvature": float(mu_curvature),
        "eta_cohesion": float(eta_cohesion),
        "initial_eta_cohesion": float(eta_cohesion),
        "repulsion_strength": float(repulsion_strength),
        "initial_repulsion_strength": float(repulsion_strength),
        "repulsion_power": float(repulsion_power),
        "initial_repulsion_power": float(repulsion_power),
        "repulsion_distance": float(initial_repulsion_distance),
        "initial_repulsion_distance": float(initial_repulsion_distance),
        "force_clip": float(force_clip),
        "initial_force_clip": float(force_clip),
        "running": False,
        "started": False,
        "time": 0.0,
        "step": 0,
        "velocities": None,
    }
    state["velocities"] = np.zeros_like(state["mobile_points"])
    state_lock = threading.Lock()

    def _safe_mobile_array(raw_points):
        pts = np.asarray(raw_points, dtype=float)
        if pts.size == 0:
            return np.zeros((0, 2), dtype=float)
        pts = pts.reshape((-1, 2))
        pts = np.clip(pts, 0.0, 1.0)
        return pts

    def _all_points(fence_points_local, mobile_points):
        if mobile_points.shape[0] == 0:
            return fence_points_local.copy()
        return np.vstack([fence_points_local, mobile_points])

    def _snapshot_locked():
        mobile = state["mobile_points"]
        fence = state["fence_points"]
        radius = float(state["sensing_radius"])
        n_fence = int(fence.shape[0])
        all_pts = _all_points(fence, mobile)
        edges, triangles, uncovered_cycles, boundary_edges = _topology_data(all_pts, radius, n_fence)
        all_vel, diagnostics = _bccf_force_model(all_pts, edges, uncovered_cycles, n_fence)
        mobile_vel = all_vel[n_fence:] if all_vel.shape[0] > n_fence else np.zeros_like(mobile)
        return {
            "fence_points": fence.tolist(),
            "mobile_points": mobile.tolist(),
            "sensing_radius": radius,
            "dt": float(dt),
            "time": float(state["time"]),
            "step": int(state["step"]),
            "running": bool(state["running"]),
            "started": bool(state["started"]),
            "lambda_shrink": float(state["lambda_shrink"]),
            "mu_curvature": float(state["mu_curvature"]),
            "eta_cohesion": float(state["eta_cohesion"]),
            "repulsion_strength": float(state["repulsion_strength"]),
            "repulsion_power": float(state["repulsion_power"]),
            "repulsion_distance": float(state["repulsion_distance"]),
            "force_clip": float(state["force_clip"]),
            "domain_area": domain_area,
            "edges": [list(map(int, e)) for e in edges],
            "triangles": [list(map(int, tri)) for tri in triangles],
            "boundary_edges": [list(map(int, e)) for e in boundary_edges],
            "uncovered_cycles": [list(map(int, cyc)) for cyc in uncovered_cycles],
            "velocities": mobile_vel.tolist(),
            "hole_count": diagnostics["hole_count"],
            "hole_area_total": diagnostics["hole_area_total"],
            "hole_perimeter_total": diagnostics["hole_perimeter_total"],
            "boundary_sensor_count": diagnostics["boundary_sensor_count"],
            "mean_force_norm": diagnostics["mean_force_norm"],
            "max_force_norm": diagnostics["max_force_norm"],
            "force_clipped_count": diagnostics["force_clipped_count"],
            "repulsion_activations": diagnostics["repulsion_activations"],
        }

    def _step_locked():
        mobile = state["mobile_points"]
        fence = state["fence_points"]
        radius = float(state["sensing_radius"])
        n_fence = int(fence.shape[0])
        all_pts = _all_points(fence, mobile)
        edges, _, uncovered_cycles, _ = _topology_data(all_pts, radius, n_fence)
        all_vel, _ = _bccf_force_model(all_pts, edges, uncovered_cycles, n_fence)
        mobile_vel = all_vel[n_fence:] if all_vel.shape[0] > n_fence else np.zeros_like(mobile)
        speed = np.linalg.norm(mobile_vel, axis=1)
        mask = speed > max_speed
        if np.any(mask):
            mobile_vel[mask] = mobile_vel[mask] * (max_speed / (speed[mask] + 1e-12))[:, None]
        state["mobile_points"] = np.clip(mobile + dt * mobile_vel, 0.0, 1.0)
        state["velocities"] = mobile_vel
        state["time"] += dt
        state["step"] += 1

    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Efficient Motion Setup + Simulation</title>
  <style>
    :root {
      --bg: #eef2f7;
      --panel: #ffffff;
      --ink: #102a43;
      --muted: #52606d;
      --accent: #0b7285;
      --edge: #c92a2a;
      --boundary: #0b7285;
      --mobile: #1c7ed6;
      --fence: #000000;
      --cycle: #fab005;
      --tri: #d9480f;
      --line: #d9e2ec;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 18% 12%, #d9f0ff 0%, transparent 42%),
        radial-gradient(circle at 84% 4%, #d5f5e3 0%, transparent 35%),
        var(--bg);
      min-height: 100vh;
      padding: 14px;
    }
    .layout {
      max-width: 1300px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: minmax(680px, 1fr) 320px;
      gap: 14px;
      align-items: start;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      box-shadow: 0 8px 18px rgba(16, 42, 67, 0.08);
      overflow: hidden;
    }
    .toolbar {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      justify-content: space-between;
    }
    .row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    button {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px 10px;
      font-weight: 600;
      background: #fff;
      color: var(--ink);
      cursor: pointer;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    button.active {
      border-color: var(--accent);
      color: var(--accent);
      background: #e6fcf5;
    }
    button:disabled {
      opacity: 0.45;
      cursor: default;
    }
    #view {
      width: 100%;
      aspect-ratio: 1 / 1;
      display: block;
      background: #ffffff;
    }
    .side {
      padding: 12px;
      display: grid;
      gap: 10px;
    }
    .meta {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #f8fbff;
      font-size: 13px;
      color: var(--muted);
      line-height: 1.35;
    }
    .legend {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #fbfdff;
      font-size: 12px;
      display: grid;
      gap: 5px;
    }
    .param-grid {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #f8fbff;
      display: grid;
      gap: 8px;
      font-size: 12px;
    }
    .param-row {
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: center;
      gap: 8px;
    }
    .param-row input[type="number"] {
      width: 120px;
      padding: 4px 6px;
      border: 1px solid var(--line);
      border-radius: 6px;
      font: inherit;
      color: var(--ink);
      background: #fff;
    }
    .param-row input[type="checkbox"] {
      width: 16px;
      height: 16px;
      accent-color: var(--accent);
    }
    .swatch {
      display: inline-block;
      width: 12px;
      height: 12px;
      margin-right: 6px;
      border-radius: 2px;
      vertical-align: middle;
    }
    .note {
      font-size: 12px;
      color: var(--muted);
    }
    @media (max-width: 1050px) {
      .layout { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <div class="card">
      <div class="toolbar">
        <div class="row">
          <button id="dragTool" class="active">Drag</button>
          <button id="addTool">Add</button>
          <button id="removeTool">Remove</button>
        </div>
        <div class="row">
          <button id="startBtn" class="primary">Start Simulation</button>
          <button id="pauseBtn">Pause</button>
          <button id="stepBtn">Step</button>
          <button id="resetBtn">Reset Setup</button>
        </div>
      </div>
      <canvas id="view" width="900" height="900"></canvas>
    </div>
    <div class="card side">
      <div class="meta" id="statusBox">Loading...</div>
      <div class="param-grid">
        <div class="param-row">
          <label for="radiusInput"><strong>r (sensing radius)</strong></label>
          <input id="radiusInput" type="number" min="0.005" step="0.005" />
        </div>
        <div class="param-row">
          <label for="lambdaInput"><strong>lambda (inward)</strong></label>
          <input id="lambdaInput" type="number" min="0.0" step="0.05" />
        </div>
        <div class="param-row">
          <label for="muInput"><strong>mu (curvature)</strong></label>
          <input id="muInput" type="number" min="0.0" step="0.05" />
        </div>
        <div class="param-row">
          <label for="etaInput"><strong>eta (cohesion)</strong></label>
          <input id="etaInput" type="number" min="0.0" step="0.05" />
        </div>
        <div class="param-row">
          <label for="repulsionStrengthInput"><strong>repulsion strength</strong></label>
          <input id="repulsionStrengthInput" type="number" min="0.0" step="0.001" />
        </div>
        <div class="param-row">
          <label for="repulsionDistanceInput"><strong>repulsion distance</strong></label>
          <input id="repulsionDistanceInput" type="number" min="0.001" step="0.01" />
        </div>
        <div class="param-row">
          <label for="forceClipInput"><strong>force clip</strong></label>
          <input id="forceClipInput" type="number" min="0.001" step="0.05" />
        </div>
      </div>
      <div class="legend">
        <div><span class="swatch" style="background:#d9480f; opacity:.32"></span>Alpha triangles</div>
        <div><span class="swatch" style="background:#ff0000;"></span>Alpha vertices</div>
        <div><span class="swatch" style="background:#c92a2a;"></span>Alpha edges</div>
        <div><span class="swatch" style="background:#0b7285;"></span>Boundary edges</div>
        <div><span class="swatch" style="background:#fab005; opacity:.45"></span>Uncovered cycles</div>
        <div><span class="swatch" style="background:#1c7ed6;"></span>Mobile sensors</div>
        <div><span class="swatch" style="background:#364fc7;"></span>BCCF force vectors</div>
        <div><span class="swatch" style="background:#000000;"></span>Fence sensors (fixed)</div>
      </div>
      <div class="note">
        Setup mode allows drag/add/remove mobile points only. Fence points are immutable defaults.
        Motion model is Boundary-Cycle Curvature Flow: inward + curvature + cohesion + short-range repulsion with force clipping.
        Start simulation to run the model and watch alpha-complex evolution.
      </div>
    </div>
  </div>

  <script>
    const canvas = document.getElementById("view");
    const ctx = canvas.getContext("2d");
    const statusBox = document.getElementById("statusBox");

    const dragTool = document.getElementById("dragTool");
    const addTool = document.getElementById("addTool");
    const removeTool = document.getElementById("removeTool");
    const startBtn = document.getElementById("startBtn");
    const pauseBtn = document.getElementById("pauseBtn");
    const stepBtn = document.getElementById("stepBtn");
    const resetBtn = document.getElementById("resetBtn");
    const radiusInput = document.getElementById("radiusInput");
    const lambdaInput = document.getElementById("lambdaInput");
    const muInput = document.getElementById("muInput");
    const etaInput = document.getElementById("etaInput");
    const repulsionStrengthInput = document.getElementById("repulsionStrengthInput");
    const repulsionDistanceInput = document.getElementById("repulsionDistanceInput");
    const forceClipInput = document.getElementById("forceClipInput");

    let state = null;
    let tool = "drag";
    let draggingIndex = -1;
    let pointerDown = false;
    let runTimer = null;
    let lastDragSync = 0;
    let lastAddSync = 0;
    let lastAddScreenPoint = null;

    const worldMin = -0.17;
    const worldMax = 1.17;
    const worldSpan = worldMax - worldMin;

    function syncCanvasSize() {
      const rect = canvas.getBoundingClientRect();
      const w = Math.max(1, Math.round(rect.width));
      const h = Math.max(1, Math.round(rect.height));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
    }

    function pointerCanvasXY(event) {
      const rect = canvas.getBoundingClientRect();
      const sx = canvas.width / Math.max(rect.width, 1);
      const sy = canvas.height / Math.max(rect.height, 1);
      return [
        (event.clientX - rect.left) * sx,
        (event.clientY - rect.top) * sy
      ];
    }

    function setTool(nextTool) {
      tool = nextTool;
      dragTool.classList.toggle("active", tool === "drag");
      addTool.classList.toggle("active", tool === "add");
      removeTool.classList.toggle("active", tool === "remove");
    }

    async function api(path, method = "GET", payload = null) {
      const options = { method, headers: { "Content-Type": "application/json" } };
      if (payload !== null) options.body = JSON.stringify(payload);
      const res = await fetch(path, options);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || ("request failed: " + path));
      return data;
    }

    function allPoints() {
      return state.fence_points.concat(state.mobile_points);
    }

    function worldToCanvas(pt) {
      const x = ((pt[0] - worldMin) / worldSpan) * canvas.width;
      const y = canvas.height - ((pt[1] - worldMin) / worldSpan) * canvas.height;
      return [x, y];
    }

    function canvasToWorld(x, y) {
      const wx = worldMin + (x / canvas.width) * worldSpan;
      const wy = worldMin + ((canvas.height - y) / canvas.height) * worldSpan;
      return [wx, wy];
    }

    function clamp01(v) {
      return Math.max(0.0, Math.min(1.0, v));
    }

    function dist2(a, b) {
      const dx = a[0] - b[0];
      const dy = a[1] - b[1];
      return dx * dx + dy * dy;
    }

    function pickMobileIndex(screenX, screenY) {
      let best = -1;
      let bestDist2 = 22 * 22;
      for (let i = 0; i < state.mobile_points.length; i++) {
        const c = worldToCanvas(state.mobile_points[i]);
        const d2 = dist2([screenX, screenY], c);
        if (d2 < bestDist2) {
          bestDist2 = d2;
          best = i;
        }
      }
      return best;
    }

    function setControlsEnabled() {
      const setupMode = !state.started;
      dragTool.disabled = !setupMode;
      addTool.disabled = !setupMode;
      removeTool.disabled = !setupMode;
      stepBtn.disabled = !state.started;
      pauseBtn.disabled = !state.running;
      startBtn.textContent = state.running ? "Running..." : (state.started ? "Resume" : "Start Simulation");
      startBtn.disabled = state.running;
      radiusInput.disabled = !setupMode;
      lambdaInput.disabled = !setupMode;
      muInput.disabled = !setupMode;
      etaInput.disabled = !setupMode;
      repulsionStrengthInput.disabled = !setupMode;
      repulsionDistanceInput.disabled = !setupMode;
      forceClipInput.disabled = !setupMode;
    }

    function updateStatus() {
      const setupMode = !state.started;
      statusBox.innerHTML =
        "<strong>Mode:</strong> " + (setupMode ? "Setup" : (state.running ? "Simulation (running)" : "Simulation (paused)")) + "<br/>" +
        "<strong>Time:</strong> " + state.time.toFixed(3) + "<br/>" +
        "<strong>Step:</strong> " + state.step + "<br/>" +
        "<strong>Mobile Sensors:</strong> " + state.mobile_points.length + "<br/>" +
        "<strong>Fence Sensors:</strong> " + state.fence_points.length + "<br/>" +
        "<strong>Uncovered Cycles:</strong> " + state.uncovered_cycles.length + " (holes=" + state.hole_count + ")<br/>" +
        "<strong>Hole Area Total:</strong> " + state.hole_area_total.toFixed(4) +
        " &nbsp;&nbsp; <strong>Hole Perimeter Total:</strong> " + state.hole_perimeter_total.toFixed(4) + "<br/>" +
        "<strong>Boundary Mobile Sensors:</strong> " + state.boundary_sensor_count + "<br/>" +
        "<strong>r:</strong> " + state.sensing_radius.toFixed(3) +
        " &nbsp;&nbsp; <strong>lambda:</strong> " + state.lambda_shrink.toFixed(3) +
        " &nbsp;&nbsp; <strong>mu:</strong> " + state.mu_curvature.toFixed(3) + "<br/>" +
        "<strong>eta:</strong> " + state.eta_cohesion.toFixed(3) +
        " &nbsp;&nbsp; <strong>repulsion k:</strong> " + state.repulsion_strength.toFixed(4) +
        " &nbsp;&nbsp; <strong>repulsion d:</strong> " + state.repulsion_distance.toFixed(3) + "<br/>" +
        "<strong>Mean/Max Force:</strong> " + state.mean_force_norm.toFixed(4) + " / " + state.max_force_norm.toFixed(4) +
        " &nbsp;&nbsp; <strong>Clipped:</strong> " + state.force_clipped_count +
        " &nbsp;&nbsp; <strong>Repulse Hits:</strong> " + state.repulsion_activations;
    }

    function syncParamInputs() {
      if (!state) return;
      if (document.activeElement !== radiusInput) {
        radiusInput.value = Number(state.sensing_radius).toFixed(4);
      }
      if (document.activeElement !== lambdaInput) {
        lambdaInput.value = Number(state.lambda_shrink).toFixed(4);
      }
      if (document.activeElement !== muInput) {
        muInput.value = Number(state.mu_curvature).toFixed(4);
      }
      if (document.activeElement !== etaInput) {
        etaInput.value = Number(state.eta_cohesion).toFixed(4);
      }
      if (document.activeElement !== repulsionStrengthInput) {
        repulsionStrengthInput.value = Number(state.repulsion_strength).toFixed(4);
      }
      if (document.activeElement !== repulsionDistanceInput) {
        repulsionDistanceInput.value = Number(state.repulsion_distance).toFixed(4);
      }
      if (document.activeElement !== forceClipInput) {
        forceClipInput.value = Number(state.force_clip).toFixed(4);
      }
    }

    function drawCircle(pt, radiusWorld, fillStyle, strokeStyle, alpha) {
      const c = worldToCanvas(pt);
      const pxRadius = (radiusWorld / worldSpan) * canvas.width;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.arc(c[0], c[1], pxRadius, 0, Math.PI * 2);
      if (fillStyle) {
        ctx.fillStyle = fillStyle;
        ctx.fill();
      }
      if (strokeStyle) {
        ctx.strokeStyle = strokeStyle;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      ctx.restore();
    }

    function draw() {
      if (!state) return;
      syncCanvasSize();
      const points = allPoints();
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // soft background grid
      ctx.save();
      ctx.strokeStyle = "rgba(90, 114, 128, 0.15)";
      ctx.lineWidth = 1;
      for (let k = 0; k <= 10; k++) {
        const t = k / 10;
        const p1 = worldToCanvas([worldMin + t * worldSpan, worldMin]);
        const p2 = worldToCanvas([worldMin + t * worldSpan, worldMax]);
        ctx.beginPath();
        ctx.moveTo(p1[0], p1[1]);
        ctx.lineTo(p2[0], p2[1]);
        ctx.stroke();
        const q1 = worldToCanvas([worldMin, worldMin + t * worldSpan]);
        const q2 = worldToCanvas([worldMax, worldMin + t * worldSpan]);
        ctx.beginPath();
        ctx.moveTo(q1[0], q1[1]);
        ctx.lineTo(q2[0], q2[1]);
        ctx.stroke();
      }
      ctx.restore();

      // uncovered cycles fill
      ctx.save();
      ctx.fillStyle = "rgba(250, 176, 5, 0.20)";
      for (const cyc of state.uncovered_cycles) {
        if (!cyc || cyc.length < 3) continue;
        const p0 = worldToCanvas(points[cyc[0]]);
        ctx.beginPath();
        ctx.moveTo(p0[0], p0[1]);
        for (let i = 1; i < cyc.length; i++) {
          const pp = worldToCanvas(points[cyc[i]]);
          ctx.lineTo(pp[0], pp[1]);
        }
        ctx.closePath();
        ctx.fill();
      }
      ctx.restore();

      // alpha triangles
      ctx.save();
      ctx.fillStyle = "rgba(217, 72, 15, 0.17)";
      for (const tri of state.triangles) {
        const p0 = worldToCanvas(points[tri[0]]);
        const p1 = worldToCanvas(points[tri[1]]);
        const p2 = worldToCanvas(points[tri[2]]);
        ctx.beginPath();
        ctx.moveTo(p0[0], p0[1]);
        ctx.lineTo(p1[0], p1[1]);
        ctx.lineTo(p2[0], p2[1]);
        ctx.closePath();
        ctx.fill();
      }
      ctx.restore();

      // alpha edges
      ctx.save();
      ctx.strokeStyle = "rgba(220, 0, 0, 0.92)";
      ctx.lineWidth = 1.9;
      for (const e of state.edges) {
        const p = worldToCanvas(points[e[0]]);
        const q = worldToCanvas(points[e[1]]);
        ctx.beginPath();
        ctx.moveTo(p[0], p[1]);
        ctx.lineTo(q[0], q[1]);
        ctx.stroke();
      }
      ctx.restore();

      // alpha vertices
      ctx.save();
      ctx.fillStyle = "rgba(220, 0, 0, 0.95)";
      const alphaNodes = new Set();
      for (const e of state.edges) {
        alphaNodes.add(e[0]);
        alphaNodes.add(e[1]);
      }
      for (const tri of state.triangles) {
        alphaNodes.add(tri[0]);
        alphaNodes.add(tri[1]);
        alphaNodes.add(tri[2]);
      }
      for (const node of alphaNodes) {
        const p = worldToCanvas(points[node]);
        ctx.beginPath();
        ctx.arc(p[0], p[1], 3.0, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      // boundary edges overlay
      ctx.save();
      ctx.strokeStyle = "rgba(11, 114, 133, 0.72)";
      ctx.lineWidth = 2.1;
      for (const e of state.boundary_edges) {
        const p = worldToCanvas(points[e[0]]);
        const q = worldToCanvas(points[e[1]]);
        ctx.beginPath();
        ctx.moveTo(p[0], p[1]);
        ctx.lineTo(q[0], q[1]);
        ctx.stroke();
      }
      ctx.restore();

      // sensing disks
      for (const fp of state.fence_points) {
        drawCircle(fp, state.sensing_radius, "rgba(108,122,137,0.10)", "rgba(108,122,137,0.5)", 1.0);
      }
      for (const mp of state.mobile_points) {
        drawCircle(mp, state.sensing_radius, "rgba(76,110,245,0.10)", "rgba(76,110,245,0.5)", 1.0);
      }

      // unit square
      const s0 = worldToCanvas([0, 0]);
      const s1 = worldToCanvas([1, 1]);
      ctx.save();
      ctx.strokeStyle = "#111";
      ctx.lineWidth = 2;
      ctx.strokeRect(s0[0], s1[1], s1[0] - s0[0], s0[1] - s1[1]);
      ctx.restore();

      // velocity arrows (setup preview + simulation)
      if (state.velocities.length === state.mobile_points.length) {
        ctx.save();
        ctx.strokeStyle = "rgba(54,79,199,0.85)";
        ctx.lineWidth = 1.5;
        for (let i = 0; i < state.mobile_points.length; i++) {
          const p = state.mobile_points[i];
          const v = state.velocities[i];
          const end = [p[0] + 0.65 * v[0], p[1] + 0.65 * v[1]];
          const a = worldToCanvas(p);
          const b = worldToCanvas(end);
          ctx.beginPath();
          ctx.moveTo(a[0], a[1]);
          ctx.lineTo(b[0], b[1]);
          ctx.stroke();
        }
        ctx.restore();
      }

      // sensors
      ctx.save();
      for (const fp of state.fence_points) {
        const p = worldToCanvas(fp);
        ctx.fillStyle = "#000";
        const s = 7;
        ctx.fillRect(p[0] - s / 2, p[1] - s / 2, s, s);
      }
      for (const mp of state.mobile_points) {
        const p = worldToCanvas(mp);
        ctx.fillStyle = "#1c7ed6";
        ctx.beginPath();
        ctx.arc(p[0], p[1], 5.2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      updateStatus();
      syncParamInputs();
      setControlsEnabled();
    }

    async function refreshState() {
      state = await api("/state", "GET");
      draw();
    }

    async function setMobilePoints(mobilePoints) {
      state = await api("/set_mobile", "POST", { mobile_points: mobilePoints });
      draw();
    }

    async function setParams(patch) {
      state = await api("/set_params", "POST", patch);
      draw();
    }

    async function addPointAtCanvas(screenX, screenY) {
      if (!state || state.started) return;
      const now = performance.now();
      if (now - lastAddSync < 28) return;
      if (lastAddScreenPoint && dist2([screenX, screenY], lastAddScreenPoint) < 9 * 9) return;
      const w = canvasToWorld(screenX, screenY);
      if (w[0] < 0 || w[0] > 1 || w[1] < 0 || w[1] > 1) return;
      const next = state.mobile_points.slice();
      next.push([clamp01(w[0]), clamp01(w[1])]);
      lastAddSync = now;
      lastAddScreenPoint = [screenX, screenY];
      await setMobilePoints(next);
    }

    async function startLoop() {
      if (runTimer !== null) return;
      runTimer = setInterval(async () => {
        try {
          state = await api("/step", "POST", {});
          draw();
          if (!state.running) stopLoop();
        } catch (err) {
          console.error(err);
          stopLoop();
        }
      }, Math.max(20, Math.round((state ? state.dt : 0.04) * 1000)));
    }

    function stopLoop() {
      if (runTimer !== null) {
        clearInterval(runTimer);
        runTimer = null;
      }
    }

    canvas.addEventListener("mousedown", async (event) => {
      if (!state || state.started) return;
      pointerDown = true;
      const [x, y] = pointerCanvasXY(event);

      if (tool === "add") {
        await addPointAtCanvas(x, y);
        return;
      }

      const idx = pickMobileIndex(x, y);
      if (idx < 0) return;

      if (tool === "remove") {
        const next = state.mobile_points.slice();
        next.splice(idx, 1);
        await setMobilePoints(next);
        return;
      }

      if (tool === "drag") {
        draggingIndex = idx;
      }
    });

    canvas.addEventListener("mousemove", async (event) => {
      const [x, y] = pointerCanvasXY(event);
      if (!state || state.started) return;

      if (tool === "add" && pointerDown) {
        await addPointAtCanvas(x, y);
        return;
      }

      if (draggingIndex < 0 || tool !== "drag") return;

      const now = performance.now();
      if (now - lastDragSync < 30) return;
      lastDragSync = now;
      const w = canvasToWorld(x, y);
      const next = state.mobile_points.slice();
      next[draggingIndex] = [clamp01(w[0]), clamp01(w[1])];
      await setMobilePoints(next);
    });

    window.addEventListener("mouseup", async (event) => {
      pointerDown = false;
      lastAddScreenPoint = null;
      if (!state || state.started || draggingIndex < 0 || tool !== "drag") {
        draggingIndex = -1;
        return;
      }
      const [x, y] = pointerCanvasXY(event);
      const w = canvasToWorld(x, y);
      const next = state.mobile_points.slice();
      next[draggingIndex] = [clamp01(w[0]), clamp01(w[1])];
      draggingIndex = -1;
      await setMobilePoints(next);
    });

    dragTool.addEventListener("click", () => setTool("drag"));
    addTool.addEventListener("click", () => setTool("add"));
    removeTool.addEventListener("click", () => setTool("remove"));

    startBtn.addEventListener("click", async () => {
      if (!state) return;
      state = await api("/start", "POST", {});
      draw();
      await startLoop();
    });

    pauseBtn.addEventListener("click", async () => {
      if (!state) return;
      state = await api("/pause", "POST", {});
      draw();
      stopLoop();
    });

    stepBtn.addEventListener("click", async () => {
      if (!state) return;
      state = await api("/step_once", "POST", {});
      draw();
    });

    resetBtn.addEventListener("click", async () => {
      state = await api("/reset", "POST", {});
      draw();
      stopLoop();
      setTool("drag");
    });

    radiusInput.addEventListener("change", async () => {
      if (!state || state.started) return;
      const value = Number(radiusInput.value);
      if (!Number.isFinite(value) || value <= 0) {
        syncParamInputs();
        return;
      }
      await setParams({ sensing_radius: value });
    });

    lambdaInput.addEventListener("change", async () => {
      if (!state || state.started) return;
      const value = Number(lambdaInput.value);
      if (!Number.isFinite(value) || value < 0) {
        syncParamInputs();
        return;
      }
      await setParams({ lambda_shrink: value });
    });

    muInput.addEventListener("change", async () => {
      if (!state || state.started) return;
      const value = Number(muInput.value);
      if (!Number.isFinite(value) || value < 0) {
        syncParamInputs();
        return;
      }
      await setParams({ mu_curvature: value });
    });

    etaInput.addEventListener("change", async () => {
      if (!state || state.started) return;
      const value = Number(etaInput.value);
      if (!Number.isFinite(value) || value < 0) {
        syncParamInputs();
        return;
      }
      await setParams({ eta_cohesion: value });
    });

    repulsionStrengthInput.addEventListener("change", async () => {
      if (!state || state.started) return;
      const value = Number(repulsionStrengthInput.value);
      if (!Number.isFinite(value) || value < 0) {
        syncParamInputs();
        return;
      }
      await setParams({ repulsion_strength: value });
    });

    repulsionDistanceInput.addEventListener("change", async () => {
      if (!state || state.started) return;
      const value = Number(repulsionDistanceInput.value);
      if (!Number.isFinite(value) || value <= 0) {
        syncParamInputs();
        return;
      }
      await setParams({ repulsion_distance: value });
    });

    forceClipInput.addEventListener("change", async () => {
      if (!state || state.started) return;
      const value = Number(forceClipInput.value);
      if (!Number.isFinite(value) || value <= 0) {
        syncParamInputs();
        return;
      }
      await setParams({ force_clip: value });
    });

    refreshState().catch((err) => {
      statusBox.textContent = "Failed to load initial state: " + err.message;
    });
  </script>
</body>
</html>
"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send_json(self, data, status=200):
            payload = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _read_json(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(raw.decode("utf-8"))

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/" or path == "/index.html":
                payload = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            if path == "/state":
                with state_lock:
                    snap = _snapshot_locked()
                self._send_json(snap, status=200)
                return

            self._send_json({"error": "not found"}, status=404)

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                payload = self._read_json()
                with state_lock:
                    if path == "/set_mobile":
                        if state["started"]:
                            self._send_json({"error": "setup is locked after simulation starts"}, status=409)
                            return
                        mobile_points = payload.get("mobile_points", [])
                        state["mobile_points"] = _safe_mobile_array(mobile_points)
                        state["velocities"] = np.zeros_like(state["mobile_points"])
                        self._send_json(_snapshot_locked(), status=200)
                        return

                    if path == "/set_params":
                        if state["started"]:
                            self._send_json({"error": "setup is locked after simulation starts"}, status=409)
                            return

                        if "sensing_radius" in payload:
                            radius = float(payload.get("sensing_radius", state["sensing_radius"]))
                            if radius <= 0:
                                self._send_json({"error": "sensing_radius must be > 0"}, status=400)
                                return
                            state["sensing_radius"] = radius
                            state["fence_points"] = _fence_points_for_radius(radius)
                            if "repulsion_distance" not in payload and "initial_repulsion_distance" in state:
                                state["repulsion_distance"] = float(2.0 * radius)

                        if "lambda_shrink" in payload:
                            val = float(payload.get("lambda_shrink", state["lambda_shrink"]))
                            if val < 0:
                                self._send_json({"error": "lambda_shrink must be >= 0"}, status=400)
                                return
                            state["lambda_shrink"] = val

                        if "mu_curvature" in payload:
                            val = float(payload.get("mu_curvature", state["mu_curvature"]))
                            if val < 0:
                                self._send_json({"error": "mu_curvature must be >= 0"}, status=400)
                                return
                            state["mu_curvature"] = val

                        if "eta_cohesion" in payload:
                            val = float(payload.get("eta_cohesion", state["eta_cohesion"]))
                            if val < 0:
                                self._send_json({"error": "eta_cohesion must be >= 0"}, status=400)
                                return
                            state["eta_cohesion"] = val

                        if "repulsion_strength" in payload:
                            val = float(payload.get("repulsion_strength", state["repulsion_strength"]))
                            if val < 0:
                                self._send_json({"error": "repulsion_strength must be >= 0"}, status=400)
                                return
                            state["repulsion_strength"] = val

                        if "repulsion_power" in payload:
                            val = float(payload.get("repulsion_power", state["repulsion_power"]))
                            if val < 0:
                                self._send_json({"error": "repulsion_power must be >= 0"}, status=400)
                                return
                            state["repulsion_power"] = val

                        if "repulsion_distance" in payload:
                            val = float(payload.get("repulsion_distance", state["repulsion_distance"]))
                            if val <= 0:
                                self._send_json({"error": "repulsion_distance must be > 0"}, status=400)
                                return
                            state["repulsion_distance"] = val

                        if "force_clip" in payload:
                            val = float(payload.get("force_clip", state["force_clip"]))
                            if val <= 0:
                                self._send_json({"error": "force_clip must be > 0"}, status=400)
                                return
                            state["force_clip"] = val

                        state["velocities"] = np.zeros_like(state["mobile_points"])
                        self._send_json(_snapshot_locked(), status=200)
                        return

                    if path == "/start":
                        state["running"] = True
                        state["started"] = True
                        self._send_json(_snapshot_locked(), status=200)
                        return

                    if path == "/pause":
                        state["running"] = False
                        self._send_json(_snapshot_locked(), status=200)
                        return

                    if path == "/step":
                        if state["running"]:
                            _step_locked()
                        self._send_json(_snapshot_locked(), status=200)
                        return

                    if path == "/step_once":
                        if not state["started"]:
                            self._send_json({"error": "click Start Simulation before stepping"}, status=409)
                            return
                        _step_locked()
                        self._send_json(_snapshot_locked(), status=200)
                        return

                    if path == "/reset":
                        state["running"] = False
                        state["started"] = False
                        state["time"] = 0.0
                        state["step"] = 0
                        state["sensing_radius"] = float(state["initial_sensing_radius"])
                        state["lambda_shrink"] = float(state["initial_lambda_shrink"])
                        state["mu_curvature"] = float(state["initial_mu_curvature"])
                        state["eta_cohesion"] = float(state["initial_eta_cohesion"])
                        state["repulsion_strength"] = float(state["initial_repulsion_strength"])
                        state["repulsion_power"] = float(state["initial_repulsion_power"])
                        state["repulsion_distance"] = float(state["initial_repulsion_distance"])
                        state["force_clip"] = float(state["initial_force_clip"])
                        state["fence_points"] = state["initial_fence_points"].copy()
                        state["mobile_points"] = state["initial_mobile_points"].copy()
                        state["velocities"] = np.zeros_like(state["mobile_points"])
                        self._send_json(_snapshot_locked(), status=200)
                        return

                self._send_json({"error": "not found"}, status=404)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)

    server = ThreadingHTTPServer((host, port), Handler)
    actual_host, actual_port = server.server_address[0], server.server_address[1]
    url = "http://{}:{}/".format(actual_host, actual_port)

    print("Efficient motion UI running at {}".format(url))
    print("Press Ctrl+C to stop.")
    if auto_open:
        time.sleep(0.2)
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()

    return url


if __name__ == "__main__":
    efficient_motion()
