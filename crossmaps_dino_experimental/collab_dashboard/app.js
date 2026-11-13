"use strict";

// Pure normalization/mapping functions are also exercised by the Node tests.
const PALETTE = [[96, 217, 209], [183, 154, 255], [240, 181, 104], [111, 177, 244]];
const IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
const robotId = (robot) => String(robot.robot_id ?? robot.id ?? "unknown");
const escapeText = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const ageLabel = (value) => !finite(value) ? "No data" : value < 1 ? `${Math.round(value * 1000)} ms` : `${value.toFixed(1)} s`;
const numberLabel = (value, digits = 2) => finite(value) ? value.toFixed(digits) : "—";
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

function matrix(value) {
  if (Array.isArray(value) && value.length === 16 && value.every(finite)) return Array.from({ length: 4 }, (_, i) => value.slice(i * 4, i * 4 + 4));
  return Array.isArray(value) && value.length === 4 && value.every((row) => Array.isArray(row) && row.length === 4 && row.every(finite)) ? value : null;
}
function transformPoint(point, transform) {
  const t = transform || IDENTITY;
  return [0, 1, 2].map((i) => t[i][0] * point[0] + t[i][1] * point[1] + t[i][2] * point[2] + t[i][3]);
}
function inverseRigid(t) {
  const result = IDENTITY.map((row) => row.slice());
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) result[i][j] = t[j][i];
    result[i][3] = -result[i].slice(0, 3).reduce((sum, value, j) => sum + value * t[j][3], 0);
  }
  return result;
}
function matrixProduct(a, b) {
  return a.map((row) => b.map((_, j) => row.reduce((sum, value, k) => sum + value * b[k][j], 0)));
}
function anchorTransform(x, y, z, yawDegrees) {
  if (![x, y, z, yawDegrees].every(finite) || Math.max(Math.abs(x), Math.abs(y), Math.abs(z)) > 10000 || Math.abs(yawDegrees) > 36000) throw new Error("Enter finite, plausible translation and yaw values.");
  const yaw = yawDegrees * Math.PI / 180, c = Math.cos(yaw), s = Math.sin(yaw);
  return [[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, z], [0, 0, 0, 1]];
}
function normalizeSnapshot(raw) {
  return { ...raw, robots: Array.isArray(raw.robots) ? raw.robots : [], submaps: Array.isArray(raw.submaps) ? raw.submaps : [], candidates: Array.isArray(raw.candidates) ? raw.candidates : [], constraints: Array.isArray(raw.constraints) ? raw.constraints : [], global_entities: Array.isArray(raw.global_entities) ? raw.global_entities : (raw.entities || []), entity_links: Array.isArray(raw.entity_links) ? raw.entity_links : [], events: Array.isArray(raw.events) ? raw.events : [], stats: raw.stats || {} };
}
function buildCloud(snapshot, source = "combined", colorMode = "source", budget = 30000) {
  const positions = [], colors = [];
  const robots = snapshot.robots || [], excluded = [];
  const chosen = (snapshot.submaps || []).filter((submap) => {
    if (source !== "combined") return String(submap.robot_id) === source;
    if (submap.aligned === true && matrix(submap.T_world_submap)) return true;
    excluded.push(submap.id); return false;
  });
  const total = chosen.reduce((sum, submap) => sum + (submap.points || []).length, 0);
  const stride = Math.max(1, Math.ceil(total / Math.max(1, budget)));
  let index = 0;
  for (const submap of chosen) {
    const robotIndex = robots.findIndex((robot) => robotId(robot) === String(submap.robot_id));
    const robot = robots[robotIndex];
    let transform = matrix(submap.T_world_submap);
    if (source !== "combined") {
      transform = matrix(submap.T_local_submap) || (transform && matrix(robot?.T_world_local) ? matrixProduct(inverseRigid(matrix(robot.T_world_local)), transform) : transform) || IDENTITY;
    }
    const sourceColor = PALETTE[Math.max(0, robotIndex) % PALETTE.length];
    for (let i = 0; i < (submap.points || []).length; i++, index++) {
      if (index % stride) continue;
      const point = submap.points[i];
      if (!Array.isArray(point) || point.length < 3 || !point.slice(0, 3).every(finite)) continue;
      const transformed = transformPoint(point, transform);
      if (!transformed.every(finite)) continue;
      positions.push(...transformed);
      let color = colorMode === "rgb" ? (submap.colors?.[i] || (point.length >= 6 ? point.slice(3, 6) : sourceColor)) : sourceColor;
      if (!Array.isArray(color) || color.length < 3 || !color.slice(0, 3).every(finite)) color = sourceColor;
      const scale = Math.max(...color) <= 1 ? 255 : 1;
      colors.push(...color.slice(0, 3).map((c) => clamp(Math.round(c * scale), 0, 255)));
    }
  }
  return { positions, colors, excluded, submaps: chosen.length };
}
function candidateConstraint(candidate, snapshot) {
  return snapshot.constraints.find((constraint) => constraint.id === candidate.constraint_id || constraint.id === candidate.id || constraint.candidate_id === candidate.id);
}
function activeEntityLink(link) {
  return Boolean(link.active && link.alignment_valid && link.compatible_current === true);
}
function modePresentation(snapshot) {
  if (snapshot.mode === "synthetic_demo") return { label: "Synthetic demo · no robots", style: "supplied" };
  if (snapshot.mode === "recorded_replay") return { label: "Recorded replay · no robots", style: "supplied" };
  return snapshot.shadow_mode !== false ? { label: "Live · shadow mode", style: "shadow" } : { label: "Live · automatic alignment", style: "verified" };
}
function imageAge(robot, view) {
  if (finite(robot.image_age_s?.[view])) return robot.image_age_s[view];
  return view === "rgb" ? robot.frame_age_s : robot.observation_age_s;
}
function orderedCandidates(snapshot) {
  const rank = (candidate) => { const constraint = candidateConstraint(candidate, snapshot); return constraint?.active ? 1 : ["verified", "corroborated"].includes(candidate.state) ? 0 : candidate.state === "rejected" ? 3 : 2; };
  return snapshot.candidates.slice().reverse().sort((a, b) => rank(a) - rank(b));
}
function healthState(robot) {
  if (robot.error || robot.last_error || robot.health?.last_error || robot.perception?.error || ["error", "processing_error", "adapter_exited", "input_rejected"].includes(robot.status)) return "error";
  if (robot.status === "pose_rejected" || robot.pose_quality?.integration_allowed === false || robot.perception?.status === "pose_rejected") return "pose_rejected";
  if (robot.status === "capacity_reached") return "capacity_reached";
  if (!finite(robot.frame_age_s)) return robot.status || "waiting";
  if (robot.frame_age_s > 8) return "stale";
  return "healthy";
}

class WorldViewer {
  constructor(canvas) {
    this.canvas = canvas; this.gl = canvas.getContext("webgl", { antialias: true, alpha: false });
    this.positions = []; this.count = 0; this.target = [0, 0, 0]; this.distance = 9; this.yaw = -.85; this.pitch = .7; this.pointSize = 3; this.autoFit = true;
    if (!this.gl) return;
    const gl = this.gl;
    const shader = (type, source) => { const result = gl.createShader(type); gl.shaderSource(result, source); gl.compileShader(result); if (!gl.getShaderParameter(result, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(result)); return result; };
    this.program = gl.createProgram();
    gl.attachShader(this.program, shader(gl.VERTEX_SHADER, "attribute vec3 p;attribute vec3 c;uniform mat4 m;uniform float s;varying vec3 col;void main(){gl_Position=m*vec4(p,1.);gl_PointSize=s;col=c;}"));
    gl.attachShader(this.program, shader(gl.FRAGMENT_SHADER, "precision mediump float;varying vec3 col;uniform float roundPoint;void main(){if(roundPoint>.5 && length(gl_PointCoord-vec2(.5))>.5)discard;gl_FragColor=vec4(col,1.);}"));
    gl.linkProgram(this.program); if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw new Error("Unable to initialize WebGL scene");
    this.attr = { p: gl.getAttribLocation(this.program, "p"), c: gl.getAttribLocation(this.program, "c"), m: gl.getUniformLocation(this.program, "m"), s: gl.getUniformLocation(this.program, "s"), round: gl.getUniformLocation(this.program, "roundPoint") };
    this.points = this.buffer([], []);
    const grid = [], gridColors = [];
    for (let i = -10; i <= 10; i++) { grid.push(i, -10, 0, i, 10, 0, -10, i, 0, 10, i, 0); for (let j = 0; j < 4; j++) gridColors.push(...(i === 0 ? [58, 81, 103] : [30, 43, 58])); }
    this.grid = this.buffer(grid, gridColors);
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("pointerdown", (event) => { canvas.setPointerCapture(event.pointerId); this.drag = { x: event.clientX, y: event.clientY, pan: event.shiftKey || event.button === 2 }; });
    canvas.addEventListener("pointermove", (event) => { if (!this.drag) return; const dx = event.clientX - this.drag.x, dy = event.clientY - this.drag.y; this.drag.x = event.clientX; this.drag.y = event.clientY; if (this.drag.pan) { const scale = this.distance * .0015; this.target[0] += (Math.sin(this.yaw) * dx + Math.cos(this.yaw) * dy) * scale; this.target[1] += (-Math.cos(this.yaw) * dx + Math.sin(this.yaw) * dy) * scale; } else { this.yaw -= dx * .006; this.pitch = clamp(this.pitch + dy * .006, -.15, 1.56); } });
    canvas.addEventListener("pointerup", () => { this.drag = null; }); canvas.addEventListener("pointercancel", () => { this.drag = null; });
    canvas.addEventListener("wheel", (event) => { event.preventDefault(); this.distance = clamp(this.distance * Math.exp(event.deltaY * .001), .2, 1000); }, { passive: false });
    requestAnimationFrame(() => this.draw());
  }
  buffer(positions, colors, old) { const gl = this.gl, item = old || { p: gl.createBuffer(), c: gl.createBuffer() }; gl.bindBuffer(gl.ARRAY_BUFFER, item.p); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.DYNAMIC_DRAW); gl.bindBuffer(gl.ARRAY_BUFFER, item.c); gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(colors), gl.DYNAMIC_DRAW); item.count = positions.length / 3; return item; }
  update(cloud) { this.positions = cloud.positions; this.count = cloud.positions.length / 3; if (!this.gl) return; this.points = this.buffer(cloud.positions, cloud.colors, this.points); if (this.autoFit && this.count) { this.fit(); this.autoFit = false; } }
  fit() { if (!this.positions.length) { this.target = [0, 0, 0]; this.distance = 9; return; } const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity]; for (let i = 0; i < this.positions.length; i++) { const axis = i % 3; min[axis] = Math.min(min[axis], this.positions[i]); max[axis] = Math.max(max[axis], this.positions[i]); } this.target = min.map((value, i) => (value + max[i]) / 2); this.distance = clamp(Math.hypot(...max.map((value, i) => value - min[i])) * 1.6, 2, 1000); }
  draw() {
    if (!this.gl) return;
    const gl = this.gl, ratio = Math.min(2, window.devicePixelRatio || 1), width = Math.round(this.canvas.clientWidth * ratio), height = Math.round(this.canvas.clientHeight * ratio);
    if (this.canvas.width !== width || this.canvas.height !== height) { this.canvas.width = width; this.canvas.height = height; }
    gl.viewport(0, 0, width, height); gl.clearColor(.047, .071, .106, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST); gl.useProgram(this.program);
    const eye = [this.target[0] + this.distance * Math.cos(this.pitch) * Math.cos(this.yaw), this.target[1] + this.distance * Math.cos(this.pitch) * Math.sin(this.yaw), this.target[2] + this.distance * Math.sin(this.pitch)];
    const z = normalize(eye.map((v, i) => v - this.target[i])), x = normalize([-z[1], z[0], 0]), y = cross(z, x);
    const view = [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1];
    const f = 1 / Math.tan(Math.PI / 8), near = .01, far = 5000, proj = [f / Math.max(.01, width / Math.max(1, height)), 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0, 2 * far * near / (near - far), 0];
    const vp = new Float32Array(16); for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) vp[i + j * 4] = proj[i] * view[j * 4] + proj[i + 4] * view[j * 4 + 1] + proj[i + 8] * view[j * 4 + 2] + proj[i + 12] * view[j * 4 + 3];
    gl.uniformMatrix4fv(this.attr.m, false, vp);
    for (const [buffer, mode] of [[this.grid, gl.LINES], [this.points, gl.POINTS]]) { gl.bindBuffer(gl.ARRAY_BUFFER, buffer.p); gl.enableVertexAttribArray(this.attr.p); gl.vertexAttribPointer(this.attr.p, 3, gl.FLOAT, false, 0, 0); gl.bindBuffer(gl.ARRAY_BUFFER, buffer.c); gl.enableVertexAttribArray(this.attr.c); gl.vertexAttribPointer(this.attr.c, 3, gl.UNSIGNED_BYTE, true, 0, 0); gl.uniform1f(this.attr.s, this.pointSize * ratio); gl.uniform1f(this.attr.round, mode === gl.POINTS ? 1 : 0); gl.drawArrays(mode, 0, buffer.count); }
    requestAnimationFrame(() => this.draw());
  }
}
const normalize = (v) => { const length = Math.hypot(...v) || 1; return v.map((n) => n / length); };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);

function evidenceRenderKey(state) {
  // Camera ages change every poll; unchanged evidence must not replace focused
  // buttons/details in the DOM while an operator is reading or clicking them.
  return JSON.stringify([state.candidates, state.constraints, state.global_entities,
    state.entity_links, state.events, state.stats, state.mode, state._dashboard?.commands_enabled]);
}

function startDashboard() {
  let lastEvidenceKey = null;
  const $ = (id) => document.getElementById(id);
  let snapshot = normalizeSnapshot({}), source = "combined", preview = true, connected = false, pollBusy = false, sequence = 0, robotKey = "", lastRenderKey = "";
  const cameraState = new Map();
  let viewer;
  try { viewer = new WorldViewer($("world")); } catch (error) { notify(error.message, true); }
  function notify(message, error = false) { $("notice").hidden = false; $("notice").textContent = message; $("notice").classList.toggle("error", error); }
  async function command(payload) {
    if (!connected || !snapshot._dashboard?.commands_enabled) throw new Error("Commands are available only from a connected local dashboard.");
    const response = await fetch("/api/command", { method: "POST", headers: { "Content-Type": "application/json", "X-CrossMaps-Token": snapshot._dashboard.command_token }, body: JSON.stringify(payload), signal: AbortSignal.timeout(15000) });
    const result = await response.json(); if (!response.ok || result.ok === false) throw new Error(result.error || "Command was rejected");
    notify(result.message || result.reason || "Command applied. Local source evidence is preserved."); await poll(); return result;
  }
  function renderRobots() {
    $("robots").innerHTML = snapshot.robots.map((robot, index) => {
      const state = healthState(robot), timing = robot.timing || {}, health = robot.health || {}, perception = robot.perception || {}, id = robotId(robot);
      const cycle = timing.cycle_avg_s ?? timing.cycle_s ?? (finite(timing.average_ms) ? timing.average_ms / 1000 : finite(timing.cycle_ms) ? timing.cycle_ms / 1000 : null);
      const pose = typeof robot.pose_quality === "object" && robot.pose_quality ? robot.pose_quality.status : robot.pose_quality;
      const error = robot.error || robot.last_error || health.error || health.last_error || perception.error;
      return `<article class="robot-card"><div class="robot-top"><div class="robot-name"><i class="source-dot ${index % 2 ? "violet" : ""}"></i>Rover ${escapeText(id)}</div><span class="badge ${["pose_rejected", "capacity_reached"].includes(state) ? "pending" : escapeText(state)}">${escapeText(state.replaceAll("_", " "))}</span></div><div class="robot-details"><div class="metric"><span>Camera age</span><strong>${ageLabel(robot.frame_age_s)}</strong></div><div class="metric"><span>Processed observation</span><strong>${ageLabel(robot.observation_age_s)}</strong></div><div class="metric"><span>Mean cycle</span><strong>${ageLabel(cycle)}</strong></div><div class="metric"><span>Keyframes / submaps</span><strong>${escapeText(robot.keyframes ?? robot.counts?.keyframes ?? 0)} / ${escapeText(robot.submaps ?? robot.counts?.submaps ?? 0)}</strong></div></div><div class="robot-meta">Domain ${escapeText(robot.domain ?? "—")} · ${escapeText(perception.stage || health.stage || robot.stage || robot.status || "Waiting for input")} · session ${escapeText(String(robot.session_id || "not received").slice(0, 18))}<br>${escapeText(pose || "External odometry; not validated")}</div>${error ? `<div class="robot-error">${escapeText(error)}</div>` : ""}</article>`;
    }).join("") || '<div class="robot-card muted">Waiting for rover configuration from the coordinator.</div>';
  }
  function ensureCameras() {
    const key = snapshot.robots.map(robotId).join("|"); if (key === robotKey) return; robotKey = key;
    for (const camera of cameraState.values()) { camera.controller?.abort(); if (camera.url) URL.revokeObjectURL(camera.url); } cameraState.clear();
    $("source-tabs").innerHTML = '<button data-source="combined">Combined</button>' + snapshot.robots.map((robot) => `<button data-source="${escapeText(robotId(robot))}">Rover ${escapeText(robotId(robot))}</button>`).join("");
    if (source !== "combined" && !snapshot.robots.some((robot) => robotId(robot) === source)) source = "combined";
    $("cameras").innerHTML = snapshot.robots.map((robot, i) => `<article class="camera-card" data-robot="${escapeText(robotId(robot))}"><div class="camera-header"><strong><i class="source-dot ${i % 2 ? "violet" : ""}"></i> Rover ${escapeText(robotId(robot))}</strong><div class="camera-controls"><select aria-label="Rover ${escapeText(robotId(robot))} camera view"><option value="rgb">RGB</option><option value="sam">SAM masks</option><option value="segments">Segments</option><option value="entities">Entities</option><option value="dino">DINO</option><option value="dino_similarity">DINO similarity</option><option value="clip">CLIP</option><option value="depth">Depth</option></select><button data-expand>Expand</button></div></div><div class="camera-image"><img alt="Rover ${escapeText(robotId(robot))} camera" hidden><span class="camera-placeholder">Waiting for first image</span></div><div class="camera-caption"><span data-image-status>Not received</span><span data-image-age></span></div></article>`).join("");
    for (const card of $("cameras").querySelectorAll(".camera-card")) {
      const id = card.dataset.robot, camera = { card, view: "rgb", busy: false, generation: 0, url: null }; cameraState.set(id, camera);
      card.querySelector("select").addEventListener("change", (event) => { camera.view = event.target.value; camera.generation++; camera.controller?.abort(); card.querySelector("img").hidden = true; card.querySelector(".camera-placeholder").hidden = false; card.querySelector(".camera-placeholder").textContent = "Loading selected view…"; });
      card.querySelector("[data-expand]").addEventListener("click", () => fullscreen(card.querySelector(".camera-image")));
    }
    for (const id of ["anchor-robot", "anchor-reference"]) { const previous = $(id).value; $(id).innerHTML = snapshot.robots.map((robot) => `<option value="${escapeText(robotId(robot))}">Rover ${escapeText(robotId(robot))}</option>`).join(""); if (snapshot.robots.some((robot) => robotId(robot) === previous)) $(id).value = previous; }
    if (snapshot.robots.length > 1 && $("anchor-robot").value === $("anchor-reference").value) $("anchor-robot").value = robotId(snapshot.robots[1]);
  }
  async function refreshCamera(id, camera) {
    if (!preview || document.hidden || camera.busy) return;
    camera.busy = true; const generation = camera.generation; camera.controller = new AbortController(); const timeout = setTimeout(() => camera.controller.abort(), 7000); let url;
    try {
      const response = await fetch(`/api/image?robot=${encodeURIComponent(id)}&view=${encodeURIComponent(camera.view)}&seq=${sequence++}`, { cache: "no-store", signal: camera.controller.signal });
      if (!response.ok) throw new Error(response.status === 404 ? "No image yet for this view" : `Image unavailable (${response.status})`);
      const blob = await response.blob(); if (!blob.type.startsWith("image/")) throw new Error("Invalid camera response");
      url = URL.createObjectURL(blob); const decoded = new Image(); decoded.src = url; await decoded.decode();
      if (generation !== camera.generation || !preview) return;
      const img = camera.card.querySelector("img"), old = camera.url; img.src = url; img.hidden = false; camera.url = url; url = null; if (old) URL.revokeObjectURL(old);
      camera.card.querySelector(".camera-placeholder").hidden = true; camera.card.querySelector("[data-image-status]").textContent = `${camera.view.toUpperCase()} · ${decoded.naturalWidth} × ${decoded.naturalHeight}`;
    } catch (error) {
      if (generation === camera.generation) { camera.card.querySelector("[data-image-status]").textContent = error.name === "AbortError" ? "Image request timed out" : error.message; if (camera.card.querySelector("img").hidden) camera.card.querySelector(".camera-placeholder").textContent = error.message; }
    } finally { if (url) URL.revokeObjectURL(url); clearTimeout(timeout); camera.busy = false; }
  }
  function renderMap() {
    for (const button of $("source-tabs").querySelectorAll("button")) button.classList.toggle("active", button.dataset.source === source);
    const cloud = buildCloud(snapshot, source, $("color-mode").value, Number($("budget").value)); viewer?.update(cloud);
    $("point-count").textContent = `${(cloud.positions.length / 3).toLocaleString()} points · ${cloud.submaps} submaps`;
    $("map-frame").textContent = source === "combined" ? "Shared world · aligned evidence only" : `Rover ${source} · local frame`;
    $("map-empty").hidden = cloud.positions.length > 0 && Boolean(viewer?.gl);
    $("map-empty").querySelector("h3").textContent = !viewer?.gl ? "WebGL is unavailable" : source === "combined" && cloud.excluded.length ? "Maps not aligned yet" : "Waiting for spatial evidence";
    $("map-empty").querySelector("p").textContent = !viewer?.gl ? "Camera feeds and evidence remain available. Enable WebGL to see the 3D map." : cloud.excluded.length ? "Choose a rover to inspect its local map. Unaligned submaps are never overlaid at a guessed origin." : "Selected keyframes and measured geometry will appear here.";
    $("alignment-summary").textContent = source === "combined" ? `${cloud.excluded.length} unaligned submap(s) excluded. ${snapshot.shadow_mode !== false ? "Shadow mode: proposals do not automatically change alignment." : "Automatic acceptance enabled; inspect constraint evidence below."}` : "Local source geometry. This view does not imply cross-rover alignment or corrected odometry.";
    $("map-legend").innerHTML = snapshot.robots.map((robot, index) => `<span><i class="source-dot ${index % 2 ? "violet" : ""}"></i>Rover ${escapeText(robotId(robot))} · ${escapeText(robot.alignment_state || (robot.aligned ? "anchored" : "unaligned"))}</span>`).join("");
  }
  function renderEvidence() {
    const evidenceKey = evidenceRenderKey(snapshot);
    $("revision").textContent = `revision ${snapshot.revision ?? "—"}`;
    if (evidenceKey === lastEvidenceKey) {
      for (const button of document.querySelectorAll("[data-action], #save, #anchor-form button")) button.disabled = !snapshot._dashboard?.commands_enabled;
      $("save").disabled = snapshot.mode === "recorded_replay" || !snapshot._dashboard?.commands_enabled;
      return;
    }
    lastEvidenceKey = evidenceKey;
    const expanded = new Set(Array.from($("candidates").querySelectorAll("details[open]")).map((element) => element.dataset.key));
    const candidateScroll = $("candidates").scrollTop, eventScroll = $("events").scrollTop;
    $("candidate-count").textContent = snapshot.candidates.length;
    const anchorCards = snapshot.constraints.filter((constraint) => constraint.kind === "anchor" && constraint.active).map((constraint) => `<article class="evidence-card"><div class="evidence-title"><strong>Rover ${escapeText(constraint.robot_id || constraint.source)} → ${escapeText(constraint.reference_robot_id || constraint.target)}</strong><span class="badge supplied">Supplied anchor</span></div><p>Operator-provided alignment. This is not an automatic geometric match.</p><details data-key="${escapeText(constraint.id)}"><summary>Inspect supplied transform</summary><pre>${escapeText(JSON.stringify({ T_reference_local: constraint.T_reference_local, T_target_source: constraint.T_target_source }, null, 2))}</pre></details><div class="evidence-actions"><button data-action="reject_constraint" data-id="${escapeText(constraint.id)}">Undo supplied anchor</button></div></article>`).join("");
    const candidateCards = orderedCandidates(snapshot).slice(0, 100).map((candidate) => {
      const verification = candidate.verification || {}, constraint = candidateConstraint(candidate, snapshot), transform = candidate.transform || constraint?.T_target_source || verification.T_target_source;
      const state = constraint?.active ? constraint.state || "accepted" : candidate.state || "proposed";
      return `<article class="evidence-card"><div class="evidence-title"><strong>${escapeText(candidate.source_robot ? `Rover ${candidate.source_robot}` : candidate.source)} ↔ ${escapeText(candidate.target_robot ? `Rover ${candidate.target_robot}` : candidate.target)}</strong><span class="badge ${escapeText(state)}">${escapeText(state)}</span></div><p>${escapeText(candidate.reason || verification.reason || "Awaiting geometric verification")}</p><div class="evidence-metrics"><span>Retrieval ${numberLabel(candidate.score)}</span><span>Inliers ${escapeText(verification.inliers ?? verification.inlier_count ?? candidate.inliers ?? "—")}</span><span>RMSE ${numberLabel(verification.rmse ?? candidate.rmse, 3)} m</span><span>Corroborations ${escapeText(constraint?.corroborations ?? 0)}</span></div><details data-key="${escapeText(candidate.id)}"><summary>Inspect evidence and transform</summary><pre>${escapeText(JSON.stringify({ source: candidate.source, target: candidate.target, cues: candidate.cues, verification, T_target_source: transform }, null, 2))}</pre></details>${constraint ? `<div class="evidence-actions">${!constraint.active && !["rejected", "invalid"].includes(constraint.state) ? `<button data-action="accept_constraint" data-id="${escapeText(constraint.id)}">Accept verified constraint</button>` : ""}<button data-action="reject_constraint" data-id="${escapeText(constraint.id)}">${constraint.active ? "Undo constraint" : "Reject constraint"}</button></div>` : ""}</article>`;
    });
    $("candidates").innerHTML = candidateCards.slice(0, 8).join("") || '<div class="empty-list">No cross-rover candidates yet.<br>Distinct keyframes and overlapping places are needed before verification.</div>';
    if (candidateCards.length > 8) $("candidates").insertAdjacentHTML("beforeend", `<details data-key="additional-candidates"><summary>Show ${candidateCards.length - 8} more candidates and rejected matches</summary>${candidateCards.slice(8).join("")}</details>`);
    $("candidates").insertAdjacentHTML("afterbegin", anchorCards);
    const shared = snapshot.global_entities.filter((entity) => (entity.members || []).length > 1);
    $("entity-count").textContent = shared.length;
    $("entities").innerHTML = shared.slice(0, 60).map((entity) => {
      const memberIds = (entity.members || []).map((member) => typeof member === "string" ? member : member.id);
      const links = snapshot.entity_links.filter((link) => activeEntityLink(link) && (link.members || []).every((id) => memberIds.includes(id)));
      return `<article class="evidence-card"><div class="evidence-title"><strong>${escapeText(entity.id)}</strong><span class="badge verified">${escapeText(finite(entity.confidence) ? `Confidence ${numberLabel(entity.confidence)}` : String(entity.confidence || "linked").replaceAll("_", " "))}</span></div><p class="members">${(entity.members || []).map((member) => escapeText(typeof member === "string" ? member : `${member.robot_id} / ${member.local_id}`)).join(" ↔ ")}</p><p>${escapeText(entity.reason || "Cross-rover identity hypothesis; source geometry remains separate.")}</p><div class="evidence-actions">${links.map((link) => `<button data-action="unlink_entities" data-id="${escapeText(link.id)}">Unlink identity</button>`).join("")}</div></article>`;
    }).join("") || '<div class="empty-list">No shared identities yet.<br>Local segments and entity hypotheses remain with their source rover.</div>';
    const proposedLinks = snapshot.entity_links.filter((link) => link.state !== "rejected" && (!link.active || !link.alignment_valid || link.compatible_current !== true)).slice(-40);
    $("entities").insertAdjacentHTML("beforeend", proposedLinks.map((link) => `<article class="evidence-card"><div class="evidence-title"><strong>${link.active ? "Suspended identity link" : "Identity proposal"}</strong><span class="badge pending">${escapeText(link.active ? "suspended" : link.state)}</span></div><p class="members">${(link.members || []).map(escapeText).join(" ↔ ")}</p>${link.reason ? `<p>${escapeText(link.reason)}</p>` : ""}<div class="evidence-metrics"><span>DINO ${numberLabel(link.dino)}</span><span>Overlap ${numberLabel(link.overlap)}</span><span>${link.alignment_valid ? "Alignment valid" : "Alignment unavailable"}</span></div><div class="evidence-actions">${!link.active && link.alignment_valid && link.compatible_current === true ? `<button data-action="accept_entity_link" data-id="${escapeText(link.id)}">Accept identity link</button>` : ""}<button data-action="unlink_entities" data-id="${escapeText(link.id)}">${link.active ? "Undo identity link" : "Reject identity link"}</button></div></article>`).join(""));
    $("events").innerHTML = snapshot.events.slice().reverse().slice(0, 60).map((event) => { const stamp = event.time ?? event.timestamp ?? event.stamp; const label = finite(stamp) ? new Date(stamp * (stamp > 1e12 ? 1 : 1000)).toLocaleTimeString() : ""; return `<div class="event"><time>${escapeText(label)}</time><span>${escapeText(typeof event === "string" ? event : event.message || event.reason || event.type || JSON.stringify(event))}</span></div>`; }).join("") || '<div class="empty-list">Coordinator events will appear here.</div>';
    $("revision").textContent = `revision ${snapshot.revision ?? "—"}`;
    $("stats").textContent = Object.entries(snapshot.stats).filter(([, value]) => typeof value !== "object").slice(0, 8).map(([key, value]) => `${key.replaceAll("_", " ")}: ${value}`).join(" · ");
    for (const button of document.querySelectorAll("[data-action], #save, #anchor-form button")) button.disabled = !snapshot._dashboard?.commands_enabled;
    $("save").disabled = snapshot.mode === "recorded_replay" || !snapshot._dashboard?.commands_enabled;
    $("save").title = snapshot.mode === "recorded_replay" ? "Replay decisions are in memory only; the source recording is unchanged" : "Checkpoint retained evidence and decisions";
    for (const details of $("candidates").querySelectorAll("details")) details.open = expanded.has(details.dataset.key);
    $("candidates").scrollTop = candidateScroll; $("events").scrollTop = eventScroll;
  }
  async function poll() {
    if (pollBusy) return; pollBusy = true;
    try {
      const response = await fetch("/api/state", { cache: "no-store", signal: AbortSignal.timeout(7000) }); if (!response.ok) throw new Error(`Coordinator returned ${response.status}`);
      snapshot = normalizeSnapshot(await response.json()); connected = true; $("connection").textContent = `Coordinator connected · ${snapshot.robots.length} rover sources`; $("connection-dot").classList.add("online");
      const mode = modePresentation(snapshot); $("mode-badge").textContent = mode.label; $("mode-badge").className = `badge ${mode.style}`;
      $("shadow").checked = snapshot.shadow_mode !== false; $("shadow").disabled = !snapshot._dashboard?.commands_enabled;
      renderRobots(); ensureCameras(); renderEvidence();
      const key = JSON.stringify([snapshot.revision, source, $("color-mode").value, $("budget").value, snapshot.submaps.map((s) => [s.id, s.aligned, s.pose_version, s.points?.length])]);
      if (key !== lastRenderKey || snapshot.revision === undefined) { renderMap(); lastRenderKey = key; }
      for (const robot of snapshot.robots) { const camera = cameraState.get(robotId(robot)); if (camera) camera.card.querySelector("[data-image-age]").textContent = snapshot.mode === "recorded_replay" ? "Recorded observation · not live" : `Source age ${ageLabel(imageAge(robot, camera.view))}`; }
    } catch (error) { connected = false; $("connection-dot").classList.remove("online"); $("connection").textContent = `Disconnected · ${error.message}`; for (const button of document.querySelectorAll("[data-action], #save, #anchor-form button")) button.disabled = true; $("shadow").disabled = true; for (const camera of cameraState.values()) camera.card.querySelector("[data-image-age]").textContent = "Age unknown · coordinator disconnected"; }
    finally { pollBusy = false; }
  }
  async function fullscreen(element) { try { if (document.fullscreenElement) await document.exitFullscreen(); else if (element.requestFullscreen) await element.requestFullscreen(); else notify("Full screen is unavailable in this browser. Increase the browser window to enlarge the view."); } catch (error) { notify(error.message, true); } }
  $("source-tabs").addEventListener("click", (event) => { const button = event.target.closest("[data-source]"); if (!button) return; source = button.dataset.source; viewer && (viewer.autoFit = true); renderMap(); });
  for (const id of ["color-mode", "budget"]) $(id).addEventListener("change", renderMap);
  $("point-size").addEventListener("input", () => { if (viewer) viewer.pointSize = Number($("point-size").value); });
  $("fit").addEventListener("click", () => viewer?.fit()); $("top").addEventListener("click", () => { if (viewer) { viewer.pitch = 1.56; viewer.yaw = -Math.PI / 2; viewer.fit(); } });
  $("map-fullscreen").addEventListener("click", () => fullscreen(document.querySelector(".map-panel")));
  $("help-toggle").addEventListener("click", () => { $("help").hidden = !$("help").hidden; $("help-toggle").setAttribute("aria-expanded", String(!$("help").hidden)); });
  $("preview-toggle").addEventListener("click", () => { preview = !preview; $("preview-toggle").textContent = preview ? "Pause preview" : "Resume preview"; $("preview-toggle").setAttribute("aria-pressed", String(preview)); for (const camera of cameraState.values()) if (!preview) camera.controller?.abort(); });
  $("save").addEventListener("click", () => command({ action: "save" }).catch((error) => notify(error.message, true)));
  $("shadow").addEventListener("change", async () => { const enabled = $("shadow").checked; if (!enabled && !window.confirm("Allow the coordinator to commit automatically verified alignments? This changes the shared map. It does not change robot localization.")) { $("shadow").checked = true; return; } try { await command({ action: "set_shadow_mode", enabled }); } catch (error) { $("shadow").checked = snapshot.shadow_mode !== false; notify(error.message, true); } });
  document.addEventListener("click", async (event) => { const button = event.target.closest("[data-action]"); if (!button) return; if (!window.confirm(`${button.textContent}? Local observations will be preserved.`)) return; button.disabled = true; try { await command({ action: button.dataset.action, id: button.dataset.id }); } catch (error) { notify(error.message, true); button.disabled = false; } });
  $("anchor-form").addEventListener("submit", async (event) => { event.preventDefault(); try { const robot = $("anchor-robot").value, reference = $("anchor-reference").value; if (!robot || !reference || robot === reference) throw new Error("Choose two different rover frames."); const transform = anchorTransform(...["x", "y", "z", "yaw"].map((axis) => Number($("anchor-" + axis).value))); if (!window.confirm(`Apply a SUPPLIED alignment from ${robot} into ${reference}? Confirm these are measured frame offsets, not guessed robot positions.`)) return; await command({ action: "anchor", robot_id: robot, reference_robot_id: reference, T_reference_local: transform }); } catch (error) { notify(error.message, true); } });
  const tick = async () => { await poll(); setTimeout(tick, 1000); }; tick();
  const images = () => { for (const [id, camera] of cameraState) refreshCamera(id, camera); setTimeout(images, 1000); }; images();
}

if (typeof module !== "undefined" && module.exports) module.exports = { robotId, escapeText, ageLabel, matrix, transformPoint, inverseRigid, matrixProduct, anchorTransform, normalizeSnapshot, buildCloud, candidateConstraint, activeEntityLink, modePresentation, imageAge, orderedCandidates, healthState, evidenceRenderKey };
if (typeof document !== "undefined") { if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", startDashboard); else startDashboard(); }
