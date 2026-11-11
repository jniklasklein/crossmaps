"use strict";

const appState = {
  config: null,
  params: {},
  activeParamGroup: null,
  selectedImage: null,
  selectedGrid: null,
  layerVisible: {},
  layerCounts: {},
  cloudStates: {},
  cloudError: "",
  lastStatus: null,
  pendingParams: {},
  paramTimers: {},
  stagedParams: {},
  liveApply: true,
  cameraStreamsEnabled: false,
  imageRequestSeq: 0,
  imageRequestController: null,
  lastImageError: "",
  lastStatusAt: 0,
  imageLargeMode: false,
  queryInputDirty: false,
  pendingQueryText: null,
  pendingQueryUntil: 0,
  viewer: null,
};

const $ = (id) => document.getElementById(id);

async function apiGet(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function apiPost(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || `${response.status} ${response.statusText}`);
  }
  return data;
}

async function apiDelete(path) {
  const response = await fetch(path, { method: "DELETE" });
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || `${response.status} ${response.statusText}`);
  }
  return data;
}

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

function fmtAge(age) {
  if (age === null || age === undefined) return "no data";
  if (age < 1) return `${Math.round(age * 1000)} ms`;
  if (age < 60) return `${age.toFixed(1)} s`;
  return `${Math.round(age)} s`;
}

function fmtCount(n) {
  const value = Number(n || 0);
  if (value >= 1000000) return `${(value / 1000000).toFixed(1)}m`;
  if (value >= 1000) return `${(value / 1000).toFixed(1)}k`;
  return String(value);
}

function timeLabel(t) {
  if (!t) return "";
  return new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function escapeText(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch];
  });
}

class GLViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext("webgl", { antialias: true, alpha: false });
    this.layers = new Map();
    this.lineSets = new Map();
    this.visible = {};
    this.pointSize = 4;
    this.pointOpacity = 0.95;
    this.colorBoost = 1.15;
    this.yaw = -0.72;
    this.pitch = 0.78;
    this.distance = 12;
    this.target = [0, 0, 0.8];
    this.robot = { available: false, position: [0, 0, 0], yaw_rad: 0 };
    this.bounds = null;
    this.drag = null;
    this.frameCount = 0;
    this.lastFpsT = performance.now();
    this.fps = 0;

    if (!this.gl) {
      $("viewer-frame").textContent = "WebGL unavailable";
      return;
    }

    this.program = this.createProgram(VERT_SHADER, FRAG_SHADER);
    this.locations = {
      position: this.gl.getAttribLocation(this.program, "aPosition"),
      color: this.gl.getAttribLocation(this.program, "aColor"),
      viewProj: this.gl.getUniformLocation(this.program, "uViewProj"),
      pointSize: this.gl.getUniformLocation(this.program, "uPointSize"),
      roundPoints: this.gl.getUniformLocation(this.program, "uRoundPoints"),
      opacity: this.gl.getUniformLocation(this.program, "uOpacity"),
      colorBoost: this.gl.getUniformLocation(this.program, "uColorBoost"),
    };

    this.installControls();
    this.createReferenceLines();
    window.addEventListener("resize", () => this.resize());
    this.resize();
    requestAnimationFrame(() => this.render());
  }

  createShader(type, source) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader) || "shader compile failed");
    }
    return shader;
  }

  createProgram(vertexSource, fragmentSource) {
    const gl = this.gl;
    const program = gl.createProgram();
    gl.attachShader(program, this.createShader(gl.VERTEX_SHADER, vertexSource));
    gl.attachShader(program, this.createShader(gl.FRAGMENT_SHADER, fragmentSource));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program) || "program link failed");
    }
    return program;
  }

  installControls() {
    const canvas = this.canvas;
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("pointerdown", (event) => {
      canvas.setPointerCapture(event.pointerId);
      this.drag = {
        pointerId: event.pointerId,
        x: event.clientX,
        y: event.clientY,
        mode: event.button === 2 || event.shiftKey ? "pan" : "orbit",
      };
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!this.drag || this.drag.pointerId !== event.pointerId) return;
      const dx = event.clientX - this.drag.x;
      const dy = event.clientY - this.drag.y;
      this.drag.x = event.clientX;
      this.drag.y = event.clientY;
      if (this.drag.mode === "pan") {
        this.pan(dx, dy);
      } else {
        this.yaw -= dx * 0.006;
        this.pitch = clamp(this.pitch + dy * 0.005, -1.35, 1.35);
      }
    });
    canvas.addEventListener("pointerup", (event) => {
      if (this.drag && this.drag.pointerId === event.pointerId) this.drag = null;
    });
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const scale = Math.exp(event.deltaY * 0.001);
      this.distance = clamp(this.distance * scale, 0.5, 250);
    }, { passive: false });
  }

  pan(dx, dy) {
    const eye = this.eye();
    const forward = normalize(sub(this.target, eye));
    const right = normalize(cross(forward, [0, 0, 1]));
    const up = normalize(cross(right, forward));
    const scale = this.distance / Math.max(240, this.canvas.clientHeight);
    const delta = add(scaleVec(right, -dx * scale), scaleVec(up, dy * scale));
    this.target = add(this.target, delta);
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const width = Math.max(2, Math.floor(rect.width * dpr));
    const height = Math.max(2, Math.floor(rect.height * dpr));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    if (this.gl) this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
  }

  createReferenceLines() {
    const grid = [];
    const gridColors = [];
    const size = 12;
    for (let i = -size; i <= size; i += 1) {
      pushLine(grid, [-size, i, 0], [size, i, 0]);
      pushLine(grid, [i, -size, 0], [i, size, 0]);
      const c = i === 0 ? [95, 95, 82] : [47, 49, 40];
      pushColorLine(gridColors, c, c);
      pushColorLine(gridColors, c, c);
    }
    this.setLineSet("grid", grid, gridColors);

    const axes = [];
    const colors = [];
    pushLine(axes, [0, 0, 0], [1.5, 0, 0]);
    pushColorLine(colors, [232, 95, 92], [232, 95, 92]);
    pushLine(axes, [0, 0, 0], [0, 1.5, 0]);
    pushColorLine(colors, [54, 182, 143], [54, 182, 143]);
    pushLine(axes, [0, 0, 0], [0, 0, 1.5]);
    pushColorLine(colors, [85, 167, 232], [85, 167, 232]);
    this.setLineSet("axes", axes, colors);
  }

  setVisible(id, visible) {
    this.visible[id] = visible;
  }

  setPointSize(value) {
    this.pointSize = Number(value) || 3;
  }

  setPointOpacity(value) {
    this.pointOpacity = clamp(Number(value) || 0.95, 0.1, 1);
  }

  setColorBoost(value) {
    this.colorBoost = clamp(Number(value) || 1, 0.5, 5);
  }

  setPointLayer(id, positions, colors) {
    if (!this.gl) return;
    const gl = this.gl;
    const count = Math.floor(positions.length / 3);
    let layer = this.layers.get(id);
    if (!layer) {
      layer = {
        positionBuffer: gl.createBuffer(),
        colorBuffer: gl.createBuffer(),
        count: 0,
      };
      this.layers.set(id, layer);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, layer.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, layer.colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(colors), gl.DYNAMIC_DRAW);
    layer.count = count;
    this.recomputeBounds();
  }

  setLineSet(id, positions, colors) {
    if (!this.gl) return;
    const gl = this.gl;
    let set = this.lineSets.get(id);
    if (!set) {
      set = {
        positionBuffer: gl.createBuffer(),
        colorBuffer: gl.createBuffer(),
        count: 0,
      };
      this.lineSets.set(id, set);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, set.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, set.colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(colors), gl.DYNAMIC_DRAW);
    set.count = Math.floor(positions.length / 3);
  }

  setGraph(graph) {
    const linePositions = [];
    const lineColors = [];
    for (const line of graph.lines || []) {
      const points = line.points || [];
      const color = line.color || [167, 124, 255];
      if (points.length < 6) continue;
      if (line.type === 5) {
        for (let i = 0; i + 5 < points.length; i += 6) {
          pushLine(linePositions, points.slice(i, i + 3), points.slice(i + 3, i + 6));
          pushColorLine(lineColors, color, color);
        }
      } else {
        for (let i = 0; i + 5 < points.length; i += 3) {
          pushLine(linePositions, points.slice(i, i + 3), points.slice(i + 3, i + 6));
          pushColorLine(lineColors, color, color);
        }
      }
    }
    this.setLineSet("graph", linePositions, lineColors);

    const nodePositions = [];
    const nodeColors = [];
    for (const node of graph.nodes || []) {
      nodePositions.push(...node.position);
      nodeColors.push(...(node.color || [167, 124, 255]));
    }
    layerPositionCache.set("graph_nodes", nodePositions);
    this.setPointLayer("graph_nodes", nodePositions, nodeColors);
  }

  setRobot(robot) {
    this.robot = robot || { available: false, position: [0, 0, 0], yaw_rad: 0 };
    const pos = this.robot.position || [0, 0, 0];
    const yaw = Number(this.robot.yaw_rad || 0);
    const lengthM = 0.58;
    const widthM = 0.42;
    const z = Number(pos[2] || 0) + 0.06;
    const local = [
      [lengthM / 2, widthM / 2, z],
      [lengthM / 2, -widthM / 2, z],
      [-lengthM / 2, -widthM / 2, z],
      [-lengthM / 2, widthM / 2, z],
    ];
    const world = local.map((p) => rotateTranslate2d(p, pos, yaw));
    const lines = [];
    const colors = [];
    const bodyColor = this.robot.available ? [242, 193, 78] : [120, 116, 94];
    for (let i = 0; i < 4; i += 1) {
      pushLine(lines, world[i], world[(i + 1) % 4]);
      pushColorLine(colors, bodyColor, bodyColor);
    }
    const nose = rotateTranslate2d([lengthM * 0.72, 0, z], pos, yaw);
    pushLine(lines, [pos[0], pos[1], z], nose);
    pushColorLine(colors, [255, 230, 130], [255, 230, 130]);
    this.setLineSet("rover", lines, colors);
    layerPositionCache.set("rover_center", [pos[0], pos[1], z]);
    this.setPointLayer("rover_center", [pos[0], pos[1], z], bodyColor);
  }

  recomputeBounds() {
    const bounds = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity], count: 0 };
    for (const [id, layer] of this.layers.entries()) {
      if (layer.count <= 0 || this.visible[id] === false) continue;
      const positions = getLayerCache(id);
      if (!positions) continue;
      for (let i = 0; i < positions.length; i += 3) {
        bounds.min[0] = Math.min(bounds.min[0], positions[i]);
        bounds.min[1] = Math.min(bounds.min[1], positions[i + 1]);
        bounds.min[2] = Math.min(bounds.min[2], positions[i + 2]);
        bounds.max[0] = Math.max(bounds.max[0], positions[i]);
        bounds.max[1] = Math.max(bounds.max[1], positions[i + 1]);
        bounds.max[2] = Math.max(bounds.max[2], positions[i + 2]);
        bounds.count += 1;
      }
    }
    this.bounds = bounds.count > 0 ? bounds : null;
  }

  resetView() {
    this.recomputeBounds();
    if (!this.bounds) {
      this.target = [0, 0, 0.8];
      this.distance = 12;
      this.yaw = -0.72;
      this.pitch = 0.78;
      return;
    }
    const min = this.bounds.min;
    const max = this.bounds.max;
    this.target = [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5, (min[2] + max[2]) * 0.5];
    const radius = Math.max(2, length(sub(max, min)) * 0.55);
    this.distance = clamp(radius * 2.2, 3, 120);
    this.yaw = -0.72;
    this.pitch = 0.78;
  }

  topView() {
    this.recomputeBounds();
    if (this.bounds) {
      const min = this.bounds.min;
      const max = this.bounds.max;
      this.target = [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5, (min[2] + max[2]) * 0.5];
      this.distance = clamp(Math.max(max[0] - min[0], max[1] - min[1], 4) * 1.25, 4, 180);
    }
    this.yaw = -Math.PI / 2;
    this.pitch = 1.34;
  }

  centerOnRover() {
    const pos = this.robot?.position || [0, 0, 0];
    this.target = [Number(pos[0] || 0), Number(pos[1] || 0), Number(pos[2] || 0) + 0.35];
    this.distance = clamp(this.distance, 3, 18);
  }

  eye() {
    const cp = Math.cos(this.pitch);
    return [
      this.target[0] + this.distance * cp * Math.cos(this.yaw),
      this.target[1] + this.distance * cp * Math.sin(this.yaw),
      this.target[2] + this.distance * Math.sin(this.pitch),
    ];
  }

  drawBuffer(set, mode, viewProj, pointSize, roundPoints) {
    const gl = this.gl;
    if (!set || set.count <= 0) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, set.positionBuffer);
    gl.enableVertexAttribArray(this.locations.position);
    gl.vertexAttribPointer(this.locations.position, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, set.colorBuffer);
    gl.enableVertexAttribArray(this.locations.color);
    gl.vertexAttribPointer(this.locations.color, 3, gl.UNSIGNED_BYTE, true, 0, 0);
    gl.uniformMatrix4fv(this.locations.viewProj, false, viewProj);
    gl.uniform1f(this.locations.pointSize, pointSize);
    gl.uniform1f(this.locations.roundPoints, roundPoints ? 1 : 0);
    gl.uniform1f(this.locations.opacity, this.pointOpacity);
    gl.uniform1f(this.locations.colorBoost, this.colorBoost);
    gl.drawArrays(mode, 0, set.count);
  }

  render() {
    if (!this.gl) return;
    this.resize();
    const gl = this.gl;
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(0.063, 0.067, 0.058, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.program);

    const aspect = this.canvas.width / Math.max(1, this.canvas.height);
    const proj = mat4Perspective(Math.PI / 4, aspect, 0.03, 600);
    const view = mat4LookAt(this.eye(), this.target, [0, 0, 1]);
    const viewProj = mat4Multiply(proj, view);

    this.drawBuffer(this.lineSets.get("grid"), gl.LINES, viewProj, 1, false);
    this.drawBuffer(this.lineSets.get("axes"), gl.LINES, viewProj, 1, false);

    if (this.visible.graph !== false) {
      this.drawBuffer(this.lineSets.get("graph"), gl.LINES, viewProj, 1, false);
    }
    if (this.visible.rover !== false) {
      this.drawBuffer(this.lineSets.get("rover"), gl.LINES, viewProj, 1, false);
    }

    let visiblePoints = 0;
    for (const [id, layer] of this.layers.entries()) {
      if (id === "graph_nodes" && this.visible.graph === false) continue;
      if (id === "rover_center" && this.visible.rover === false) continue;
      if (id !== "graph_nodes" && this.visible[id] === false) continue;
      visiblePoints += layer.count;
      const size = id === "graph_nodes" || id === "rover_center" ? Math.max(7, this.pointSize + 3) : this.pointSize;
      this.drawBuffer(layer, gl.POINTS, viewProj, size, true);
    }

    this.frameCount += 1;
    const now = performance.now();
    if (now - this.lastFpsT > 700) {
      this.fps = Math.round((this.frameCount * 1000) / (now - this.lastFpsT));
      this.frameCount = 0;
      this.lastFpsT = now;
      $("viewer-count").textContent = `${fmtCount(visiblePoints)} pts`;
      $("viewer-frame").textContent = `${this.fps} fps`;
    }

    requestAnimationFrame(() => this.render());
  }
}

const layerPositionCache = new Map();
function getLayerCache(id) {
  return layerPositionCache.get(id);
}

function pushLine(out, a, b) {
  out.push(a[0], a[1], a[2], b[0], b[1], b[2]);
}

function pushColorLine(out, a, b) {
  out.push(a[0], a[1], a[2], b[0], b[1], b[2]);
}

function rotateTranslate2d(p, origin, yaw) {
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  return [
    Number(origin[0] || 0) + p[0] * c - p[1] * s,
    Number(origin[1] || 0) + p[0] * s + p[1] * c,
    p[2],
  ];
}

function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scaleVec(v, s) {
  return [v[0] * s, v[1] * s, v[2] * s];
}

function length(v) {
  return Math.hypot(v[0], v[1], v[2]);
}

function normalize(v) {
  const n = length(v) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function mat4Perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

function mat4LookAt(eye, center, up) {
  const z = normalize(sub(eye, center));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
}

function mat4Multiply(a, b) {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i += 1) {
    for (let j = 0; j < 4; j += 1) {
      out[i + j * 4] =
        a[i] * b[j * 4] +
        a[i + 4] * b[j * 4 + 1] +
        a[i + 8] * b[j * 4 + 2] +
        a[i + 12] * b[j * 4 + 3];
    }
  }
  return out;
}

const VERT_SHADER = `
attribute vec3 aPosition;
attribute vec3 aColor;
uniform mat4 uViewProj;
uniform float uPointSize;
varying vec3 vColor;
void main() {
  gl_Position = uViewProj * vec4(aPosition, 1.0);
  gl_PointSize = uPointSize;
  vColor = aColor;
}`;

const FRAG_SHADER = `
precision mediump float;
uniform float uRoundPoints;
uniform float uOpacity;
uniform float uColorBoost;
varying vec3 vColor;
void main() {
  if (uRoundPoints > 0.5) {
    vec2 p = gl_PointCoord - vec2(0.5);
    if (dot(p, p) > 0.25) discard;
  }
  float boost = max(uColorBoost, 1.0);
  vec3 color = pow(max(vColor, vec3(0.0)), vec3(1.0 / boost));
  gl_FragColor = vec4(color, uOpacity);
}`;

async function init() {
  appState.viewer = new GLViewer($("viewer-canvas"));
  bindTopbar();

  try {
    appState.config = await apiGet("/api/config");
    $("target-node").textContent = appState.config.targetNode;
    appState.cameraStreamsEnabled = Boolean(appState.config.cameraStreamsEnabled);
    $("camera-streams-toggle").checked = appState.cameraStreamsEnabled;
    buildLayerControls();
    buildImageTabs();
    buildGridTabs();
    buildParamUI();
    updateStagedUI();
    await Promise.allSettled([refreshParams(true), loadPresets()]);
    startEventStream();
    startPolling();
    setConnection(true, "connected");
  } catch (err) {
    setConnection(false, err.message);
  }
}

function bindTopbar() {
  $("query-input").addEventListener("input", () => {
    appState.queryInputDirty = true;
  });
  $("query-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const value = $("query-input").value.trim();
    if (!value) return;
    appState.queryInputDirty = false;
    appState.pendingQueryText = value;
    appState.pendingQueryUntil = Date.now() + 6000;
    setQueryInputValue(value, { force: true, preservePending: true });
    try {
      await setParams({ query_text: value });
    } catch (err) {
      appState.pendingQueryText = null;
      appState.pendingQueryUntil = 0;
      throw err;
    }
  });
  $("refresh-params").addEventListener("click", () => refreshParams(true));
  $("reset-view").addEventListener("click", () => appState.viewer.resetView());
  $("point-size").addEventListener("input", (event) => appState.viewer.setPointSize(event.target.value));
  $("point-glow").addEventListener("input", (event) => appState.viewer.setColorBoost(event.target.value));
  $("point-opacity").addEventListener("input", (event) => appState.viewer.setPointOpacity(event.target.value));
  $("point-budget").addEventListener("change", () => fetchClouds());
  $("top-view").addEventListener("click", () => appState.viewer.topView());
  $("rover-view").addEventListener("click", () => appState.viewer.centerOnRover());
  $("live-apply-toggle").addEventListener("change", (event) => {
    appState.liveApply = event.target.checked;
    if (appState.liveApply) applyStagedParams();
    updateStagedUI();
  });
  $("apply-staged-params").addEventListener("click", () => applyStagedParams());
  $("camera-streams-toggle").addEventListener("change", (event) => setCameraStreams(event.target.checked));
  $("image-large-toggle").addEventListener("click", toggleImageLargeMode);
  $("image-fullscreen-toggle").addEventListener("click", toggleImageFullscreen);
  $("live-image").addEventListener("click", selectSimilarityPoint);
  window.addEventListener("resize", updateSimilarityMarker);
  document.addEventListener("fullscreenchange", updateImageViewModeUI);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && appState.imageLargeMode && !document.fullscreenElement) {
      setImageLargeMode(false);
    }
  });
  updateImageViewModeUI();
}

function setImageLargeMode(enabled) {
  appState.imageLargeMode = Boolean(enabled);
  document.body.classList.toggle("image-large-mode", appState.imageLargeMode);
  updateImageViewModeUI();
  window.requestAnimationFrame(updateSimilarityMarker);
}

function toggleImageLargeMode() {
  setImageLargeMode(!appState.imageLargeMode);
}

async function toggleImageFullscreen() {
  const section = document.querySelector(".media-section");
  if (!section || !document.fullscreenEnabled) return;
  try {
    if (document.fullscreenElement === section) {
      await document.exitFullscreen();
    } else {
      await section.requestFullscreen();
    }
  } catch (err) {
    console.warn("Failed to change image fullscreen mode", err);
  }
}

function updateImageViewModeUI() {
  const section = document.querySelector(".media-section");
  const fullscreen = Boolean(section && document.fullscreenElement === section);
  const largeButton = $("image-large-toggle");
  const fullscreenButton = $("image-fullscreen-toggle");
  largeButton.classList.toggle("active", appState.imageLargeMode);
  largeButton.textContent = appState.imageLargeMode ? "Restore" : "Large";
  fullscreenButton.classList.toggle("active", fullscreen);
  fullscreenButton.textContent = fullscreen ? "Exit Full Screen" : "Full Screen";
  fullscreenButton.disabled = !document.fullscreenEnabled;
  window.requestAnimationFrame(updateSimilarityMarker);
}

async function selectSimilarityPoint(event) {
  const img = $("live-image");
  const rect = containedImageRect(img);
  if (rect.width <= 0 || rect.height <= 0) return;
  if (
    event.clientX < rect.left || event.clientX > rect.right ||
    event.clientY < rect.top || event.clientY > rect.bottom
  ) return;
  const x = clamp((event.clientX - rect.left) / rect.width, 0, 1);
  const y = clamp((event.clientY - rect.top) / rect.height, 0, 1);
  appState.params.diagnostic_similarity_x = x;
  appState.params.diagnostic_similarity_y = y;
  updateSimilarityMarker();
  try {
    await setParams({ diagnostic_similarity_x: x, diagnostic_similarity_y: y });
  } catch (err) {
    console.warn("Failed to update feature similarity selection", err);
  }
}

function containedImageRect(img) {
  const box = img.getBoundingClientRect();
  const naturalWidth = Number(img.naturalWidth || 0);
  const naturalHeight = Number(img.naturalHeight || 0);
  if (box.width <= 0 || box.height <= 0 || naturalWidth <= 0 || naturalHeight <= 0) {
    return {
      left: box.left,
      top: box.top,
      right: box.right,
      bottom: box.bottom,
      width: box.width,
      height: box.height,
    };
  }
  const scale = Math.min(box.width / naturalWidth, box.height / naturalHeight);
  const width = naturalWidth * scale;
  const height = naturalHeight * scale;
  const left = box.left + 0.5 * (box.width - width);
  const top = box.top + 0.5 * (box.height - height);
  return { left, top, right: left + width, bottom: top + height, width, height };
}

function updateSimilarityMarker() {
  const img = $("live-image");
  const marker = $("similarity-marker");
  const frame = img.parentElement;
  if (!frame || img.style.display === "none" || !img.getAttribute("src")) {
    marker.style.display = "none";
    return;
  }
  const imageRect = containedImageRect(img);
  const frameRect = frame.getBoundingClientRect();
  if (imageRect.width <= 0 || imageRect.height <= 0) {
    marker.style.display = "none";
    return;
  }
  const x = clamp(Number(appState.params.diagnostic_similarity_x ?? 0.5), 0, 1);
  const y = clamp(Number(appState.params.diagnostic_similarity_y ?? 0.5), 0, 1);
  marker.style.left = `${imageRect.left - frameRect.left + x * imageRect.width}px`;
  marker.style.top = `${imageRect.top - frameRect.top + y * imageRect.height}px`;
  marker.style.display = "block";
}

function setQueryInputValue(value, options = {}) {
  const input = $("query-input");
  const text = String(value ?? "");
  const force = Boolean(options.force);
  if (!force) {
    const now = Date.now();
    if (appState.pendingQueryText && text !== appState.pendingQueryText && now < appState.pendingQueryUntil) {
      return;
    }
    if (document.activeElement === input && appState.queryInputDirty && input.value !== text) {
      return;
    }
  }
  if (!options.preservePending && appState.pendingQueryText && text === appState.pendingQueryText) {
    appState.pendingQueryText = null;
    appState.pendingQueryUntil = 0;
  }
  input.value = text;
}

function setConnection(online, text) {
  const dot = $("connection-dot");
  dot.classList.toggle("online", online);
  dot.classList.toggle("offline", !online);
  $("connection-text").textContent = text;
}

function buildLayerControls() {
  const root = $("layer-controls");
  root.innerHTML = "";
  for (const layer of appState.config.cloudLayers) {
    appState.layerVisible[layer.id] = Boolean(layer.defaultVisible);
    appState.viewer.setVisible(layer.id, appState.layerVisible[layer.id]);
    root.appendChild(layerButton(layer.id, layer.label, layer.color));
  }
  appState.layerVisible.graph = true;
  appState.viewer.setVisible("graph", true);
  root.appendChild(layerButton("graph", "Graph", [167, 124, 255]));
  appState.layerVisible.rover = true;
  appState.viewer.setVisible("rover", true);
  root.appendChild(layerButton("rover", "Rover", [242, 193, 78]));
}

function layerButton(id, label, color) {
  const button = document.createElement("button");
  button.className = `layer-toggle ${appState.layerVisible[id] ? "active" : ""}`;
  button.dataset.layer = id;
  button.setAttribute("aria-pressed", String(Boolean(appState.layerVisible[id])));
  const layer = appState.config.cloudLayers.find(item => item.id === id);
  button.title = layer?.description || label;
  button.innerHTML = `<span class="swatch" style="background: rgb(${color.join(",")})"></span><span>${escapeText(label)}</span>`;
  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      await toggleCloudLayer(id);
    } finally {
      button.disabled = false;
    }
  });
  return button;
}

async function toggleCloudLayer(id) {
  const enabled = !appState.layerVisible[id];
  const layer = appState.config.cloudLayers.find(item => item.id === id);
  appState.cloudError = "";
  try {
    if (enabled && layer?.publishParam) {
      const disabled = (layer.requiresParams || []).filter(name => appState.params[name] === false);
      if (disabled.length) throw new Error(`Enable ${disabled.join(", ")} in Parameters first.`);
      // A view click is an explicit publication request, even with Live Apply off.
      // Do not switch on model/memory processing or turn off other consumers.
      await setParams({ [layer.publishParam]: true });
      // Co-located point clouds otherwise obscure each other's ID colors.
      for (const other of appState.config.cloudLayers) appState.layerVisible[other.id] = false;
    }
    appState.layerVisible[id] = enabled;
    for (const [key, visible] of Object.entries(appState.layerVisible)) {
      appState.viewer.setVisible(key, visible);
    }
    for (const button of $("layer-controls").querySelectorAll("button")) {
      const visible = Boolean(appState.layerVisible[button.dataset.layer]);
      button.classList.toggle("active", visible);
      button.setAttribute("aria-pressed", String(visible));
    }
    appState.viewer.recomputeBounds();
  } catch (err) {
    appState.cloudError = `${layer?.label || id}: ${err.message}`;
  }
  updateCloudNotice();
}

function updateCloudNotice() {
  const messages = appState.cloudError ? [appState.cloudError] : [];
  for (const layer of appState.config.cloudLayers) {
    if (!layer.publishParam || !appState.layerVisible[layer.id]) continue;
    const state = appState.cloudStates[layer.id] || {};
    let detail;
    if (state.state === "disabled") detail = `disabled by ${(state.disabled_by || []).join(", ")}; enable in Parameters or reselect the layer`;
    else if (!state.available) detail = "waiting for publisher (restart both experimental node and dashboard after updating)";
    else if (state.state === "stale") detail = `stale (${fmtAge(state.age_s)}); displaying last received geometry`;
    else if (!state.count) detail = "no surfaces pass the 3D support/confidence/observation filters yet";
    else detail = `${fmtCount(state.count)} points · ${fmtAge(state.age_s)}`;
    if (layer.id === "entity_voxels") detail += " · gray: unassigned · amber: ambiguous · colors indicate hypotheses, not verified whole objects";
    messages.push(`${layer.label}: ${detail}`);
  }
  const notice = $("cloud-notice");
  notice.textContent = messages.join("\n");
  notice.hidden = messages.length === 0;
}

function buildImageTabs() {
  const root = $("image-tabs");
  root.innerHTML = "";
  for (const layer of appState.config.imageLayers) {
    const button = document.createElement("button");
    button.className = "mini-tab";
    button.dataset.image = layer.id;
    button.textContent = layer.label;
    button.addEventListener("click", () => selectImage(layer.id));
    root.appendChild(button);
    if (!appState.selectedImage && layer.defaultVisible) appState.selectedImage = layer.id;
  }
  selectImage(appState.selectedImage || appState.config.imageLayers[0]?.id);
}

function selectImage(id) {
  appState.imageRequestController?.abort();
  appState.imageRequestController = null;
  appState.imageRequestSeq++;
  appState.lastImageError = "";
  appState.selectedImage = id;
  const caption = $("image-caption");
  if (caption) {
    caption.hidden = !["segment_projection", "entity_projection"].includes(id);
    caption.textContent = id === "entity_projection"
      ? "Entity hypotheses · dominant color per mask · dashed amber / ?: provisional · raw/mixed evidence under Segments"
      : "Surface IDs · dominant color per mask · dashed amber / ?: provisional · gray: unassigned · raw/mixed evidence under Segments";
  }
  clearLiveImage();
  for (const button of $("image-tabs").querySelectorAll("button")) {
    button.classList.toggle("active", button.dataset.image === id);
  }
  refreshImage();
  updateImageHealth();
}

function clearLiveImage() {
  const img = $("live-image");
  const oldUrl = img.dataset.objectUrl || "";
  if (oldUrl) URL.revokeObjectURL(oldUrl);
  delete img.dataset.objectUrl;
  img.removeAttribute("src");
  img.style.display = "none";
  $("similarity-marker").style.display = "none";
  $("image-empty").style.display = "block";
}

function buildGridTabs() {
  const root = $("grid-tabs");
  root.innerHTML = "";
  for (const layer of appState.config.gridLayers) {
    const button = document.createElement("button");
    button.className = "mini-tab";
    button.dataset.grid = layer.id;
    button.textContent = layer.label;
    button.addEventListener("click", () => selectGrid(layer.id));
    root.appendChild(button);
  }
  selectGrid(appState.config.gridLayers[0]?.id);
}

function selectGrid(id) {
  appState.selectedGrid = id;
  for (const button of $("grid-tabs").querySelectorAll("button")) {
    button.classList.toggle("active", button.dataset.grid === id);
  }
  refreshGrid();
}

function buildParamUI() {
  const tabs = $("param-tabs");
  const groups = $("param-groups");
  tabs.innerHTML = "";
  groups.innerHTML = "";
  appState.activeParamGroup = appState.config.paramGroups[0]?.id;

  for (const group of appState.config.paramGroups) {
    const tab = document.createElement("button");
    tab.className = `tab-button ${group.id === appState.activeParamGroup ? "active" : ""}`;
    tab.dataset.group = group.id;
    tab.textContent = group.label;
    tab.addEventListener("click", () => activateParamGroup(group.id));
    tabs.appendChild(tab);

    const section = document.createElement("section");
    section.className = `param-group ${group.id === appState.activeParamGroup ? "active" : ""}`;
    section.dataset.group = group.id;
    for (const spec of group.params) {
      section.appendChild(renderParamRow(spec));
    }
    groups.appendChild(section);
  }
}

function activateParamGroup(id) {
  appState.activeParamGroup = id;
  for (const button of $("param-tabs").querySelectorAll(".tab-button")) {
    button.classList.toggle("active", button.dataset.group === id);
  }
  for (const section of $("param-groups").querySelectorAll(".param-group")) {
    section.classList.toggle("active", section.dataset.group === id);
  }
}

function renderParamRow(spec) {
  const row = document.createElement("div");
  row.className = "param-row";
  row.dataset.param = spec.name;
  if (spec.description) row.title = spec.description;
  const label = document.createElement("label");
  label.textContent = spec.label || spec.name;
  row.appendChild(label);

  const control = document.createElement("div");
  control.className = "param-control";
  row.appendChild(control);

  if (spec.type === "bool") {
    control.classList.add("single");
    const toggle = document.createElement("label");
    toggle.className = "toggle";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.addEventListener("change", () => stageOrSendParam(spec.name, input.checked));
    toggle.appendChild(input);
    control.appendChild(toggle);
  } else if (spec.control === "select") {
    control.classList.add("single");
    const select = document.createElement("select");
    for (const choice of spec.choices || []) {
      const option = document.createElement("option");
      option.value = choice;
      option.textContent = choice;
      select.appendChild(option);
    }
    select.addEventListener("change", () => stageOrSendParam(spec.name, select.value));
    control.appendChild(select);
  } else if (spec.type === "str") {
    control.classList.add("single");
    const input = document.createElement("input");
    input.type = "text";
    input.autocomplete = "off";
    input.spellcheck = false;
    input.addEventListener("change", () => stageOrSendParam(spec.name, input.value));
    control.appendChild(input);
  } else {
    const range = document.createElement("input");
    range.type = "range";
    range.min = spec.min ?? 0;
    range.max = spec.max ?? 1;
    range.step = spec.step ?? (spec.type === "int" ? 1 : 0.01);
    const number = document.createElement("input");
    number.type = "number";
    number.min = range.min;
    number.max = range.max;
    number.step = range.step;
    range.addEventListener("input", () => {
      number.value = range.value;
      stageOrDebounceParam(spec.name, coerceControlValue(spec, range.value));
    });
    number.addEventListener("change", () => {
      range.value = number.value;
      stageOrSendParam(spec.name, coerceControlValue(spec, number.value));
    });
    control.appendChild(range);
    control.appendChild(number);
  }
  return row;
}

function coerceControlValue(spec, value) {
  if (spec.type === "int") return Math.round(Number(value));
  if (spec.type === "float") return Number(value);
  if (spec.type === "bool") return Boolean(value);
  return String(value);
}

function debounceParam(name, value) {
  appState.pendingParams[name] = value;
  window.clearTimeout(appState.paramTimers[name]);
  appState.paramTimers[name] = window.setTimeout(() => {
    const payload = { [name]: appState.pendingParams[name] };
    delete appState.pendingParams[name];
    sendParams(payload);
  }, 260);
}

function sendParam(name, value) {
  return sendParams({ [name]: value });
}

function stageOrSendParam(name, value) {
  if (appState.liveApply) return sendParam(name, value);
  appState.stagedParams[name] = value;
  updateLocalParamValue(name, value);
  updateStagedUI();
  return Promise.resolve();
}

function stageOrDebounceParam(name, value) {
  if (appState.liveApply) {
    debounceParam(name, value);
    return;
  }
  appState.stagedParams[name] = value;
  updateLocalParamValue(name, value);
  updateStagedUI();
}

function updateLocalParamValue(name, value) {
  appState.params[name] = value;
}

async function applyStagedParams() {
  const staged = { ...appState.stagedParams };
  if (Object.keys(staged).length === 0) return;
  appState.stagedParams = {};
  updateStagedUI();
  await sendParams(staged);
}

function updateStagedUI() {
  const count = Object.keys(appState.stagedParams).length;
  $("staged-count").textContent = `${count} staged`;
  $("apply-staged-params").disabled = count === 0;
  for (const row of document.querySelectorAll(".param-row")) {
    row.classList.toggle("staged", Object.prototype.hasOwnProperty.call(appState.stagedParams, row.dataset.param));
  }
}

async function setParams(values) {
  return sendParams(values);
}

async function sendParams(values) {
  try {
    const data = await apiPost("/api/params", { updates: values });
    const rejected = [];
    for (const [name, value] of Object.entries(values)) {
      if (data.results?.[name]?.ok === true) {
        appState.params[name] = value;
        delete appState.stagedParams[name];
      } else {
        rejected.push(`${name}: ${data.results?.[name]?.reason || "not acknowledged"}`);
      }
    }
    updateParamValues();
    updateStagedUI();
    if (rejected.length) throw new Error(rejected.join("; "));
    if (values.query_text !== undefined) setQueryInputValue(values.query_text, { force: true, preservePending: true });
    return data;
  } catch (err) {
    setConnection(false, err.message);
    throw err;
  }
}

async function refreshParams(refresh) {
  const data = await apiGet(`/api/params${refresh ? "?refresh=1" : ""}`);
  appState.params = data.params || {};
  Object.assign(appState.params, appState.stagedParams);
  updateParamValues();
  updateStagedUI();
  if (appState.params.query_text !== undefined) {
    setQueryInputValue(appState.params.query_text);
  }
}

async function setCameraStreams(enabled) {
  try {
    const data = await apiPost("/api/camera_streams", { enabled: Boolean(enabled) });
    appState.cameraStreamsEnabled = Boolean(data.enabled);
    $("camera-streams-toggle").checked = appState.cameraStreamsEnabled;
    await fetchStatus();
  } catch (err) {
    $("camera-streams-toggle").checked = appState.cameraStreamsEnabled;
    setConnection(false, err.message);
  }
}

function updateParamValues() {
  for (const group of appState.config.paramGroups) {
    for (const spec of group.params) {
      const value = appState.params[spec.name];
      const row = document.querySelector(`.param-row[data-param="${spec.name}"]`);
      if (!row || value === undefined || value === null) continue;
      if (spec.type === "bool") {
        const input = row.querySelector("input[type='checkbox']");
        if (input) input.checked = Boolean(value);
      } else if (spec.control === "select") {
        const select = row.querySelector("select");
        if (select) select.value = String(value);
      } else if (spec.type === "str") {
        const input = row.querySelector("input[type='text']");
        if (input && document.activeElement !== input) input.value = String(value);
      } else {
        const range = row.querySelector("input[type='range']");
        const number = row.querySelector("input[type='number']");
        const numeric = Number(value);
        if (range && document.activeElement !== range) range.value = String(numeric);
        if (number && document.activeElement !== number) number.value = String(numeric);
      }
    }
  }
  updateSimilarityMarker();
}

async function loadPresets() {
  const payload = await apiGet("/api/presets");
  renderPresets(payload.presets || []);
}

function collectPresetValues() {
  const values = {};
  for (const group of appState.config.paramGroups) {
    for (const spec of group.params) {
      if (appState.params[spec.name] !== undefined) values[spec.name] = appState.params[spec.name];
    }
  }
  Object.assign(values, appState.pendingParams, appState.stagedParams);
  return values;
}

function selectedPresetName() {
  return $("preset-name").value.trim() || $("preset-select").value.trim();
}

function renderPresets(presets) {
  const select = $("preset-select");
  const nameInput = $("preset-name");
  const previous = select.value || nameInput.value.trim();
  select.innerHTML = "";
  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = "configs";
  select.appendChild(empty);
  for (const preset of presets) {
    const option = document.createElement("option");
    option.value = preset.name;
    option.textContent = preset.name;
    select.appendChild(option);
  }
  if (previous && presets.some((preset) => preset.name === previous)) {
    select.value = previous;
    nameInput.value = previous;
  }

  select.onchange = () => {
    nameInput.value = select.value;
  };
  $("preset-apply").onclick = async () => {
    const name = selectedPresetName();
    if (!name) return;
    await apiPost("/api/presets/apply", { name });
    select.value = name;
    nameInput.value = name;
    await refreshParams(true);
  };
  $("preset-save").onclick = async () => {
    const name = selectedPresetName();
    if (!name) {
      nameInput.focus();
      return;
    }
    const values = collectPresetValues();
    const payload = await apiPost("/api/presets", { name, values });
    renderPresets(payload.presets || []);
    $("preset-select").value = name;
    nameInput.value = name;
  };
  $("preset-delete").onclick = async () => {
    const name = select.value || nameInput.value.trim();
    if (!name) return;
    const payload = await apiDelete(`/api/presets?name=${encodeURIComponent(name)}`);
    renderPresets(payload.presets || []);
    nameInput.value = "";
  };
}

function startEventStream() {
  if (!window.EventSource) {
    window.setInterval(fetchStatus, 1200);
    return;
  }
  const source = new EventSource("/api/events");
  source.onmessage = (event) => {
    try {
      updateStatus(JSON.parse(event.data));
      setConnection(true, "connected");
    } catch (err) {
      setConnection(false, err.message);
    }
  };
  source.onerror = () => {
    setConnection(false, "reconnecting");
  };
}

function startPolling() {
  fetchStatus();
  fetchClouds();
  fetchGraph();
  refreshImage();
  refreshGrid();
  window.setInterval(fetchClouds, 1100);
  window.setInterval(fetchGraph, 1600);
  window.setInterval(refreshImage, 350);
  window.setInterval(refreshGrid, 1800);
  window.setInterval(() => refreshParams(false).catch(() => {}), 4000);
}

async function fetchStatus() {
  try {
    updateStatus(await apiGet("/api/status?light=1"));
  } catch (err) {
    setConnection(false, err.message);
  }
}

function updateStatus(status) {
  appState.lastStatus = status;
  appState.lastStatusAt = Date.now();
  updateImageHealth();
  if (status.active_query) {
    setQueryInputValue(status.active_query);
  }
  updateTopicStrip(status.topics || {});
  updateEvents(status.events || []);
  if (status.graph) {
    $("graph-node-count").textContent = `${status.graph.node_count || 0} nodes`;
    $("graph-edge-count").textContent = `${status.graph.edge_count || 0} edges`;
  }
  updateObservationCycle(status.observation_cycle);
  updateBattery(status.battery);
  if (status.robot) appState.viewer.setRobot(status.robot);
}

function updateObservationCycle(cycle) {
  const el = $("cycle-status");
  if (!cycle || cycle.average_ms === null || cycle.average_ms === undefined) {
    el.textContent = "Cycle --";
    el.title = "No completed observation frame received";
    return;
  }
  const avg = Number(cycle.average_ms);
  const hz = Number(cycle.average_hz || 0);
  const stale = Number(cycle.age_s || 0) > Math.max(10, 3*avg/1000);
  el.textContent = stale ? `Cycle STALE · ${fmtAge(cycle.age_s)}` : `Cycle ${avg.toFixed(0)} ms`;
  el.title = [
    `total average: ${avg.toFixed(2)} ms (${hz.toFixed(2)} Hz)`,
    `recent average: ${Number(cycle.rolling_average_ms || avg).toFixed(2)} ms`,
    `last: ${Number(cycle.last_ms || 0).toFixed(2)} ms`,
    `p95: ${Number(cycle.p95_ms || 0).toFixed(2)} ms`,
    `window: ${cycle.window_count || 0}/${cycle.window_size || 0}`,
    `mode: ${cycle.mode || "unknown"}`,
    `age: ${fmtAge(cycle.age_s)}`,
  ].join(" / ");
}

function updateBattery(battery) {
  const el = $("battery-status");
  el.classList.remove("good", "low", "critical");
  if (!battery) {
    el.textContent = "Battery --";
    el.title = "No BatteryState topic seen";
    return;
  }
  const pct = battery.percentage === null || battery.percentage === undefined ? null : Math.round(battery.percentage * 100);
  const volts = battery.voltage === null || battery.voltage === undefined ? "" : ` / ${Number(battery.voltage).toFixed(1)} V`;
  el.textContent = pct === null ? `Battery${volts}` : `Battery ${pct}%${volts}`;
  el.title = `${battery.topic || "battery"} / ${fmtAge(battery.age_s)}`;
  if (pct !== null && pct <= 15) el.classList.add("critical");
  else if (pct !== null && pct <= 35) el.classList.add("low");
  else el.classList.add("good");
}

function updateTopicStrip(topics) {
  const root = $("topic-strip");
  const ids = [
    "camera_rgb",
    "esam_overlay",
    "voxel_fused",
    "object_query",
    "voxel_stm",
    "voxel_ltm",
    "segment_projection",
    "segment_voxels",
    "entity_voxels",
  ];
  root.innerHTML = "";
  for (const id of ids) {
    const topic = topics[id];
    const div = document.createElement("div");
    div.className = "topic-pill";
    const count = topic?.count !== undefined ? ` / ${fmtCount(topic.count)}` : "";
    const source = topic?.source_topic ? ` / ${topic.source_topic}` : "";
    div.innerHTML = `<strong>${escapeText(topic?.label || id)}</strong><span>${fmtAge(topic?.age_s)}${count}${escapeText(source)}</span>`;
    root.appendChild(div);
  }
}

function updateEvents(events) {
  const root = $("event-log");
  root.innerHTML = "";
  for (const event of events.slice().reverse()) {
    const row = document.createElement("div");
    row.className = `event ${event.level || "info"}`;
    row.innerHTML = `<time>${timeLabel(event.t)}</time><div class="event-message">${escapeText(event.message)}${event.names ? `: ${escapeText(event.names)}` : ""}</div>`;
    root.appendChild(row);
  }
}

async function fetchClouds() {
  try {
    const budget = Number($("point-budget").value || 16000);
    const payload = await apiGet(`/api/clouds?max_points=${budget}`);
    let firstFrame = "";
    for (const layer of payload.layers || []) {
      appState.cloudStates[layer.id] = layer;
      appState.layerCounts[layer.id] = layer.count || 0;
      if (layer.available) {
        layerPositionCache.set(layer.id, layer.positions);
        appState.viewer.setPointLayer(layer.id, layer.positions, layer.colors);
        firstFrame = firstFrame || layer.frame_id || "";
      } else {
        layerPositionCache.set(layer.id, []);
        appState.viewer.setPointLayer(layer.id, [], []);
      }
    }
    if (payload.graph) appState.viewer.setGraph(payload.graph);
    if (payload.robot) appState.viewer.setRobot(payload.robot);
    if (firstFrame) $("viewer-frame").textContent = firstFrame;
    updateCloudNotice();
  } catch (err) {
    setConnection(false, err.message);
  }
}

async function fetchGraph() {
  try {
    const payload = await apiGet("/api/graph");
    renderGraph(payload.descriptor || {});
  } catch (err) {
    // Graph can legitimately be absent while CrossMaps is warming up.
  }
}

function renderGraph(descriptor) {
  const root = $("graph-list");
  const evidenceOpen = Boolean(root.querySelector?.("#grouping-evidence")?.open);
  const attributionOpen = Boolean(root.querySelector?.("#mask-evidence")?.open);
  const nodes = Array.isArray(descriptor.nodes) ? descriptor.nodes.slice() : [];
  nodes.sort((a, b) => Number(b.query_score || 0) - Number(a.query_score || 0));
  root.innerHTML = "";
  const identity = descriptor.identity;
  if (identity) {
    const summary = document.createElement("div");
    summary.className = "identity-summary";
    summary.textContent = `${identity.entities || 0} hypotheses · ${identity.pending_segments || 0} pending · ${identity.ambiguous_segments || 0} ambiguous · update ${Number(identity.update_ms || 0).toFixed(1)} ms`;
    if (identity.whole_object_grouping === false) summary.textContent += " · surface continuity only; object grouping disabled";
    summary.title = "Persistent IDs are tracking hypotheses, not verified whole objects. Gray in Entities: unassigned; amber: ambiguous. Rooms and semantic part-of inference are not enabled.";
    root.appendChild(summary);
    const assembly = identity.assembly;
    if (assembly && assembly.enabled) {
      const settings = document.createElement("div");
      settings.className = "identity-summary";
      settings.textContent = `DINO grouping: ${assembly.joined_hypotheses || 0} joins · sensitivity ${Number(assembly.sensitivity).toFixed(2)} · DINO ≥ ${Number(assembly.min_dino_cos).toFixed(2)} · gap ≤ ${Number(assembly.max_gap_m).toFixed(2)} m${assembly.budget_limited ? " · pair budget reached" : ""}`;
      settings.title = "Tune Identity → Grouping Sensitivity. Higher accepts lower DINO similarity and larger gaps. Actual surface contact, repeated observations and object-size safeguards still apply. Groups are reversible hypotheses; no segments are merged.";
      root.appendChild(settings);
      const pairs = Array.isArray(assembly.pairs) ? assembly.pairs : [];
      if (pairs.length) {
        const inspector = document.createElement("details");
        inspector.id = "grouping-evidence";
        inspector.open = evidenceOpen;
        inspector.className = "identity-summary";
        const heading = document.createElement("summary");
        heading.textContent = `Grouping evidence (${pairs.length} of ${assembly.pair_count ?? pairs.length} candidate pairs)`;
        inspector.appendChild(heading);
        for (const pair of pairs) {
          const row = document.createElement("div");
          row.className = "identity-pending";
          const status = pair.group_blocker || (pair.confirmed ? `confirmed · ${pair.reason}` : pair.reason);
          row.textContent = `E${pair.a} + E${pair.b}: ${String(status || "pending").replaceAll("_", " ")} · DINO ${pair.dino_cos == null ? "n/a" : Number(pair.dino_cos).toFixed(2)} · gap ${Number(pair.gap_m || 0).toFixed(2)} m · ${pair.hits || 0} hits / ${(pair.views || []).length} views`;
          if (pair.contact_fraction != null) row.textContent += ` · contact ${(100*pair.contact_fraction).toFixed(0)}% of samples`;
          row.title = "Grouping uses interior DINO, not the tracking context descriptor or a clicked heatmap token. Contact is sampled voxel evidence, not a calibrated probability.";
          inspector.appendChild(row);
        }
        root.appendChild(inspector);
      }
    }
  }
  if (Array.isArray(descriptor.mask_attribution)) {
    const masks = descriptor.mask_attribution;
    const assigned = masks.filter(mask => (mask.segments || []).length > 0).length;
    const summary = document.createElement("div");
    summary.className = "identity-summary";
    summary.textContent = `SAM attribution: ${assigned}/${masks.length} masks have segment support`;
    const reasons = {};
    for (const mask of masks) if (!(mask.segments || []).length) reasons[mask.reason || "unknown"] = (reasons[mask.reason || "unknown"] || 0) + 1;
    for (const [reason, count] of Object.entries(reasons)) summary.textContent += ` · ${count} ${reason.replaceAll("_", " ")}`;
    summary.title = "One SAM mask may cover several segments. Camera boundaries come from current SAM masks; the ID is a temporal/depth-supported hypothesis, not an assertion of dense 3D ownership.";
    root.appendChild(summary);
    const inspector = document.createElement("details");
    inspector.id = "mask-evidence";
    inspector.open = attributionOpen;
    inspector.className = "identity-summary";
    const heading = document.createElement("summary");
    heading.textContent = "Mask ownership evidence";
    inspector.appendChild(heading);
    for (const mask of masks.slice(0, 100)) {
      const row = document.createElement("div");
      row.className = "identity-pending";
      row.textContent = `Mask ${mask.mask ?? "?"} → ${mask.mask_owner == null ? "unresolved" : `S${mask.mask_owner}`} · ${String(mask.association_basis || mask.reason || "unknown").replaceAll("_", " ")} · support: ${(mask.segments || []).map(s => `S${s}`).join(", ") || "none"} · ${mask.depth_points || 0} depth samples`;
      if (mask.display_owner != null) row.textContent += ` · display: S${mask.display_owner}${mask.display_provisional ? "? (provisional)" : ""}`;
      inspector.appendChild(row);
    }
    root.appendChild(inspector);
  }
  if (descriptor.surface_tracking?.active_mask_tracks != null) {
    const summary = document.createElement("div");
    summary.className = "identity-summary";
    summary.textContent = `Mask tracking: ${descriptor.surface_tracking.active_mask_tracks} active · ${descriptor.surface_tracking.mask_owner_switches || 0} confirmed rebindings · ${descriptor.surface_tracking.temporal_holds || 0} temporal holds`;
    summary.title = "Diagnostic event counts, not ground-truth tracking accuracy. Temporary holds require current depth and world-position agreement.";
    root.appendChild(summary);
  }
  if (descriptor.identity_recording) {
    const recording = descriptor.identity_recording;
    const summary = document.createElement("div");
    summary.className = "identity-summary";
    summary.textContent = `Identity replay: ${recording.state} · ${recording.frames}/${recording.max_frames} frames${recording.error ? ` · ${recording.error}` : ""}`;
    summary.title = recording.path || "";
    root.appendChild(summary);
  }
  if (nodes.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = identity ? "No tracks pass the graph display filters yet." : "waiting for graph";
    root.appendChild(empty);
  }
  for (const node of nodes.slice(0, 28)) {
    const row = document.createElement("div");
    row.className = "graph-item";
    row.title = node.uid || "";
    const entityView = node.node_type === "object_hypothesis" || descriptor.entity_type === "entity";
    const id = `${entityView ? "E" : "S"}${node.id}`;
    const support = node.support_count ?? 0;
    const obs = node.observations ?? 0;
    const score = Number(node.query_score || 0);
    const conf = Number(node.confidence || 0);
    row.innerHTML = `
      <div class="graph-id">${escapeText(id)}</div>
      <div class="graph-main">
        <strong>${escapeText(node.state || "tentative")} / ${entityView ? "object hypothesis" : "surface segment"}</strong>
        <span>${fmtCount(support)} voxels / ${obs} obs / conf ${conf.toFixed(2)}</span>
        <span class="graph-members">${entityView ? `segments: ${escapeText((node.segment_ids || []).join(", "))}` : `entity: ${node.entity_id == null ? "unassigned" : `E${escapeText(node.entity_id)}`}`} / ${Number(node.view_count || 0)} views</span>
      </div>
      <div class="graph-score">${score.toFixed(2)}</div>
    `;
    root.appendChild(row);
  }
  for (const pending of (descriptor.pending_memberships || []).slice(0, 12)) {
    const row = document.createElement("div");
    row.className = "identity-pending";
    row.textContent = `S${pending.segment_id}: ${String(pending.reason || "pending").replaceAll("_", " ")}${pending.candidate == null ? "" : ` → E${pending.candidate} (${pending.hits} observations)`}`;
    const context = (pending.neighborhood_scores || []).find(candidate => candidate.entity_id === pending.candidate && candidate.neighbor_segment_id != null);
    if (context) row.textContent += ` · neighbor S${context.neighbor_segment_id} · overlap ${Number(context.overlap).toFixed(2)} · context ${Number(context.neighborhood_strength).toFixed(2)}`;
    root.appendChild(row);
  }
}

function imageHealthMessage(status, selected, statusAgeSeconds = 0) {
  if (!status) return "Waiting for dashboard status";
  if (statusAgeSeconds > 5) return "Dashboard connection stale — image freshness unknown";
  const health = status.pipeline_health;
  const topic = status.topics?.[selected];
  const threshold = Math.max(10, 3*Number(status.observation_cycle?.average_ms || 0)/1000);
  const imageAge = topic?.age_s == null ? null : Number(topic.age_s)+statusAgeSeconds;
  let message = imageAge == null ? "Waiting for camera image" : imageAge > threshold
    ? `STALE IMAGE · ${fmtAge(imageAge)} old — showing last received frame` : "";
  if (health) {
    if (Number(health.perception_publishers || 0) > 1) message += `${message ? " · " : ""}MULTIPLE PERCEPTION NODES — conflicting maps/IDs; stop the extra node`;
    if (health.stationary_pose_lock) message += `${message ? " · " : ""}STATIONARY POSE LOCK — disable before moving the robot`;
    if (Number(health.age_s || 0)+statusAgeSeconds > 5) {
      message += `${message ? " · " : ""}Perception heartbeat stale`;
    } else if (!["healthy", "processing"].includes(health.state)) {
      message += `${message ? " · " : ""}${String(health.state).replaceAll("_", " ")}`;
      if (health.last_error || health.last_rejection) message += `: ${health.last_error || health.last_rejection}`;
    }
    if (message && health.stage !== "idle") message += ` · ${health.stage} (${Number(health.stage_age_s || 0).toFixed(1)}s)`;
  }
  return message;
}

function updateImageHealth() {
  const notice = $("image-health");
  if (!notice) return;
  const elapsed = appState.lastStatusAt ? (Date.now()-appState.lastStatusAt)/1000 : 0;
  const message = imageHealthMessage(appState.lastStatus, appState.selectedImage, elapsed);
  notice.textContent = [message, appState.lastImageError].filter(Boolean).join(" · ");
  notice.hidden = !notice.textContent;
  const health = appState.lastStatus?.pipeline_health;
  notice.title = health ? Object.entries(health.inputs || {}).map(([name,s]) =>
    `${name}: ${s.hz} Hz, age ${s.age_s == null ? "never" : fmtAge(Number(s.age_s)+elapsed)}`).join(" / ") : "";
}

async function refreshImage() {
  updateImageHealth();
  const id = appState.selectedImage;
  if (!id || appState.imageRequestController) return;
  const seq = ++appState.imageRequestSeq;
  const controller = new AbortController();
  appState.imageRequestController = controller;
  const timeout = setTimeout(() => controller.abort(), 5000);
  const img = $("live-image");
  const empty = $("image-empty");
  empty.textContent = "waiting for image";
  try {
    const response = await fetch(`/api/image/${encodeURIComponent(id)}?t=${Date.now()}`, { cache: "no-store", signal: controller.signal });
    if (seq !== appState.imageRequestSeq || id !== appState.selectedImage) return;
    if (!response.ok) {
      appState.lastImageError = response.status === 404 ? "Image not yet published" : `Image request failed (${response.status})`;
      if (!img.dataset.objectUrl) {
        img.style.display = "none";
        empty.style.display = "block";
      }
      return;
    }
    const blob = await response.blob();
    if (seq !== appState.imageRequestSeq || id !== appState.selectedImage) return;
    const oldUrl = img.dataset.objectUrl || "";
    appState.lastImageError = "";
    const url = URL.createObjectURL(blob);
    img.onload = () => {
      if (oldUrl) URL.revokeObjectURL(oldUrl);
      updateSimilarityMarker();
    };
    img.dataset.objectUrl = url;
    img.src = url;
    img.style.display = "block";
    empty.style.display = "none";
  } catch (err) {
    if (seq !== appState.imageRequestSeq || id !== appState.selectedImage) return;
    appState.lastImageError = err.name === "AbortError" ? "Image request timed out" : "Image request failed";
    if (!img.dataset.objectUrl) {
      img.style.display = "none";
      empty.style.display = "block";
    }
  } finally {
    clearTimeout(timeout);
    if (appState.imageRequestController === controller) appState.imageRequestController = null;
    updateImageHealth();
  }
}

function refreshGrid() {
  const id = appState.selectedGrid;
  if (!id) return;
  const img = $("grid-image");
  const empty = $("grid-empty");
  img.onload = () => {
    img.style.display = "block";
    empty.style.display = "none";
  };
  img.onerror = () => {
    img.style.display = "none";
    empty.style.display = "block";
  };
  img.src = `/api/grid/${encodeURIComponent(id)}?t=${Date.now()}`;
}

document.addEventListener("DOMContentLoaded", init);
