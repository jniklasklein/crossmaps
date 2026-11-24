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
const evaluationCount = value => Number.isSafeInteger(value) && value >= 0 ? value : null;
function simulationEvaluationModel(snapshot, connected = true) {
  const data = snapshot.simulation_evaluation;
  if (!data || data.enabled !== true) return null;
  const recorded = data.recorded === true || snapshot.mode === "recorded_replay";
  const status = !connected ? "disconnected" : data.schema !== 1 ? "invalid" : ["ready", "waiting", "stale", "invalid"].includes(data.status) ? data.status : "waiting";
  return { ...data, recorded, status, measured: status === "ready", coverage: data.coverage || {},
    local: Array.isArray(data.local) ? data.local : [], shared: data.shared || {}, optimization: data.optimization || {}, loops: data.loops || {},
    reason: !connected ? `Coordinator disconnected; ${recorded ? "recorded evaluation cannot be refreshed" : "live evaluation is unavailable"}.` : data.schema !== 1 ? "Unsupported evaluation schema; measurements are unavailable." : data.reason || "",
    notes: Array.isArray(data.notes) ? data.notes : [] };
}
function evaluationInterval(value) {
  const low = Array.isArray(value) ? value[0] : value?.min, high = Array.isArray(value) ? value[1] : value?.max;
  if (!finite(low) || !finite(high) || low <= 0 || high < low) return "interval unavailable";
  return `${low.toFixed(1)}–${high.toFixed(1)} s between retained captures`;
}
function simulationEvaluationContent(model) {
  if (!model) return "";
  const measured = model.measured, coverage = model.coverage;
  const count = value => evaluationCount(value) ?? "—";
  const metric = (value, units, n, digits = 3) => measured && evaluationCount(n) > 0 && finite(value) && value >= 0 ? `${value.toFixed(digits)} ${units}` : "Unavailable";
  const pairMetric = row => `${metric(row.translation_rmse_m, "m", row.pairs)} / ${metric(row.rotation_rmse_deg, "°", row.pairs, 2)}`;
  const state = model.recorded ? `Recorded evaluation${model.status === "ready" ? "" : " · unavailable"}` : {ready:"Ground truth connected", waiting:"Waiting for evaluation", stale:"Ground truth stale", invalid:"Evaluation unavailable", disconnected:"Evaluation disconnected"}[model.status];
  const captured = finite(model.updated_unix) && Math.abs(model.updated_unix) <= 8.64e12 ? new Date(model.updated_unix * 1000).toISOString() : "time unavailable";
  const reason = model.recorded ? `Snapshot captured: ${captured}. Saved measurements, not a live ground-truth connection. ${model.reason}` : model.reason || "Capture-time ground truth is used only for scoring, never for mapping or alignment.";
  const confirmed = model.shared.confirmed || {}, provisional = model.shared.provisional || {}, opt = model.optimization, loops = model.loops;
  const percentage = (value, n) => measured && evaluationCount(n) > 0 && finite(value) && value >= 0 && value <= 1 ? `${(100*value).toFixed(1)}%` : "Unavailable";
  const precision = percentage(loops.precision, loops.evaluated), acceptedPrecision = percentage(loops.accepted_precision, loops.accepted_evaluated);
  const rows = model.local.filter(row => row && typeof row === "object").map(row => `<tr><th scope="row">${escapeText(row.robot_id || "Unknown rover")}<small>${count(row.samples)} associated keyframes</small></th>
    <td>${metric(row.position_rmse_m, "m", row.samples)}<small>${metric(row.rotation_rmse_deg, "°", row.samples, 2)} rotation</small></td>
    <td>${metric(row.rpe_translation_rmse_m, "m", row.rpe_pairs)}<small>${metric(row.rpe_rotation_rmse_deg, "°", row.rpe_pairs, 2)} rotation · ${count(row.rpe_pairs)} pairs</small></td>
    <td>${escapeText(evaluationInterval(row.rpe_interval_s))}</td></tr>`).join("");
  return `<div class="evaluation-status"><span class="badge ${model.recorded ? "reference" : model.status === "ready" ? "healthy" : "pending"}">${state}</span><p title="${escapeText(reason)}">${escapeText(reason)}</p></div>
    <p class="panel-note evaluation-coverage">${model.recorded ? "Truth age at recording" : "Truth age"}: ${ageLabel(coverage.truth_age_s)} · Associated retained keyframes: ${count(coverage.associated)} / ${count(coverage.keyframes)} · Missing truth: ${count(coverage.missing)}</p>
    <div class="evaluation-grid">
      <article class="evaluation-card"><h3>Cross-rover keyframe relative pose</h3><p class="evaluation-label">Translation / rotation RMSE</p><dl><div><dt>Confirmed</dt><dd>${pairMetric(confirmed)}<small>${count(confirmed.pairs)} scored pairs</small></dd></div><div><dt>Provisional</dt><dd>${pairMetric(provisional)}<small>${count(provisional.pairs)} scored pairs</small></dd></div></dl><p>Compared at each keyframe’s own capture time; not a simultaneous live robot-position measurement.</p></article>
      <article class="evaluation-card"><h3>Local baseline vs optimized map</h3><p class="evaluation-label">Same reference · ${count(opt.samples)} keyframes</p><dl><div><dt>Raw local poses</dt><dd>${metric(opt.raw_position_rmse_m, "m", opt.samples)}<small>${metric(opt.raw_rotation_rmse_deg, "°", opt.samples, 2)} rotation RMSE</small></dd></div><div><dt>Optimized poses</dt><dd>${metric(opt.optimized_position_rmse_m, "m", opt.samples)}<small>${metric(opt.optimized_rotation_rmse_deg, "°", opt.samples, 2)} rotation RMSE</small></dd></div></dl><p>Raw local poses use the current first-keyframe alignment. These are ground-truth errors, not graph solver cost; this is not a controlled PGO on/off experiment.</p></article>
      <article class="evaluation-card"><h3>Cross-rover closure measurements</h3><p class="evaluation-label">Geometric correctness of scored constraints</p><dl><div><dt>Verified / accepted</dt><dd>${precision}<small>${count(loops.correct)} / ${count(loops.evaluated)} within tolerance</small></dd></div><div><dt>Accepted only</dt><dd>${acceptedPrecision}<small>${count(loops.accepted_correct)} / ${count(loops.accepted_evaluated)} within tolerance</small></dd></div><div><dt>Measurement RMSE</dt><dd>${metric(loops.translation_rmse_m, "m", loops.evaluated)}<small>${metric(loops.rotation_rmse_deg, "°", loops.evaluated, 2)} rotation</small></dd></div></dl><p>Tolerance: ${finite(loops.translation_threshold_m) ? `${loops.translation_threshold_m.toFixed(2)} m` : "unavailable"} and ${finite(loops.rotation_threshold_deg) ? `${loops.rotation_threshold_deg.toFixed(1)}°` : "unavailable"}. This is not loop-detection recall.</p></article>
    </div>
    <div class="evaluation-local"><h3>Per-rover local odometry</h3><div class="evaluation-table-wrap"><table><thead><tr><th scope="col">Rover</th><th scope="col">First-pose anchored RMSE</th><th scope="col">Relative pose error RMSE</th><th scope="col">RPE sampling interval</th></tr></thead><tbody>${rows || '<tr><td colspan="4">Waiting for timestamp-associated captures.</td></tr>'}</tbody></table></div><p>Local baselines are anchored independently at each rover’s first pose. They do not measure cross-rover alignment. RPE uses adjacent retained captures with the intervals shown, not a fixed one-second benchmark.</p></div>
    <div class="panel-note evaluation-limits"><strong>Not yet measured:</strong> observed-surface IoU / F-score, loop-detection recall, and SLAM Toolbox’s single-rover loop closures. Sparse-map overlap alone is not localization accuracy.${model.notes.length ? `<ul>${model.notes.map(note => `<li>${escapeText(note)}</li>`).join("")}</ul>` : ""}</div>`;
}
function renderSimulationEvaluation(panel, content, snapshot, connected = true) {
  const model = simulationEvaluationModel(snapshot, connected);
  panel.hidden = !model;
  content.innerHTML = simulationEvaluationContent(model);
}
const byteLabel = value => !finite(value) || value < 0 ? "Unknown" : value < 1024 ? `${Math.round(value)} B` : value < 1024 ** 2 ? `${(value / 1024).toFixed(1)} KiB` : value < 1024 ** 3 ? `${(value / 1024 ** 2).toFixed(1)} MiB` : `${(value / 1024 ** 3).toFixed(2)} GiB`;
const storageRunId = value => typeof value === "string" && /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/.test(value);
function storageModel(snapshot, connected = true) {
  const data = snapshot.storage || {}, stats = snapshot.stats || {}, limits = stats.limits || {};
  const current = storageRunId(data.current_run_id) ? data.current_run_id : null;
  const runs = (Array.isArray(data.runs) ? data.runs : []).filter(run => run && storageRunId(run.id));
  const enabled = connected && snapshot.mode !== "recorded_replay" && data.readonly !== true && snapshot._dashboard?.commands_enabled === true && Boolean(current);
  return { ...data, free_bytes: data.free_bytes ?? data.disk_free_bytes, reason: data.reason || stats.capacity_reason,
    current_run_id: current, map_generation: snapshot.map_generation, enabled, runs: runs.map(run => ({ ...run, reason: run.reason || run.protected_reason,
    deletable: run.deletable === true && run.active !== true && run.id !== current })),
    keyframes: finite(stats.keyframes) ? stats.keyframes : null, max_keyframes: limits.max_keyframes,
    evidence_bytes: stats.store_bytes, max_evidence_bytes: limits.max_store_bytes,
    unavailable: !current ? "Storage information is not available from this coordinator." : !connected ? "Reconnect to the coordinator before changing storage."
      : snapshot.mode === "recorded_replay" || data.readonly === true ? "Recorded replay is read-only. Open a live coordinator to manage storage."
      : !snapshot._dashboard?.commands_enabled ? "Storage changes require a local dashboard session." : "" };
}
function storageOverview(model) {
  const count = finite(model.keyframes) ? model.keyframes : "—", limit = finite(model.max_keyframes) ? model.max_keyframes : "—";
  return `<p class="storage-current">Current run: <code>${escapeText(model.current_run_id || "Unavailable")}</code></p><div class="storage-summary">
    <div><span>Retained keyframes</span><strong>${count} / ${limit}</strong></div>
    <div><span>Current evidence / limit</span><strong>${byteLabel(model.evidence_bytes)} / ${byteLabel(model.max_evidence_bytes)}</strong></div>
    <div><span>All recordings / budget</span><strong>${byteLabel(model.total_bytes)} / ${byteLabel(model.max_total_bytes)}</strong></div></div>
    <p class="storage-current">Current recording: ${byteLabel(model.current_bytes)} · Disk free: ${byteLabel(model.free_bytes)} · Reserved free space: ${byteLabel(model.min_free_bytes)}.</p>
    ${finite(model.managed_runs) && finite(model.max_runs) ? `<p class="storage-current">Managed recordings: ${model.managed_runs} / ${model.max_runs}. No automatic deletion.${finite(model.reserved_bytes) && model.reserved_bytes > 0 ? ` In-flight write reservations: ${byteLabel(model.reserved_bytes)}.` : ""}</p>` : ""}`;
}
function storageRunsContent(model, selected) {
  return model.runs.map(run => `<label class="storage-run"><input type="checkbox" data-storage-run="${escapeText(run.id)}" ${selected.has(run.id) ? "checked" : ""} ${!model.enabled || !run.deletable ? "disabled" : ""}>
    <span>${escapeText(run.id)}<small>${escapeText(run.id === model.current_run_id ? "Current run — use Rebuild current map" : run.active ? "Active — protected" : !run.deletable ? run.reason || "Protected recording" : run.managed === false ? "Recognized legacy recording — explicit removal only" : "Inactive recording")}</small></span><span class="storage-bytes">${byteLabel(run.bytes)}</span></label>`).join("") || '<p class="empty-list">No recordings listed.</p>';
}
function storagePlanContent(plan) {
  return `<p>${escapeText(plan.description || (plan.operation === "rebuild_current" ? "Permanently discard current CrossMaps map evidence and rebuild from new observations." : "Permanently delete the selected old CrossMaps recordings."))}</p>
    <p>Affected recording${plan.run_ids.length === 1 ? "" : "s"}: <strong>${plan.run_ids.length}</strong> · Prepared size: <strong>${byteLabel(plan.bytes)}</strong></p>
    <ul>${plan.run_ids.map(id => `<li>${escapeText(id)}</li>`).join("")}</ul>
    <p>Only the listed CrossMaps data is affected. Robot SLAM, Gazebo and Nav2 are unchanged.</p>`;
}
function storageResult(result) {
  const details = result.result || {};
  if (details.ok === false) throw new Error(details.error || "The storage operation did not finish.");
  const warnings = [result.warning, details.warning, ...(Array.isArray(details.warnings) ? details.warnings : [])].filter(Boolean).map(String);
  const message = result.message || (details.reset_applied ? "Current CrossMaps map cleared; rebuilding from fresh observations."
    : result.operation === "delete_runs" ? "Selected old recordings removed." : "Storage operation completed.");
  return { message: `${message}${finite(details.freed_bytes) ? ` Freed ${byteLabel(details.freed_bytes)}.` : ""}${warnings.length ? ` Warning: ${warnings.join("; ")}` : ""}`, warning: warnings.length > 0 };
}
// Preparing never deletes. A server-issued scope token is required for the
// separate confirmation. Polls cannot silently switch the reviewed targets.
function storageController(send, changed = () => {}) {
  const state = { model: storageModel({}), selected: new Set(), plan: null, busy: false, message: "", error: false };
  let generation = 0;
  const emit = () => changed(state);
  const note = (message, error = false) => { state.message = message; state.error = error; };
  return {
    state,
    update(snapshot, connected) {
      const next = storageModel(snapshot, connected);
      if ((state.model.current_run_id && next.current_run_id !== state.model.current_run_id) ||
          (state.model.map_generation && next.map_generation !== state.model.map_generation)) {
        const pending = state.plan || state.busy;
        generation++; state.plan = null; state.busy = false; state.selected.clear();
        if (pending) note("The active map changed. Review storage again before confirming.");
      }
      state.model = next;
      state.selected = new Set([...state.selected].filter(id => next.runs.some(run => run.id === id && run.deletable)));
      emit();
    },
    select(id, selected) {
      if (state.busy || state.plan || !state.model.enabled || !state.model.runs.some(run => run.id === id && run.deletable)) return;
      if (selected) state.selected.add(id); else state.selected.delete(id);
      emit();
    },
    cancel() { generation++; state.plan = null; state.busy = false; note(""); emit(); },
    async prepare(operation) {
      if (state.busy || state.plan) return;
      if (!state.model.enabled) { note(state.model.unavailable, true); emit(); return; }
      if (!["rebuild_current", "delete_runs"].includes(operation)) return;
      const ids = operation === "delete_runs" ? [...state.selected].sort() : [state.model.current_run_id];
      if (!ids.length) { note("Select at least one inactive recording.", true); emit(); return; }
      const request = ++generation;
      state.busy = true; note("Checking the selected data. Nothing has been removed."); emit();
      try {
        const result = await send({ action: "prepare_storage", operation, ...(operation === "delete_runs" ? { run_ids: ids } : {}) });
        if (request !== generation) return;
        const plan = result.plan;
        if (!plan || typeof plan.token !== "string" || !plan.token || plan.operation !== operation ||
            !Array.isArray(plan.run_ids) || !plan.run_ids.every(storageRunId) ||
            JSON.stringify([...plan.run_ids].sort()) !== JSON.stringify(ids) || !finite(plan.bytes) || plan.bytes < 0)
          throw new Error("The prepared removal scope does not match your selection. Nothing was confirmed.");
        state.plan = { ...plan, run_ids: plan.run_ids.slice() };
        note("Review the exact recordings below. Nothing is removed until you confirm.");
      } catch (error) { if (request === generation) note(error.message, true); }
      finally { if (request === generation) { state.busy = false; emit(); } }
    },
    async confirm() {
      if (state.busy || !state.plan) return;
      if (!state.model.enabled) { note(state.model.unavailable, true); emit(); return; }
      const request = ++generation, plan = state.plan;
      state.busy = true; note("Applying the confirmed storage operation…"); emit();
      try {
        const result = await send({ action: "confirm_storage", token: plan.token });
        if (request !== generation) return;
        const outcome = storageResult(result);
        state.selected.clear(); note(outcome.message, outcome.warning);
      } catch (error) { if (request === generation) note(error.message + " Review and prepare again before retrying.", true); }
      finally { if (request === generation) { state.plan = null; state.busy = false; emit(); } }
    }
  };
}
function installStorageDialog($, getSnapshot, isConnected, send) {
  const dialog = $("storage-dialog");
  let runsKey = "", planKey = "";
  const controller = storageController(send, state => {
    if (!dialog.open) return;
    const { model, plan, busy } = state;
    $("storage-status").textContent = state.message || model.unavailable || model.reason || "Redundant views are skipped; stored evidence remains bounded. Use these controls when you want to free it.";
    $("storage-status").className = state.error || model.reason ? "error" : "";
    $("storage-overview").innerHTML = storageOverview(model);
    const nextRuns = JSON.stringify([model.current_run_id, model.enabled, model.runs.map(({ id, active, managed, deletable, reason }) => [id, active, managed, deletable, reason])]);
    if (runsKey !== nextRuns && !plan && !busy) {
      const scroll = $("storage-runs").scrollTop;
      $("storage-runs").innerHTML = storageRunsContent(model, state.selected); $("storage-runs").scrollTop = scroll; runsKey = nextRuns;
    }
    for (const input of $("storage-runs").querySelectorAll("[data-storage-run]")) {
      input.checked = state.selected.has(input.dataset.storageRun);
      input.disabled = busy || !model.enabled || !model.runs.some(run => run.id === input.dataset.storageRun && run.deletable);
    }
    $("storage-controls").hidden = Boolean(plan); $("storage-confirmation").hidden = !plan;
    const nextPlan = plan ? JSON.stringify(plan) : "";
    if (planKey !== nextPlan) { $("storage-plan").innerHTML = plan ? storagePlanContent(plan) : ""; planKey = nextPlan; }
    $("storage-rebuild").disabled = busy || !model.enabled;
    $("storage-delete").disabled = busy || !model.enabled || !state.selected.size;
    $("storage-confirm").disabled = busy || !model.enabled || !plan;
    $("storage-confirm").textContent = plan?.operation === "rebuild_current" ? "Delete current evidence & rebuild" : "Permanently delete recordings";
    $("storage-cancel").disabled = busy;
    $("storage-close").disabled = busy;
  });
  const refresh = () => controller.update(getSnapshot(), isConnected());
  $("storage-open").addEventListener("click", () => { dialog.showModal(); refresh(); });
  $("storage-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", () => controller.cancel());
  dialog.addEventListener("cancel", event => { if (controller.state.busy) event.preventDefault(); });
  $("storage-runs").addEventListener("change", event => {
    const input = event.target.closest("[data-storage-run]"); if (input) controller.select(input.dataset.storageRun, input.checked);
  });
  $("storage-rebuild").addEventListener("click", () => controller.prepare("rebuild_current"));
  $("storage-delete").addEventListener("click", () => controller.prepare("delete_runs"));
  $("storage-cancel").addEventListener("click", () => controller.cancel());
  $("storage-confirm").addEventListener("click", () => controller.confirm());
  return { refresh, controller };
}
function buildCloud(snapshot, source = "combined", colorMode = "source", budget = 30000, showProvisional = true) {
  const positions = [], colors = [];
  const robots = snapshot.robots || [], excluded = [], confirmed = [], previews = [];
  for (const submap of snapshot.submaps || []) {
    if (source !== "combined") {
      const robot = robots.find(r => robotId(r) === source);
      if (String(submap.robot_id) === source && !(submap.session_id && robot?.session_id && submap.session_id !== robot.session_id)) confirmed.push(submap);
    } else if (submap.aligned === true && matrix(submap.T_world_submap)) {
      confirmed.push(submap);
    } else if (submap.aligned === false && snapshot.shadow_mode === false &&
               submap.alignment_state === "provisional" && matrix(submap.T_preview_world_submap)) {
      previews.push(submap);
    } else {
      excluded.push(submap.id);
    }
  }
  // Keep previews in a contiguous GPU draw range (amber-edged diamonds).
  // Never fall back to the local pose/identity for a combined-world preview.
  const chosen = showProvisional ? confirmed.concat(previews) : confirmed;
  const total = chosen.reduce((sum, submap) => sum + (submap.points || []).length, 0);
  const stride = Math.max(1, Math.ceil(total / Math.max(1, budget)));
  let index = 0, previewPoints = 0;
  for (const [n, submap] of chosen.entries()) {
    const robotIndex = robots.findIndex((robot) => robotId(robot) === String(submap.robot_id));
    const robot = robots[robotIndex];
    const tentative = n >= confirmed.length;
    let transform = matrix(tentative ? submap.T_preview_world_submap : submap.T_world_submap);
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
      if (tentative) previewPoints++;
      let color = colorMode === "rgb" ? (submap.colors?.[i] || (point.length >= 6 ? point.slice(3, 6) : sourceColor)) : sourceColor;
      if (!Array.isArray(color) || color.length < 3 || !color.slice(0, 3).every(finite)) color = sourceColor;
      const scale = Math.max(...color) <= 1 ? 255 : 1;
      colors.push(...color.slice(0, 3).map((c) => clamp(Math.round(c * scale), 0, 255)));
    }
  }
  return { positions, colors, excluded, submaps: chosen.length, previewPoints, previewAvailable: previews.length,
    previewShown: showProvisional ? previews.length : 0 };
}
// Poses are measurements attached to a submap, not permanent world coordinates.
// Re-project them after each optimization, alignment promotion or revocation.
function buildRoverOverlay(snapshot, source = "combined", { showProvisional = true, showMarkers = true, showTrails = true, connected = true } = {}) {
  const maps = new Map((snapshot.submaps || []).map(s => [s.id, s]));
  const markers = [], trails = [];
  for (const [i, robot] of (snapshot.robots || []).entries()) {
    const id = robotId(robot), trajectory = robot.trajectory, color = PALETTE[i % PALETTE.length];
    if (!trajectory || (source !== "combined" && source !== id)) continue;
    const placement = (submapId) => {
      const s = maps.get(submapId);
      if (!s || String(s.robot_id) !== id || !robot.session_id || s.session_id !== robot.session_id) return null;
      let transform, tentative = false;
      if (source !== "combined") transform = matrix(s.T_local_submap);
      else if (s.aligned === true) transform = matrix(s.T_world_submap);
      else if (s.aligned === false && showProvisional && snapshot.shadow_mode === false && s.alignment_state === "provisional") {
        transform = matrix(s.T_preview_world_submap); tentative = true;
      }
      return transform ? { transform, tentative } : null;
    };
    const isPoseFrame = frame => frame === "base" || frame === "camera_optical";
    const latest = trajectory.latest, pose = matrix(latest?.T_submap_pose), at = placement(latest?.submap_id);
    if (showMarkers && pose && at && isPoseFrame(latest.frame)) {
      const t = matrixProduct(at.transform, pose);
      const axis = latest.frame === "base" ? 0 : 2;
      const position = t.slice(0, 3).map(row => row[3]), forward = t.slice(0, 3).map(row => row[axis]);
      const historical = snapshot.mode === "recorded_replay" || snapshot.mode === "synthetic_demo";
      const stale = !connected || trajectory.tracking_lost === true || mappingPaused(robot) ||
        (!historical && finite(robot.map_observation_age_s) && robot.map_observation_age_s > 8);
      if (position.every(finite) && forward.every(finite) && Math.hypot(...forward) > .5)
        markers.push({ id, position, forward: normalize(forward), frame: latest.frame, color, stale, historical, tentative: at.tentative });
    }
    if (!showTrails) continue;
    let run = null;
    for (const segment of trajectory.segments || []) {
      const at = placement(segment?.submap_id);
      if (!at || !isPoseFrame(segment.frame) || !Array.isArray(segment.points) || !segment.points.length) { run = null; continue; }
      if (segment.break_before !== false || run?.frame !== segment.frame || run.tentative !== at.tentative) run = null;
      for (const p of segment.points) {
        if (!Array.isArray(p) || p.length !== 3 || !p.every(finite)) { run = null; continue; }
        const world = transformPoint(p, at.transform);
        if (!world.every(finite)) { run = null; continue; }
        if (!run) { run = { id, frame: segment.frame, color, tentative: at.tentative, points: [] }; trails.push(run); }
        run.points.push(world);
      }
    }
  }
  return { markers, trails: trails.filter(run => run.points.length >= 2) };
}

// vp is column-major (the same projection passed to WebGL).
function projectMarker(p, vp, width, height) {
  const component = row => p[0]*vp[row] + p[1]*vp[row+4] + p[2]*vp[row+8] + vp[row+12];
  const w = component(3), z = component(2);
  if (!(w > 0 && z >= -w && z <= w)) return null;
  return [(component(0)/w + 1)*width/2, (1 - component(1)/w)*height/2];
}

function drawRoverOverlay(canvas, ctx, overlay, vp, width, height, ratio) {
  if (!ctx) return;
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const w = width/ratio, h = height/ratio;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, w, h);
  ctx.lineJoin = ctx.lineCap = "round";
  for (const run of overlay.trails) {
    ctx.beginPath();
    let connected = false;
    for (const p of run.points) {
      const xy = projectMarker(p, vp, w, h);
      if (!xy) { connected = false; continue; }
      if (connected) ctx.lineTo(...xy); else ctx.moveTo(...xy);
      connected = true;
    }
    ctx.setLineDash(run.tentative ? [5, 4] : []);
    ctx.lineWidth = 4; ctx.strokeStyle = "#071019d9"; ctx.stroke();
    ctx.lineWidth = 2; ctx.strokeStyle = run.tentative ? "#efb668" : `rgb(${run.color.join(",")})`; ctx.stroke();
  }
  ctx.setLineDash([]);
  for (const marker of overlay.markers) {
    const pos = projectMarker(marker.position, vp, w, h);
    if (!pos || pos[0] < 0 || pos[0] > w || pos[1] < 0 || pos[1] > h) continue;
    const tip = projectMarker(marker.position.map((v,i) => v + marker.forward[i]*.2), vp, w, h);
    const hasBearing = tip && Math.hypot(tip[0]-pos[0], tip[1]-pos[1]) > .1;
    const color = `rgb(${marker.color.join(",")})`;
    ctx.save(); ctx.translate(...pos);
    if (hasBearing) ctx.rotate(Math.atan2(tip[1]-pos[1], tip[0]-pos[0]));
    ctx.beginPath();
    if (hasBearing) { ctx.moveTo(12,0); ctx.lineTo(-7,-6); ctx.lineTo(-4,0); ctx.lineTo(-7,6); ctx.closePath(); }
    else ctx.arc(0,0,6,0,2*Math.PI); // Looking directly into/out of screen: never invent a yaw.
    ctx.strokeStyle = "#09121d"; ctx.lineWidth = 5; ctx.stroke();
    ctx.fillStyle = marker.stale ? "#111b2b" : color; ctx.fill();
    ctx.strokeStyle = marker.tentative ? "#f6b958" : color; ctx.lineWidth=2; ctx.stroke(); ctx.restore();
    ctx.font = "600 11px system-ui, sans-serif"; ctx.lineWidth=3; ctx.strokeStyle="#09121d";
    const tx = clamp(pos[0]+14, 2, Math.max(2,w-ctx.measureText(marker.id).width-2)), ty=clamp(pos[1]-8,13,Math.max(13,h-2));
    ctx.strokeText(marker.id, tx, ty); ctx.fillStyle = "#f0f6fa"; ctx.fillText(marker.id, tx, ty);
  }
}

function roverPoseLegend(robot, marker, source, showMarkers) {
  const alignment = source !== "combined" ? "local frame" : robot.aligned ? "anchored"
    : robot.alignment_state === "provisional" ? (marker ? "provisional" : "provisional placement") : "unaligned";
  if (!showMarkers) return alignment;
  if (!marker) return alignment + (source === "combined" && !robot.aligned ? " · pose not shown" : " · no placed pose");
  const origin = marker.frame === "base" ? "base pose" : "camera pose (base unavailable)";
  return alignment + " · " + (marker.stale ? "last known " : marker.historical ? "recorded " : "") + origin;
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
  const age = robot.image_age_s?.[view];
  if (finite(age) && age >= 0) return age;
  // A recent observation does not mean this overlay was produced. In particular,
  // pose rejection may keep SAM fresh while segments retain their last map frame.
  return !robot.image_age_s && view === "rgb" && finite(robot.frame_age_s) ? robot.frame_age_s : null;
}
function orderedCandidates(snapshot) {
  const rank = (candidate) => { const constraint = candidateConstraint(candidate, snapshot); return constraint?.active ? 1 : ["verified", "corroborated"].includes(candidate.state) ? 0 : candidate.state === "rejected" ? 3 : 2; };
  return snapshot.candidates.slice().reverse().sort((a, b) => rank(a) - rank(b));
}
function currentComponent(robot, key) {
  const value = robot[key] || {};
  return value.session_id && robot.session_id && value.session_id !== robot.session_id ? {} : value;
}
function mappingPaused(robot) {
  const perception = currentComponent(robot, "perception");
  return robot.mapping_paused === true || robot.integration_allowed === false
    || robot.pose_quality?.integration_allowed === false || perception.pose_quality?.integration_allowed === false
    || [robot.status, robot.mapping_status, perception.status].some((value) => ["pose_rejected", "pose_inconsistent", "mapping_paused"].includes(value));
}
function healthPresentation(robot) {
  const transport = currentComponent(robot, "health"), perception = currentComponent(robot, "perception");
  const inputStatus = transport.status || "", status = robot.status || "";
  const pose = perception.pose_quality || robot.pose_quality;
  const poseReason = typeof pose === "object" && pose ? pose.status : pose;
  const result = { state: "healthy", label: "Healthy", style: "healthy", detail: "",
    input: inputStatus || (finite(robot.frame_age_s) ? "receiving" : "waiting_for_input"),
    perception: perception.stage || perception.status || (status === "processing" ? "processing" : "waiting"),
    mapping: mappingPaused(robot) ? "paused" : robot.mapping_status || "awaiting_observation" };
  if (transport.camera_transport === "ssh_compressed_pair") {
    result.input += " · compressed SSH (" + (transport.camera_relay?.state || "connecting") + ")";
  } else if (transport.camera_transport === "dds_compressed") {
    result.input += " · compressed DDS";
    if (transport.camera_relay_probing) result.input += " · trying SSH recovery";
  }
  // Errors belong to the currently failing component; historical last_error
  // strings must not paint a recovered source red forever.
  if (robot.error || ["error", "processing_error", "adapter_exited"].includes(status)
      || (perception.status === "error" && !(status === "healthy" && finite(robot.last_processed_stamp_ns)
          && finite(perception.stamp_ns) && robot.last_processed_stamp_ns >= perception.stamp_ns))) {
    return { ...result, state: "error", style: "error", label: status === "adapter_exited" ? "Adapter stopped" : "Perception error",
      detail: robot.error || robot.last_error || perception.error || "Processing failed; inspect coordinator events." };
  }
  if (transport.clock_mode === "simulation" && ["waiting_for_clock", "paused_or_stalled", "reset_restart_required"].includes(transport.simulation_clock_status)) {
    const clock = transport.simulation_clock_status;
    return { ...result, state: "simulation_" + clock, style: clock === "reset_restart_required" ? "error" : "pending",
      label: {waiting_for_clock: "Waiting for sim clock", paused_or_stalled: "Simulation stopped", reset_restart_required: "Simulation reset"}[clock],
      mapping: "paused", detail: transport.last_error || "Simulation time is not advancing." };
  }
  if (["depth_stale", "rgbd_stale"].includes(inputStatus)) {
    return { ...result, state: "stale", style: "stale", label: inputStatus === "depth_stale" ? "Depth stale" : "RGB-D stale",
      mapping: "awaiting a fresh pair", detail: transport.camera_relay?.last_error || transport.camera_relay_last_error ||
        transport.last_error || "Waiting for synchronized compressed RGB and depth." };
  }
  if (inputStatus === "rgb_stale" || (finite(robot.frame_age_s) && robot.frame_age_s > 8)) {
    const relayError = transport.camera_relay?.last_error || transport.camera_relay_last_error;
    if (relayError) return { ...result, state: "stale", style: "stale", label: "Camera stale", mapping: "awaiting fresh RGB",
      detail: "Camera recovery: " + String(relayError) };
    return { ...result, state: "stale", style: "stale", label: "Camera stale", mapping: "awaiting fresh RGB",
      detail: inputStatus === "rgb_stale" ? `Compressed RGB stalled while depth advances. Reader retries: ${transport.rgb_reader_attempts ?? 0}. Waiting for a new camera timestamp.` : "No advancing camera timestamp; displayed images may be old." };
  }
  if (inputStatus === "waiting_for_camera_relay") return { ...result, state: inputStatus,
    style: "pending", label: "Camera recovery", mapping: "awaiting fresh pair",
    detail: transport.camera_relay?.last_error || transport.last_error || "Waiting for fresh synchronized compressed relay images." };
  const inputRejected = inputStatus === "input_rejected" || status === "input_rejected";
  const expiredPose = inputRejected && (transport.timing_status === "timestamped_tf_wait_expired" ||
    /timestamped TF wait expired/i.test(transport.last_error || ""));
  if (inputStatus === "waiting_for_tf" || expiredPose || (poseReason === "timestamped_tf_unavailable" &&
      !inputRejected && !inputStatus.startsWith("waiting_for_"))) {
    return { ...result, state: "waiting_for_tf", style: "pending", label: "Waiting for TF",
      mapping: "paused", detail: transport.last_error || "Camera processing can continue; mapping waits for a valid image-time pose." };
  }
  if (inputRejected) {
    return { ...result, state: "input_rejected", style: "pending", label: "Input rejected",
      detail: transport.last_error || "The latest input failed synchronization or calibration checks." };
  }
  if (inputStatus.startsWith("waiting_for_") || status === "waiting_for_input") {
    return { ...result, state: inputStatus || status, style: "pending", label: (inputStatus || status).replaceAll("_", " "), detail: transport.last_error || "" };
  }
  if (mappingPaused(robot)) return { ...result, state: "mapping_paused", style: "pending", label: "Mapping paused", detail: String(poseReason || "Pose evidence is not currently safe to integrate.").replaceAll("_", " ") };
  if ([status, robot.mapping_status].includes("capacity_reached")) return { ...result, state: "capacity_reached", style: "pending", label: "Evidence store full", detail: "New retained keyframes are paused; existing evidence is preserved." };
  if (status === "recorded") return { ...result, state: "recorded", style: "supplied", label: "Recorded", mapping: "recorded evidence" };
  if (!finite(robot.frame_age_s)) return { ...result, state: status || "waiting", style: "pending", label: (status || "Waiting").replaceAll("_", " ") };
  if (status === "processing" || perception.status === "processing") return { ...result, state: "processing", style: "reference", label: "Processing" };
  return result;
}
function healthState(robot) { return healthPresentation(robot).state; }

// Upstream errors can contain ROS frame names, timestamps and multiline text.
// Keep the overview short; preserve the original message in the disclosure.
function roverHealthCard(robot) {
  const health = healthPresentation(robot), sensor = currentComponent(robot, "health");
  const pose = currentComponent(robot, "perception").pose_quality || robot.pose_quality;
  const timing = robot.timing || {}, relay = sensor.camera_relay || {};
  const cycle = timing.cycle_avg_s ?? timing.cycle_s ?? (finite(timing.average_ms) ? timing.average_ms / 1000 : finite(timing.cycle_ms) ? timing.cycle_ms / 1000 : null);
  const transport = {dds_compressed: "Compressed DDS", ssh_compressed_pair: "Compressed SSH"}[sensor.camera_transport];
  const localAge = robot.local_map_update_age_s ?? robot.map_update_age_s ?? robot.map_observation_age_s;
  const mapAge = Object.hasOwn(robot, "shared_evidence_age_s") ? robot.shared_evidence_age_s : robot.map_update_age_s ?? robot.map_observation_age_s;
  const expired = sensor.timing_status === "timestamped_tf_wait_expired" || /TF wait expired/i.test(health.detail);
  const shortSync = health.state === "waiting_for_tf" && sensor.status === "waiting_for_tf" && !expired &&
    finite(sensor.pending_wait_s) && sensor.pending_wait_s >= 0 && sensor.pending_wait_s < 1 &&
    finite(localAge) && localAge >= 0 && localAge < 6;
  let summary = health.label;
  switch (health.state) {
    case "healthy": case "processing": summary = robot.evidence_status === "redundant" || robot.evidence?.outcome === "redundant"
      ? "Live processing; redundant views are not stored again." : "No current warning."; break;
    case "stale":
      summary = health.label === "Depth stale" ? "Depth has stopped updating." :
        health.label === "RGB-D stale" ? "RGB and depth have stopped updating." : "RGB has stopped updating.";
      if (sensor.camera_relay_probing || sensor.status === "waiting_for_camera_relay") summary += " Trying the backup feed.";
      break;
    case "waiting_for_camera_relay": summary = "Waiting for fresh RGB and depth from the backup feed."; break;
    case "waiting_for_tf":
      summary = sensor.camera_relay_probing && sensor.camera_relay_probe_reason === "tf" ? "Checking the backup feed for an image-time pose."
        : shortSync ? "Synchronizing image and pose." : expired ? "Image-time pose unavailable; frame skipped."
        : "Waiting for the pose at image capture time.";
      break;
    case "input_rejected": summary = "Frame failed input checks; see diagnostics."; break;
    case "mapping_paused": summary = "Pose check blocked map integration."; break;
    case "capacity_reached": summary = "Shared evidence limit reached. Open Storage & maps to manage it."; break;
    case "error": summary = health.label === "Adapter stopped" ? "Input adapter stopped; see diagnostics." : "Processing failed; see diagnostics."; break;
    case "recorded": summary = "Stored observations; not a live source."; break;
    case "simulation_waiting_for_clock": summary = "Waiting for Gazebo's simulation clock."; break;
    case "simulation_paused_or_stalled": summary = "Simulation paused or clock publisher stopped."; break;
    case "simulation_reset_restart_required": summary = "World reset detected; restart the CrossMaps simulation session."; break;
    default: summary = human(health.label) + "…";
  }
  const rows = [
    ["Session", robot.session_id], ["Input", human(health.input)], ["Perception", human(health.perception)],
    ["Mapping", human(health.mapping)], ["Pose check", typeof pose === "object" ? pose?.status : pose],
    ["Pose reason", typeof pose === "object" ? pose?.reason : null],
    ["Timing", sensor.timing_status], ["Last mapped input age (s)", robot.map_observation_age_s],
    ["Local fusion completed (s ago)", localAge], ["Shared evidence retained (s ago)", robot.shared_evidence_age_s],
    ["Evidence decision", robot.evidence_status || robot.evidence?.reason || robot.evidence?.outcome],
    ["Clock mode", sensor.clock_mode], ["Simulation clock", sensor.simulation_clock_status],
    ["Last pose source", {dds_tf:"ROS on laptop", source_tf:"capture-time ROS pose from rover"}[sensor.last_pose_source]],
    ["Frames with rover-supplied pose", sensor.source_tf_pairs], ["Last measured pose coverage lag (s)", sensor.tf_coverage_lag_s],
    ["Frame waiting for pose (s)", sensor.pending_wait_s],
    ["TF wait budget (s)",sensor.tf_wait_budget_s], ["Last resolved pose wait (s)", sensor.tf_resolved_wait_s],
    ["Delayed poses recovered", sensor.tf_delayed_recovered],
    ["TF wait expirations", sensor.tf_wait_expired], ["RGB age at arrival (s)", sensor.rgb_header_age_at_receipt_s],
    ["Depth age at arrival (s)", sensor.depth_header_age_at_receipt_s],
    ["Camera relay", relay.state], ["Recovery probe", sensor.camera_relay_probing ? sensor.camera_relay_probe_reason : null],
    ["Relay reconnect attempts", relay.attempts], ["Stale relay pairs", relay.stale_pairs],
    ["Status detail", health.detail], ["Latest adapter message", sensor.last_error !== health.detail ? sensor.last_error : null],
    ["Last relay error", relay.last_error || sensor.camera_relay_last_error]
  ];
  return { label: shortSync || ["healthy","processing"].includes(health.state) ? "Live" : health.label,
    style: shortSync || health.state === "processing" ? "healthy" : health.style, summary,
    meta: [`Domain ${robot.domain ?? "—"}`, transport, Object.hasOwn(robot, "local_map_update_age_s") ? `Local fusion: ${ageLabel(localAge)} ago` : null].filter(Boolean).join(" · "),
    values: [ageLabel(robot.frame_age_s), ageLabel(robot.observation_age_s), ageLabel(mapAge), ageLabel(cycle),
      `${robot.keyframes ?? robot.counts?.keyframes ?? 0} / ${robot.submaps ?? robot.counts?.submaps ?? 0}`],
    diagnostics: rows.filter(([, v]) => v != null && v !== "").map(([k, v]) => `${k}: ${v}`).join("\n") };
}
function human(value) { return String(value ?? "").replaceAll("_", " "); }

function roverCardRenderer(container) {
  const labels = ["Camera age", "Perception completed", "Shared evidence saved", "Mean cycle", "Keyframes / submaps"];
  let key, cards = [];
  const put = (el, text) => { if (el.textContent !== text) el.textContent = text; };
  return (robots) => {
    const next = JSON.stringify(robots.map(robotId));
    if (next !== key) {
      key = next;
      container.innerHTML = robots.map((r,i) => `<article class="robot-card"><div class="robot-top"><div class="robot-name"><i class="source-dot ${i%2 ? "violet" : ""}"></i>Rover ${escapeText(robotId(r))}</div><span class="badge"></span></div>
        <div class="robot-details">${labels.map(l => `<div class="metric"><span>${l}</span><strong></strong></div>`).join("")}</div>
        <div class="robot-meta"></div><details class="rover-diagnostics"><summary><span class="rover-summary"></span><span class="rover-details-label">Diagnostics</span></summary>
        <pre tabindex="0" aria-label="Full diagnostics for rover ${escapeText(robotId(r))}"></pre></details></article>`).join("") || '<div class="robot-card muted">Waiting for rover configuration from the coordinator.</div>';
      cards = [...container.querySelectorAll(".robot-card")].slice(0, robots.length).map(card => {
        const c = { badge:card.querySelector(".badge"), values:card.querySelectorAll(".metric strong"), meta:card.querySelector(".robot-meta"),
          details:card.querySelector("details"), summary:card.querySelector(".rover-summary"), raw:card.querySelector("pre") };
        c.details.addEventListener("toggle", () => { if (c.details.open) put(c.raw, c.latest); });
        return c;
      });
    }
    robots.forEach((r,i) => {
      const c = cards[i], model = roverHealthCard(r);
      // Close diagnostics for a replaced source session; retain the DOM/focus on
      // ordinary polls. While open, freeze the diagnostic snapshot for reading.
      if (c.session !== r.session_id) { c.details.open = false; c.session = r.session_id; }
      c.badge.className = `badge ${model.style}`; put(c.badge, model.label);
      model.values.forEach((v,j) => put(c.values[j],v)); put(c.meta, model.meta);
      put(c.summary, model.summary); c.summary.className = `rover-summary ${model.style}`;
      c.latest = "Snapshot · close and reopen to refresh\n" + model.diagnostics;
      if (!c.details.open) put(c.raw, c.latest);
    });
  };
}
function displayedImageAge(camera, nowMs) {
  return finite(camera.displayedAge) && finite(camera.displayedAt) ? camera.displayedAge + Math.max(0, nowMs - camera.displayedAt) / 1000 : null;
}
// Prefer the server's age for these exact bytes. A state poll can describe a
// different frame, and fetching the same JPEG again does not make it younger.
function receivedImageAge(header, headerAt, snapshotAge, snapshotAt, displayedAt) {
  const raw = typeof header === "string" && header.trim() !== "" ? Number(header) : NaN;
  return finite(raw) && raw >= 0
    ? displayedImageAge({displayedAge: raw, displayedAt: headerAt}, displayedAt)
    : displayedImageAge({displayedAge: snapshotAge, displayedAt: snapshotAt}, displayedAt);
}
// Aborting fetch does not abort HTMLImageElement.decode() on its blob URL.
// Race the full decode against the same deadline so Safari can retry a hung image.
function abortableDecode(image, signal) {
  return new Promise((resolve, reject) => {
    function done(callback, value) {
      signal.removeEventListener("abort", abort);
      callback(value);
    }
    function abort() { done(reject, new DOMException("Image decode aborted", "AbortError")); }
    if (signal.aborted) { abort(); return; }
    signal.addEventListener("abort", abort, { once: true });
    Promise.resolve().then(() => image.decode()).then(() => done(resolve), error => done(reject, error));
  });
}
function frameWarning(view, { mode, connected, preview, available, age, simulationClock }) {
  if (!available || mode === "recorded_replay" || !preview) return "";
  if (!connected) return "Last frame · dashboard disconnected";
  if (simulationClock === "paused_or_stalled") return "Simulation paused or clock stopped";
  if (simulationClock === "reset_restart_required") return "World reset · restart session";
  if (view === "rgb" && finite(age) && age > 5) return `Stale RGB · last frame ${ageLabel(age)} old`;
  return "";
}
function imageCaption(robot, view, { mode, preview = true, available = false, age = null } = {}) {
  if (!available) return "No image for this view/session";
  if (mode === "recorded_replay") return "Recorded observation · not live";
  const clock = currentComponent(robot, "health");
  if (clock.clock_mode === "simulation" && clock.simulation_clock_status === "paused_or_stalled")
    return "Last simulation frame · Gazebo paused or clock stopped";
  if (clock.clock_mode === "simulation" && clock.simulation_clock_status === "reset_restart_required")
    return "Previous simulation epoch · restart session after world reset";
  const mapped = ["segments", "entities"].includes(view);
  const kind = view === "rgb" ? "Camera frame" : mapped ? "Last mapped overlay" : "Perception image";
  const label = finite(age) ? `${ageLabel(age)} old` : "age unknown";
  return `${preview ? "" : "Preview paused · "}${kind} · ${label}${mapped && mappingPaused(robot) ? " · mapping paused" : ""}`;
}
function shadowDescription(enabled) {
  return enabled !== false
    ? "Shadow mode keeps local perception and mapping running. Provisional overlays and automatic acceptance are off; existing and manually accepted links remain active."
    : "Automatic alignment is enabled. Agreeing RGB-D matches can preview a provisional pose; a distinct viewpoint can confirm it. Only confirmed geometry supports shared identities. Existing corroborated entity proposals can be accepted immediately; robot localization is unchanged.";
}
function previewRecords(snapshot) {
  const records = snapshot.alignment_evidence?.provisional;
  return snapshot.shadow_mode === false && Array.isArray(records) ? records.filter(r => r?.status === "provisional") : [];
}
function alignmentEvidence(snapshot) {
  const candidates = snapshot.candidates || [], constraints = snapshot.constraints || [];
  const supplied = snapshot.alignment_evidence || {};
  const count = (key, fallback) => finite(supplied[key]) ? Math.max(0, Math.floor(supplied[key])) : fallback;
  const retrieved = count("retrieved", candidates.length);
  const checked = count("checked", candidates.filter((candidate) => candidate.verification && typeof candidate.verification.ok === "boolean").length);
  const verified = count("verified", candidates.filter((candidate) => candidate.verification?.ok === true || ["verified", "corroborated", "accepted"].includes(candidate.state)).length);
  const active = count("active", constraints.filter((constraint) => constraint.kind === "loop" && constraint.active).length);
  const anchors = constraints.filter((constraint) => constraint.kind === "anchor" && constraint.active).length;
  const failed = candidates.filter((candidate) => candidate.verification?.ok === false)
    .sort((a, b) => (b.verification.inliers || 0) - (a.verification.inliers || 0));
  const best = Object.hasOwn(supplied, "best_rejection") ? supplied.best_rejection : failed[0]?.verification;
  const previews = previewRecords(snapshot);
  const diagnostics = snapshot.shadow_mode === false && Array.isArray(supplied.provisional_diagnostics)
    ? supplied.provisional_diagnostics.filter(r => r && ["waiting", "withheld"].includes(r.status)) : [];
  let title, next;
  if (previews.length) {
    title = "Provisional alignment · awaiting a distinct view";
    next = "A tentative overlay from agreeing RGB-D matches is available in Combined. It is not a pose-graph constraint and cannot link objects. A compatible verified match from a visibly different view on either rover can promote the alignment automatically.";
    if (snapshot.mode === "recorded_replay") next += " Recorded replay does not collect new views.";
  } else if (active > 0 || anchors > 0) {
    title = anchors > 0 && active === 0 ? "Supplied alignment active" : "Cross-rover alignment active";
    next = "Object identity still needs its own aligned geometric and appearance evidence; alignment does not merge every segment.";
  } else if (verified === 0) {
    title = retrieved ? "First verified pair still needed" : "Waiting for overlapping keyframes";
    next = "Retrieval similarity is not alignment confidence. First establish a geometrically verified overlapping pair; repeated stationary frames do not replace that check.";
  } else if (snapshot.shadow_mode !== false) {
    title = "Verified evidence awaiting a decision";
    next = "Shadow mode prevents automatic previews and acceptance. Disable shadow to allow a tentative overlay from agreeing matches, or accept a verified constraint explicitly. One rover changing viewpoint is sufficient for automatic confirmation.";
  } else {
    title = "Distinct-view corroboration still needed";
    next = "Moving one rover is sufficient for a distinct-view confirmation. Stationary repeats cannot confirm alignment; a provisional overlay may appear if enough strong frame pairs agree.";
  }
  const reasons = { insufficient_distinctive_mutual_matches: "too few distinctive visual matches", insufficient_correspondences: "too few matches with valid depth", insufficient_geometric_inliers: "too few consistent 3D matches", localized_or_ambiguous_correspondences: "correspondences are too localized or ambiguous", insufficient_depth_overlap: "insufficient depth overlap or excessive residual", icp_disagrees_with_visual_geometry: "depth refinement disagrees with visual geometry" };
  const beforeFit = ["visual_features", "distinctive_matching", "depth_correspondences"].includes(best?.verification_stage);
  const support = beforeFit ? ` · ${best.visual_matches ?? 0} visual matches; 3D fit not run` : finite(best?.inliers) ? ` · ${best.inliers} inliers` : "";
  const rejection = best ? `${reasons[best.reason] || String(best.reason || "verification failed").replaceAll("_", " ")}${support}` : "";
  return { retrieved, checked, verified, active, previews, diagnostics, unchecked: Math.max(0, retrieved - checked), title, next, rejection,
    scope: snapshot.alignment_evidence ? "Retained candidate pairs" : "Displayed candidate pairs" };
}
function candidateReason(candidate) {
  return candidate.reason === "retrieved_not_verified" && !candidate.verification
    ? "Not checked within the per-keyframe verification budget. Skipped candidates may be revisited; this is not a failed match or a verified alignment."
    : candidate.reason || candidate.verification?.reason || "Awaiting geometric verification";
}

class WorldViewer {
  constructor(canvas, overlayCanvas) {
    this.canvas = canvas; this.gl = canvas.getContext("webgl", { antialias: true, alpha: false });
    this.overlayCanvas = overlayCanvas; this.overlayContext = overlayCanvas?.getContext("2d"); this.overlay = { markers: [], trails: [] };
    this.positions = []; this.count = 0; this.previewPoints = 0; this.target = [0, 0, 0]; this.distance = 9; this.yaw = -.85; this.pitch = .7; this.pointSize = 3; this.autoFit = true;
    if (!this.gl) return;
    const gl = this.gl;
    const shader = (type, source) => { const result = gl.createShader(type); gl.shaderSource(result, source); gl.compileShader(result); if (!gl.getShaderParameter(result, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(result)); return result; };
    this.program = gl.createProgram();
    gl.attachShader(this.program, shader(gl.VERTEX_SHADER, "attribute vec3 p;attribute vec3 c;uniform mat4 m;uniform float s;varying vec3 col;void main(){gl_Position=m*vec4(p,1.);gl_PointSize=s;col=c;}"));
    gl.attachShader(this.program, shader(gl.FRAGMENT_SHADER, `precision mediump float;varying vec3 col;uniform float roundPoint;
      void main(){vec3 color=col;if(roundPoint>1.5){vec2 q=abs(gl_PointCoord-vec2(.5));float d=q.x+q.y;
        if(d>.5)discard;if(d>.30)color=vec3(.98,.72,.30);
      }else if(roundPoint>.5 && length(gl_PointCoord-vec2(.5))>.5)discard;gl_FragColor=vec4(color,1.);}`));
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
  update(cloud) {
    this.positions = cloud.positions; this.count = cloud.positions.length / 3; this.previewPoints = cloud.previewPoints;
    if (!this.gl) return; this.points = this.buffer(cloud.positions, cloud.colors, this.points);
    if (this.autoFit && (this.count || this.overlay.markers.length || this.overlay.trails.length)) { this.fit(); this.autoFit = false; }
  }
  fit() {
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    const expand = (v, i) => { min[i] = Math.min(min[i], v); max[i] = Math.max(max[i], v); };
    this.positions.forEach((v, i) => expand(v, i%3));
    for (const marker of this.overlay.markers) marker.position.forEach(expand);
    for (const trail of this.overlay.trails) for (const p of trail.points) p.forEach(expand);
    if (!finite(min[0])) { this.target = [0, 0, 0]; this.distance = 9; return; }
    this.target = min.map((v,i) => (v + max[i])/2);
    this.distance = clamp(Math.hypot(...max.map((v,i) => v-min[i]))*1.6, 2, 1000);
  }
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
    const solid = this.points.count - this.previewPoints;
    for (const [buffer, mode, first, count, style] of [[this.grid, gl.LINES, 0, this.grid.count, 0], [this.points, gl.POINTS, 0, solid, 1], [this.points, gl.POINTS, solid, this.previewPoints, 2]]) {
      if (!count) continue;
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer.p); gl.enableVertexAttribArray(this.attr.p); gl.vertexAttribPointer(this.attr.p, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer.c); gl.enableVertexAttribArray(this.attr.c); gl.vertexAttribPointer(this.attr.c, 3, gl.UNSIGNED_BYTE, true, 0, 0);
      gl.uniform1f(this.attr.s, (style === 2 ? Math.max(5, this.pointSize + 2) : this.pointSize) * ratio);
      gl.uniform1f(this.attr.round, style); gl.drawArrays(mode, first, count);
    }
    // Only re-project traces when data, camera or viewport changed; idle polling
    // should not repaint hundreds of path vertices at display refresh rate.
    const viewKey = [width,height,ratio,...this.target,this.distance,this.pitch,this.yaw].join(",");
    if (viewKey !== this.overlayViewKey || this.drawnOverlay !== this.overlay) {
      drawRoverOverlay(this.overlayCanvas, this.overlayContext, this.overlay, vp, width, height, ratio);
      this.overlayViewKey = viewKey; this.drawnOverlay = this.overlay;
    }
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
    state.entity_links, state.events, state.stats, state.mode, state.shadow_mode,
    state.alignment_evidence, state._dashboard?.commands_enabled]);
}

function identityUrl(...ids) {
  return "/api/identity/" + ids.map(id => encodeURIComponent(String(id))).join("/");
}
function inspectButton(link) {
  return `<button type="button" aria-haspopup="dialog" data-inspect-identity="${escapeText(link.id)}">Compare evidence</button>`;
}
function witnessOption(member, view) {
  const count = finite(view.projected_points) ? ` · ${view.projected_points} points` : "";
  const distinct = (member.distinct_pair || []).includes(view.frame_id) ? " · corroborating view" : "";
  return `Keyframe ${view.seq ?? view.frame_id}${count}${distinct}`;
}
function selectedWitness(member, remembered) {
  const views = member.views || [];
  return views.find(view => view.frame_id === remembered) ||
    views.find(view => view.projected_points > 0 && (member.distinct_pair || []).includes(view.frame_id)) ||
    views.find(view => view.projected_points > 0) || views[0];
}
function witnessImage(linkId, member, view) {
  if (!view) return '<p class="witness-meta">No retained keyframe for this source.</p>';
  const caption = `Frame ${view.frame_id} · surface IDs: ${(view.surface_ids || []).join(", ") || "none"}`;
  const img = view.projected_points > 0
    ? `<img alt="Recorded RGB with projected support for ${escapeText(member.robot_id)} entity ${escapeText(member.local_id)}" src="${escapeText(identityUrl(linkId, member.id, view.frame_id))}.jpg">`
    : '<p class="witness-meta">No projected surface support for this keyframe.</p>';
  return `<figure class="witness-image">${img}<figcaption class="witness-meta">${escapeText(caption)}</figcaption></figure>`;
}
function identityContent(detail, remembered) {
  const cards = detail.members.map((member, index) => {
    const view = selectedWitness(member, remembered.get(member.id));
    return `<section class="witness-card"><h3>${escapeText(member.robot_id || "Unknown rover")} · local entity ${escapeText(member.local_id ?? "—")}</h3>
      <p class="witness-meta">Session ${escapeText(member.session_id || "not retained")}<br>Member ${escapeText(member.id)}</p>
      <p>${member.distinct_views ? "Distinct viewpoints recorded" : "Distinct viewpoints not confirmed"}</p>
      ${view ? `<label>Recorded witness<select data-witness-member="${index}">${member.views.map(v => `<option value="${escapeText(v.frame_id)}"${v === view ? " selected" : ""}>${escapeText(witnessOption(member, v))}</option>`).join("")}</select></label>` : ""}
      <div data-witness-image>${witnessImage(detail.id, member, view)}</div></section>`;
  }).join("");
  const reasons = (detail.blocking_reasons || []).map(reason => `<li>${escapeText(reason)}</li>`).join("");
  const { corroboration_translation: movement, corroboration_rotation_deg: turn } = detail.thresholds || {};
  return `<p><strong>${detail.merged ? "Shared identity active" : "Not merged"}</strong> · ${escapeText(detail.state || "unknown")} · ${detail.alignment_valid ? "alignment valid" : "alignment unavailable"}</p>
    <div class="evidence-metrics"><span>DINO ${numberLabel(detail.dino)}</span><span>CLIP ${numberLabel(detail.clip)}</span><span>3D overlap ${numberLabel(detail.overlap)}</span></div>
    ${detail.evidence_scope === "last_compatible" ? '<p class="robot-warning">Historical evidence from the last compatible comparison. These scores do not establish a current match.</p>' : ''}
    ${reasons ? `<h3>What blocks automatic linking?</h3><ul class="identity-blockers">${reasons}</ul>` : ""}
    ${finite(movement) && finite(turn) ? `<p class="inspector-note">Each rover needs two visibly different contributing views with a pose change of at least ${numberLabel(movement)} m or ${numberLabel(turn, 1)}°. Repeated parked frames don’t corroborate identity.</p>` : ""}
    <p class="inspector-note">${escapeText(detail.support_note || "Marks are projected surface samples, not full segmentation masks. These are recorded frames, not live cameras.")}</p>
    <div class="witness-grid">${cards || "<p>Source evidence is no longer available.</p>"}</div>`;
}

// Only an explicit open/refresh requests details. Closing or changing links
// invalidates the previous request even if a late network response still arrives.
function identityRequestReader(read) {
  let controller;
  return {
    cancel() { controller?.abort(); controller = null; },
    async load(id, receive, fail) {
      this.cancel();
      const request = controller = new AbortController();
      try {
        const detail = await read(id, request.signal);
        if (!detail || detail.id !== id || !Array.isArray(detail.members)) throw new Error("Evidence does not match the selected link.");
        if (controller === request) receive(detail);
      } catch (error) { if (controller === request) fail(error); }
    }
  };
}
function identityLinkSignature(snapshot, id) {
  const link = (snapshot.entity_links || []).find(item => item.id === id);
  return link ? JSON.stringify([snapshot.mode, snapshot.shadow_mode, link]) : null;
}
function identityFromHash(hash) {
  return /^#identity\/([0-9a-f]{20})$/.exec(hash)?.[1] || null;
}
function installIdentityInspector($, getSnapshot) {
  const dialog = $("identity-inspector"), body = $("identity-content"), status = $("identity-status");
  let linkId = null, signature = null, detail = null, snapshotReady = false;
  const remembered = new Map();
  const reader = identityRequestReader(async (id, signal) => {
    const response = await fetch(identityUrl(id), { signal, cache: "no-store" });
    if (!response.ok) throw new Error(response.status === 404 ? "Evidence is not retained for this link." : `Coordinator returned ${response.status}.`);
    return response.json();
  });
  function load() {
    signature = identityLinkSignature(getSnapshot(), linkId);
    status.textContent = "Loading stored identity evidence…";
    $("identity-reload").disabled = true;
    reader.load(linkId, data => {
      detail = data;
      body.innerHTML = identityContent(data, remembered);
      status.textContent = "Stored evidence · " + modePresentation(getSnapshot()).label;
      $("identity-reload").disabled = false;
      checkSnapshot();
    }, error => {
      status.textContent = `Could not load evidence: ${error.message}${detail ? " Still showing the previously loaded snapshot." : ""}`;
      $("identity-reload").disabled = false;
    });
  }
  function checkSnapshot() {
    if (!snapshotReady) { snapshotReady = true; followHash(); }
    if (!dialog.open || !detail || $("identity-reload").disabled) return;
    const current = identityLinkSignature(getSnapshot(), linkId);
    if (current === null || current !== signature) status.textContent = current === null
      ? "This link is no longer in the latest snapshot. Showing previously loaded evidence."
      : "The proposal changed. Showing the snapshot you opened; refresh to compare current evidence.";
  }
  function openLink(id) {
    if (dialog.open && linkId === id) return;
    linkId = id;
    detail = null; remembered.clear(); body.replaceChildren();
    if (!dialog.open) dialog.showModal();
    load();
  }
  function followHash() {
    if (!snapshotReady) return;
    const id = identityFromHash(window.location.hash);
    if (id) openLink(id);
    else if (dialog.open) dialog.close();
  }
  window.addEventListener("hashchange", followHash);
  document.addEventListener("click", event => {
    const button = event.target.closest("[data-inspect-identity]");
    if (button) openLink(button.dataset.inspectIdentity);
  });
  $("identity-close").addEventListener("click", () => dialog.close());
  $("identity-reload").addEventListener("click", load);
  dialog.addEventListener("close", () => {
    if (dialog.open) return;
    reader.cancel();
    const previous = linkId;
    if (previous && identityFromHash(window.location.hash) === previous) {
      window.history.replaceState(window.history.state, "", window.location.pathname + window.location.search);
    }
    linkId = null; detail = null; remembered.clear(); body.replaceChildren();
    Array.from(document.querySelectorAll("[data-inspect-identity]")).find(el => el.dataset.inspectIdentity === previous)?.focus({ preventScroll: true });
  });
  body.addEventListener("change", event => {
    const select = event.target.closest("[data-witness-member]");
    const member = select && detail?.members[Number(select.dataset.witnessMember)];
    if (!member) return;
    remembered.set(member.id, select.value);
    select.closest(".witness-card").querySelector("[data-witness-image]").innerHTML = witnessImage(linkId, member, selectedWitness(member, select.value));
  });
  body.addEventListener("error", event => {
    if (!event.target.matches?.(".witness-image img")) return;
    const msg = document.createElement("p"); msg.className = "witness-meta"; msg.textContent = "Marked image unavailable for this recorded keyframe.";
    event.target.replaceWith(msg);
  }, true);
  return { checkSnapshot, reset() { reader.cancel(); if (dialog.open) dialog.close(); detail = null; remembered.clear(); body.replaceChildren(); } };
}

function startDashboard() {
  let lastEvidenceKey = null;
  const $ = (id) => document.getElementById(id);
  let snapshot = normalizeSnapshot({}), snapshotAt = null, source = "combined", preview = true, connected = false, pollBusy = false, sequence = 0, robotKey = "", lastRenderKey = "";
  const cameraState = new Map();
  const inspector = installIdentityInspector($, () => snapshot);
  let viewer;
  try { viewer = new WorldViewer($("world"), $("map-poses")); } catch (error) { notify(error.message, true); }
  function notify(message, error = false) { $("notice").hidden = false; $("notice").textContent = message; $("notice").classList.toggle("error", error); }
  async function command(payload, { quiet = false, refresh = true } = {}) {
    if (!connected || !snapshot._dashboard?.commands_enabled) throw new Error("Commands are available only from a connected local dashboard.");
    const response = await fetch("/api/command", { method: "POST", headers: { "Content-Type": "application/json", "X-CrossMaps-Token": snapshot._dashboard.command_token }, body: JSON.stringify(payload), signal: AbortSignal.timeout(15000) });
    const result = await response.json(); if (!response.ok || result.ok === false) throw new Error(result.error || "Command was rejected");
    if (!quiet) notify(result.message || result.reason || "Command applied. Local source evidence is preserved."); if (refresh) await poll(); return result;
  }
  const storage = installStorageDialog($, () => snapshot, () => connected, async payload => {
    const result = await command(payload, { quiet: true, refresh: false });
    // Let the controller show any committed-reset/partial-cleanup warning
    // before a new map generation arrives and clears stale review state.
    if (payload.action === "confirm_storage") setTimeout(poll, 0);
    return result;
  });
  const renderRobots = roverCardRenderer($("robots"));
  function invalidateCamera(camera, message) {
    camera.generation++; camera.controller?.abort();
    if (camera.url) URL.revokeObjectURL(camera.url);
    camera.url = null; camera.displayedAge = null; camera.displayedAt = null;
    const img = camera.card.querySelector("img"); img.hidden = true; img.removeAttribute("src");
    camera.card.querySelector(".camera-placeholder").hidden = false;
    camera.card.querySelector(".camera-placeholder").textContent = message;
    camera.card.querySelector("[data-image-status]").textContent = message;
    camera.card.querySelector("[data-image-age]").textContent = "No image for this view/session";
    camera.card.querySelector("[data-frame-warning]").hidden = true;
  }
  function refreshCameraCaptions() {
    for (const robot of snapshot.robots) {
      const camera = cameraState.get(robotId(robot));
      if (!camera) continue;
      const data = {mode: snapshot.mode, connected, preview, available: Boolean(camera.url), age: displayedImageAge(camera, performance.now()),
        simulationClock: currentComponent(robot, "health").clock_mode === "simulation" ? currentComponent(robot, "health").simulation_clock_status : null};
      camera.card.querySelector("[data-image-age]").textContent = connected ? imageCaption(robot, camera.view, data) : "Coordinator disconnected · last received frame";
      const badge = camera.card.querySelector("[data-frame-warning]");
      badge.textContent = frameWarning(camera.view, data); badge.hidden = !badge.textContent;
    }
  }
  function ensureCameras() {
    const key = snapshot.robots.map(robotId).join("|");
    if (key === robotKey) {
      for (const robot of snapshot.robots) {
        const camera = cameraState.get(robotId(robot));
        if (camera && camera.session !== robot.session_id) { camera.session = robot.session_id; invalidateCamera(camera, "New source session · waiting for selected view"); }
      }
      return;
    }
    robotKey = key;
    for (const camera of cameraState.values()) invalidateCamera(camera, "Source list changed");
    cameraState.clear();
    $("source-tabs").innerHTML = '<button data-source="combined">Combined</button>' + snapshot.robots.map((robot) => `<button data-source="${escapeText(robotId(robot))}">Rover ${escapeText(robotId(robot))}</button>`).join("");
    if (source !== "combined" && !snapshot.robots.some((robot) => robotId(robot) === source)) source = "combined";
    $("cameras").innerHTML = snapshot.robots.map((robot, i) => `<article class="camera-card" data-robot="${escapeText(robotId(robot))}"><div class="camera-header"><strong><i class="source-dot ${i % 2 ? "violet" : ""}"></i> Rover ${escapeText(robotId(robot))}</strong><div class="camera-controls"><select aria-label="Rover ${escapeText(robotId(robot))} camera view"><option value="rgb">RGB</option><option value="sam">SAM masks</option><option value="segments">Segments</option><option value="entities">Entities</option><option value="dino">DINO</option><option value="dino_similarity">DINO similarity</option><option value="clip">CLIP</option><option value="depth">Depth</option></select><button data-expand>Expand</button></div></div><div class="camera-image"><img alt="Rover ${escapeText(robotId(robot))} camera" hidden><span class="camera-placeholder">Waiting for first image</span><span class="frame-warning" data-frame-warning hidden></span></div><div class="camera-caption"><span data-image-status>Not received</span><span data-image-age></span></div></article>`).join("");
    for (const card of $("cameras").querySelectorAll(".camera-card")) {
      const id = card.dataset.robot, camera = { card, view: "rgb", busy: false, generation: 0, url: null,
        session: snapshot.robots.find((robot) => robotId(robot) === id)?.session_id, displayedAge: null, displayedAt: null }; cameraState.set(id, camera);
      card.querySelector("select").addEventListener("change", (event) => { camera.view = event.target.value; invalidateCamera(camera, preview ? "Loading selected view…" : "Preview paused · resume to load selected view"); });
      card.querySelector("[data-expand]").addEventListener("click", () => fullscreen(card.querySelector(".camera-image")));
    }
    for (const id of ["anchor-robot", "anchor-reference"]) { const previous = $(id).value; $(id).innerHTML = snapshot.robots.map((robot) => `<option value="${escapeText(robotId(robot))}">Rover ${escapeText(robotId(robot))}</option>`).join(""); if (snapshot.robots.some((robot) => robotId(robot) === previous)) $(id).value = previous; }
    if (snapshot.robots.length > 1 && $("anchor-robot").value === $("anchor-reference").value) $("anchor-robot").value = robotId(snapshot.robots[1]);
  }
  async function refreshCamera(id, camera) {
    if (!preview || !connected || document.hidden || camera.busy) return;
    camera.busy = true; const generation = camera.generation, view = camera.view, sampledAt = snapshotAt;
    const sourceAge = imageAge(snapshot.robots.find((robot) => robotId(robot) === id) || {}, view);
    const controller = new AbortController(); camera.controller = controller;
    const timeout = setTimeout(() => controller.abort(), 7000); let url;
    try {
      const response = await fetch(`/api/image?robot=${encodeURIComponent(id)}&view=${encodeURIComponent(view)}&seq=${sequence++}`, { cache: "no-store", signal: controller.signal });
      const header = response.headers.get("X-CrossMaps-Image-Age-S"), headerAt = performance.now();
      if (!response.ok) { const error = new Error(response.status === 404 ? "No image yet for this view/session" : `Image unavailable (${response.status})`); error.missing = response.status === 404; throw error; }
      const blob = await response.blob(); if (!blob.type.startsWith("image/")) throw new Error("Invalid camera response");
      url = URL.createObjectURL(blob); const decoded = new Image(); decoded.src = url; await abortableDecode(decoded, controller.signal);
      if (generation !== camera.generation || !preview || !connected) return;
      const img = camera.card.querySelector("img"), old = camera.url; img.src = url; img.hidden = false; camera.url = url; url = null; if (old) URL.revokeObjectURL(old);
      camera.displayedAt = performance.now(); camera.displayedAge = receivedImageAge(header, headerAt, sourceAge, sampledAt, camera.displayedAt);
      camera.card.querySelector(".camera-placeholder").hidden = true; camera.card.querySelector("[data-image-status]").textContent = `${camera.view.toUpperCase()} · ${decoded.naturalWidth} × ${decoded.naturalHeight}`;
      refreshCameraCaptions();
    } catch (error) {
      if (generation === camera.generation) {
        if (error.missing) invalidateCamera(camera, error.message);
        else if (preview) { camera.card.querySelector("[data-image-status]").textContent = error.name === "AbortError" ? "Image request timed out · keeping last frame" : error.message; if (camera.card.querySelector("img").hidden) camera.card.querySelector(".camera-placeholder").textContent = error.message; }
      }
    } finally { if (url) URL.revokeObjectURL(url); clearTimeout(timeout); camera.busy = false; }
  }
  function renderMap() {
    for (const button of $("source-tabs").querySelectorAll("button")) button.classList.toggle("active", button.dataset.source === source);
    const showProvisional = $("show-provisional").checked, showMarkers = $("show-rovers").checked;
    const roverOverlay = buildRoverOverlay(snapshot, source, { showProvisional, showMarkers, showTrails: $("show-trails").checked, connected });
    const cloud = buildCloud(snapshot, source, $("color-mode").value, Number($("budget").value), showProvisional);
    if (viewer) { viewer.overlay = roverOverlay; viewer.update(cloud); }
    $("overlay-control").hidden = !cloud.previewAvailable;
    $("point-count").textContent = `${(cloud.positions.length / 3).toLocaleString()} points · ${cloud.submaps} submaps`;
    $("map-frame").textContent = source === "combined" ? (cloud.previewShown ? "Shared world · provisional overlay" : "Shared world · confirmed geometry only") : `Rover ${source} · local frame`;
    $("map-frame").classList.toggle("tentative", Boolean(cloud.previewShown));
    $("map-empty").hidden = Boolean(viewer?.gl && (cloud.positions.length || roverOverlay.markers.length || roverOverlay.trails.length));
    $("map-empty").querySelector("h3").textContent = !viewer?.gl ? "WebGL is unavailable" : source === "combined" && cloud.excluded.length ? "Maps not aligned yet" : "Waiting for spatial evidence";
    $("map-empty").querySelector("p").textContent = !viewer?.gl ? "Camera feeds and evidence remain available. Enable WebGL to see the 3D map." : cloud.excluded.length ? "Choose a rover to inspect its local map. Unaligned submaps are never overlaid at a guessed origin." : "Selected keyframes and measured geometry will appear here.";
    const previewNote = cloud.previewAvailable ? (cloud.previewShown
      ? `${cloud.previewShown} provisional submap(s) shown as amber-edged diamonds. This overlay can change and does not enable shared object IDs.`
      : `${cloud.previewAvailable} provisional submap(s) hidden by the display filter.`)
      : snapshot.shadow_mode !== false ? "Shadow mode: provisional display and automatic acceptance are off." : "See the alignment evidence below.";
    $("alignment-summary").textContent = source === "combined" ? `${cloud.excluded.length} unaligned submap(s) excluded. ${previewNote}` : "Local source geometry. This view does not imply cross-rover alignment or corrected odometry.";
    $("map-legend").innerHTML = snapshot.robots.map((robot, index) => {
      if (source !== "combined" && robotId(robot) !== source) return "";
      const tentative = !robot.aligned && robot.alignment_state === "provisional" && snapshot.shadow_mode === false;
      const label = roverPoseLegend(robot, roverOverlay.markers.find(m => m.id === robotId(robot)), source, showMarkers);
      return `<span><i class="source-dot ${index % 2 ? "violet" : ""} ${source === "combined" && tentative ? "tentative" : ""}"></i>Rover ${escapeText(robotId(robot))} · ${label}</span>`;
    }).join("");
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
    const candidateScroll = $("candidates").scrollTop, eventScroll = $("events").scrollTop, entityScroll = $("entities").scrollTop;
    const alignment = alignmentEvidence(snapshot);
    $("candidate-count").textContent = alignment.retrieved;
    const previewRows = alignment.previews.map(r => `<p><strong>Preview: ${escapeText(r.source_robot)} ↔ ${escapeText(r.target_robot)}</strong> · ${escapeText(r.support_count)} agreeing frame pairs.</p>`).join("");
    const pendingRows = alignment.diagnostics.map(r => `<p>Provisional preview ${escapeText(r.status)} · ${escapeText(r.source_robot)} ↔ ${escapeText(r.target_robot)}: ${escapeText(r.reason)}${finite(r.support_count) && finite(r.required_count) ? ` (${r.support_count}/${r.required_count} pairs)` : ""}.</p>`).join("");
    $("alignment-evidence").innerHTML = `<div class="evidence-metrics"><span>${escapeText(alignment.scope)}: <strong>${alignment.retrieved}</strong></span><span>Checked: <strong>${alignment.checked}</strong></span><span>Verified: <strong>${alignment.verified}</strong></span><span>Active constraints (confirmed): <strong>${alignment.active}</strong></span></div><h3>${escapeText(alignment.title)}</h3><p>${escapeText(alignment.next)}</p>${previewRows}${pendingRows}${alignment.rejection ? `<p>Best failed verification: ${escapeText(alignment.rejection)}.</p>` : ""}${alignment.unchecked ? `<p class="muted">${alignment.unchecked} pair(s) not checked within the verification budget. A high retrieval score alone cannot activate them.</p>` : ""}`;
    const anchorCards = snapshot.constraints.filter((constraint) => constraint.kind === "anchor" && constraint.active).map((constraint) => `<article class="evidence-card"><div class="evidence-title"><strong>Rover ${escapeText(constraint.robot_id || constraint.source)} → ${escapeText(constraint.reference_robot_id || constraint.target)}</strong><span class="badge supplied">Supplied anchor</span></div><p>Operator-provided alignment. This is not an automatic geometric match.</p><details data-key="${escapeText(constraint.id)}"><summary>Inspect supplied transform</summary><pre>${escapeText(JSON.stringify({ T_reference_local: constraint.T_reference_local, T_target_source: constraint.T_target_source }, null, 2))}</pre></details><div class="evidence-actions"><button data-action="reject_constraint" data-id="${escapeText(constraint.id)}">Undo supplied anchor</button></div></article>`).join("");
    const candidateCards = orderedCandidates(snapshot).slice(0, 100).map((candidate) => {
      const verification = candidate.verification || {}, constraint = candidateConstraint(candidate, snapshot), transform = candidate.transform || constraint?.T_target_source || verification.T_target_source;
      const state = constraint?.active ? constraint.state || "accepted" : candidate.state || "proposed";
      return `<article class="evidence-card"><div class="evidence-title"><strong>${escapeText(candidate.source_robot ? `Rover ${candidate.source_robot}` : candidate.source)} ↔ ${escapeText(candidate.target_robot ? `Rover ${candidate.target_robot}` : candidate.target)}</strong><span class="badge ${escapeText(state)}">${escapeText(state)}</span></div><p>${escapeText(candidateReason(candidate))}</p><div class="evidence-metrics"><span>Retrieval score ${numberLabel(candidate.score)} · not confidence</span><span>Inliers ${escapeText(verification.inliers ?? verification.inlier_count ?? candidate.inliers ?? "—")}</span><span>RMSE ${numberLabel(verification.rmse ?? candidate.rmse, 3)} m</span><span>Corroborations ${escapeText(constraint?.corroborations ?? 0)}</span></div><details data-key="${escapeText(candidate.id)}"><summary>Inspect evidence and transform</summary><pre>${escapeText(JSON.stringify({ source: candidate.source, target: candidate.target, cues: candidate.cues, verification, T_target_source: transform }, null, 2))}</pre></details>${constraint ? `<div class="evidence-actions">${!constraint.active && !["rejected", "invalid"].includes(constraint.state) ? `<button data-action="accept_constraint" data-id="${escapeText(constraint.id)}">Accept verified constraint</button>` : ""}<button data-action="reject_constraint" data-id="${escapeText(constraint.id)}">${constraint.active ? "Undo constraint" : "Reject constraint"}</button></div>` : ""}</article>`;
    });
    $("candidates").innerHTML = candidateCards.slice(0, 8).join("") || '<div class="empty-list">No cross-rover candidates yet.<br>Distinct keyframes and overlapping places are needed before verification.</div>';
    const schedule = snapshot.stats?.verification_scheduler;
    if (schedule) {
      const coverage = document.createElement("p");
      coverage.className = "muted";
      coverage.textContent = `Across this run: ${schedule.checked_unique_pairs} unique frame pairs checked across ${schedule.distinct_checked_submap_pairs} submap pairs. ${schedule.backfill_checks} deferred checks completed; ${schedule.pending_pairs} queued.`;
      $("alignment-evidence").appendChild(coverage);
    }
    if (candidateCards.length > 8) $("candidates").insertAdjacentHTML("beforeend", `<details data-key="additional-candidates"><summary>Show ${candidateCards.length - 8} more candidates and rejected matches</summary>${candidateCards.slice(8).join("")}</details>`);
    $("candidates").insertAdjacentHTML("afterbegin", anchorCards);
    const shared = snapshot.global_entities.filter((entity) => (entity.members || []).length > 1);
    $("entity-count").textContent = shared.length;
    $("entities").innerHTML = shared.slice(0, 60).map((entity) => {
      const memberIds = (entity.members || []).map((member) => typeof member === "string" ? member : member.id);
      const links = snapshot.entity_links.filter((link) => activeEntityLink(link) && (link.members || []).every((id) => memberIds.includes(id)));
      return `<article class="evidence-card"><div class="evidence-title"><strong>${escapeText(entity.id)}</strong><span class="badge verified">${escapeText(finite(entity.confidence) ? `Confidence ${numberLabel(entity.confidence)}` : String(entity.confidence || "linked").replaceAll("_", " "))}</span></div><p class="members">${(entity.members || []).map((member) => escapeText(typeof member === "string" ? member : `${member.robot_id} / ${member.local_id}`)).join(" ↔ ")}</p><p>${escapeText(entity.reason || "Cross-rover identity hypothesis; source geometry remains separate.")}</p><div class="evidence-actions">${links.map((link) => `${inspectButton(link)}<button data-action="unlink_entities" data-id="${escapeText(link.id)}">Unlink identity</button>`).join("")}</div></article>`;
    }).join("") || '<div class="empty-list">No shared identities yet.<br>Local segments and entity hypotheses remain with their source rover.</div>';
    const memberLabels = new Map(snapshot.global_entities.flatMap(group => (group.members || [])
      .filter(m => m && typeof m === "object" && m.robot_id && m.local_id != null)
      .map(m => [m.id, `${m.robot_id} · local entity ${m.local_id}`])));
    const proposedLinks = snapshot.entity_links.filter((link) => link.state !== "rejected" && (!link.active || !link.alignment_valid || link.compatible_current !== true)).slice(-40);
$("entities").insertAdjacentHTML("beforeend", proposedLinks.map((link) => `<article class="evidence-card"><div class="evidence-title"><strong>${link.active ? "Suspended identity link" : "Identity proposal"}</strong><span class="badge pending">${escapeText(link.active ? "suspended" : link.state)}</span></div><p class="members">${(link.members || []).map(id => escapeText(memberLabels.get(id) || id)).join(" ↔ ")}</p>${link.reason ? `<p>${escapeText(link.reason)}</p>` : ""}<div class="evidence-metrics"><span>DINO ${numberLabel(link.dino)}</span><span>Overlap ${numberLabel(link.overlap)}</span><span>${link.alignment_valid ? "Alignment valid" : "Alignment unavailable"}</span></div><div class="evidence-actions">${inspectButton(link)}${!link.active && link.alignment_valid && link.compatible_current === true ? `<button data-action="accept_entity_link" data-id="${escapeText(link.id)}">Accept identity link</button>` : ""}<button data-action="unlink_entities" data-id="${escapeText(link.id)}">${link.active ? "Undo identity link" : "Reject identity link"}</button></div></article>`).join(""));
    $("events").innerHTML = snapshot.events.slice().reverse().slice(0, 60).map((event) => { const stamp = event.time ?? event.timestamp ?? event.stamp; const label = finite(stamp) ? new Date(stamp * (stamp > 1e12 ? 1 : 1000)).toLocaleTimeString() : ""; return `<div class="event"><time>${escapeText(label)}</time><span>${escapeText(typeof event === "string" ? event : event.message || event.reason || event.type || JSON.stringify(event))}</span></div>`; }).join("") || '<div class="empty-list">Coordinator events will appear here.</div>';
    $("revision").textContent = `revision ${snapshot.revision ?? "—"}`;
    $("stats").textContent = Object.entries(snapshot.stats).filter(([, value]) => typeof value !== "object").slice(0, 8).map(([key, value]) => `${key.replaceAll("_", " ")}: ${value}`).join(" · ");
    for (const button of document.querySelectorAll("[data-action], #save, #anchor-form button")) button.disabled = !snapshot._dashboard?.commands_enabled;
    $("save").disabled = snapshot.mode === "recorded_replay" || !snapshot._dashboard?.commands_enabled;
    $("save").title = snapshot.mode === "recorded_replay" ? "Replay decisions are in memory only; the source recording is unchanged" : "Checkpoint retained evidence and decisions";
    for (const details of $("candidates").querySelectorAll("details")) details.open = expanded.has(details.dataset.key);
    $("candidates").scrollTop = candidateScroll; $("events").scrollTop = eventScroll; $("entities").scrollTop = entityScroll;
  }
  async function poll() {
    if (pollBusy) return; pollBusy = true;
    try {
      const response = await fetch("/api/state", { cache: "no-store", signal: AbortSignal.timeout(7000) }); if (!response.ok) throw new Error(`Coordinator returned ${response.status}`);
      const previousGeneration = snapshot.map_generation;
      snapshot = normalizeSnapshot(await response.json()); connected = true; snapshotAt = performance.now();
      if (previousGeneration && snapshot.map_generation !== previousGeneration) {
        lastEvidenceKey = null; lastRenderKey = ""; inspector.reset();
        for (const camera of cameraState.values()) invalidateCamera(camera, "Map rebuilt · waiting for a fresh view…");
        if (viewer) viewer.autoFit = true;
      }
      $("connection").textContent = `Coordinator connected · ${snapshot.robots.length} rover sources`; $("connection-dot").classList.add("online");
      const mode = modePresentation(snapshot); $("mode-badge").textContent = mode.label; $("mode-badge").className = `badge ${mode.style}`;
      $("shadow").checked = snapshot.shadow_mode !== false; $("shadow").disabled = !snapshot._dashboard?.commands_enabled;
      $("shadow-explanation").textContent = shadowDescription(snapshot.shadow_mode);
      renderRobots(snapshot.robots); ensureCameras(); renderEvidence(); inspector.checkSnapshot(); storage.refresh();
      renderSimulationEvaluation($("simulation-evaluation"), $("simulation-evaluation-content"), snapshot, connected);
      const key = JSON.stringify([snapshot.revision, snapshot.shadow_mode, source, $("color-mode").value, $("budget").value,
        snapshot.robots.map(r => [r.session_id, r.trajectory?.latest?.seq, r.trajectory?.tracking_lost, mappingPaused(r), r.map_observation_age_s > 8]),
        snapshot.submaps.map((s) => [s.id, s.aligned, s.alignment_state, s.T_preview_world_submap, s.pose_version, s.points?.length])]);
      if (key !== lastRenderKey || snapshot.revision === undefined) { renderMap(); lastRenderKey = key; }
      refreshCameraCaptions();
    } catch (error) { const wasConnected = connected; connected = false; $("connection-dot").classList.remove("online"); $("connection").textContent = `Disconnected · ${error.message}`; for (const button of document.querySelectorAll("[data-action], #save, #anchor-form button")) button.disabled = true; $("shadow").disabled = true; storage.refresh(); renderSimulationEvaluation($("simulation-evaluation"), $("simulation-evaluation-content"), snapshot, connected); if (wasConnected) { lastRenderKey = ""; renderMap(); } refreshCameraCaptions(); }
    finally { pollBusy = false; }
  }
  async function fullscreen(element) { try { if (document.fullscreenElement) await document.exitFullscreen(); else if (element.requestFullscreen) await element.requestFullscreen(); else notify("Full screen is unavailable in this browser. Increase the browser window to enlarge the view."); } catch (error) { notify(error.message, true); } }
  $("source-tabs").addEventListener("click", (event) => { const button = event.target.closest("[data-source]"); if (!button) return; source = button.dataset.source; viewer && (viewer.autoFit = true); renderMap(); });
  for (const id of ["color-mode", "budget", "show-provisional", "show-rovers", "show-trails"]) $(id).addEventListener("change", renderMap);
  $("point-size").addEventListener("input", () => { if (viewer) viewer.pointSize = Number($("point-size").value); });
  $("fit").addEventListener("click", () => viewer?.fit()); $("top").addEventListener("click", () => { if (viewer) { viewer.pitch = 1.56; viewer.yaw = -Math.PI / 2; viewer.fit(); } });
  $("map-fullscreen").addEventListener("click", () => fullscreen(document.querySelector(".map-panel")));
  $("help-toggle").addEventListener("click", () => { $("help").hidden = !$("help").hidden; $("help-toggle").setAttribute("aria-expanded", String(!$("help").hidden)); });
  $("preview-toggle").addEventListener("click", () => { preview = !preview; $("preview-toggle").textContent = preview ? "Pause preview" : "Resume preview"; $("preview-toggle").setAttribute("aria-pressed", String(preview)); for (const camera of cameraState.values()) if (!preview) { camera.generation++; camera.controller?.abort(); } refreshCameraCaptions(); });
  $("save").addEventListener("click", () => command({ action: "save" }).catch((error) => notify(error.message, true)));
  $("shadow").addEventListener("change", async () => { const enabled = $("shadow").checked; if (!enabled && !window.confirm("Enable automatic acceptance of corroborated cross-rover alignments and entity links? Existing corroborated entity proposals may be accepted immediately. This changes the shared world model, not robot localization.")) { $("shadow").checked = true; return; } try { await command({ action: "set_shadow_mode", enabled }); } catch (error) { $("shadow").checked = snapshot.shadow_mode !== false; notify(error.message, true); } });
  document.addEventListener("click", async (event) => { const button = event.target.closest("[data-action]"); if (!button) return; if (!window.confirm(`${button.textContent}? Local observations will be preserved.`)) return; button.disabled = true; try { await command({ action: button.dataset.action, id: button.dataset.id }); } catch (error) { notify(error.message, true); button.disabled = false; } });
  $("anchor-form").addEventListener("submit", async (event) => { event.preventDefault(); try { const robot = $("anchor-robot").value, reference = $("anchor-reference").value; if (!robot || !reference || robot === reference) throw new Error("Choose two different rover frames."); const transform = anchorTransform(...["x", "y", "z", "yaw"].map((axis) => Number($("anchor-" + axis).value))); if (!window.confirm(`Apply a SUPPLIED alignment from ${robot} into ${reference}? Confirm these are measured frame offsets, not guessed robot positions.`)) return; await command({ action: "anchor", robot_id: robot, reference_robot_id: reference, T_reference_local: transform }); } catch (error) { notify(error.message, true); } });
  const tick = async () => { await poll(); setTimeout(tick, 1000); }; tick();
  const images = () => { refreshCameraCaptions(); for (const [id, camera] of cameraState) refreshCamera(id, camera); setTimeout(images, 1000); }; images();
}

if (typeof module !== "undefined" && module.exports) module.exports = { robotId, escapeText, ageLabel, matrix, transformPoint, inverseRigid, matrixProduct, anchorTransform, normalizeSnapshot, buildCloud, buildRoverOverlay, projectMarker, drawRoverOverlay, roverPoseLegend, candidateConstraint, activeEntityLink, modePresentation, imageAge, orderedCandidates, currentComponent, mappingPaused, healthPresentation, healthState, roverHealthCard, roverCardRenderer, displayedImageAge, receivedImageAge, abortableDecode, frameWarning, imageCaption, shadowDescription, alignmentEvidence, candidateReason, evidenceRenderKey, identityUrl, selectedWitness, witnessImage, identityContent, identityRequestReader, identityLinkSignature, identityFromHash, installIdentityInspector, byteLabel, storageModel, storageOverview, storageRunsContent, storagePlanContent, storageResult, storageController, installStorageDialog, simulationEvaluationModel, simulationEvaluationContent, evaluationInterval, renderSimulationEvaluation };
if (typeof document !== "undefined") { if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", startDashboard); else startDashboard(); }
