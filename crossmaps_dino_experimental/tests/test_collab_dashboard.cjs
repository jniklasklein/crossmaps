"use strict";
const assert = require("node:assert/strict");
const { robotId, escapeText, ageLabel, matrix, transformPoint, inverseRigid, matrixProduct, anchorTransform, normalizeSnapshot, buildCloud, candidateConstraint, activeEntityLink, modePresentation, imageAge, orderedCandidates, healthState, healthPresentation, currentComponent, mappingPaused, displayedImageAge, imageCaption, shadowDescription, alignmentEvidence, candidateReason } = require("../collab_dashboard/app.js");
let checks = 0;
function check(name, fn) { fn(); checks++; console.log(`ok ${checks} - ${name}`); }
const I = anchorTransform(0, 0, 0, 0);
const T = anchorTransform(10, 2, 0, 90);
function close(a, b) { assert.ok(Math.abs(a - b) < 1e-8, `${a} ≠ ${b}`); }
const state = normalizeSnapshot({ robots: [{ robot_id: "A", T_world_local: I }, { robot_id: "B", T_world_local: T }], submaps: [
  { id: "a1", robot_id: "A", aligned: true, T_world_submap: I, T_local_submap: I, points: [[1, 0, 0]], colors: [[255, 0, 0]] },
  { id: "b1", robot_id: "B", aligned: false, T_world_submap: T, T_local_submap: I, points: [[1, 0, 0]], colors: [[0, 255, 0]] },
] });
check("normalizes optional fields and robot IDs", () => { assert.equal(robotId({ id: "B" }), "B"); assert.equal(robotId({ robot_id: "A", id: "B" }), "A"); assert.deepEqual(normalizeSnapshot({}).candidates, []); });
check("escapes externally supplied text", () => assert.equal(escapeText('<script>"&'), "&lt;script&gt;&quot;&amp;"));
check("missing ages never look fresh", () => { assert.equal(ageLabel(null), "No data"); assert.equal(ageLabel(undefined), "No data"); assert.equal(ageLabel(0), "0 ms"); assert.equal(ageLabel(2), "2.0 s"); });
check("known anchor has correct direction and yaw", () => { const p = transformPoint([1, 0, 0], T); close(p[0], 10); close(p[1], 3); close(p[2], 0); });
check("rigid inverse composes to identity", () => { const result = matrixProduct(T, inverseRigid(T)); for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) close(result[i][j], I[i][j]); });
check("rejects nonfinite and implausible supplied poses", () => { assert.throws(() => anchorTransform(NaN, 0, 0, 0)); assert.throws(() => anchorTransform(10001, 0, 0, 0)); assert.equal(matrix([[1]]), null); assert.equal(matrix([Infinity, ...Array(15).fill(0)]), null); });
check("combined view excludes unaligned maps", () => { const cloud = buildCloud(state); assert.deepEqual(cloud.positions, [1, 0, 0]); assert.deepEqual(cloud.excluded, ["b1"]); assert.equal(cloud.submaps, 1); });
check("unaligned maps remain inspectable in their own frame", () => { const cloud = buildCloud(state, "B"); assert.deepEqual(cloud.positions, [1, 0, 0]); assert.deepEqual(cloud.colors, [183, 154, 255]); });
check("accepted transform places B in world, not at A origin", () => { const accepted = { ...state, submaps: state.submaps.map((s) => ({ ...s, aligned: true })) }; const cloud = buildCloud(accepted); close(cloud.positions[3], 10); close(cloud.positions[4], 3); });
check("local source view strips global transform when local transform absent", () => { const submap = { ...state.submaps[1] }; delete submap.T_local_submap; const cloud = buildCloud({ ...state, submaps: [submap] }, "B"); close(cloud.positions[0], 1); close(cloud.positions[1], 0); });
check("RGB and source colors are separate display choices", () => { assert.deepEqual(buildCloud(state, "A", "rgb").colors, [255, 0, 0]); assert.deepEqual(buildCloud(state, "A", "source").colors, [96, 217, 209]); });
check("invalid geometry is omitted and budget bounded", () => { const large = { ...state, submaps: [{ ...state.submaps[0], points: [...Array(99)].map((_, i) => [i, 0, 0]).concat([[NaN, 0, 0]]) }] }; assert.ok(buildCloud(large, "combined", "source", 10).positions.length <= 30); });
check("missing transform is never implicitly aligned", () => { const s = { ...state, submaps: [{ ...state.submaps[0], T_world_submap: null }] }; assert.equal(buildCloud(s).positions.length, 0); });
check("candidate actions target concrete verified constraint IDs", () => { const constraint = { id: "verified-1" }; assert.equal(candidateConstraint({ id: "candidate-1", constraint_id: "verified-1" }, { constraints: [constraint] }), constraint); assert.equal(candidateConstraint({ id: "unknown" }, { constraints: [constraint] }), undefined); });
check("camera health derives from actual source age", () => { assert.equal(healthState({ frame_age_s: 20 }), "stale"); assert.equal(healthState({ frame_age_s: .1 }), "healthy"); assert.equal(healthState({ status: "waiting" }), "waiting"); assert.equal(healthState({ frame_age_s: 0, error: "lost" }), "error"); });
check("historical acceptance is not current shared identity evidence", () => { assert.equal(activeEntityLink({ active: true, alignment_valid: true, compatible_current: true }), true); assert.equal(activeEntityLink({ active: true, alignment_valid: true, compatible_current: false }), false); assert.equal(activeEntityLink({ active: true, alignment_valid: false, compatible_current: true }), false); assert.equal(activeEntityLink({ active: false, alignment_valid: true, compatible_current: true }), false); });
check("demo and replay cannot be mistaken for live observations", () => { assert.match(modePresentation({ mode: "synthetic_demo" }).label, /Synthetic demo/); assert.match(modePresentation({ mode: "recorded_replay" }).label, /Recorded replay/); assert.match(modePresentation({ mode: "live", shadow_mode: true }).label, /shadow/); });
check("view age belongs only to that produced image", () => { const robot = { frame_age_s: 1, observation_age_s: 2, image_age_s: { rgb: 1, sam: 30 } }; assert.equal(imageAge(robot, "sam"), 30); assert.equal(imageAge(robot, "rgb"), 1); assert.equal(imageAge(robot, "entities"), null); assert.equal(imageAge({ image_age_s: { sam: null }, observation_age_s: 0 }, "sam"), null); });
check("input and perception failures are distinguished despite fresh RGB", () => { assert.equal(healthState({ frame_age_s: .1, health: { status: "waiting_for_tf", last_error: "TF unavailable" } }), "waiting_for_tf"); assert.equal(healthState({ frame_age_s: .1, status: "pose_rejected" }), "mapping_paused"); assert.equal(healthState({ frame_age_s: .1, perception: { status: "error", error: "model failed" } }), "error"); });
check("actionable verification is sorted before unverified proposals", () => { const snapshot = normalizeSnapshot({ candidates: [{ id: "v", state: "verified" }, { id: "p", state: "proposed" }, { id: "r", state: "rejected" }], constraints: [] }); assert.deepEqual(orderedCandidates(snapshot).map((candidate) => candidate.id), ["v", "p", "r"]); });
check("polling ages does not invalidate evidence controls", () => {
  const { evidenceRenderKey } = require("../collab_dashboard/app.js");
  const first = normalizeSnapshot({ revision: 1, robots: [{ robot_id: "A", frame_age_s: 1 }], candidates: [{ id: "v", state: "verified" }] });
  const next = { ...first, revision: 2, robots: [{ robot_id: "A", frame_age_s: 2 }] };
  assert.equal(evidenceRenderKey(first), evidenceRenderKey(next));
  assert.notEqual(evidenceRenderKey(first), evidenceRenderKey({ ...next, candidates: [{ id: "v", state: "accepted" }] }));
});
check("recovered component status suppresses historical error strings", () => {
  const robot = { status: "healthy", frame_age_s: .1, last_error: "old worker failure", health: { status: "receiving", last_error: "old TF failure" }, perception: { status: "processed", error: "old model failure" } };
  assert.equal(healthState(robot), "healthy"); assert.equal(healthPresentation(robot).detail, "");
});
check("stalled compressed RGB is not masked by the last frame's TF error", () => {
  const badge = healthPresentation({ frame_age_s: 280, health: { status: "rgb_stale", rgb_reader_attempts: 2 },
    pose_quality: { status: "timestamped_tf_unavailable" } });
  assert.equal(badge.label, "Camera stale"); assert.match(badge.detail, /retries: 2/);
});
check("session replacement cannot inherit old perception failure or pose pause", () => {
  const robot = { session_id: "new", status: "healthy", frame_age_s: .1, perception: { session_id: "old", status: "error", error: "failed", pose_quality: { integration_allowed: false } } };
  assert.deepEqual(currentComponent(robot, "perception"), {}); assert.equal(healthState(robot), "healthy"); assert.equal(mappingPaused(robot), false);
});
check("TF wait and rejected input are warnings rather than process crashes", () => {
  assert.equal(healthPresentation({ frame_age_s: .1, health: { status: "waiting_for_tf", last_error: "pose missing" } }).style, "pending");
  assert.equal(healthState({ frame_age_s: .1, health: { status: "input_rejected", last_error: "calibration wrong" } }), "input_rejected");
  assert.equal(healthState({ frame_age_s: .1, status: "processing_error", last_error: "model crashed" }), "error");
});
check("mapping pause does not claim fresh model outputs are stale map outputs", () => {
  const robot = { status: "pose_rejected", map_observation_age_s: 50, observation_age_s: .2, image_age_s: { sam: .2, segments: 50 } };
  assert.match(imageCaption(robot, "segments", { available: true, age: imageAge(robot, "segments") }), /Last mapped overlay.*50.0 s old.*mapping paused/);
  assert.match(imageCaption(robot, "sam", { available: true, age: imageAge(robot, "sam") }), /Perception image.*200 ms old/);
  assert.doesNotMatch(imageCaption(robot, "sam", { available: true, age: .2 }), /mapped overlay/);
});
check("missing views never borrow recent perception freshness", () => {
  assert.equal(imageAge({ observation_age_s: 0, image_age_s: {} }, "entities"), null);
  assert.equal(imageCaption({}, "entities", { available: false, age: 0 }), "No image for this view/session");
  assert.match(imageCaption({}, "entities", { available: true, age: null }), /age unknown/);
});
check("pausing preview ages the displayed frame even if backend is updating", () => {
  const camera = { displayedAge: 2, displayedAt: 1000 };
  assert.equal(displayedImageAge(camera, 6000), 7);
  assert.match(imageCaption({}, "rgb", { preview: false, available: true, age: displayedImageAge(camera, 6000) }), /Preview paused.*7.0 s old/);
  assert.equal(displayedImageAge({ displayedAge: null, displayedAt: 1000 }, 6000), null);
});
check("shadow description distinguishes perception from automatic world-model decisions", () => {
  assert.match(shadowDescription(true), /local perception and mapping running/);
  assert.match(shadowDescription(true), /manually accepted links remain active/);
  assert.match(shadowDescription(false), /entity proposals can be accepted immediately/);
});
check("alignment funnel uses complete retained counts instead of capped display list", () => {
  const summary = alignmentEvidence(normalizeSnapshot({ shadow_mode: false, alignment_evidence: { retrieved: 209, checked: 90, verified: 0, active: 0, best_rejection: { reason: "insufficient_geometric_inliers", inliers: 11 } }, candidates: [{ state: "proposed", score: .96 }] }));
  assert.equal(summary.retrieved, 209); assert.equal(summary.unchecked, 119);
  assert.equal(summary.title, "First verified pair still needed"); assert.match(summary.rejection, /11 inliers/);
  assert.match(summary.next, /not alignment confidence/); assert.doesNotMatch(summary.next, /second compatible/);
});
check("verified-but-unaccepted alignment calls for one rover's distinct view", () => {
  const summary = alignmentEvidence({ shadow_mode: false, alignment_evidence: { retrieved: 10, checked: 8, verified: 1, active: 0 } });
  assert.equal(summary.title, "Distinct-view corroboration still needed"); assert.match(summary.next, /Moving one rover is sufficient/); assert.match(summary.next, /Stationary repeats/);
});
check("shadow decision and supplied anchor are not mistaken for failed verification", () => {
  const shadow = alignmentEvidence({ shadow_mode: true, alignment_evidence: { retrieved: 1, checked: 1, verified: 1, active: 0 } });
  assert.match(shadow.title, /awaiting a decision/); assert.match(shadow.next, /Shadow mode prevents/);
  const anchored = alignmentEvidence({ constraints: [{ kind: "anchor", active: true }], candidates: [] });
  assert.equal(anchored.title, "Supplied alignment active"); assert.match(anchored.next, /Object identity still needs/);
});
check("skipped verification is labeled as a budget decision, not a failed match", () => {
  assert.match(candidateReason({ reason: "retrieved_not_verified" }), /per-keyframe verification budget/);
  assert.match(candidateReason({ reason: "retrieved_not_verified" }), /not a failed match/);
  assert.equal(candidateReason({ verification: { reason: "rgbd_verified" } }), "rgbd_verified");
});
check("alignment summary changes invalidate evidence rendering", () => {
  const { evidenceRenderKey } = require("../collab_dashboard/app.js");
  const state = normalizeSnapshot({ alignment_evidence: { retrieved: 100, checked: 20 } });
  assert.notEqual(evidenceRenderKey(state), evidenceRenderKey({ ...state, alignment_evidence: { retrieved: 101, checked: 20 } }));
});
const preview = { source_robot: "B", target_robot: "A", status: "provisional", support_count: 4 };
const stationary = { ...state, shadow_mode: false, alignment_evidence: { provisional: [preview], verified: 4, active: 0 },
  submaps: [state.submaps[0], { ...state.submaps[1], alignment_state: "provisional", T_world_submap: I, T_preview_world_submap: T }] };
check("tentative geometry uses its separate transform and tagged draw range; never changes source or RGB", () => {
  const cloud = buildCloud(stationary, "combined", "rgb");
  assert.equal(cloud.previewPoints, 1); assert.equal(cloud.previewShown, 1); assert.equal(cloud.previewAvailable, 1);
  close(cloud.positions[3], 10); close(cloud.positions[4], 3);
  assert.deepEqual(cloud.colors, [255, 0, 0, 0, 255, 0]);
  const local = buildCloud(stationary, "B"); assert.deepEqual(local.positions, [1, 0, 0]); assert.equal(local.previewPoints, 0);
});
check("preview display filter and shadow mode both omit tentative geometry, not the anchored source", () => {
  const hidden = buildCloud(stationary, "combined", "source", 100, false);
  assert.deepEqual(hidden.positions, [1, 0, 0]); assert.equal(hidden.previewShown, 0); assert.equal(hidden.previewAvailable, 1);
  const shadow = { ...stationary, shadow_mode: true };
  assert.deepEqual(buildCloud(shadow).excluded, ["b1"]); assert.equal(alignmentEvidence(shadow).previews.length, 0);
});
check("declared preview without transform or withdrawn preview with stale transform cannot be projected", () => {
  for (const submap of [{ ...stationary.submaps[1], T_preview_world_submap: null }, { ...stationary.submaps[1], alignment_state: "unaligned" }]) {
    const cloud = buildCloud({ ...stationary, submaps: [submap] });
    assert.deepEqual(cloud.positions, []); assert.deepEqual(cloud.excluded, ["b1"]);
  }
});
check("on promotion, confirmed geometry replaces rather than duplicates or offsets the preview", () => {
  const promoted = { ...stationary, submaps: [state.submaps[0], { ...stationary.submaps[1], aligned: true, T_world_submap: I }],
    alignment_evidence: { active: 1, verified: 4, provisional: [] } };
  assert.deepEqual(buildCloud(promoted).positions, [1, 0, 0, 1, 0, 0]); assert.equal(buildCloud(promoted).previewPoints, 0);
  assert.equal(alignmentEvidence(promoted).title, "Cross-rover alignment active");
});
check("provisional evidence distinguishes preview from confirmation, promotion from replay, and missing from conflicting support", () => {
  const summary = alignmentEvidence(stationary); assert.match(summary.title, /Provisional/); assert.equal(summary.active, 0);
  assert.equal(summary.previews[0].support_count, 4); assert.match(summary.next, /cannot link objects/);
  assert.match(alignmentEvidence({ ...stationary, mode: "recorded_replay" }).next, /does not collect new views/);
  const held = { ...stationary, alignment_evidence: { provisional: [], provisional_diagnostics: [
    { status: "withheld", reason: "Two competing placements." }, { status: "waiting", reason: "Only one qualified pair." }] } };
  assert.deepEqual(alignmentEvidence(held).diagnostics.map(item => item.status), ["withheld", "waiting"]);
});
console.log(`${checks} collaborative dashboard checks passed`);
