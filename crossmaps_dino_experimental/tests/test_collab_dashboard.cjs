"use strict";
const assert = require("node:assert/strict");
const { robotId, escapeText, ageLabel, matrix, transformPoint, inverseRigid, matrixProduct, anchorTransform, normalizeSnapshot, buildCloud, candidateConstraint, activeEntityLink, modePresentation, imageAge, orderedCandidates, healthState } = require("../collab_dashboard/app.js");
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
check("view age follows that image rather than last processed observation", () => { const robot = { frame_age_s: 1, observation_age_s: 2, image_age_s: { sam: 30 } }; assert.equal(imageAge(robot, "sam"), 30); assert.equal(imageAge(robot, "rgb"), 1); assert.equal(imageAge(robot, "entities"), 2); });
check("input and perception failures stay visible despite fresh RGB", () => { assert.equal(healthState({ frame_age_s: .1, last_error: "TF unavailable" }), "error"); assert.equal(healthState({ frame_age_s: .1, status: "pose_rejected" }), "pose_rejected"); assert.equal(healthState({ frame_age_s: .1, perception: { error: "model failed" } }), "error"); });
check("actionable verification is sorted before unverified proposals", () => { const snapshot = normalizeSnapshot({ candidates: [{ id: "v", state: "verified" }, { id: "p", state: "proposed" }, { id: "r", state: "rejected" }], constraints: [] }); assert.deepEqual(orderedCandidates(snapshot).map((candidate) => candidate.id), ["v", "p", "r"]); });
check("polling ages does not invalidate evidence controls", () => {
  const { evidenceRenderKey } = require("../collab_dashboard/app.js");
  const first = normalizeSnapshot({ revision: 1, robots: [{ robot_id: "A", frame_age_s: 1 }], candidates: [{ id: "v", state: "verified" }] });
  const next = { ...first, revision: 2, robots: [{ robot_id: "A", frame_age_s: 2 }] };
  assert.equal(evidenceRenderKey(first), evidenceRenderKey(next));
  assert.notEqual(evidenceRenderKey(first), evidenceRenderKey({ ...next, candidates: [{ id: "v", state: "accepted" }] }));
});
console.log(`${checks} collaborative dashboard checks passed`);
