"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { abortableDecode, receivedImageAge, frameWarning, healthPresentation, roverHealthCard, roverCardRenderer } = require("../collab_dashboard/app.js");

test("status summaries separate camera recovery, missing pose and an expired wait without dumping ROS exceptions inline", () => {
  const robot = {robot_id:"A", domain:0, session_id:"new", frame_age_s:.5};
  const issue = "Timestamped TF unavailable: Lookup would require extrapolation into the future. Requested time 1795183709.404347 …";
  const waiting = roverHealthCard({...robot, health:{status:"waiting_for_tf", last_error:issue, tf_coverage_lag_s:.117}});
  assert.match(waiting.summary,/pose at image capture time/);
  assert.doesNotMatch(waiting.summary,/1795183709/);
  assert.ok(waiting.diagnostics.includes(issue));
  assert.match(waiting.diagnostics,/coverage lag \(s\): 0.117/);
  const expired = roverHealthCard({...robot, health:{status:"waiting_for_tf", last_error:"Timestamped TF wait expired",tf_wait_expired:2}});
  assert.match(expired.summary,/frame skipped/);
  assert.match(expired.diagnostics,/expirations: 2/);
  const recovering = roverHealthCard({...robot, health:{status:"waiting_for_camera_relay"}, pose_quality:{status:"timestamped_tf_unavailable"}});
  assert.equal(recovering.label,"Camera recovery");
  assert.match(recovering.summary,/backup feed/);
  const rejected = {...robot, pose_quality:{status:"timestamped_tf_unavailable"},
    health:{status:"input_rejected",timing_status:"frame_stale_on_arrival",last_error:"source frame arrived too late"}};
  assert.equal(roverHealthCard(rejected).label,"Input rejected");
  assert.equal(roverHealthCard({...rejected,health:{...rejected.health,timing_status:"timestamped_tf_wait_expired"}}).label,"Waiting for TF");
  assert.equal(roverHealthCard({...robot, status:"processing"}).label, roverHealthCard(robot).label);
});

test("short TF lookup is routine only with a recently completed map update; long or expired waits remain visible", () => {
  const robot = {frame_age_s:.3, map_update_age_s:1, map_observation_age_s:13,
    health:{status:"waiting_for_tf",pending_wait_s:.2}};
  const sync = roverHealthCard(robot);
  assert.equal(sync.label,"Live"); assert.match(sync.summary,/Synchronizing/);
  assert.equal(sync.values[2],"1.0 s"); assert.match(sync.diagnostics,/Last mapped input age \(s\): 13/);
  for (const bad of [{map_update_age_s:20}, {map_update_age_s:undefined},
    {health:{...robot.health,pending_wait_s:2}}, {health:{...robot.health,timing_status:"timestamped_tf_wait_expired"}}])
    assert.equal(roverHealthCard({...robot,...bad}).label,"Waiting for TF");
});

test("polls retain the same focusable details node and an open snapshot; a new session discards old diagnostics", () => {
  const fields = new Map(), metrics = Array.from({length:5}, () => ({}));
  const card = {querySelector: k => { if (!fields.has(k)) fields.set(k,new EventTarget()); return fields.get(k); },
    querySelectorAll: () => metrics};
  let mounts = 0;
  const render = roverCardRenderer({set innerHTML(value) {mounts++; assert.ok(value.includes("<summary>"));},
    querySelectorAll: () => [card]});
  const robot = {robot_id:"A", session_id:"initial", frame_age_s:.5, health:{status:"waiting_for_tf",last_error:"first failure"}};
  render([robot]);
  const raw = fields.get("pre"), details = fields.get("details");
  assert.match(raw.textContent,/first failure/);
  details.open = true; details.dispatchEvent(new Event("toggle"));
  render([{...robot,frame_age_s:11,health:{status:"rgb_stale",last_error:"second failure"}}]);
  assert.equal(mounts,1); assert.ok(details.open); assert.match(raw.textContent,/first failure/);
  assert.equal(fields.get(".badge").textContent,"Camera stale");
  assert.equal(metrics[0].textContent,"11.0 s");
  details.open = false; details.open = true; details.dispatchEvent(new Event("toggle"));
  assert.match(raw.textContent,/second failure/);
  render([{...robot,session_id:"replacement",health:{status:"receiving"}}]);
  assert.equal(details.open,false); assert.doesNotMatch(raw.textContent,/first failure|second failure/);
  assert.equal(mounts,1);
});

test("a relay probe leaves the healthy DDS stream active and reports which input stalled", () => {
  const health = {status:"depth_stale",camera_transport:"dds_compressed",camera_relay_probing:true};
  const card = healthPresentation({health,frame_age_s:.2});
  assert.equal(card.label,"Depth stale");
  assert.match(card.input,/DDS.*trying SSH/);
});

test("fresh snapshot cannot reset the age of an old JPEG", () => {
  assert.equal(receivedImageAge("281.5", 9000, 1, 8000, 10000), 282.5);
  assert.equal(receivedImageAge("0", 9000, 200, 8000, 10000), 1);
  for (const header of [null, "", " ", "-1", "invalid", "Infinity"]) {
    assert.equal(receivedImageAge(header, 9000, 2, 1000, 10000), 11);
    assert.equal(receivedImageAge(header, 9000, null, 1000, 10000), null);
  }
});

test("live card warns about stale RGB even when JPEG downloads keep succeeding", () => {
  const live = {mode:"live", connected:true, preview:true, available:true, age:280};
  assert.match(frameWarning("rgb", live), /Stale RGB.*280.0 s/);
  assert.match(frameWarning("rgb", {...live,connected:false}), /dashboard disconnected/);
  for (const override of [{mode:"recorded_replay"}, {preview:false}, {available:false}, {age:1}])
    assert.equal(frameWarning("rgb", {...live,...override}), "");
  assert.equal(frameWarning("segments", live), "");
});

test("a stalled Safari image decode cannot block subsequent refreshes", async () => {
  const control = new AbortController();
  let completeLate;
  const old = abortableDecode({decode:() => new Promise(resolve => { completeLate = resolve; })}, control.signal);
  await Promise.resolve();
  control.abort();
  await assert.rejects(old, {name:"AbortError"});
  await abortableDecode({decode: async () => {}}, new AbortController().signal);
  completeLate(); // does not reinstate stale data after the timeout
  await assert.rejects(abortableDecode({decode: () => assert.fail("already expired")}, control.signal), {name:"AbortError"});
  await assert.rejects(abortableDecode({decode: async () => { throw Error("bad JPEG"); }}, new AbortController().signal), /bad JPEG/);
});
