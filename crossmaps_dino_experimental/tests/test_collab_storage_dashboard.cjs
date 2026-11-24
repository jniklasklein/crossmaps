"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const { byteLabel, storageModel, storageOverview, storageRunsContent, storagePlanContent, storageResult, storageController, installStorageDialog, roverHealthCard } = require("../collab_dashboard/app.js");

function snapshot(overrides = {}) {
  return { mode: "live", map_generation: "generation-1", _dashboard: { commands_enabled: true },
    stats: { keyframes: 256, store_bytes: 180 * 1024 ** 2, limits: { max_keyframes: 256, max_store_bytes: 2 * 1024 ** 3 } },
    storage: { current_run_id: "current", current_bytes: 200 * 1024 ** 2, total_bytes: 400 * 1024 ** 2,
      max_total_bytes: 4 * 1024 ** 3, free_bytes: 90 * 1024 ** 3, min_free_bytes: 1024 ** 3,
      runs: [{ id: "current", bytes: 200, active: true, managed: true, deletable: true },
        { id: "old", bytes: 300, active: false, managed: true, deletable: true },
        { id: "legacy", bytes: 150, active: false, managed: false, deletable: true },
        { id: "other-live", bytes: 50, active: true, managed: true, deletable: true },
        { id: "protected", bytes: 80, active: false, managed: false, deletable: false, reason: "Unrecognized evidence" }] }, ...overrides };
}
function plan(operation, ids) { return { ok: true, plan: { token: "server-scoped-token", operation, run_ids: ids, bytes: 300, description: "Delete exactly the listed evidence." } }; }

test("storage overview shows real quota/count information without calling absent limits unlimited", () => {
  const html = storageOverview(storageModel(snapshot()));
  for (const text of ["256 / 256", "180.0 MiB / 2.00 GiB", "400.0 MiB / 4.00 GiB", "90.00 GiB"]) assert.ok(html.includes(text), text);
  assert.equal(byteLabel(null), "Unknown"); assert.equal(byteLabel(NaN), "Unknown"); assert.equal(byteLabel(-1), "Unknown");
  assert.match(storageOverview(storageModel({})), /Unavailable/);
});

test("only server-authorized inactive IDs are eligible; current and active cannot be selected even if flagged deletable", () => {
  const model = storageModel(snapshot());
  assert.deepEqual(model.runs.filter(r => r.deletable).map(r => r.id), ["old", "legacy"]);
  const html = storageRunsContent(model, new Set(["old"]));
  assert.match(html, /data-storage-run="old" checked/);
  assert.match(html, /Recognized legacy recording/); assert.match(html, /Unrecognized evidence/);
  assert.match(html, /data-storage-run="current"\s+disabled/);
});

test("paths and injected markup are never sent as run IDs and all server descriptions are escaped", () => {
  const s = snapshot(); s.storage.runs.push({ id: "../elsewhere", deletable: true }, { id: "<script>", deletable: true });
  assert.equal(storageModel(s).runs.length, 5);
  assert.doesNotMatch(storageRunsContent(storageModel(s), new Set()), /\.\.\/elsewhere|<script>/);
  const html = storagePlanContent({ operation: "delete_runs", run_ids: ["safe"], bytes: 3, description: "<img onerror='bad'>" });
  assert.match(html, /&lt;img/); assert.doesNotMatch(html, /<img/);
});

test("native inventory names and protection reasons remain visible", () => {
  const data = snapshot(); delete data.storage.free_bytes; data.storage.disk_free_bytes = 15 * 1024 ** 3;
  data.storage.managed_runs = 4; data.storage.max_runs = 10; data.storage.reserved_bytes = 500;
  data.storage.runs[4].reason = undefined; data.storage.runs[4].protected_reason = "Run is locked by another coordinator";
  const model = storageModel(data);
  assert.match(storageOverview(model), /Disk free: 15.00 GiB/);
  assert.match(storageOverview(model), /Managed recordings: 4 \/ 10/);
  assert.match(storageOverview(model), /In-flight write reservations: 500 B/);
  assert.match(storageRunsContent(model, new Set()), /locked by another coordinator/);
});

test("committed rebuild with incomplete disk cleanup is explicit and warning survives the new-generation poll", async () => {
  const result = {ok:true,operation:"rebuild_current",result:{ok:true,reset_applied:true,freed_bytes:10,warning:"Old evidence cleanup pending: disk error"}};
  const c = storageController(async command => command.action === "prepare_storage" ? plan("rebuild_current", ["current"]) : result);
  c.update(snapshot(), true); await c.prepare("rebuild_current"); await c.confirm();
  assert.match(c.state.message, /rebuilding from fresh observations/); assert.match(c.state.message, /Freed 10 B/);
  assert.match(c.state.message, /cleanup pending/); assert.equal(c.state.error,true);
  c.update(snapshot({map_generation:"generation-2"}),true);
  assert.match(c.state.message, /cleanup pending/); assert.equal(c.state.error,true);
  assert.throws(()=>storageResult({result:{ok:false,error:"Partial deletion failed"}}), /Partial deletion/);
});

test("opening and selecting only prepare scope; permanent removal requires a distinct confirmation call", async () => {
  const calls = [], c = storageController(async command => { calls.push(command); return command.action === "prepare_storage" ? plan(command.operation, command.run_ids) : { ok: true, message: "Freed 300 bytes." }; });
  c.update(snapshot(), true); c.select("current", true); c.select("other-live", true); c.select("protected", true); c.select("old", true);
  assert.deepEqual(calls, []); assert.deepEqual([...c.state.selected], ["old"]);
  await c.prepare("delete_runs");
  assert.deepEqual(calls, [{ action: "prepare_storage", operation: "delete_runs", run_ids: ["old"] }]);
  assert.equal(c.state.plan.token, "server-scoped-token");
  await c.confirm();
  assert.deepEqual(calls[1], { action: "confirm_storage", token: "server-scoped-token" });
  assert.equal(c.state.plan, null); assert.equal(c.state.message, "Freed 300 bytes.");
});

test("rebuild prepares only the current run and cannot silently become old-run cleanup", async () => {
  const calls = [], c = storageController(async command => { calls.push(command); return plan("rebuild_current", ["current"]); });
  c.update(snapshot(), true); c.select("old", true); await c.prepare("rebuild_current");
  assert.deepEqual(calls, [{ action: "prepare_storage", operation: "rebuild_current" }]);
  assert.deepEqual(c.state.plan.run_ids, ["current"]);
});

test("periodic age, size and readiness changes never replace the reviewed removal plan", async () => {
  const c = storageController(async () => plan("delete_runs", ["old"]));
  c.update(snapshot(), true); c.select("old", true); await c.prepare("delete_runs");
  const original = c.state.plan;
  const changed = snapshot(); changed.storage.total_bytes += 100; changed.storage.runs[1].bytes += 50;
  c.update(changed, true); c.select("legacy", true);
  assert.equal(c.state.plan, original); assert.equal(c.state.plan.bytes, 300);
  assert.deepEqual([...c.state.selected], ["old"]);
});

test("new recording or map generation invalidates stale confirmation", async () => {
  for (const field of ["current_run_id", "map_generation"]) {
    const calls = [], c = storageController(async command => { calls.push(command); return plan("delete_runs", ["old"]); });
    c.update(snapshot(), true); c.select("old", true); await c.prepare("delete_runs");
    const next = snapshot(); if (field === "current_run_id") next.storage.current_run_id = "new"; else next.map_generation = "generation-2";
    c.update(next, true); await c.confirm();
    assert.equal(c.state.plan, null); assert.equal(calls.length, 1); assert.match(c.state.message, /active map changed/);
  }
});

test("replay, remote dashboard, missing storage and disconnection cannot prepare or confirm removal", async () => {
  for (const [value, connected] of [[snapshot({mode:"recorded_replay"}), true], [snapshot({_dashboard:{commands_enabled:false}}), true],
    [snapshot({storage:undefined}), true], [snapshot(), false]]) {
    const calls = [], c = storageController(async command => calls.push(command));
    c.update(value, connected); await c.prepare("rebuild_current"); await c.confirm();
    assert.equal(calls.length, 0); assert.equal(c.state.model.enabled, false);
  }
  const calls = [], c = storageController(async command => { calls.push(command); return plan("delete_runs", ["old"]); });
  c.update(snapshot(), true); c.select("old", true); await c.prepare("delete_runs"); c.update(snapshot(), false); await c.confirm();
  assert.equal(calls.length, 1); assert.ok(c.state.plan); assert.match(c.state.message, /Reconnect/);
});

test("mismatched, missing and nonfinite prepared scope never produces a confirmable plan", async () => {
  const valid = plan("delete_runs", ["old"]);
  for (const result of [{ok:true}, {plan:{...valid.plan,run_ids:["legacy"]}}, {plan:{...valid.plan,operation:"rebuild_current"}},
    {plan:{...valid.plan,bytes:NaN}}, {plan:{...valid.plan,token:""}}, {plan:{...valid.plan,run_ids:["old","old"]}}]) {
    const c = storageController(async () => result); c.update(snapshot(), true); c.select("old", true); await c.prepare("delete_runs");
    assert.equal(c.state.plan, null); assert.equal(c.state.error, true); assert.match(c.state.message, /scope/);
  }
});

test("closing cancels a pending preparation; late responses cannot reopen destructive controls", async () => {
  let resolve; const c = storageController(() => new Promise(r => resolve = r));
  c.update(snapshot(), true); c.select("old", true); const pending = c.prepare("delete_runs"); c.cancel();
  resolve(plan("delete_runs", ["old"])); await pending;
  assert.equal(c.state.plan, null); assert.equal(c.state.busy, false);
});

test("confirmation cannot be double-submitted and an uncertain failure requires preparing again", async () => {
  let resolve; const calls = [], c = storageController(command => { calls.push(command); return command.action === "prepare_storage" ? Promise.resolve(plan("delete_runs", ["old"])) : new Promise(r => resolve = r); });
  c.update(snapshot(), true); c.select("old", true); await c.prepare("delete_runs");
  const pending = c.confirm(); await c.confirm(); assert.equal(calls.length, 2);
  resolve({ok:true}); await pending; assert.equal(c.state.plan, null);
  const failing = storageController(async command => { if (command.action === "confirm_storage") throw new Error("Connection lost"); return plan("delete_runs", ["old"]); });
  failing.update(snapshot(), true); failing.select("old", true); await failing.prepare("delete_runs"); await failing.confirm();
  assert.equal(failing.state.plan, null); assert.match(failing.state.message, /prepare again/);
});

test("shared evidence age never borrows a recent local fusion time; both meanings remain visible", () => {
  const robot = { frame_age_s:.1, shared_evidence_age_s:90, local_map_update_age_s:.2, map_update_age_s:.2 };
  const card = roverHealthCard(robot);
  assert.equal(card.values[2], "90.0 s"); assert.match(card.meta, /Local fusion: 200 ms ago/);
  assert.match(card.diagnostics, /Shared evidence retained \(s ago\): 90/);
  assert.equal(roverHealthCard({...robot,shared_evidence_age_s:null}).values[2], "No data");
  assert.match(roverHealthCard({...robot,mapping_status:"capacity_reached"}).summary, /Storage & maps/);
});

test("dialog has explicit destructive labels, a cancel step and keeps its reviewed plan across polls", async () => {
  function element() { return Object.assign(new EventTarget(), { innerHTML:"", textContent:"", className:"", scrollTop:0,
    querySelectorAll:()=>[], showModal(){this.open=true;}, close(){this.open=false;this.dispatchEvent(new Event("close"));} }); }
  const els = new Map(), $ = id => { if (!els.has(id)) els.set(id, element()); return els.get(id); };
  let current = snapshot(); const sends = [];
  const ui = installStorageDialog($, () => current, () => true, async command => { sends.push(command); return plan("rebuild_current", ["current"]); });
  $("storage-open").dispatchEvent(new Event("click"));
  assert.equal($("storage-dialog").open, true); assert.equal(sends.length, 0);
  await ui.controller.prepare("rebuild_current");
  assert.equal($("storage-confirmation").hidden, false); assert.equal($("storage-controls").hidden, true);
  assert.equal($("storage-confirm").textContent, "Delete current evidence & rebuild");
  const html = $("storage-plan").innerHTML; current = snapshot(); current.storage.total_bytes += 1000; ui.refresh();
  assert.equal($("storage-plan").innerHTML, html);
  $("storage-cancel").dispatchEvent(new Event("click")); assert.equal($("storage-confirmation").hidden, true); assert.equal(sends.length, 1);
});
