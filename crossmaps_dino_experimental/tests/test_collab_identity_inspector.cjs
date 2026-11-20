"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const { identityContent, selectedWitness, witnessImage, identityRequestReader, identityLinkSignature, identityFromHash, installIdentityInspector } = require("../collab_dashboard/app.js");

const member = {
  id: "member", robot_id: "rover <a>", session_id: "session", local_id: 1,
  distinct_views: true, distinct_pair: ["f2", "f1"],
  views: [{ frame_id: "fresh", seq: 3, projected_points: 0 }, { frame_id: "f2", seq: 2, projected_points: 12, surface_ids: [5, 6] },
    { frame_id: "f1", seq: 1, projected_points: 10, surface_ids: [5] }]
};
const detail = { id: "link", state: "proposed", alignment_valid: true, evidence_scope: "last_compatible",
  blocking_reasons: ["No distinct views on rover B.", "<untrusted>"], dino: .92, clip: null, overlap: .8,
  members: [member, { id: "other", robot_id: "B", session_id: "other session", local_id: 0, views: [] }] };

test("stored comparison shows source, scores, actual blockers and historical status without presenting commands", () => {
  const html = identityContent(detail, new Map());
  for (const s of ["rover &lt;a&gt;", "local entity 0", "other session", "DINO 0.92", "CLIP —", "3D overlap 0.80",
    "No distinct views on rover B.", "Historical evidence", "No retained keyframe", "corroborating view", "surface IDs: 5, 6"]) assert.ok(html.includes(s), s);
  assert.doesNotMatch(html, /data-action|<untrusted>/);
  assert.equal((html.match(/<img /g) || []).length, 1);
  assert.match(html, /src="\/api\/identity\/link\/member\/f2.jpg"/);
});
test("preserves selected witness on refresh, falls back if evicted, and supports explicitly selecting an image without pixels", () => {
  assert.equal(selectedWitness(member, "evicted").frame_id, "f2");
  assert.equal(selectedWitness(member, "fresh").frame_id, "fresh");
  assert.match(identityContent(detail, new Map([["member", "f1"]])), /f1.jpg/);
  assert.doesNotMatch(witnessImage("link", member, selectedWitness(member, "fresh")), /<img/);
  assert.match(witnessImage("link", member, { frame_id: "x/y", projected_points: 1 }), /member\/x%2Fy.jpg/);
});
test("polling unrelated changes does not invalidate an open comparison; review changes, shadow changes and eviction do", () => {
  const s = { revision: 1, shadow_mode: false, entity_links: [{ id: "link", state: "proposed" }] };
  assert.equal(identityLinkSignature(s, "link"), identityLinkSignature({ ...s, revision: 8, entity_links: [...s.entity_links, { id: "elsewhere" }] }, "link"));
  assert.notEqual(identityLinkSignature(s, "link"), identityLinkSignature({ ...s, shadow_mode: true }, "link"));
  assert.notEqual(identityLinkSignature(s, "link"), identityLinkSignature({ ...s, entity_links: [{ id: "link", state: "accepted" }] }, "link"));
  assert.equal(identityLinkSignature({ ...s, entity_links: [] }, "link"), null);
});
test("details are demand-loaded; switching or closing before completion cannot display stale evidence", async () => {
  const pending = [], seen = [];
  const reader = identityRequestReader((id, signal) => new Promise((resolve, reject) => pending.push({ id, signal, resolve, reject })));
  const open = id => reader.load(id, data => seen.push(data.id), err => seen.push(err.message));
  assert.equal(pending.length, 0);
  const a = open("a"), b = open("b");
  assert.ok(pending[0].signal.aborted);
  pending[1].resolve({ id: "b", members: [] }); await b;
  pending[0].resolve({ id: "a", members: [] }); await a;
  const c = open("c"); reader.cancel(); pending[2].reject(new Error("late")); await c;
  assert.deepEqual(seen, ["b"]);
  const d = open("d"); pending[3].resolve({ id: "e", members: [] }); await d;
  assert.match(seen[1], /does not match/);
});
test("deep links are exact, read-only and scoped to one link", () => {
  const id = "0123456789abcdef0123";
  assert.equal(identityFromHash("#identity/" + id), id);
  for (const hash of ["", "#identity/no", "#identity/" + id + ".jpg", "#identity/" + id + "/extra", "#other/" + id, "#identity/%2Fetc"]) assert.equal(identityFromHash(hash), null);
});
test("direct link opens once after the first state, survives polling, clears on close, and honors later hash navigation", async (t) => {
  const a = "0123456789abcdef0123", b = "1123456789abcdef0123", requests = [];
  function target(extra = {}) {
    return Object.assign(new EventTarget(), { replaceChildren() {}, showModal() { this.open = true; }, close() { this.open = false; this.dispatchEvent(new Event("close")); } }, extra);
  }
  const location = { hash: "#identity/" + a, pathname: "/replay", search: "?s=test" }, urls = [];
  const win = target({ location, history: { replaceState(_, __, url) { urls.push(url); location.hash = ""; } } });
  const els = new Map(), $ = (id) => { if (!els.has(id)) els.set(id, target()); return els.get(id); };
  const before = Object.fromEntries(["document", "window", "fetch"].map(k => [k, Object.getOwnPropertyDescriptor(globalThis, k)]));
  t.after(() => { for (const [k, desc] of Object.entries(before)) { if (desc) Object.defineProperty(globalThis, k, desc); else delete globalThis[k]; } });
  globalThis.document = target({ querySelectorAll: () => [] }); globalThis.window = win;
  globalThis.fetch = async (url, opts) => {
    requests.push(url); assert.equal(opts.method || "GET", "GET");
    return { ok: true, json: async () => ({ id: url.split("/").pop(), members: [] }) };
  };
  const inspector = installIdentityInspector($, () => ({ entity_links: [{ id: a }, { id: b }] }));
  assert.deepEqual(requests, []);
  inspector.checkSnapshot(); inspector.checkSnapshot();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal($("identity-inspector").open, true); assert.deepEqual(requests, ["/api/identity/" + a]);
  $("identity-inspector").close(); inspector.checkSnapshot();
  assert.deepEqual(urls, ["/replay?s=test"]); assert.equal($("identity-inspector").open, false);
  location.hash = "#identity/" + b; win.dispatchEvent(new Event("hashchange"));
  assert.deepEqual(requests, ["/api/identity/" + a, "/api/identity/" + b]);
  location.hash = ""; win.dispatchEvent(new Event("hashchange"));
  assert.equal($("identity-inspector").open, false);
});
