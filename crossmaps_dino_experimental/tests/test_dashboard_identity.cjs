const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../crossmaps_dashboard/app.js"), "utf8");
const renderSource = source.slice(source.indexOf("function renderGraph(descriptor)"), source.indexOf("async function refreshImage()"));
function element() {
  return {children: [], className: "", textContent: "", html: "",
    set innerHTML(value) { this.html = value; this.children = []; },
    get innerHTML() { return this.html; },
    appendChild(child) { this.children.push(child); },
    querySelector(selector) { return this.children.find(child => `#${child.id}` === selector); }};
}
const root = element();
const context = vm.createContext({
  document: {createElement: element},
  $: () => root,
  fmtCount: String,
  escapeText: value => String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;"),
});
vm.runInContext(renderSource, context);

context.renderGraph({identity: {entities: 1, pending_segments: 2, ambiguous_segments: 1, whole_object_grouping: false}, nodes: [],
  pending_memberships: [{segment_id: 42, reason: "ambiguous_support"}]});
assert.equal(root.children.length, 3);
assert.match(root.children[0].textContent, /1 hypotheses/);
assert.match(root.children[0].textContent, /object grouping disabled/);
assert.match(root.children[1].textContent, /display filters/);
assert.match(root.children[2].textContent, /S42: ambiguous support/);

context.renderGraph({entity_type: "entity", nodes: [{id: 7, node_type: "object_hypothesis",
  segment_ids: [101, 202], observations: 4, view_count: 2, confidence: 0.8}]});
assert.match(root.children[0].innerHTML, />E7</);
assert.match(root.children[0].innerHTML, /segments: 101, 202/);
assert.match(root.children[0].innerHTML, /object hypothesis/);

context.renderGraph({entity_type: "segment", nodes: [{id: 101, entity_id: 7, node_type: "surface_segment"}]});
assert.match(root.children[0].innerHTML, />S101</);
assert.match(root.children[0].innerHTML, /entity: E7/);

context.renderGraph({entity_type: "entity", nodes: [{id: 7, state: "<script>bad()</script>", segment_ids: ["<img>"]}]});
assert.doesNotMatch(root.children[0].innerHTML, /<script>|<img>/);
context.renderGraph({identity: {entities: 2, whole_object_grouping: true,
  assembly: {enabled: true, sensitivity: .5, min_dino_cos: .8, max_gap_m: .15, joined_hypotheses: 1,
    pairs: [{a: 1, b: 2, dino_cos: .91, gap_m: .04, hits: 3, views: [[0,0,0]], confirmed: true,
      reason: "dino_contact", group_blocker: "<script>bad()</script>"}]}},
  mask_attribution: [{segments: [101], reason: "attributed"}, {segments: [], reason: "no_valid_depth"}], nodes: []});
assert.match(root.children[1].textContent, /1 joins · sensitivity 0.50 · DINO ≥ 0.80 · gap ≤ 0.15 m/);
const inspector = root.children[2];
assert.match(inspector.children[1].textContent, /E1 \+ E2: <script>bad\(\)<\/script> · DINO 0.91/);
assert.equal(inspector.children[1].innerHTML, ""); // Dynamic diagnostics use textContent, never HTML.
assert.match(root.children[3].textContent, /1\/2 masks have segment support · 1 no valid depth/);
inspector.open = true;
root.querySelector("#mask-evidence").open = true;
context.renderGraph({identity: {assembly: {enabled: true, pairs: [{a:1,b:2,hits:1}]}},
  mask_attribution: [{mask:0, mask_owner:101, segments:[101], association_basis:"temporal_depth_hold"}],
  identity_recording: {state:"stopped",frames:12,max_frames:120,path:"local trace"}, nodes: []});
assert.equal(root.querySelector("#grouping-evidence").open, true);
assert.equal(root.querySelector("#mask-evidence").open, true);
assert.ok(root.children.some(child => /Identity replay: stopped · 12\/120/.test(child.textContent)));
context.renderGraph({nodes: [], mask_attribution: [{mask:0, segments:[101,202],
  display_owner:101, display_provisional:true}], pending_memberships: [{segment_id:303,
  candidate:2, hits:2, reason:"neighborhood_tiebreak", neighborhood_scores:[{entity_id:2,
  neighbor_segment_id:202, overlap:.75, neighborhood_strength:.9}]}]});
assert.match(root.querySelector("#mask-evidence").children[1].textContent, /display: S101\? \(provisional\)/);
assert.ok(root.children.some(child => /neighbor S202 · overlap 0.75 · context 0.90/.test(child.textContent)));
console.log("Dashboard identity rendering: 10 checks passed.");
