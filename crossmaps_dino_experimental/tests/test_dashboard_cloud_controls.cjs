const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../crossmaps_dashboard/app.js"), "utf8");
const cloudSource = source.slice(source.indexOf("async function toggleCloudLayer"), source.indexOf("function buildImageTabs"));
const paramsSource = source.slice(source.indexOf("async function sendParams"), source.indexOf("async function refreshParams"));
const imageSource = source.slice(source.indexOf("function containedImageRect"), source.indexOf("function setQueryInputValue"));
const layers = [
  {id: "voxel_fused"},
  {id: "segment_voxels", label: "Segments", publishParam: "publish_segment_voxels", requiresParams: ["use_segment_memory"]},
  {id: "entity_voxels", label: "Entities", publishParam: "publish_entity_voxels", requiresParams: ["use_segment_memory"]},
];
const controls = layers.map(layer => ({dataset: {layer: layer.id}, classList: {toggle() {}}, setAttribute() {}}));
const notice = {};
const appState = {config: {cloudLayers: layers}, params: {}, cloudStates: {}, stagedParams: {},
  layerVisible: {voxel_fused: true, segment_voxels: false, entity_voxels: false, graph: true},
  viewer: {setVisible() {}, recomputeBounds() {}}};
const requests = [];
const context = vm.createContext({
  appState, document: {querySelectorAll: () => []},
  $: id => id === "cloud-notice" ? notice : {querySelectorAll: () => controls},
  setParams: async values => { requests.push(values); Object.assign(appState.params, values); },
  fmtAge: age => `${age} s`, fmtCount: String,
  updateParamValues() {}, updateStagedUI() {}, setConnection() {}, setQueryInputValue() {},
});
vm.runInContext(cloudSource + paramsSource, context);

(async () => {
  await context.toggleCloudLayer("segment_voxels");
  assert.equal(requests[0].publish_segment_voxels, true);
  assert.equal(appState.layerVisible.voxel_fused, false);
  assert.equal(appState.layerVisible.segment_voxels, true);
  assert.equal(appState.layerVisible.graph, true);
  assert.match(notice.textContent, /waiting for publisher/);

  await context.toggleCloudLayer("entity_voxels");
  assert.equal(appState.layerVisible.segment_voxels, false);
  assert.equal(appState.layerVisible.entity_voxels, true);
  assert.match(notice.textContent, /gray: unassigned/);
  await context.toggleCloudLayer("entity_voxels");
  assert.equal(requests.length, 2, "Hiding a layer must not disable its ROS publisher for other clients");

  appState.params.use_segment_memory = false;
  await context.toggleCloudLayer("segment_voxels");
  assert.equal(requests.length, 2);
  assert.equal(appState.layerVisible.segment_voxels, false);
  assert.match(notice.textContent, /Enable use_segment_memory/);
  appState.params.use_segment_memory = true;
  context.setParams = async () => {throw new Error("publisher rejected");};
  await context.toggleCloudLayer("entity_voxels");
  assert.equal(appState.layerVisible.entity_voxels, false);
  assert.match(notice.textContent, /publisher rejected/);

  appState.cloudError = "";
  appState.layerVisible.entity_voxels = true;
  for (const [state, expected] of [["disabled", /disabled by publish_entity_voxels/],
    ["stale", /stale/], ["empty", /no surfaces pass/], ["ready", /2 points/]]) {
    appState.cloudStates.entity_voxels = {state, available: true, count: state === "empty" ? 0 : 2,
      disabled_by: ["publish_entity_voxels"], age_s: 11};
    context.updateCloudNotice();
    assert.match(notice.textContent, expected);
  }

  context.apiPost = async () => ({results: {publish_entity_voxels: {ok: false, reason: "unknown parameter"},
    publish_segment_voxels: {ok: true}}});
  appState.params.publish_entity_voxels = false;
  await assert.rejects(context.sendParams({publish_entity_voxels: true, publish_segment_voxels: true}), /unknown parameter/);
  assert.equal(appState.params.publish_entity_voxels, false, "Rejected values must not appear applied");
  assert.equal(appState.params.publish_segment_voxels, true);

  const image = {getBoundingClientRect: () => ({left: 0, top: 0, width: 400, height: 400}),
    naturalWidth: 1600, naturalHeight: 900};
  vm.runInContext(imageSource, context);
  const rect = context.containedImageRect(image);
  assert.equal(rect.width, 400);
  assert.equal(rect.height, 225);
  assert.equal(rect.top, 87.5);
  assert.equal(rect.bottom, 312.5);
  image.naturalWidth = 900;
  image.naturalHeight = 1600;
  assert.equal(context.containedImageRect(image).left, 87.5);
  console.log("Dashboard clouds: publication, errors, exclusivity, diagnostics and letterboxing checks passed.");
})().catch(error => { console.error(error); process.exitCode = 1; });
