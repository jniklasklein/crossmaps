const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../crossmaps_dashboard/app.js'), 'utf8');
const elements = Object.fromEntries(['image-health', 'live-image', 'image-empty'].map(id =>
  [id, {dataset: {}, style: {}, textContent: ''}]));
let calls = 0, resolveFetch, rejectFetch, timeout;
const state = {selectedImage: 'rgb', imageRequestSeq: 0, imageRequestController: null,
  lastStatusAt: 0, lastImageError: '', lastStatus: null};
const context = vm.createContext({
  appState: state, $: id => elements[id], fmtAge: age => `${age}s`, Date, AbortController,
  setTimeout: callback => { timeout = callback; return 1; }, clearTimeout: () => {},
  URL: {createObjectURL: () => 'blob:frame', revokeObjectURL: () => {}},
  updateSimilarityMarker: () => {},
  fetch: (_, options) => { calls++; return new Promise((resolve, reject) => {
    resolveFetch = resolve; rejectFetch = reject;
    options.signal.addEventListener('abort', () => reject(Object.assign(new Error(), {name:'AbortError'})));
  }); },
});
vm.runInContext(source.slice(source.indexOf('function imageHealthMessage('), source.indexOf('function refreshGrid(')), context);
const status = {topics: {rgb: {age_s: 1}}, pipeline_health: {state:'healthy', stage:'idle', age_s:0}};
assert.equal(context.imageHealthMessage(status, 'rgb'), '');
status.pipeline_health.perception_publishers = 2;
assert.match(context.imageHealthMessage(status, 'rgb'), /MULTIPLE PERCEPTION NODES/);
status.pipeline_health.perception_publishers = 1;
status.pipeline_health.stationary_pose_lock = true;
assert.match(context.imageHealthMessage(status, 'rgb'), /STATIONARY POSE LOCK.*disable before moving/);
status.pipeline_health.stationary_pose_lock = false;
assert.match(context.imageHealthMessage(status, 'rgb', 6), /connection stale/);
status.topics.rgb.age_s = 25;
assert.match(context.imageHealthMessage(status, 'rgb'), /STALE IMAGE/);
status.pipeline_health.state = 'processing_stalled';
status.pipeline_health.stage = 'DINO';
assert.match(context.imageHealthMessage(status, 'rgb'), /processing stalled.*DINO/);
status.pipeline_health.age_s = 12;
assert.match(context.imageHealthMessage(status, 'rgb'), /heartbeat stale/);

(async () => {
  // Poll ticks during a slow fetch must not invalidate its eventual response.
  const first = context.refreshImage();
  await context.refreshImage();
  await context.refreshImage();
  assert.equal(calls, 1);
  resolveFetch({ok:true, blob:async () => ({})});
  await first;
  assert.equal(elements['live-image'].src, 'blob:frame');
  assert.equal(state.imageRequestController, null);

  const timed = context.refreshImage();
  timeout();
  await timed;
  assert.match(state.lastImageError, /timed out/);
  assert.equal(state.imageRequestController, null);
  assert.equal(elements['live-image'].style.display, 'block'); // Keep last frame.

  const staleTab = context.refreshImage();
  state.selectedImage = 'sam';
  state.imageRequestSeq++;
  state.imageRequestController.abort();
  state.imageRequestController = null;
  const newTab = context.refreshImage();
  const newController = state.imageRequestController;
  await staleTab;
  assert.equal(state.imageRequestController, newController);
  resolveFetch({ok:false, status:404});
  await newTab;
  assert.match(state.lastImageError, /not yet published/);

  const failed = context.refreshImage();
  rejectFetch(new Error('offline'));
  await failed;
  assert.match(state.lastImageError, /request failed/);
  assert.equal(state.imageRequestController, null);
  console.log('Dashboard freshness and single-flight image requests: passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
