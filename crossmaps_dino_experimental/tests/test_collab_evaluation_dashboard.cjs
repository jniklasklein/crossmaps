"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { simulationEvaluationModel, simulationEvaluationContent, evaluationInterval, renderSimulationEvaluation } = require("../collab_dashboard/app.js");

function snapshot(overrides = {}) {
  return { simulation_evaluation: { enabled: true, schema: 1, status: "ready", reason: "Independent evaluation",
    coverage: { keyframes: 12, associated: 10, missing: 2, truth_age_s: .25 },
    local: [{ robot_id: "rover_a", samples: 6, position_rmse_m: .012, rotation_rmse_deg: .2,
      rpe_translation_rmse_m: .008, rpe_rotation_rmse_deg: .15, rpe_pairs: 5, rpe_interval_s: { min: 7.8, max: 8.4 } }],
    shared: { confirmed: { pairs: 4, translation_rmse_m: .032, rotation_rmse_deg: 1.2 },
      provisional: { pairs: 2, translation_rmse_m: .041, rotation_rmse_deg: 1.8 } },
    optimization: { samples: 8, raw_position_rmse_m: .067, optimized_position_rmse_m: .025,
      raw_rotation_rmse_deg: 2.3, optimized_rotation_rmse_deg: .9 },
    loops: { evaluated: 5, correct: 4, precision: .8, translation_rmse_m: .019, rotation_rmse_deg: .7,
      translation_threshold_m: .1, rotation_threshold_deg: 5, accepted_evaluated: 2, accepted_correct: 2, accepted_precision: 1 },
    notes: ["Ground truth does not enter the graph."], ...overrides } };
}
const content = (s = snapshot(), connected = true) => simulationEvaluationContent(simulationEvaluationModel(s, connected));

test("physical snapshots and disabled evaluation hide the entire panel", () => {
  assert.equal(simulationEvaluationModel({}), null);
  assert.equal(simulationEvaluationModel({ simulation_evaluation: null }), null);
  assert.equal(simulationEvaluationModel(snapshot({ enabled: false })), null);
  const panel = {}, body = {};
  renderSimulationEvaluation(panel, body, snapshot()); assert.equal(panel.hidden, false);
  renderSimulationEvaluation(panel, body, {}); assert.equal(panel.hidden, true); assert.equal(body.innerHTML, "");
});

test("available metrics have explicit units, counts, capture association and distinct states", () => {
  const html = content();
  for (const text of ["250 ms", "10 / 12", "Missing truth: 2", "0.032 m / 1.20 °", "0.041 m / 1.80 °", "Confirmed", "Provisional", "0.067 m", "0.025 m", "80.0%", "4 / 5 within tolerance", "0.10 m", "5.0°", "Accepted only", "100.0%", "2 / 2 within tolerance"])
    assert.ok(html.includes(text), text);
});

test("local anchored error and variable-interval RPE are not called fitted ATE or simultaneous robot error", () => {
  const html = content();
  for (const text of ["First-pose anchored RMSE", "7.8–8.4 s", "5 pairs", "not a simultaneous live robot-position measurement", "not a controlled PGO on/off experiment", "not a fixed one-second benchmark"])
    assert.ok(html.includes(text), text);
  assert.equal(evaluationInterval(null), "interval unavailable");
  assert.equal(evaluationInterval({ min: 0, max: 8 }), "interval unavailable");
  assert.equal(evaluationInterval({ min: 9, max: 8 }), "interval unavailable");
});

test("missing sample populations cannot display fabricated zero error or zero precision", () => {
  const html = content(snapshot({ local: [{robot_id: "rover_a", samples: 0, position_rmse_m: 0, rotation_rmse_deg: 0}],
    shared: { confirmed: { pairs: 0, translation_rmse_m: 0, rotation_rmse_deg: 0 } },
    optimization: { samples: 0, raw_position_rmse_m: 0, optimized_position_rmse_m: 0 },
    loops: { evaluated: 0, correct: 0, precision: 0, translation_rmse_m: 0, rotation_rmse_deg: 0 } }));
  assert.match(html, /Unavailable/); assert.doesNotMatch(html, /0\.000 m|0\.00 °|0\.0%/);
});

test("measured zero error remains a legitimate measurement when samples exist", () => {
  const html = content(snapshot({ shared: { confirmed: { pairs: 2, translation_rmse_m: 0, rotation_rmse_deg: 0 } } }));
  assert.match(html, /0\.000 m \/ 0\.00 °/);
});

test("stale, invalid, waiting and disconnected states suppress live metric values", () => {
  for (const status of ["stale", "invalid", "waiting"]) {
    const html = content(snapshot({ status, reason: "No capture coverage" }));
    assert.match(html, /No capture coverage/); assert.match(html, /Unavailable/);
    assert.doesNotMatch(html, /0\.032 m|0\.025 m|80\.0%/);
  }
  const disconnected = content(snapshot(), false);
  assert.match(disconnected, /Coordinator disconnected/); assert.doesNotMatch(disconnected, /0\.032 m|80\.0%/);
  assert.match(content(snapshot({schema: 2})), /Unsupported evaluation schema/);
});

test("nonfinite, negative and malformed measurements are unavailable, not formatted into evidence", () => {
  const html = content(snapshot({ local: [{robot_id:"rover_a",samples:2,position_rmse_m:NaN,rotation_rmse_deg:-1}],
    loops: {evaluated:2,precision:1.5}, shared: {confirmed:{pairs:2,translation_rmse_m:Infinity,rotation_rmse_deg:-2}} }));
  assert.doesNotMatch(html, /NaN|Infinity|150\.0%|-1\.00|-2\.00/);
  assert.match(html, /Unavailable/);
});

test("all external IDs, reasons and notes are escaped", () => {
  const html = content(snapshot({reason:'<img src=x onerror="bad">',notes:["<script>bad</script>"],local:[{robot_id:'<iframe>',samples:1}]}));
  assert.doesNotMatch(html, /<img|<script|<iframe/);
  assert.match(html, /&lt;img/); assert.match(html, /&lt;script/); assert.match(html, /&lt;iframe/);
});

test("limitations explicitly exclude unmeasured IoU, recall and single-rover SLAM closures", () => {
  const html = content();
  for (const text of ["Not yet measured", "observed-surface IoU / F-score", "loop-detection recall", "SLAM Toolbox", "Sparse-map overlap alone is not localization accuracy"])
    assert.ok(html.includes(text), text);
});

test("simulation panel is hidden initially and has a ground-truth isolation label", () => {
  const html = fs.readFileSync(path.join(__dirname, "../collab_dashboard/index.html"), "utf8");
  assert.match(html, /id="simulation-evaluation"[^>]*hidden/);
  assert.match(html, /Truth never feeds mapping/);
  assert.match(html, /id="evaluation-heading"/);
  const css = fs.readFileSync(path.join(__dirname, "../collab_dashboard/styles.css"), "utf8");
  assert.match(css, /evaluation-status[^}]*min-height/);
  assert.match(css, /evaluation-table-wrap\{overflow-x:auto/);
});

test("recorded reports show captured time and historical truth age, never a live connection", () => {
  const captured = Date.parse("2026-11-24T12:30:00Z") / 1000;
  const data = snapshot({recorded: true, updated_unix: captured});
  const html = content(data);
  assert.match(html, /Recorded evaluation/);
  assert.match(html, /Snapshot captured: 2026-11-24T12:30:00\.000Z/);
  assert.match(html, /Saved measurements, not a live ground-truth connection/);
  assert.match(html, /Truth age at recording: 250 ms/);
  assert.match(html, /0\.032 m \/ 1\.20 °/); // Valid recorded metrics remain inspectable.
  assert.doesNotMatch(html, /Ground truth connected|>Truth age:/);
  assert.match(content({...snapshot({updated_unix: captured}),mode:"recorded_replay"}), /Recorded evaluation/);
  assert.match(content(snapshot({recorded:true,updated_unix:Infinity})), /Snapshot captured: time unavailable/);
  const offline = content(data, false);
  assert.match(offline, /Recorded evaluation · unavailable/);
  assert.match(offline, /Truth age at recording/);
  assert.doesNotMatch(offline, /Ground truth connected|0\.032 m/);
});
