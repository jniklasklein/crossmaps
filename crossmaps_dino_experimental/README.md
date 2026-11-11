# CrossMaps v2

An RGB-D world-model pipeline with persistent surface segments, reversible
object-entity grouping, open-vocabulary semantics, and a live inspection dashboard.

Development branch: **`crossmaps-v2`**.

For the maintained file inventory, launch commands, external dependencies, and
what is intentionally kept local, see [FILES.md](FILES.md). This directory is
the current implementation; similarly named scripts and backup directories at
the repository root are not used by it.

The perception node combines four deliberately separate signals:

- EfficientSAM: per-frame mask proposals
- DINOv2: instance appearance and cross-view correspondence
- CLIP: open-vocabulary semantics and text queries
- RGB-D geometry: reprojection, spatial gating, support, and free-space removal

DINO and CLIP descriptors are never averaged together. A segment keeps a CLIP
semantic memory and an independent DINO appearance memory.

## Quick start

From the repository root, with the ROS 2/model environment active, run these
in separate terminals:

```sh
python3 crossmaps_dino_experimental/crossmaps_v2_node.py
```

```sh
python crossmaps_dino_experimental/crossmaps_dashboard.py
```

Open **http://127.0.0.1:8765**. The robot's RGB-D camera and TF publishers must
already be running. Start only one perception node. External dependencies and
weight locations are listed in [FILES.md](FILES.md).

The browser dashboard is the primary interface; RViz2 is not required. Activate
the same ROS/Python environment in both terminals (on the existing MacBook,
`conda activate ros_humble_env`). Both `python3` and `python` must resolve to
that environment. The dashboard targets `/crossmaps_v2_node` by default.

### Naming and compatibility

`crossmaps_v2_node.py` is the maintained perception entry point. The dashboard
entry point remains `crossmaps_dashboard.py`; helper modules use descriptive
names such as `identity_graph.py` and `dominant_overlay.py`.

The directory name `crossmaps_dino_experimental/` is retained for path
compatibility: local `.runtime/` maps, logs, model-weight lookup, and existing
workspace references stay in place. The ROS node is now `/crossmaps_v2_node`;
the dashboard defaults to that target. Use underscores, not hyphens, in ROS
node names. `/crossmaps/...` topics, source IDs and serialized schema versions
remain unchanged. Update any custom parameter commands or node-keyed YAML
configuration to use the new node name.
Update launch commands to use `crossmaps_v2_node.py`; the old dated entry-point
filename is no longer used. This naming cleanup changes no inference defaults
or association behavior, and does not claim that upstream yaw drift is solved.

## Runtime defaults

The defaults retain the existing MacBook inference setup:

- one aspect-preserving `dinov2_vits14` full-frame pass at 518 px
- `ViT-L-14` CLIP for the stronger open-vocabulary semantic representation
- EfficientSAM `vitt`; no precision/model changes in the identity rework
- one CLIP crop per SAM mask; no additional model passes for identity
- asynchronous semantic inference
- wireless-friendly JPEG RGB and lossless 16UC1 PNG depth transport (native resolution)
- exclusive per-point mask fusion
- global one-to-one fallback association
- automatic destructive segment merging **disabled**
- appearance-only ownership carving **disabled**
- persistent entity hypotheses with independent IDs and delayed births
- reversible DINO/contact grouping above those tracks, with live sensitivity
- graph edges requiring two distinct, jointly observed endpoint updates

No per-instance DINO crops are used. Every SAM mask reuses the same dense
feature map. Pooling uses fractional patch coverage, rejects contaminated
boundary tokens, and combines the mask interior with an interior-minus-local-
ring residual. Small or low-purity masks receive lower DINO quality, so their
appearance cannot veto stronger geometry. Each 3D segment maintains a bounded,
confidence-weighted EMA across viewpoints rather than an unbounded descriptor
average.

The DINO model is loaded through the official DINOv2 Torch Hub entry. The first
run may download weights when `dinov2_allow_download=true`. For an offline
robot, point `dinov2_local_repo` at a local DINOv2 checkout and optionally set
`dinov2_checkpoint`. If DINO cannot initialize or infer, the node continues
with CLIP and geometry.

## Persistent identity and object grouping

`identity_graph.py` owns observation-driven entity memory, independently of ROS,
the neural models, and visualization. Keep all nine runtime helper modules
listed in [FILES.md](FILES.md) beside the node when launching.

- **Surface segments** retain their own geometry and CLIP/DINO descriptors.
  A small, strongly reprojected mask can update a larger existing surface;
  mask-size changes alone no longer require a new segment ID.
- **Entities** are persistent object hypotheses with their own monotonic IDs.
  They are not automatically certified whole physical objects. A membership is
  `member_of`, not a semantic claim such as “chair leg part_of chair.”
- **Areas** are deliberately deferred. The descriptor reports `areas: []` and
  `area_inference: deferred`; there is no placeholder room clustering.

An unassigned surface needs three fresh observations before an entity is born.
Reassociation to an existing base entity needs clear overlap with its prior **3D
voxel support**, a margin over alternatives, and two fresh observations. DINO
can reject inconsistent matches in this continuity layer. The separate assembly
layer below can group disjoint parts using combined DINO/contact evidence.
Up to four cached DINO exemplars preserve appearance diversity
without extra inference. This does not solve arbitrary semantic part–whole
grouping, moving-object re-identification, or unobservable boundaries between
touching identical objects.

Pending/ambiguous fragments remain unassigned. Existing entity records are not
merged by clustering. A disconnected maintenance split records `split_from`
lineage but does not prove that its children belong to one object. A segment
replacement supported by the same old geometry can retain the entity ID.
Dormant records retain bounded support for a configurable interval after their
last observation, allowing conservative reappearance matching. Unobserved
retained tracks do not acquire extra observations or disappear just because
they are outside the camera view.

The normal fine-image assignment path protects other segment owners when a
SAM mask spans multiple existing objects. Fallback matching now requires
actual support overlap or substantial 3D box IoU, and rejects close score ties.
Shared XY cells alone cannot match vertically stacked objects. A final fusion
guard also rejects ownership conflicts when no trustworthy projected context
is available. Appearance-only carving (`segment_semantic_conflict_carve`) is
off by default: a changing descriptor is not proof of changed physical
ownership. The optional legacy merge routine is additionally barred from
merging independent entities. Split surfaces inherit their existing surface
evidence so they remain projectable, but do not receive a new observation
sequence or bypass entity-birth probation.

Entity membership is reversible without deleting geometry. For an explicit
correction, publish a `std_msgs/msg/String` on `/crossmaps/entities/corrections`:

```json
{"action": "detach_segment", "segment_id": 101}
```

This detaches S101, excludes its rejected entity from automatic reassociation,
and lets subsequent observations establish a separate hypothesis. Segment and
entity IDs have separate namespaces; use the **segment** ID in this command.

### Mask attribution and stable camera colors

The fine-image path now uses `segment_mask_min_depth_points=3` and
`segment_mask_min_support_voxels=1`, independently of `esam_min_depth_points=25`
for dense semantic fusion. A 400-pixel mask sampled at stride 8 may contain only
about six depth samples; requiring 25 previously discarded such masks. Small
surface tracks do not bypass the stricter entity-birth/support requirements.
Bad/missing depth and geometry outside the volume still cannot create tracks.

Current reprojected owners remain attributed even if a mask is fragmented,
missing, or fails an appearance update gate. Visibility refreshes track age,
not its semantic observation count. One broad mask can retain several segment
owners; it does not steal or collapse their support. Maintenance now tolerates
a one-voxel sampling gap (`segment_split_gap_vox=1`) and keeps tiny components
(`segment_split_drop_tiny_components=false`) instead of deleting them.

The default camera path now uses **current SAM silhouettes and mask-level ID
consensus**, not nearest-seed/Voronoi patches. Reliable interior depth votes
select the ID. A previous mask ID can hold through depth-quantization noise
only with current valid depth, world-position agreement, image overlap, and
appearance approval. Three agreeing observations confirm an owner change;
missing/ambiguous updates interrupt that confirmation. Duplicate frames do not
advance it. No old screen pixels are propagated.

Two sufficiently supported owners under one SAM mask remain mixed in the
underlying evidence. The default display picks one stable dominant hypothesis,
marked with a dashed amber outline and `?` when provisional. Turn off **Stable
Dominant Overlay** to inspect mixed colors instead. Gray means unassigned.
White outlines and S/E labels expose the current mask boundary and identity. If SAM misses a
surface, its 3D track can remain alive without painting stale pixels into the
normal camera view. Exact SAM outlines are still only as good as SAM itself.

Use **Segments → Raw Support Debug View** (`segment_overlay_raw_support=true`)
to see sparse projected/measured IDs separately. Full-mask coloring is an
association hypothesis, **not dense 3D ownership**. It never creates geometry.
`segment_overlay_fill_masks=true` and `segment_temporal_consensus=true` select
the new default view; the legacy interpolation path remains only for contexts
without consensus. Adjust `segment_overlay_alpha` (default 0.78) for opacity.
The graph inspector explains each mask's owner, decision basis and depth support.

Geometry changes are separate and conservative. Four consecutive, strong
interior-depth votes can correct a small foreign boundary patch. An owner
supported by another current mask is protected, as are explicit exclusions.
Corrections have a lifetime budget of 10% of the source segment's support at
its first correction proposal, persist that budget across restarts, and record
lineage. They cannot erase an entire track or merge whole segments. Disable
`segment_boundary_repair_enabled` to keep all existing owners immutable.

A mixed mask no longer copies its broad appearance descriptor into every
component. Existing per-component CLIP descriptors are retained; DINO can be
re-pooled from each component's current depth-supported footprint in the
**same already-computed feature grid**. The round-robin budget is
`segment_mixed_dino_pool_budget=12`; low-quality footprints do not provide
fresh grouping evidence. This adds CPU pooling, not crops or neural passes.

## Observation evidence and persistence

Identity/edge updates and segment maintenance run after accepted semantic
fusion, not in the publishing timer. Entity observations count once per cycle,
not once per child. Duplicate/out-of-order job sequences and repeated positive
depth timestamps are rejected. A zero timestamp falls back to sequence checks.
Restart the node when intentionally replaying a sensor clock backwards.

An edge receives evidence only when both endpoints were actually updated in
that observation. Missing edges count as negative evidence only when both
endpoints were evaluated again; occlusion, graph filters, and display budgets
are not misses. Repeated stationary observations can stabilize a track, but
are not presented as multi-view proof of objectness. View diversity is reported
separately using camera-position cells, and still depends on trustworthy poses.

Map-state version 3 saves entity IDs, memberships, pending evidence, bounded
support, lineage, a persistent world UUID, and boundary-correction budgets.
Per-segment DINO dimensions survive save/load even while a context-layout
change is being refreshed. Voxel geometry and feature-model metadata are checked
before a version-3 map is loaded; incompatible state is rejected before memory
is replaced. Relation confirmation restarts after loading. Version-2
segment-only state remains readable: entities are conservatively established
from new observations. Legacy maps lack this extra metadata and must be used
with their original voxel/model settings. The default state path is isolated here:
`crossmaps_dino_experimental/.runtime/map_state.pkl`. No old shared state is
automatically read or migrated. An explicit `map_state_path` override still wins.

The graph JSON schema is `crossmaps_hierarchical_scene_graph_v3`, with typed
nodes, `memberships`, `pending_memberships`, `lineage`, and `identity` diagnostics.
`nodes`/`edges` remain the selected display view. Node/edge budgets only limit
publication; they do not allocate, merge, or remove persistent identities.
`use_entity_grouping=true` selects the entity view; false selects surface nodes.
The old `entity_group_*` clustering thresholds remain accepted for old parameter
sets but no longer affect association.

### World-model contract

The v3 export adds namespaced `uid`, `segment_uids`, `component_entity_uids`,
typed membership endpoints and reversible `group_aliases`. A UID combines the
saved world UUID, node kind, and monotonic local ID; restarting a saved map
retains it. New maps have different namespaces. `world_model_source_id` is a
human-readable source name; a separate observation-source UUID identifies the
current node session. Evidence includes the accepted observation sequence,
sensor timestamp, declared geometry frame, and feature-model metadata.

This is a **filtered snapshot, not a complete map**. Absent nodes/edges must not
be interpreted as deletions. Confidence is explicitly uncalibrated evidence
strength, not a probability of physical objectness. Query similarity is not a
class label (`semantic_labels` remains empty). Large planar support/structural
surface roles are geometric candidates, not inferred room or floor semantics.

This contract is groundwork for later reasoning and collaboration, not an
implemented multi-robot map fusion protocol. Different worlds still need
verified frame alignment and cross-source identity reconciliation. The export
marks pose quality as external/unvalidated and does not claim calibrated pose
uncertainty, automatic semantic descriptions, or room/building inference.

## Important diagnostics

The timing profile reports:

- `*.dinov2_dense`
- `*.dinov2_mask_pool`
- `*.segments_projected_matched`
- `*.segments_fallback_matched`
- `segments.maintenance_merged`
- `segments.maintenance_retired`
- `identity.update` (CPU identity/graph update cost, no neural inference)
- `identity.pending_segments`

The node also publishes `/crossmaps/diagnostics/observation_cycle`. The
dashboard's `Cycle` badge shows the cumulative average for a complete accepted
observation update, including preprocessing, asynchronous queue time, model
inference, association, and memory fusion. Hover it for the last, recent, and
p95 timings.

The image dashboard includes five feature diagnostics:

- `DINO`: a display-only bilinear rendering of the shared patch descriptors
- `DINO Sim`: raw-token cosine similarity to the selected camera location
- `DINO Inst`: SAM masks colored by their mask-pooled DINO descriptors
- `CLIP`: SAM masks colored by their existing CLIP crop descriptors
- `CLIP Sim`: cosine similarity from the selected SAM mask to all other masks

Click anywhere in the live image to update the shared similarity location. SAM
boundaries and a selection marker remain visible over the diagnostics. DINO
smoothing is display-only: association still uses raw tokens, fractional mask
pooling, and multi-view accumulation. CLIP diagnostics do not enable the
optional dense CLIP path or add another CLIP model pass.

Use `Large` to expand the complete Images section over the dashboard and
`Restore` or Escape to return. `Full Screen` uses the browser fullscreen API;
its button changes to `Exit Full Screen` while active. Both modes upscale the
selected image while preserving aspect ratio, and similarity clicks ignore
letterbox margins.

The additional **Entities** camera tab uses the same accepted observation:
stable entity colors, gray for unassigned surfaces, dashed amber outlines and
`?` for provisional display choices. **Segments** remains the separate low-level ID view. Neither view
changes descriptors or triggers inference.

The old dashboard **Instances** button is replaced by independent 3D layers:

- **Segments**: `/crossmaps/segments/voxels`, stable surface-ID colors.
- **Entities**: `/crossmaps/entities/voxels`, the same geometry colored by
  persistent entity membership. Gray means unassigned; amber means ambiguous.

`publish_segment_voxels` and `publish_entity_voxels` default to true. No cloud
geometry is built without subscribers, and both outputs share one geometry
pass. A layer click enables that publisher (also when Live Apply is off), but
does not enable disabled segment memory or inference. It hides other cloud
layers to avoid overlapping colors; graph/rover stay visible. Hiding a layer
does not stop publication for other clients. The viewer reports waiting,
disabled, empty, stale, and ready states. Disabled layers do not display cached
geometry. These clouds are independent of graph visibility, node budgets, and
`use_entity_grouping`; their own support/confidence/observation and point-budget
filters still apply. The Segments parameter group exposes the main 3D limits.

The optional legacy `/crossmaps/objects/instance_voxels` topic remains supported
by `publish_object_instance_voxels` and `instance_color_by_entity`; neither
switch controls the new layers. Restart **both** the experimental node and
dashboard, then reload the page, to obtain the new publishers and controls.

Camera controls now wrap above a separate aspect-ratio image area. The whole
image is contained in normal, Large, and fullscreen modes, with letterboxing
when needed. Narrow layouts retain the camera/map/graph panels below the 3D
view rather than hiding them. These CSS changes still need visual verification
in the target browser; the automated tests check letterbox coordinates, not a
browser layout engine.

The graph panel lists entity-to-segment memberships, pending reasons, separate
observation/view counts, and identity-update time. The **Identity** parameter
group exposes birth and reassociation thresholds. Old saved parameter profiles
override defaults: check `use_entity_grouping=true`, `segment_merge_enabled=false`,
`segment_assoc_require_overlap=true`, and `object_graph_min_view_count=1` when
testing this rework. One view is sufficient to display a hypothesis; it does
not certify a whole object.
Also keep `segment_semantic_conflict_carve=false` for the conservative defaults;
saved profiles may still contain the previous value of `true`.

### Reversible whole-object hypotheses and sensitivity

Previously, assigned entities were never reconsidered for grouping, so distinct
sofa surfaces could remain separate indefinitely despite strong DINO similarity.
`object_assembly.py` now proposes reversible aliases over stable base entities.
Segments, base entity records, geometry and descriptors are never merged. The
camera, 3D entity colors and graph share the same canonical entity ID (the oldest
base ID). The graph exposes `component_entity_ids` for inspection/correction.
Disabling **Identity → DINO Object Grouping** restores the underlying IDs.

**Identity → Grouping Sensitivity** (`entity_grouping_sensitivity`, 0–1) controls
two coupled thresholds; higher is more permissive:

| Sensitivity | Minimum DINO cosine | Maximum bbox/surface gap |
| --- | --- | --- |
| 0.0 | 0.92 | 0.03 m |
| 0.5 (default) | 0.80 | 0.15 m |
| 1.0 | 0.68 | 0.27 m |

The formulas are `0.92 - 0.24*s` and `0.03 + 0.24*s`. These are experimental
tuning ranges, not calibrated probabilities. Start at 0.5 and inspect the
**Graph → Grouping evidence** foldout before increasing it. It shows the actual
cached **interior-only DINO** cosine, contact fraction, gap, observation/view
counts and blocking rule. Tracking retains its context-contrast descriptor,
but grouping excludes that contrast term: different local surroundings should
not automatically prevent sofa parts from grouping. A clicked-token camera
similarity is still not the same as a mask/track-pooled cosine. Effective thresholds update after the next accepted
observation, not on graph republishing.

Safeguards remain independent of sensitivity:

- Both base entities must have fresh fused evidence and quality-approved DINO
  updates in the same cycle. Missing/stale features cannot confirm a new join,
  but are not evidence for separating an existing group. Default
  confirmation takes three distinct observations, not three display refreshes.
- Bounding-box proximity is insufficient: measured voxel supports must also be
  near. At least two sampled contacts and 4% of the sampled smaller support
  must agree; one stray voxel is insufficient. Search samples at most 128 source
  voxels per pair, so thin contacts can be missed. Pair evaluations are capped
  at 256/cycle and report budget limits. Broad planar support/structural surfaces
  are excluded from automatic object-part grouping.
- Similarly shaped/sized parts require a shared SAM mask covering at least 65%
  of each visible owner's depth support, from two camera-position cells 30 cm
  apart. Stationary, identical neighbouring boxes stay separate in tests.
  This deliberately also delays grouping identical sofa cushions.
- Groups have a 3.5 m maximum axis extent and at most 12 base members. New joins
  require complete cross-group appearance compatibility (default minimum cosine
  0.55, **Identity → Group Compatibility Floor**), in addition to the stronger
  direct-join threshold controlled by sensitivity. This permits appearance
  variation across parts while rejecting incompatible endpoints, avoiding simple
  transitive similarity chains. Explicit separation cannot be bypassed by a
  third entity. Three contradictory joint observations undo an established
  join; missing DINO or occlusion alone is not contradictory evidence.
- A maintenance split does not automatically group its children, but allows
  independent assembly evidence. Explicit detach corrections also veto assembly.

Grouping state and exclusions are saved with entity memory. Legacy assembly-v1
confirmations are cleared once because they used different descriptors/contact
rules; base identities, segment geometry, and explicit exclusions are retained.
Fresh observations must establish groups again. For correction,
publish one of these messages on `/crossmaps/entities/corrections`, using **base
entity IDs** from the inspector or `component_entity_ids`:

```json
{"action": "separate_entities", "entity_ids": [1, 2]}
{"action": "allow_entity_group", "entity_ids": [1, 2]}
```

`allow_entity_group` removes that explicit veto; it does not force a merge and
requires fresh evidence. The dashboard inspector is read-only; manual correction
buttons and semantic part-of/room inference remain future work.

This adds no model passes. It is still a heuristic object hypothesis, not proof
of physical identity: touching similar objects of different shapes can fool it,
and real parts with dissimilar descriptors or missing contact may stay separate.
Validate the actual sofa **and** neighbouring boxes, revisits and occlusion.
Drifting poses undermine both tracking and grouping; no sensitivity setting
fixes that upstream problem.

The segment log should show fallback matches after viewpoint changes. A stream
with many new segments but almost no projected or fallback matches usually
indicates pose/depth misalignment, poor masks, or association thresholds that
are too strict.

Use the dashboard's SAM, DINOv2, Segments, Identity, and Graph groups for live tuning.

## Camera freshness and bounded scheduling

### Camera transport on the Mac/robot link

The default inputs are now `/oak/rgb/image_rect/compressed` (`rgb_compressed=true`)
and `/oak/stereo/image_raw/compressedDepth` (`depth_compressed=true`). RGB uses the
camera's existing JPEG transport; depth retains exact uint16 millimeter samples.
Resolution, camera calibration, model sizes and inference precision are unchanged.
The existing OAK image-transport publishers provide these topics; no extra relay
or neural inference is needed. Topic/type parameters are read at node startup,
so changing them requires a **perception-node restart**.
Default eSAM checkpoint resolution for both `vitt` and `vits` is relative to the
script (`../weights/efficient_sam_VARIANT.pt`), not the launching shell's directory.
An explicit `esam_checkpoint` still takes precedence.

This addresses the observed raw-transport failure: CrossMaps stopped at three RGB
and seven depth receipts, while an independent compressed subscriber sustained
70 synchronized pairs over 30 seconds. A short robot ping test also showed 40%
packet loss. The evidence implicates raw image delivery over the link, not frozen
model inference; it does not prove the network's underlying fault is repaired.
The measured compressed payloads were roughly ten times smaller per frame.

For a reliable wired link, raw inputs remain available by explicitly setting all
four parameters at startup:

```sh
python3 crossmaps_v2_node.py --ros-args \
  -p rgb_topic:=/oak/rgb/image_rect -p rgb_compressed:=false \
  -p depth_topic:=/oak/stereo/image_raw -p depth_compressed:=false
```

`32FC1` **compressed** inverse-depth is explicitly rejected rather than silently
interpreted as millimeters; raw floating-point depth remains supported. Camera
info stays subscribed for accurate health diagnostics; unchanged calibration no
longer produces repeated logs.

From this directory in the ROS Python environment, isolate transport without
models or changes to the running node:

```sh
ROS_LOG_DIR="$PWD/.runtime/diagnostics" python3 -B camera_transport_probe.py --seconds 30
# Optional comparison; raw subscriptions increase network traffic:
ROS_LOG_DIR="$PWD/.runtime/diagnostics" python3 -B camera_transport_probe.py --seconds 20 --raw
```

The bounded probe reports five-second receipt counters, age, synchronized pairs,
payload bytes and decoded sizes. It never restarts hardware or changes settings.

### Observation scheduling and diagnostics

The default `observation_latest_frame=true` keeps one latest synchronized RGB-D
pair, replacing older pending pairs while inference is busy. A 100 ms dispatcher
starts work when the worker is available, capped by `observation_max_hz=2.0`.
Pairs waiting more than `observation_max_frame_age_s=2.0` seconds **since local
receipt** are discarded. This bounds pending work, not camera/network latency
or inference duration. `embed_every_n=10` now applies only when latest-frame
scheduling is disabled. Sparse camera delivery no longer needs ten synchronized
pairs before an observation can start. No additional neural passes are introduced.

Voxel query/confidence scoring is batched. Entity/query/graph publications share
scores only within one publication callback; subsequent callbacks recompute from
current memory. `publish_max_duty_cycle=0.35` adaptively skips publication ticks
when recent publication cost would exceed that budget, leaving more executor
time for camera callbacks. This is a scheduling target, not a hard latency bound:
individual callbacks remain non-preemptible. Result fusion has its own dispatcher
and does not wait for a skipped display-publication tick.

`/crossmaps/diagnostics/pipeline_health` reports input receipt and timestamp
advancement ages, recent input rates, synchronized pairs, worker stage/age,
queue sizes, dropped pending frames, and errors. Diagnostic thresholds default
to `pipeline_stale_after_s=10` and `pipeline_processing_timeout_s=30`. They label
stalls; they do not kill GPU work, restart hardware, or fabricate observations.
With synchronous inference, the executor may also delay the heartbeat itself.

The dashboard distinguishes stale images, missing input, processing errors,
long-running processing, and a missing heartbeat. Slow image requests are
single-flight with a five-second timeout; the last image stays visible with a
warning. Normal HTTP client disconnects no longer produce request tracebacks;
unexpected server errors remain visible. With **Streams** off, RGB is the last
prepared observation preview, not a full-rate live camera feed.

To load these changes, restart **both the experimental perception node and its
dashboard**, not just the dashboard. Keep `pipeline_runtime.py` and
`voxel_scoring.py` beside the node along with the existing helper modules. No
robot restart is required merely to deploy these code changes. Camera/network
delivery interruptions still require separate diagnosis; these safeguards do
not establish that the upstream camera problem is fixed.

## Local evidence capture and validation

Under **Identity**, enable **Record Identity Replay** for a bounded grouping trace.
The default is off; enabling records at most `identity_replay_max_frames=120`
accepted updates (hard limits: 1,000 frames / 32 MiB per file). The graph shows
progress and the path. Toggle off and on to start a new file. Recordings stay in
`.runtime/identity_replays/` inside this experimental directory. They include
geometry, descriptors, thresholds, decisions and source metadata, **not RGB**.
They can nevertheless contain sensitive spatial/appearance information.
Recording errors are reported without stopping perception.

Replay a captured trace, using its actual filename:

```sh
python3 -B replay_identity.py --assembly-input .runtime/identity_replays/FILE.jsonl
```

Replay restores the initial assembly state and checks decisions against the
recorded aliases. It exercises grouping deterministically, **not** the complete
RGB-D/SAM/tracking pipeline. Decision reproduction is not grouping accuracy;
unlabeled accuracy remains `null`. Keep a labeled RGB-D recording for end-to-end
evaluation. Capture the stationary sofa, neighbouring similar boxes, occlusion,
and a revisit before increasing grouping sensitivity.

Evaluate tracking separately from semantic querying:

1. Record a short RGB-D sequence that revisits the same objects from several
   viewpoints.
2. Count ID switches, duplicate confirmed tracks, and missed instances.
3. Inspect the eSAM and segment-ID projection overlays together.
4. Only after identities are stable, evaluate text-query retrieval and graph
   relations.

Run offline tests from this directory (Python with NumPy/OpenCV, and Node.js):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
node tests/test_dashboard_identity.cjs
node tests/test_dashboard_cloud_controls.cjs
node tests/test_dashboard_image_health.cjs
node --check crossmaps_dashboard/app.js
PYTHONDONTWRITEBYTECODE=1 python3 replay_identity.py --demo
```

Tests exercise the actual experimental node's association/fusion/graph methods
with transport and inference stubbed, plus standalone identity replay. They do
not initialize ROS, download weights, or access other project directories.
Cases include stationary observations, mask fragmentation, two identical
adjacent boxes, a mask spanning both, occlusion/reappearance, conflicting DINO,
split lineage, repeated publication, display filtering, and serialization.
Additional tests cover small-mask attribution, display-only filling, SAM
dropouts, complementary-part grouping, sensitivity, hysteresis, explicit
separation, incompatible similarity chains, 80-frame noisy mask-ID replay,
40-frame depth-quantization replay, bounded boundary repair, descriptor
contamination, restart namespaces, recording failures and trace reproduction.
They do not establish real
sofa grouping accuracy or MacBook end-to-end timing.

`replay_identity.py --input .runtime/replay.jsonl` accepts saved post-segmentation
observations; its module docstring specifies the format. Optional `truth_id`
labels enable ID-switch, duplicate-entity-frame, and false-group-entity-frame
counts. Unlabeled accuracy metrics are `null`, not an implied perfect score.
Replay identity timing is separate from optional recorded full-cycle timing.
This isolates identity logic; use RGB-D recordings on the live pipeline to
evaluate segmentation, pose robustness, and actual end-to-end latency.

For manual browser checks, `python3 -B tests/dashboard_fixture.py` serves the
actual UI with synthetic images/clouds on a printed localhost URL. It never
connects to ROS, loads weights/maps/presets, or writes settings to the real node.
Check all four image corners and all tabs at desktop, narrow, Large, and
fullscreen sizes, then verify similarity clicks at the letterboxed edges.

The stationary yaw-drift problem is **not fixed by this rework**. Validate
timestamp alignment, calibration, and the external pose/SLAM source before
loosening geometry gates. Automatic rooms, semantic part-of inference, and
loop-closure-aware map deformation remain later milestones.

The descriptor JSON now reports geometry in its declared `frame_id`; older
experimental output incorrectly labeled internal-frame coordinates as the
publish frame.

## Surface stability and stationary validation (2026-09-26)

See [YAW_DRIFT_DIAGNOSIS.md](YAW_DRIFT_DIAGNOSIS.md) for the subsequent live
RF2O/base-node drift investigation, measured evidence, and read-only pose probe.

### Dominant camera overlays and neighborhood tie-breaking

The default Segments/Entities camera overlays now show one stable dominant
color per current SAM silhouette. A dashed amber outline and `?` identify a
provisional display hypothesis; gray remains unassigned. Entity votes are
aggregated by the current object group **before** selecting its color. An
incumbent survives small vote fluctuations; a stronger challenger needs three
observations. Missing depth, expired history, world-position disagreement and
appearance conflicts prevent unsupported display carry-over.

This display state never enters fusion, segment ownership, entity membership,
or grouping evidence. It updates once per observation, even without image
subscribers; requesting or refreshing an image cannot accumulate votes.
Under Segments, turn off **Stable Dominant Overlay** for the previous mixed
ownership diagnostic view, or enable **Raw Support Debug View** for measured
projected IDs. Graph mask diagnostics distinguish display choices from actual
association decisions.

Under Identity, **Neighborhood Tie-Break** is enabled by default. Only otherwise
ambiguous candidates above the existing geometric overlap minimum qualify.
The **Tie-Break Overlap Window** defaults to 0.10 (score difference, not calibrated
probability). A bounded bonus of at most 0.15 uses fresh interior-DINO similarity
and real voxel contact with an already established, currently observed member.
Exact overlapping support does not count again as neighborhood contact. Three
consecutive observations with the same supporting neighbor are required.
Structural surfaces, incompatible CLIP, repeated similar shapes, excessive
object extent, missing/stale features, and explicit exclusions cannot supply
this bonus. Grouping Sensitivity controls the minimum DINO similarity here too.
Pending diagnostics expose overlap, neighbor and context strength.

This does not reassign established segments or destructively merge surfaces.
Joining already established sofa parts is still the reversible DINO/contact
assembly layer's job. Similarity alone cannot distinguish all touching objects;
the new display is not proof that a whole physical object has been recognized.
No extra neural-network inference is added, and pose drift is not corrected.

### Duplicate-publisher and temporal-allocation fix

Two simultaneous perception processes were found publishing different world
namespaces and segment IDs to the same `/crossmaps` topics. This caused whole
scene ID/colour alternation even when each process's own switch counter was low.
The older stationary test process was stopped; the user's newer process was
retained and `stationary_pose_lock` was explicitly set false. Per-process ID
switch counters alone cannot establish dashboard identity stability.

Startup now takes an OS file lock scoped to the experimental directory and ROS
domain, and checks for existing perception publishers before loading models.
Duplicates fail with an actionable error instead of silently mixing maps. The
dashboard also warns when health reports multiple perception publishers. Stop
the previous perception node before launching a replacement; no manual lockfile
deletion is needed after a crash. The OS releases its lock on process exit.
The discovery check also detects older builds and remote publishers already
visible at startup; it cannot atomically prevent simultaneous launches on two
different machines. The runtime warning is a secondary safeguard for that case.

A geometry- and appearance-validated temporal match now actually controls
allocation of the same mask's unowned remainder. Previously a few guided points
could update the incumbent while the remaining pixels birthed a duplicate ID.
Reuse stays local and extension-budgeted, preserves other owners, and still
requires current depth/world-position and appearance agreement. It is not an
unconditional merge of all similar-looking adjacent objects. Existing erroneous
geometry or unresolved upstream drift is not repaired by this association fix.

Geometric identity now outlives semantic-cache eviction. Mature surface tracks
with support become dormant rather than expiring on a short timer; memory is
bounded by `segment_memory_max_tracks` (512 by default). Persistent, consecutive
free-space evidence can still remove obsolete support. Current measured depth
endpoints and their one-voxel neighbourhood take precedence over clearing rays
in the same observation. Intermittent misses no longer accumulate indefinitely.
Temporal mask guidance has a local extension budget, and a conflicting boundary
voxel no longer prevents updating all the valid support in a mask. Descriptor
sums and capped weights remain consistent over long observation sequences.

Entity assembly uses the dominant 26-connected surface component (at least 60%
of measured support) for contact and extent tests. Isolated depth outliers no
longer inflate otherwise compact sofa parts into room-sized objects. Original
map support is retained; excluded counts are diagnostic, not deleted evidence.
Fragmented geometry remains uncertain. Fresh interior-DINO agreement, sustained
contact, extent, structural-surface and similar-box guards still apply. One bad
extent observation no longer immediately dissolves an established assembly.
These are reversible grouping hypotheses, not verified semantic part-of labels.

In the optional mixed-ownership diagnostic view, amber means competing supported
owners, not a segment waiting for a timer. Mixed masks preserve locally supported
colours and hatch unresolved pixels. The default dominant view uses dashed
outlines and `?` instead. Filling is display-only. SAM's outline
does not imply that all pixels within it have one established world identity.

`depth_use_latest_tf=false` and `depth_allow_latest_tf_fallback=false` require
image-time TF by default; unavailable transforms cause skipped observations,
not fusion with a silently different pose.

`stationary_pose_lock=true` is an **explicit stationary-test override**, default
off. It fixes the camera pose at the first valid observation and exposes a
dashboard warning. **Do not move the robot while it is enabled.** Before moving,
archive the stationary-test map and start a fresh map with the override disabled
(or implement a properly validated alignment). This does not fix upstream yaw
drift or support mobile mapping. Saved anchored maps retain their anchor and
refuse loading with the override disabled. Robot motion between sessions still
invalidates a stationary anchor; do not reload it after moving the robot.

Regression coverage includes semantic eviction, dormant tracks, intermittent
misses, bounded descriptors and mask growth, pose anchoring/save-load, noisy
assembly bounds, connected-core extraction, and honest mixed-mask rendering.
Live stationary and recorded grouping checks are useful diagnostics, not a
ground-truth accuracy benchmark or evidence of robust mobile SLAM.

Deployment check: world `97670843-255e-4465-90d5-57e2ab4c860b`, first 50
stationary accepted observations: zero reported mask-owner switches, 11 base
entity births with no subsequent births in sampled updates, two sustained
assembly joins (including the visible sofa back/front), zero pipeline errors,
and about 2.53 s average full cycle on this MacBook. Mixed plant/boundary support
remained uncertain. These counters do not measure all possible ID errors.
Prior corrupt maps were preserved as
`.runtime/map_before_stability_20260926_{live,checkpoint}.pkl`; the intermediate
stationary test was archived as `.runtime/map_stationary_firstpass_20260926.pkl`.
The intermediate archive predates anchor serialization: retain it for evidence,
not automatic live-map resumption. Only the experimental perception node was
restarted; the robot and dashboard were left running. Refresh the dashboard
page to load the updated static captions and stationary-lock warning.
