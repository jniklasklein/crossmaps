# CrossMaps collaborative experiment

Branch: **`experimental/crossmaps-collaborative`**.

The laptop observes two independent rover ROS domains, runs one shared set of
models, maintains separate local surface/entity memories, and builds a shared
map from verified or explicitly supplied relationships. The existing single-rover
v2 entry points remain available; they are not needed alongside this server.

This is an experimental mapping system, not a robot localization replacement.
It never publishes movement commands or changes either rover's TF tree.

## Start with the two rovers

On each rover, start its existing camera, depth, calibration, odometry and TF
publishers as usual. Rover A must use ROS domain **0**, and rover B domain **1**.
The Jetsons' normal SLAM launch includes sensor recovery. For a stalled feed,
use the [rover-side health check](rover_recovery/README.md) before restarting
SLAM; it distinguishes a live publisher from a sensor actually producing data.
For a rotating map with fresh sensors, see the [odometry patch and parked/turn
checks](rover_odometry/README.md). The diagnostic RF2O heading is not the
authoritative mapping pose on a rover with that patch enabled.
The laptop must be able to discover both domains over the network. Use the
existing ROS/model environment on the laptop:

```sh
conda activate ros_humble_env
cd /Users/niklasklein/Documents/Projekte/HPI/crossmaps-ws
python3 crossmaps_dino_experimental/crossmaps_collaborative.py --check
python3 crossmaps_dino_experimental/crossmaps_collaborative.py
```

Open **http://127.0.0.1:8766** in Safari or another WebGL-capable browser.
The dashboard is built into that one command: **do not additionally start
`crossmaps_dashboard.py` or two `crossmaps_v2_node.py` processes.** Stop any old
local perception runs first to avoid unnecessary duplicate GPU work. The rover
publishers stay running.

The server explicitly selects each domain in its own ROS context; exporting
one `ROS_DOMAIN_ID` in the laptop terminal is not how the two sources are
selected. The central diagnostic ROS domain defaults to **42**, separate from
both inputs. These values are configurable.

`--check` validates configuration, the main Python dependencies and the eSAM
checkpoint path. It does **not** connect to the rovers, validate live TF or
camera calibration, initialize every neural model, or guarantee that uncached
model weights are available. Initial model loading may take time.

### First experiment

1. Confirm both rover cards receive fresh RGB and processed observations. Check
   the actual RGB scenes and depth registration before trusting geometry.
2. Inspect **Rover A** and **Rover B** separately in the spatial viewer. Combined
   deliberately excludes unaligned submaps. The first observed source session
   defines the shared coordinate reference; no GPS/global origin is implied.
3. Point both cameras at a common textured scene with details at several depths.
   For a first trial, start with similar headings and wait for a verified match.
   Leave one rover in place, move the other about 20–30 cm while keeping those
   details visible, then pause. Check that **Last map integration** updates and a
   new keyframe appears. Automatic alignment needs two geometrically consistent
   matches, with a genuine viewpoint change on at least one rover. Repeated
   observations from the same two positions provide only one viewpoint pair.
4. Leave **Shadow mode on** initially. It retrieves candidates and computes
   registration evidence without automatically accepting alignments or identity
   links. Inspect inliers, overlap, residuals and rejection reasons. A verified
   constraint can be explicitly accepted in the dashboard.
5. Alternatively, use **Supply a measured anchor** for the known-alignment
   baseline. Enter a measured transform, not a guess from where the rover bodies
   currently appear to be. The dashboard labels it *supplied*, not automatically
   recognized. Undo removes that constraint while retaining source observations.
6. Inspect shared-entity proposals only after alignment is credible. Their
   membership remains reversible; source segment/entity IDs are not overwritten.

Turning Shadow mode off allows automatic acceptance once cross-rover evidence
is corroborated. Existing eligible entity links can activate when the coordinator
re-evaluates them. Local mapping and camera previews run in either mode.

## Configuration

The default file is
[`config/collaborative_two_rovers.json`](config/collaborative_two_rovers.json).
Use a separate JSON file with `--config path/to/config.json` for a different
setup. Missing settings inherit validated defaults from `collab_config.py`.

| Setting | Default / meaning |
| --- | --- |
| `robots[].id` | `rover_a`, `rover_b`; stable source names, independent of domain numbers |
| `robots[].domain` | `0`, `1`; each adapter has an isolated ROS context and TF buffer |
| `central_domain` | `42`; publishes `/crossmaps/collaboration/state` diagnostics |
| `dashboard` | Loopback `127.0.0.1:8766`; override the port with `--port` |
| `robots[].local_frame` | `odom`; the source-local frame used for timestamped camera poses |
| `robots[].rgb_topic` | `/oak/rgb/image_rect/compressed` |
| `robots[].depth_topic` | `/oak/stereo/image_raw/compressedDepth` |
| `robots[].camera_info_topic` | `/oak/rgb/camera_info` |
| `robots[].depth_camera_info_topic` | `/oak/stereo/camera_info` |
| `robots[].depth_registration` | `auto`; see the calibration requirements below |
| `input.max_hz` | `2.0` prepared input packets/s per source; not a promised inference rate |
| `input.max_age_s` | `2.0`; reject stale input before processing |
| `input.sync_slop_s` | `0.08`; maximum paired RGB/depth timestamp skew |
| `backend.shadow_mode` | `true`; automatic cross-source acceptance disabled initially |
| `pose_guard.enabled` | `true`; pauses integration on a supported pose contradiction |
| `pose_guard.rgbd_recovery` | `true`; allows a local mapping pose to be recovered from a previously observed RGB-D view after repeated geometric and image agreement |
| `perception.parameters` | Shared v2 model/processing defaults; robot-specific non-model settings can use `robots[].parameters` |

The model defaults remain **eSAM ViT-T**, **CLIP ViT-L-14 / OpenAI**, and
**DINOv2 ViT-S/14**. There is one full-frame DINO feature pass and one configured
CLIP crop per mask, not an extra set of per-instance DINO model passes.
Model-weight loading/caches use the existing v2 environment. Do not configure
incompatible model variants for the two sources: descriptors must share a
feature space to be comparable. Stationary pose lock is explicitly unsupported
in collaborative mode.

### Timestamp and calibration requirements

- Keep laptop and rover clocks synchronized. The adapter checks a paired
  depth frame's age on arrival and tracks time spent waiting for TF separately.
  Network delivery and sensor buffering also contribute to age on arrival.
- RGB/depth pairs and TF must use a consistent clock. TF is looked up at the
  observation timestamp. The adapter waits briefly for valid data; it does not
  substitute an arbitrary latest transform.
- RGB must be rectified with matching `CameraInfo`. Image and calibration frame
  IDs must agree.
- `depth_registration: "auto"` uses same-optical-frame depth only when its pixel
  grid matches RGB. Different optical frames require depth calibration and a
  valid depth-to-RGB extrinsic; depth is geometrically projected with a
  nearest-surface z-buffer. **Depth is not resized to fake registration.**
- `"project"` explicitly requests calibrated reprojection. `"registered"`
  explicitly trusts that the supplied depth already uses the RGB pixel grid;
  choose it only after confirming the camera driver's actual output convention.
- A source clock reset creates a new source session. Old segment IDs are not
  silently reused as identities from a new run.

For diagnosis from the ROS environment, inspect each domain independently, e.g.
`ROS_DOMAIN_ID=0 ros2 topic list` and `ROS_DOMAIN_ID=1 ros2 topic list`. These
commands inspect discovery only; seeing topic names does not prove advancing
frames, valid depth, or image-time TF.

## What runs where

```text
Rover A / domain 0 → input adapter A + TF buffer ─┐
                                                ├→ fair latest-frame scheduler
Rover B / domain 1 → input adapter B + TF buffer ─┘             │
                                                              ▼
                                              one shared eSAM / DINO / CLIP set
                                                  │                     │
                                           A local memory        B local memory
                                                  └─────────┬───────────┘
                                                            ▼
                                      retained RGB-D keyframes + short submaps
                                                            │
                             candidate retrieval → RGB-D verification → pose graph
                                                            │
                               source-owned evidence + reversible identity links
                                                            ▼
                                            built-in collaborative dashboard
```

The input processes subscribe only to their own source domains. Topic names and
TF frame IDs may be identical on both rovers because their TF buffers never mix.
The coordinator processes sources fairly with at most one pending observation
per rover. Older pending observations are replaced instead of accumulating an
unbounded inference backlog. RGB and depth previews use the compressed camera
streams and update independently of model completion. The depth preview uses
the camera's depth grid. Mapping uses depth registered to RGB.
If compressed RGB stops advancing for 12 seconds while valid depth still
advances, the laptop adapter replaces just its RGB reader, clears stale pair
queues, and retries at most once per 30 seconds. The last image keeps its
original age until a fresh source timestamp arrives. This does not restart
the rover, switch to raw images, or reset the source session. A persistent
outage still warrants checking on-rover publication and wireless delivery.
One serialized inference worker shares model objects; local v2 segmentation,
ownership and entity state are separate for each source/session.

The retained unit is a pose-anchored RGB-D keyframe in a short local submap.
Retrieval uses appearance, surface descriptors and geometric summaries; the
current scene graph is not required to be correct before finding a candidate.
Verification uses distinctive mutual visual correspondences, depth-backed robust
rigid fitting, bounded ICP, spatial support and overlap checks. Appearance alone
never establishes a shared pose. Candidate transforms, supporting observations,
acceptance state and provenance remain inspectable.

ORB is the inexpensive first pass. When it cannot initialize a pose, the
verifier proposes spatially distributed mutual RootSIFT/AKAZE features and
uses calibrated depth (including two-direction PnP when depth has holes).
The fallback checks held-out image matches and original, mutually visible
depth in both directions. ICP can veto a pose, but cannot slide the returned
pose along a wall to improve its residual at the expense of image alignment.
The dashboard distinguishes retrieval, visual matches, geometric verification,
and activation; a second pose-supported view is still required for automatic
corroboration. If two verification slots are available, one may test a
recent frame from the other rover rather than another old lookalike.

To compare ORB with the current verifier without touching the live run:

```sh
python3 crossmaps_dino_experimental/replay_alignment.py \
  crossmaps_dino_experimental/.runtime/collaboration/RUN --selection nearest --limit 12
```

Replace `RUN` with the recorded run directory. Use `--selection checked`
to inspect previously checked pairs, or repeat `--pair SOURCE_ID:TARGET_ID`
for a fixed comparison. The replay reports failed stages, verified camera
poses in a common direction, and deliberately mismatched controls. Agreement
across stationary frames is not a physical accuracy measurement or a map merge.

All retained keyframes participate in the pose graph. Conservatively weighted
within-source odometry edges preserve temporal structure; accepted RGB-D matches
constrain the actual matched keyframe pairs, and a supplied anchor connects
submap-origin frames. Optimized keyframe poses update submap poses **and their
internal geometry**. Affected voxel clouds and entity support are rebuilt from
the original camera-frame observations; corrected positions are not recursively
fused over already corrected points.

Rejecting a cross-source constraint recomputes the graph and geometry; a
disconnected source becomes unaligned rather than being displayed at a guessed
identity transform. Source-local views use the internally corrected submap
geometry at its original local origin; Combined uses its optimized world pose.
The latest `T_world_local` diagnostic is not a promise that one rigid transform
aligns an entire deformed trajectory. Global identities require aligned
geometric support and compatible appearance. A changed alignment can suspend an
identity link without deleting the original decision or either rover's source
observations. Keyframe/submap/source-local IDs are preserved through correction.

### Dashboard interpretation

- **Rover cards:** camera/processed-observation ages, mean completed cycle time,
  keyframe/submap counts, source domain/session and pose status.
- **Spatial view:** Combined or one rover's local frame; source coloring or
  measured RGB; orbit, pan, zoom, top view, fit and expansion. Unaligned data is
  deliberately absent from Combined but available in its source view.
- **Camera cards:** independent RGB, SAM, segments, entities, DINO, DINO similarity,
  CLIP and depth views. Images preserve their aspect ratio and can expand. A
  decoded preview is not proof that semantic inference or map integration is
  current; inspect the corresponding ages/status.
- **Alignment candidates:** retrieval cues, geometric verification and transform
  details, explicit acceptance/rejection and undo for supplied anchors.
- **Shared entities:** cross-source membership, proposed links, suspended links
  and reversible identity decisions. Single-source hypotheses are not counted
  as already shared entities.
- **Events / checkpoint:** coordinator events and durable evidence save. The
  HTTP command interface is loopback-only, same-origin and token-protected.

## Recordings, stopping and resuming

Each normal launch creates a new run below:

```text
crossmaps_dino_experimental/.runtime/collaboration/<UTC-time>-<unique-id>/
  config.json
  ros_logs/
  evidence/
    state.json
    frame-<evidence-id>.npz
```

The exact run path is printed at startup. These generated files are ignored by
Git and do not overwrite v2 maps. Retained keyframes and graph decisions are
checkpointed during operation. **Ctrl-C** stops only this server and its own
input adapter children; the rover publishers are left alone. It attempts a final
checkpoint. If model inference cannot finish promptly during shutdown, the last
completed checkpoint remains available; an in-flight frame is not guaranteed.

To resume retained evidence, pass the existing run directory explicitly. Its
recorded configuration is selected automatically:

```sh
python3 crossmaps_dino_experimental/crossmaps_collaborative.py \
  --run-dir crossmaps_dino_experimental/.runtime/collaboration/RUN \
  --resume
```

Replace `RUN` with the actual directory name. The new adapters create new source
sessions; resuming evidence is **not** permission to assume a restarted rover has
the same odometry origin. Re-establish alignment for those sessions. An existing
run directory is rejected without `--resume`. Run directories must remain below
the experimental collaboration runtime directory. An explicitly supplied config
cannot silently change the sources, models, central domain or input conventions
of a resumed run; use a fresh run for those changes.

Storage and computational budgets are bounded. Default backend limits include
256 retained keyframes, 64 submaps and a 2 GiB evidence-file budget. On reaching
capacity the system preserves retained evidence and stops adding new keyframes;
preview/perception can continue. Inspect the capacity message and start a new
run or deliberately adjust limits. This is not an unlimited lifelong-map store.

## Try the UI without robots

```sh
python3 crossmaps_dino_experimental/crossmaps_collaborative.py --demo --check
python3 crossmaps_dino_experimental/crossmaps_collaborative.py --demo
```

The synthetic two-source demo needs NumPy/OpenCV but does not initialize ROS or
neural models and does not connect to the robots. It has the same dashboard and
evidence backend. It validates plumbing and interaction, **not live perception
accuracy**. `--duration 20` is useful for a short smoke test; `--port 8767` allows
an alternative dashboard port. Run only one coordinator per configured central
domain at a time, including demos.

### Inspect a recorded run without changing it

```sh
python3 crossmaps_dino_experimental/crossmaps_collaborative.py \
  --replay crossmaps_dino_experimental/.runtime/collaboration/RUN \
  --port 8767
```

Open `http://127.0.0.1:8767`. Replay loads the retained checkpoint and source
observations without ROS connections or model initialization. The dashboard
clearly labels it **Recorded replay**. Alignment and identity decisions can be
explored in memory; the source recording is unchanged and checkpoint/save is
disabled. The camera cards offer the latest retained RGB/depth observations, not
fresh inference or a time-stepped video reconstruction. SAM/DINO/CLIP overlay
views are not regenerated in this mode. Replay uses its recorded configuration
and can use a separate dashboard port while another run is active.

## Known limits and troubleshooting

| Symptom | Check |
| --- | --- |
| One or both rover cards receive nothing | Domain IDs, reachable DDS discovery, active camera topics, ROS environment and network/firewall settings |
| RGB and depth visible but SAM, DINO or CLIP missing | Check pairing, calibration, depth registration and the model worker. These views require a calibrated RGB-D observation. |
| Frame already old on arrival | Compare sensor stamps, host clocks and delivery delay. |
| Waiting for odometry TF or inconsistent visual pose | SAM, DINO and CLIP can update from registered RGB-D. Segments and Entities retain the last mapped overlays until a frame passes pose validation. |
| Depth registration rejected | Check true optical frames, matching CameraInfo and calibrated depth-to-RGB transform; do not resize depth or guess extrinsics |
| RGB live, processed age grows | Check incoming RGB-D pairs, model initialization and inference throughput. Each feed carries its own image age. |
| B absent from Combined | Expected until its source/session is connected by an accepted verified constraint or supplied anchor |
| No shared identities after an anchor | Alignment alone does not prove object identity; inspect proposals, geometry/appearance and distinct viewpoints |
| `visual_odometry_contradiction` / `pose_rejected` | Measured scene evidence disagrees with supplied odometry; integration pauses rather than smearing geometry |
| Already-running/port error | Stop the previous coordinator normally or choose a different port and nonconflicting central domain for an intentionally separate run |

When odometry disagrees with the camera, local mapping can resume if multiple
RGB-D observations register reliably to the same trusted view. The checkpoint
records both the supplied and recovered poses for those observations. Set
`pose_guard.rgbd_recovery` to `false` for the original pause-on-contradiction
behavior. The guard operates on the laptop. Upstream RF2O/base heading issues
still affect the rovers' own localization. Accepted visual
constraints can correct retained keyframe poses and rebuild internal submap
geometry, but only where valid evidence constrains the trajectory. Short submaps
and optimization cannot recover missing measurements or guarantee that every
false alignment is detected. Room/building inference, lidar place recognition,
distributed model inference, navigation feedback and a general multi-hypothesis
SLAM system are not implemented by this experiment.

Automated regression tests exercise isolation, transform conventions, scheduling,
verification, reversible decisions, persistence, camera transport, HTTP security
and display behavior. A synthetic demonstration or passing tests is not a
substitute for measured two-rover trials with overlap, no overlap, repeated
objects, lighting changes, motion, reconnections and clock resets.

## Validation of this implementation

The full Python suite passed **287 tests**, including real ROS adapter tests on
isolated localhost domains **86/87**, with no connections to the physical rovers.
The collaborative dashboard passed **21 JavaScript checks**; the existing v2
dashboard checks also passed. Safari was used to inspect the rendered cameras
and map, and to apply the synthetic scene's known frame alignment in read-only
replay. Recording immutability, source separation, timestamped/delayed TF,
geometry correction/rollback, and command access controls have regression tests.

From `crossmaps_dino_experimental/`, run:

```sh
CROSSMAPS_ROS_SMOKE=1 ROS_LOCALHOST_ONLY=1 \
  ROS_LOG_DIR="$PWD/.runtime/diagnostics" \
  python3 -B -m unittest discover -s tests -q
node tests/test_collab_dashboard.cjs
```

The environment assignments above apply only to the test command. Do not leave
`ROS_LOCALHOST_ONLY=1` exported for a live run: it prevents network discovery of
the rovers. The actual shared neural-model pipeline's two-rover throughput and
real-scene registration quality still need the first physical trial; synthetic
timings are not estimates of eSAM/DINO/CLIP performance.
