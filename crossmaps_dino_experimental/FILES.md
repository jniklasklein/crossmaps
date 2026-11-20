# CrossMaps — collaborative and v2 file inventory

The two-rover experimental branch is **`experimental/crossmaps-collaborative`**.
Its complete startup and operating guide is [COLLABORATIVE.md](COLLABORATIVE.md).

## Collaborative entry point and modules

| File | Purpose |
| --- | --- |
| `collab_relay_auth.py` | Host-scoped SSH/Keychain preflight and optional local enrollment for camera recovery. |
| `collab_retrieval.py` | Bounded queue and verification coverage across different frame and submap pairs. |
| `crossmaps_collaborative.py` | One-command coordinator, input process lifecycle, shared inference scheduling, built-in dashboard, checkpoints. |
| `config/collaborative_two_rovers.json` | Two-rover live defaults: input domains 0/1, central domain 42, dashboard port 8766, automatic alignment. |
| `collab_config.py` | Configuration validation and experimental-owned run directories. |
| `collab_transport.py` | Read-only per-domain ROS input adapters, isolated TF, compressed RGB-D decoding and calibrated depth registration. |
| `collab_camera_relay.py` / `collab_camera_relay_source.py` | Optional authenticated, bounded compressed RGB-D fallback for Wi-Fi; laptop receiver and ephemeral sensor-only sampler inside the running rover container. |
| `collab_runtime.py` | Fair bounded latest-frame scheduling and completed-cycle timing. |
| `collab_perception.py` | Shared model objects with separate source/session v2 memories; pose contradiction guard and observation evidence extraction. |
| `collab_backend.py` | Keyframes/submaps, candidate retrieval, source/session provenance, pose constraints, reversible global identity links and persistence. |
| `collab_provisional.py` | Display-only provisional alignment from agreeing, verified RGB-D captures; confirmed graph and object identities stay separate. |
| `collab_identity_inspector.py` | Read-only identity review: contributing camera frames, projected surface samples and explanations of pending or suspended links. |
| `collab_geometry.py` | RGB-D correspondence verification, robust fitting, ICP, geometry checks and bounded sparse keyframe pose-graph optimization. |
| `collab_dashboard.py` | Loopback HTTP server, protected command API, source camera endpoints and static asset allowlist. |
| `collab_dashboard/index.html` | Collaborative dashboard layout and alignment controls. |
| `collab_dashboard/app.js` | Per-rover camera/health views, source-local/combined WebGL map and evidence review. |
| `collab_dashboard/styles.css` | Responsive, uncropped camera and map presentation. |
| `collab_demo.py` | Synthetic observations for no-robot integration/demo runs; not a learned-perception accuracy test. |
| `COLLABORATIVE.md` | Two-rover architecture, startup, configuration, evidence workflow and limitations. |

The collaborative perception wrapper also imports the v2 entry point and helper
modules listed below. Keep both sets together. No sibling backup directory is
used, and no duplicate model-loaded ROS perception nodes need to be launched.

From the repository root with the existing ROS/model environment active:

```sh
python3 crossmaps_dino_experimental/crossmaps_collaborative.py --check
python3 crossmaps_dino_experimental/crossmaps_collaborative.py
```

Open `http://127.0.0.1:8766`. The server includes the dashboard. Generated evidence,
recorded configuration and logs live under `.runtime/collaboration/` and stay
local/ignored. `--demo` requires NumPy/OpenCV but no ROS or model initialization.
`--replay RUN_DIR` inspects a retained run without ROS/models or writing the source
recording; decisions in that dashboard are in-memory only.

## Single-rover v2 foundation

This is the CrossMaps v2 DINO/CLIP world-model pipeline with separate surface
segments, reversible entity groups, stable dominant camera overlays, and bounded
neighborhood tie-breaking. The sofa grouping was reported working in live use;
this is not a general accuracy or mobile-localization guarantee.

## Files used by the single-rover foundation

| File | Purpose |
| --- | --- |
| `crossmaps_v2_node.py` | Perception entry point: RGB-D transport, eSAM/CLIP/DINO inference, fusion, ROS publications. |
| `identity_graph.py` | Persistent entity membership, near-tie evidence, and temporal graph relations. |
| `object_assembly.py` | Reversible DINO/contact grouping of established parts; retains separate segments. |
| `surface_tracking.py` | Temporal surface consensus and bounded ownership corrections. |
| `dominant_overlay.py` | Stable, display-only dominant ownership hypotheses. |
| `segment_overlay.py` | Mask-aligned colors, outlines, provisional labels, and diagnostic overlays. |
| `world_model.py` | World IDs, descriptor helpers, geometry checks, evidence recording. |
| `voxel_scoring.py` | Vectorized voxel scoring shared with diagnostic tests. |
| `pipeline_runtime.py` | Bounded latest-frame scheduling and pipeline-health tracking. |
| `pipeline_instance.py` | Single-pipeline process lock and duplicate-publisher startup check. |
| `crossmaps_dashboard.py` | Separate ROS/HTTP dashboard server and parameter interface. |
| `crossmaps_dashboard/index.html` | Dashboard page structure. |
| `crossmaps_dashboard/app.js` | Dashboard controls, images, maps, graph and 3D rendering. |
| `crossmaps_dashboard/styles.css` | Dashboard layout and presentation. |

Keep the Python helpers beside the perception entry point and the three static
assets together in `crossmaps_dashboard/`. No sibling backup directory, alternate
root-level perception script, or root-level dashboard is imported by this version.

## Development and diagnostic files

- `tests/`: Python regression tests, Node.js dashboard tests, and the offline
  `dashboard_fixture.py` UI fixture. Not required for normal perception runs.
- `replay_identity.py`: offline identity/assembly replay and synthetic demo.
- `camera_transport_probe.py`: optional RGB-D transport investigation.
- `pose_drift_probe.py`: optional read-only ROS pose/TF investigation.
- `fixed_scan_probe.py`: optional fixed-grid lidar experiment; not part of the
  normal pipeline. Read its CLI and the yaw investigation before using it.
- `README.md`: architecture, parameters, validation and limitations.
- `YAW_DRIFT_DIAGNOSIS.md`: historical diagnosis, including unresolved upstream
  localization issues. Referenced local recordings are not shipped.
- `.gitignore`: excludes generated runtime data, caches and model artifacts.

## Start the single-rover version

Use the existing ROS 2 environment with the dependencies below available.
From the repository root, in separate terminals:

```sh
python3 crossmaps_dino_experimental/crossmaps_v2_node.py
```

```sh
python crossmaps_dino_experimental/crossmaps_dashboard.py
```

Open `http://127.0.0.1:8765`. The dashboard is the primary interface; RViz2 is
not required. Use the same ROS/Python environment in both terminals. Start only
one CrossMaps perception process.
The robot's camera, depth, camera-info and TF publishers must already be running.
Defaults retain eSAM `vitt`, CLIP `ViT-L-14`/`openai`, DINOv2 `vits14`, compressed
RGB/depth transport, and **stationary pose lock off**. Default map autoload and
autosave are off. Robot-side sources and launch configuration are not part of
this snapshot.

The single-rover baseline branch is `crossmaps-v2`. The existing `crossmaps_dino_experimental/`
directory, topics, and stored-map schema identifiers are deliberately retained
for compatibility. The ROS node is `/crossmaps_v2_node`, and the dashboard's
default target matches it. The renamed entry
point does not move or delete local data. See README.md for naming details.

## External prerequisites, not repository source

- Python with NumPy, OpenCV, Pillow, PyTorch/torchvision, `open_clip`, and the
  EfficientSAM Python package.
- ROS 2 Python packages: `rclpy`, `rcl_interfaces`, `tf2_ros`, `message_filters`,
  `cv_bridge`, and the standard sensor/navigation/geometry/visualization message
  packages imported by the entry points.
- EfficientSAM weights. Default path is
  `weights/efficient_sam_vitt.pt` relative to the repository root, or provide
  `--ros-args -p esam_checkpoint:=/absolute/path/to/efficient_sam_vitt.pt`.
- CLIP weights and the DINOv2 Torch Hub repository/weights. They are cached or
  downloaded by their loaders. Offline DINO configuration is documented in README.
- Node.js is needed for dashboard regression tests, not for serving the UI.

This is a source snapshot, not a locked environment export. Existing local
packages and model files have not been moved, deleted or bundled into Git.

## Kept local

- `.runtime/`: maps/checkpoints, screenshots, recordings, logs and lock files.
- `.runtime/collaboration/`: independent collaborative run configurations,
  keyframe evidence, graph decisions and ROS logs; not copied into Git.
- Python caches, OS metadata and any model binaries.
- The dashboard's default preset file is
  `~/.crossmaps/crossmaps_dashboard_presets.json`; it is external to this folder.
  A source checkout alone does not reproduce arbitrary unsaved live parameters.
- Other experimental directories and unrelated root-level work remain local
  and are not included in the CrossMaps v2 source snapshot commit.

## Verification

From this directory:

```sh
python3 -B -m unittest discover -s tests -q
node tests/test_dashboard_identity.cjs
node tests/test_dashboard_cloud_controls.cjs
node tests/test_dashboard_image_health.cjs
node --check crossmaps_dashboard/app.js
node tests/test_collab_dashboard.cjs
node --check collab_dashboard/app.js
python3 -B replay_identity.py --demo
```

At the 2026-11-11 launch-documentation update, 182 Python tests and the dashboard checks
passed using the existing local environment. The tests do not initialize the
neural models or validate live mobile mapping. No robot/node restart is needed
to create this Git snapshot.
