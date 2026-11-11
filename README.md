# CrossMaps v2

RGB-D world modeling with persistent surface segments, DINO-based object-entity
grouping, CLIP semantics, and a live browser dashboard.

This branch is **`crossmaps-v2`**. The current implementation lives in
[`crossmaps_dino_experimental/`](crossmaps_dino_experimental/). The directory
name is retained to preserve existing local paths; it contains the v2 code.
The **browser dashboard is the primary interface**. RViz2 is optional and is
not needed for the workflow below.

## Run CrossMaps v2 and the dashboard

Run both commands from the repository root (`crossmaps-ws` in the existing
local workspace), in **two separate terminals**. Start the robot's camera and
localization/TF services first, and use the same ROS domain/network settings
on the robot and computer.

Activate your ROS 2/model Python environment in **each** terminal. On the
existing MacBook setup:

```sh
conda activate ros_humble_env
```

Both `python3` and `python` below must resolve to that environment, not a system
Python without ROS. The environment and model weights must already be installed;
see [dependencies and file inventory](crossmaps_dino_experimental/FILES.md).

### Terminal 1 — perception and world model

```sh
python3 crossmaps_dino_experimental/crossmaps_v2_node.py
```

### Terminal 2 — browser dashboard

```sh
python crossmaps_dino_experimental/crossmaps_dashboard.py
```

### Open the dashboard

Go to **http://127.0.0.1:8765** in Safari or another browser on this computer.
The dashboard targets ROS node **`/crossmaps_v2_node`** automatically. An explicit
equivalent dashboard command is:

```sh
python crossmaps_dino_experimental/crossmaps_dashboard.py --target-node /crossmaps_v2_node --port 8765
```

Use the dashboard to:

- Enter semantic queries and tune SAM, DINO, CLIP, mapping, and identity settings.
- Inspect RGB, SAM, DINO/CLIP features and similarity, Segments, Entities, and depth.
- Compare separate surface segments with their higher-level object entities.
- Inspect 3D layers, 2D maps, graph evidence, input freshness, and full-cycle timing.
- Enlarge the image section using **Large** or **Full Screen**.

Model overlays update when a processed observation completes, not at video frame
rate. Keep direct camera **Streams** disabled unless you need them; they are not
required to inspect the model's processed camera views.

## What changed from the previous launch commands?

Only the perception script was renamed:

- Previous: `crossmaps_dino_experimental/crossmaps_3_node_3d_revised-exp.py`
- Current: **`crossmaps_dino_experimental/crossmaps_v2_node.py`**
- Dashboard: **`python crossmaps_dino_experimental/crossmaps_dashboard.py`** — unchanged.

The ROS node is now `/crossmaps_v2_node` (ROS names use underscores, not hyphens).
Update custom ROS parameter commands and node-keyed YAML files accordingly.
The `/crossmaps/...` topics, local `.runtime/` directory, and model defaults
remain unchanged. For example:

```sh
ros2 param set /crossmaps_v2_node query_text sofa
```

Start only **one** perception node and **one** dashboard. Stop old processes
before restarting; duplicate publishers can mix different maps and IDs. Ctrl+C
in each terminal is the normal shutdown method. Do not launch the similarly
named root-level legacy scripts for v2.

## Defaults and local data

- EfficientSAM `vitt`, CLIP `ViT-L-14`/`openai`, and DINOv2 `vits14`.
- Compressed RGB/depth input from the robot; camera-info and valid TF are required.
- Stationary pose lock **off**; map autoload and autosave **off** by default.
- Runtime maps, screenshots, recordings, and logs remain local under
  `crossmaps_dino_experimental/.runtime/` and are not committed to Git.
- Saved dashboard presets default to `~/.crossmaps/crossmaps_dashboard_presets.json`.
- The default eSAM weights path is `weights/efficient_sam_vitt.pt` relative to
  the repository root. Weights and the installed Python/ROS environment are
  separate prerequisites, not included in the source checkout.

Yaw drift in upstream localization remains unresolved; the new naming and
dashboard do not fix pose errors. See the detailed limitations before mobile tests.

## Documentation

- [Architecture, parameters, validation, and limitations](crossmaps_dino_experimental/README.md)
- [Runtime files, dependencies, and test commands](crossmaps_dino_experimental/FILES.md)
- [Historical yaw-drift investigation](crossmaps_dino_experimental/YAW_DRIFT_DIAGNOSIS.md)
