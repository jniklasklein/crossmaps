# CrossMaps two-rover Gazebo laboratory

Native Gazebo Harmonic on Apple Silicon, feeding the real CrossMaps perception
and collaborative backend through localhost ROS 2 Humble. This is separate
from the physical-rover profile: domains **80 / 81**, central domain **82**, and
dashboard **http://127.0.0.1:8767**. No rover SSH, hardware commands or Keychain
credentials are used. Start a new simulation run instead of resuming a physical
recording.

## Start and stop

From the repository root:

```sh
conda activate ros_humble_env
python simulation/gazebo/run.py
```

Alternatively, double-click `simulation/gazebo/start.command` in Finder. It
opens a terminal and uses the same launcher and existing ROS environment.

This generates a furnished room using bundled offline assets, builds the small native Gazebo bridge,
starts the simulator server, waits for advancing calibrated RGB-D and wheel
odometry from **both** rovers, opens Gazebo's separate GUI and starts CrossMaps
with automatic/provisional alignment enabled. Initial inference model loading
can take longer than sensor startup. Open the dashboard URL above yourself.

**Ctrl-C in this terminal stops this run's processes**, including the bridges
and coordinator, and saves the coordinator's usual evidence checkpoint. It
does not stop other dashboards or either physical robot. Do not close the
Gazebo GUI as a way to leave a hidden run active; use Ctrl-C in the launcher.

The launcher automatically locates `~/miniforge3/envs/ros_humble_env/bin/python3`.
For another installation, set `CROSSMAPS_ROS_PYTHON` to the ROS-enabled Python
executable. It does not install anything into that environment or edit shell
startup files.

```sh
# Dependencies/configuration only, without starting a simulator
python simulation/gazebo/run.py --check

# Sensor/physics trial without neural inference
python simulation/gazebo/run.py --sensors-only

# Camera rendering without the Gazebo editor window
python simulation/gazebo/run.py --no-gui

# Bounded trial; all owned processes are stopped afterwards
python simulation/gazebo/run.py --no-gui --duration 120

# Repeatable uncalibrated sensor-noise stress test (not an OAK-D emulator)
python simulation/gazebo/run.py --sensor-profile noisy --noise-seed 2026
```

The server uses Ogre2/Metal by default on macOS. `--render-engine ogre` selects
the alternative renderer for diagnosis; it is not an instruction to disable
macOS security or install Linux graphics shims.

## Drive the simulated robots

### Nav2 and click-to-drive

Double-click **`simulation/gazebo/start-navigation.command`**, or run:

```sh
conda activate ros_humble_env
python simulation/gazebo/run.py --nav2 --rviz both
```

This starts the simulation, CrossMaps, and a separate SLAM Toolbox + Nav2 stack
and RViz window for each rover. To use less compute while practising navigation,
add `--sensors-only` (no learned CrossMaps perception). `--rviz rover_a` opens
only A's window; both navigation stacks still run.

Wait for the terminal to report **Both Nav2 stacks ready** and for the laser map
to appear. In the window for the intended rover, select **Nav2 Goal**, click a
free destination on the map, drag to set the final heading, and release. Nav2
plans a path and drives the simulated wheels; it does not teleport the model.
Use **Cancel** in the **Navigation 2** panel to cancel a goal. Ctrl-C in the
launcher stops the entire simulation, including navigation and all owned GUIs.
The panel's **Pause / Resume** suspends/reactivates navigation without stopping
Gazebo or SLAM. **Reset / Startup** cleans and reconfigures the Nav2 servers;
it does not reset the simulated world or the laser map.
With navigation enabled, a clock/scan/odometry stall lasting five wall-clock
seconds stops the navigation run; restart the launcher after a long Gazebo pause.
Costmaps also stop treating laser observations as current after 0.5 simulation
seconds without a scan.

RViz's **Localization: inactive** indicator refers to an AMCL lifecycle manager,
which this online-SLAM setup deliberately does not start. SLAM Toolbox supplies
localization; the map, current rover transform and **Navigation: active** are the
relevant checks here. The per-rover Mac app wrappers are generated locally for
clear Dock/window names using unchanged installed RViz binaries. They are not
signed redistributable applications; start them through the launcher, not by
copying the generated `.app` folders to another Mac.

The occupancy map starts from live laser scans; no initial-pose click is needed
for this fresh online SLAM session. Begin with nearby goals in observed free
space. Unknown/unreachable areas, narrow gaps and blocked goals may be refused.
Each window has its **own map coordinates**, not the CrossMaps shared-world
coordinates. CrossMaps alignment is not a prerequisite for local navigation.

To open a window later for an already-running stack:

```sh
python simulation/gazebo/navigation.py --robot rover_b --rviz-only
```

Nav2 uses wheel odometry, a simulated 360° laser, an approximately rover-sized
footprint, inflation clearance and collision-aware local control. SLAM Toolbox
owns `map → odom`; the bridge owns `odom → base_footprint`. No ground-truth world
pose is fed to SLAM or CrossMaps. The two stacks see obstacles independently;
they are **not a coordinated multi-robot traffic planner**. Avoid sending both
robots into a narrow passage at once.

Native dependency versions and the guarded installer are documented in
[`NAVIGATION_DEPENDENCIES.md`](NAVIGATION_DEPENDENCIES.md).

### Keyboard driving

Start `run.py` **without `--nav2`** for manual control. Do not run keyboard
control against an active Nav2 stack; the keyboard tool refuses competing
controllers.

In another interactive terminal at the repository root:

```sh
python simulation/gazebo/teleop.py --robot rover_a
# or --robot rover_b
```

W/S drive, A/D rotate, Space stops, Q exits. Commands expire when key repeats
stop; the native bridge has a separate 500 ms wall-clock deadman. These commands
are restricted to domains 80/81 with localhost-only discovery. The real-rover
domains 0/1 are never used.

Keep overlapping views initially. A stationary pair may establish a
**provisional** alignment; move one rover slowly while preserving overlap to
collect distinct-view corroboration. Ground-truth spawn poses are not supplied
to CrossMaps' matcher, and no verification thresholds are weakened for the sim.

## What is being simulated

The robot description is based on the user's copied Waveshare
`~/ugv_ws/src/urdf/ugv_rover.urdf` and CAD assets, rather than requiring
Waveshare's entire Ubuntu virtual-machine image. The native
Harmonic systems replace the older Gazebo plugin integration; the original
workspace is not modified. See the vendor provenance and generated
`world/scene.json` for the model/calibration assumptions and source hashes.

The supplied `ws_0924` image is a VMware **x86-64 Ubuntu 22.04.5** guest;
Apple Silicon cannot hardware-virtualize it directly. CPU emulation or a
separate ARM installation would be needed, so this setup instead runs Gazebo
natively. The image was inspected read-only, not booted or modified. The
[official source](https://github.com/waveshareteam/ugv_ws/tree/f0b3ad9c16e7acb6d6145f261116d71bf6338f4a/src/ugv_main)
was checked separately: its rover description agrees with the local CAD, but
its Gazebo Classic drive plugin specifies a 50 mm wheel diameter while the
CAD and firmware describe 80 mm. This port uses the CAD/firmware value, four
driven wheels and two passive middle wheels, plus the recorded per-rover camera
calibration. It does not claim unchanged VM/plugin or full-navigation-stack
parity. Exact comparisons, assumptions and pinned source hashes are in
[`upstream_comparison.json`](upstream_comparison.json).

The indoor scene is reproducible and furnished: photographic wood, plaster
and fabric PBR materials; bevelled furniture, desk/monitor/keyboard, caster
chair, books, bins, unlabelled cartons, window/blinds, radiator, door hardware
and a scanned potted plant. There are **no artificial posters or printed feature
targets**. The photographic materials and plant are bundled CC0 Poly Haven
assets with source URLs and SHA-256 manifests under `material_assets/` and
`object_assets/`. Runtime does not download assets. Collision geometry remains
simple to keep physics fast; materials improve appearance but do not establish
sensor or contact accuracy. This is **not a validated digital twin**.
The CAD, wheel geometry, saved camera calibration and recorded static TF are
useful evidence; they do not establish measured motor dynamics, friction,
chassis payload, rolling shutter or OAK-D stereo failure characteristics.

Gazebo computes rendered RGB/depth and physical wheel motion. The bridge
publishes wheel odometry with independent zero origins, not world ground truth.
Consequently wheel slip can create odometry error. With `--nav2`, each rover runs
laser-based SLAM Toolbox and Nav2 alongside the actual CrossMaps RGB-D mapping,
pose checks and collaborative matching. This does **not** reproduce the physical
robot's full RTAB-Map/IMU stack. The laser uses 720 simultaneous GPU rays at 10 Hz,
0.12–12 m range and an uncalibrated 1 cm Gaussian noise model. Its mount follows
the CAD with an assumed sensing-plane offset; it does not model spinning-scan
distortion, reflectivity or real sensor dropout. A clean-camera baseline should pass before
introducing artificial noise. Synthetic noise must not be mistaken for a
calibrated model of the real camera.

## Time, transport and isolation

- Simulated acquisition stamps are preserved. `/clock` drives ROS sensor/TF
  comparisons; wall-clock timers still detect transport loss and expire motion.
- Pausing Gazebo is labelled as a stopped simulation clock, not bad robot time.
- **After resetting world time, restart the whole run.** Backward time jumps
  latch a mapping block instead of mixing old and new odometry/map epochs.
- Gazebo's raw rendered pixels cross only a local process pipe. ROS publishes
  JPEG RGB and lossless PNG `compressedDepth`; no raw image ROS topics or SSH
  fallback are used.
- Camera calibration and optical TF come from the same generated scene as the
  renderer. The two ROS domains use the same familiar `/oak/...` names without
  colliding. Gazebo Transport has a private partition and loopback IP.
- Homebrew's Gazebo C++ libraries run in a separate process from the existing
  Conda ROS/PyTorch environment, avoiding Python/Protobuf ABI conflicts.

## Checks and logs

With the simulator running, this **read-only** probe checks both rovers for
advancing simulated time, compressed image decoding, matching calibration,
usable depth, image variation and capture-time TF:

```sh
python simulation/gazebo/smoke_check.py --duration 20 --output simulation/gazebo/.runtime/probe
```

For an explicitly opted-in **simulated** forward/turn test (stop keyboard
teleop first):

```sh
python simulation/gazebo/motion_check.py --execute --robot rover_a --output simulation/gazebo/.runtime/motion-check
```

This moves the simulated rover a few centimetres and roughly 13 degrees,
checks rendered image changes, wheel odometry and capture-time TF, and sends
stop commands on exit. It does not assert that CrossMaps accepted every pose;
inspect the dashboard's pose checks separately. Without `--execute` it only
prints the plan.

For an opted-in Nav2 test, start with `--nav2`, leave keyboard control off, and
ensure no other navigation goal is active:

```sh
python simulation/gazebo/navigation_check.py --execute --robot rover_a --forward .5 --lateral 0 --output simulation/gazebo/.runtime/nav-a.json
python simulation/gazebo/navigation_check.py --execute --cancel --robot rover_b --forward .5 --lateral 0 --output simulation/gazebo/.runtime/nav-b-cancel.json
```

Without `--execute`, this only prints the bounded test plan. The checker observes
the planned path, actual transform, fresh scans and controller stop; it cancels
only its own goal on timeout/interruption. It never publishes a fake stop to
make a test pass.

The launcher prints its unique `.runtime/<run>/` folder. Inspect `gazebo.log`,
`rover_a.log`, `rover_b.log`, `crossmaps.log` and the ready JSON files. With Nav2,
also inspect `navigation_rover_a.log` / `navigation_rover_b.log`; these print
their individual navigation log folders.
Coordinator recordings remain in the ordinary
`crossmaps_dino_experimental/.runtime/collaboration/` directory.

### Longer runs and map storage

CrossMaps keeps six time-spaced bootstrap captures of a parked view, then skips
near-identical evidence rather than filling the store simply because time passes.
Motion, meaningful RGB-D changes and better same-view depth/image detail can
still add evidence. Local perception continues on skipped captures, so the
**Shared evidence saved** age can grow normally while both cameras remain live.

The graph remains bounded at 256 keyframes across both rovers, 64 submaps and
2 GiB of compressed evidence. Aggregate recording admission also checks 4 GiB,
a 1 GiB free-disk reserve and ten managed recordings; nothing is silently evicted.
Use the dashboard's **Storage & maps** dialog to preview and explicitly confirm
rebuilding the current CrossMaps map or deleting selected inactive recordings.
A rebuild keeps loaded models and does **not** restart/reset Gazebo, Nav2 or SLAM
Toolbox. Ordinary launches start a fresh run without importing previous maps.

New launches also compact old diagnostic log lines above 8 MiB/file or
32 MiB/owned run directory, retaining recent tails. Checks are periodic; this is
not a hard whole-disk quota or lossless logging. World assets, map evidence and
historical unmanaged simulator folders are not removed by this log policy.
See the [complete retention and cleanup policy](../../crossmaps_dino_experimental/COLLABORATIVE.md#evidence-retention-and-storage-management).

```sh
python -m unittest discover -s simulation/gazebo -p 'test_*.py' -v
```

## Validation on this Mac (2026-11-23)

Tested with native Harmonic 8.15 on the M4 Pro, Ogre2/Metal, and the existing
ROS/PyTorch environment. The new default is 960×540 at 3 Hz per camera, with
4× antialiasing. `--camera-width 640` is lighter; `--camera-width 1280` renders
more detail. Intrinsics scale with resolution, preserving the recorded camera
field of view. This is not the physical camera's full frame-rate profile.

The furnished-room sensors-only render check delivered 36 advancing, decodable
RGB-D pairs per rover in 12 seconds, all with capture-time TF, at approximately
real time. Sample images/report: `.runtime/realistic-probe/` on this Mac.

Native navigation validation in the furnished room:

- Both seven-node Nav2 lifecycle stacks became active with scan-built maps.
- Both rovers completed a 0.5 m requested goal and stopped within the configured
  0.10 m arrival tolerance (measured final errors about 0.092 m).
- Rover B accepted cancellation of a moving goal, returned CANCELED, and the
  actual command stream returned to zero.
- Both compressed cameras continued at about 3 Hz with exact-time TF and
  approximately real-time simulation while both navigation stacks ran.
- With CrossMaps also running, A completed a turning route (0.6 m forward,
  0.25 m sideways target), stopped with about 0.078 m position error, and neither
  rover recorded rejected poses in the following check. The furnished room
  produced geometrically verified cross-rover candidates; this is not a claim
  that automatic global alignment was confirmed in that stationary/short trial.
- RViz's actual map, live scans, footprint, heading and Nav2 Goal tool were
  inspected. Goal execution and cancellation were verified through Nav2's
  action interface; the UI automation backend could not perform the canvas
  drag gesture itself.
- The navigation lifecycle completed Pause → Resume and Reset → Startup in
  a live trial, without restarting Gazebo or SLAM Toolbox. The final bounded
  run shut down all of its simulation/navigation/viewer processes; the
  separate physical-rover dashboard was left running.

Reports are in `.runtime/20261123-131843-87458/nav-a-check.json`,
`nav-b-check.json`, `nav-b-cancel-fixed.json`, and `.runtime/nav2-sensor-probe/`.
These short tests do not establish robust navigation through every layout,
dynamic obstacle, narrow passage or sensor failure.

The earlier controlled-scene validation (before the furnished-room revision):

- Both compressed RGB-D streams sustained approximately 3 Hz, with matched
  capture-time TF for every pair in the final 12-second probe; simulation ran
  at approximately real time while the actual perception models were active.
- A stationary pair formed an automatic provisional alignment. After driving
  simulated A forward about 6.7 cm and turning about 12°, CrossMaps promoted
  the alignment automatically. Driving B through the same bounded test also
  preserved mapping. Both reported zero rejected poses in this final trial.
- Pause/resume reports stopped simulation time and then resumes fresh input,
  without substituting wall-clock timestamps or restarting a camera relay.
- Simulator startup, bounded shutdown, immediate restart, and native helper
  EOF/deadman handling were exercised. Physical rover domains remain untouched.

The first, mostly planar-texture scene exposed an **existing temporal pose
guard weakness**: unrefined planar EPnP produced a false visual/odometry
contradiction. Independent simulation truth showed only about 7 mm wheel-odom
error, whereas the visual estimate disagreed by 164 mm. The failure capture
is preserved in `.runtime/motion-validation/after_motion_rgbd.npz` on this Mac
(runtime artifacts are intentionally untracked). Offline refinement reduced
that disagreement to about 30 mm, but **the physical pose guard and its
thresholds have not been changed**. That controlled room included ordinary
printed objects at several depths, now replaced by natural furniture/material
detail. Its passing trial does not establish
robustness for all planar, low-texture, dynamic or real-world scenes.

Final probe and movement reports/images are in `.runtime/final-probe/`,
`.runtime/final-motion-a/` and `.runtime/final-motion-b/` on this Mac.

## Installing on another Mac

```sh
brew tap osrf/simulation
brew trust osrf/simulation
brew install gz-harmonic
```

This is a substantial Homebrew dependency installation, separate from Conda.
If Homebrew reports unsupported dependency install steps, update Homebrew
itself (`brew update`) before retrying. Do not use a blanket `brew upgrade` or
remove unrelated packages to troubleshoot this project.

References: [Waveshare simulation tutorial](https://www.waveshare.com/wiki/UGV_Rover_PI_ROS2_11._Gazebo_Simulation_debugging),
[Harmonic macOS installation](https://gazebosim.org/docs/harmonic/install_osx/),
[Gazebo macOS troubleshooting](https://gazebosim.org/docs/harmonic/troubleshooting/).
