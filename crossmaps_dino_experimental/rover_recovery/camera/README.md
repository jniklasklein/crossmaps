# OAK recovery for the Jetson/Humble SLAM bringup

Run `python3 install.py --workspace /home/ws/ugv_ws` in the SLAM container
for a read-only install plan. Add `--apply --backup-dir /home/ws/<unique-backup>/oak`
to install. Both rovers use a host-mounted, symlink-installed `ugv_vision`.
The installer refuses an unknown vendor launch file or a non-symlink-installed
overlay. It enables `camera.i_restart_on_diagnostics_error` while retaining the
other YAML values. The vendor launch is preserved as `crossmaps_oak_native.launch.py`.
No package manager or colcon build is needed for this camera change.

On the next normal SLAM launch the guard starts the unchanged vendor camera launch
in a new, private process group. The group on these rovers contains the DepthAI
component, camera robot-state publisher, and image rectifier. The rest of SLAM,
including RTAB-Map and the rover's own state publisher, is outside the group.
Do **not** start a standalone second camera when the SLAM launch owns one.

- Checks the acquisition timestamps of `/oak/rgb/image_rect/compressed` and
  `/oak/stereo/image_raw/compressedDepth` on the rover with best-effort, depth-one
  subscriptions. A health check sees at least two advancing, nonzero stamps per
  stream, then unsubscribes for 12 seconds. No raw images go to a laptop.
- Allows a 75-second initial window and 35 seconds per subsequent probe for the
  built-in diagnostics restart to work. If either stream keeps repeating the same
  timestamp, never starts, or stops, the guard recycles its own camera launch.
  It also retries if the private launch exits. Retry delay is 3, 6, 12, 24, 48,
  then at most 60 seconds. Physical disconnection may therefore require a wait
  after reattachment. The guard does not power-cycle USB or start another lidar.
- Uses a per-domain, per-camera lifetime lock and checks for an already-owned
  local camera container before spawning. Signals or the end of the owning SLAM
  launch terminate only the guard's private process group. On the stock single-OAK,
  no-namespace, Humble launch, use the defaults.

For an already-running launch on which the camera was not supervised, operators
can hand off **only the camera component**. Verify the active SLAM launch PID and
the exact old `component_container __node:=oak_container` process PID, in that
container's PID namespace, before doing anything. Never signal its inherited
process group. Once the old camera component is stopped and no twin is present,
use the original launch's ROS environment and domain to run
`python3 -u /home/ws/ugv_ws/src/ugv_main/ugv_vision/launch/crossmaps_oak_guard.py --owner-pid <slam-launch-pid>`.
The supervisor watches the original PID **and** its start time, so it exits when
that SLAM session ends. Verify fresh stamps, camera parameters and RTAB-Map input
on this rover; a process or ROS publisher existing does not establish health.

Rollback: after stopping just the camera supervisor, restore the backed-up
`launch/oak_d_lite.launch.py` and `config/oak_d_lite.yaml` under
`/home/ws/ugv_ws/src/ugv_main/ugv_vision`. The helper files can remain unused.
Restoring files doesn't replace a live camera process. No RTAB-Map DB is read or
written by this installer or supervisor.

To test the fallback without touching a physical camera, source ROS Humble and
run `CROSSMAPS_OAK_E2E=1 python3 -B test_camera_guard_e2e.py` from this folder
in a Linux container. The test hardcodes localhost-only domain 74 and unique
synthetic topic names. It continues publishing messages with frozen acquisition
timestamps, checks for a single camera restart and advancing timestamps, then
checks that ending the synthetic parent cleans up the private camera group
without stopping an unrelated process. It does not exercise the actual USB link.
