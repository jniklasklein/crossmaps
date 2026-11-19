# Rover-side sensor recovery

This is for the Jetson ROS Humble container `ugv_jetson_ros_humble` on rover A
(`.244`, domain 0) and B (`.249`, domain 1). It is independent of the laptop
coordinator. No new command is needed for normal operation: the existing
`ros2 launch ugv_slam rtabmap_rgbd.launch.py use_rviz:=false` includes the
repaired lidar and supervised camera. Do not also start separate sensor launches.
The vendor mapping launch passes **`-d`** to RTAB-Map: restarting the full
mapping launch may delete the previous database. It is not the sensor recovery
procedure.

- **LD19** retains its ROS publisher, closes a connection after 5 seconds without
  usable scans (8 seconds on startup), and opens a fresh driver on the rover's
  stable USB serial link, with 1–15 second retry delays. A real process exit is
  covered by ROS launch respawn. The scan frame, baud rate and angle crop are
  unchanged. The serial IDs differ; never copy one rover's launch to the other.
- **OAK** tries its built-in diagnostic restart first. A rover-local supervisor
  checks for advancing compressed RGB and depth acquisition stamps; a persistent
  stall or camera launch exit recycles only the camera launch, with backoff.
  See [camera recovery](camera/README.md) for the exact probe windows.
- **Scope:** no change to the base controller, command topics, odometry, TF,
  RTAB-Map settings, databases, or CrossMaps matching thresholds. Recovery
  depends on the Linux host seeing the reattached hardware. It cannot fix an
  underpowered hub, a damaged cable, lost observations, or a bad loop closure.

## Verify from the laptop without streaming raw images

From this repository root, set `ROVER_HOST` to rover A's hostname or IP and run
the following. For B change the host **and** the domain to `1`. Sampling runs on the rover; SSH
only returns a JSON report. Run it for 30–60 seconds if the system is very busy.

```sh
ssh "jetson@${ROVER_HOST:?Set ROVER_HOST to the rover hostname or IP}" \
  "docker exec -i -e ROS_DOMAIN_ID=0 ugv_jetson_ros_humble bash -c 'source /opt/ros/humble/setup.bash && source /home/ws/ugv_ws/install/setup.bash && python3 - --seconds 16 --require scan rgb depth odom rf2o info'" \
  < crossmaps_dino_experimental/rover_recovery/probe.py
```

Look for advancing timestamps and one publisher on each sensor and odometry
topic. Fresh `/odom` **alone** does not demonstrate that laser odometry or
RTAB-Map is receiving observations. RTAB-Map Info is an update signal, not a
fixed-rate heartbeat. For a physical acceptance test park one rover, note the
RTAB-Map PID, briefly unplug and reconnect its sensor hub, then repeat the
probe after the retry window. Verify all required streams recover, no duplicate
publishers appear and the RTAB-Map PID is unchanged. Do not do this while driving.

## Reproducing the install on the inspected vendor release

Copy this directory to a fresh persistent directory under `/home/ws` **inside
the container**, outside the ROS source tree. Work there in a sourced Humble +
UGV workspace with the correct domain. Use a separate backup directory for each
rollout. The installers validate the inspected vendor files and refuse unknown
revisions; inspect differences rather than overwriting modified sources.

```sh
# A only; on B the serial ends in 5AB0170376-if00.
python3 deploy_lidar.py --serial-id usb-1a86_USB_Single_Serial_5AB0170410-if00 \
  --stage "$PWD/ld19-build" --apply
python3 camera/install.py --apply --backup-dir "$PWD/oak-backup"
```

The lidar installer stages + compiles the whole driver, tests the executable with
two simulated USB ports (localhost-only ROS domain 73), and installs only the
driver sources, launch and executable. The camera installer validates the
installed symlinks and keeps the native camera launch. For an already-running
session, `live_handoff.py --help` describes the one-time sensor replacement
using verified container PIDs; do not reuse old PIDs. Its replacements exit
when that instance of the original SLAM launch stops. Subsequent starts do not
need the handoff tool.

**Rollback:** the lidar installer saves originals, destination paths and
before/after hashes in `ld19-build/backup/manifest.json`. With the sensor
stopped, confirm the live hashes still match the manifest's `after` hashes,
then restore the listed backup files to their corresponding destinations.
Restore the camera's original launch and YAML from `oak-backup/launch` and
`oak-backup/config` to their matching source-package locations. The added
camera helper files are inactive once the original launch is restored. Do not
overwrite later changes or restart the mapping launch to apply a rollback.
