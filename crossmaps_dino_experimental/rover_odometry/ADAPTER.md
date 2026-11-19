# Jetson odometry adapter

The adapter consumes absolute cumulative left/right travel in metres from the
headerless `/odom/odom_raw` (the first two floats) and body Z angular velocity
from `/imu/data_raw` in `base_imu_link`. It never consumes the RF2O-derived
`/imu/data` and never publishes to a motor or serial-control topic. Odometry
timestamps are ROS-clock receipt times of actual encoder messages, not fabricated
hardware timestamps. Output `/odom` and dynamic `odom → base_footprint` TF use
the exact same pose and stamp. Raw physical gyro time and frame are validated.

Modes inside the sourced ROS container:

| Mode | Topics | Initial reference |
|---|---|---|
| `--mode shadow` | `/crossmaps/odometry_shadow/odom` and private status only, no TF | Local origin; compare motion deltas to vendor |
| `--mode live` | `/odom`, TF unless launch turns it off, `/crossmaps/odometry/status` | Origin for normal *fresh* bringup. Refuses an existing odom publisher or odom-to-base TF |
| `--mode handoff` | Listens first; same outputs as live only after incumbent exits | Verified recent legacy pose, parked robot only |

All modes require `--owner-pid PID` and exit when that process (PID **and**
Linux start time) ends. `--config /path/file.json` accepts explicit estimator
config overrides. Heading integrates differential wheel travel; the physical
gyro supplies a consistency check, not a second yaw source. Verify sign and
wheel geometry with directed physical turns before tuning.

Run `python3 install.py --workspace /home/ws/ugv_ws` for a dry run. It checks
the deployed vendor launch, source launch, and actual Python overlay. Add
`--apply --backup-dir /home/ws/<new-dedicated-backup>` to install. For a
trial, `--modules-only` installs the adapter but leaves the default launch
unchanged. A normal install changes only the legacy base odometry action in
`ugv_bringup/launch/bringup_lidar.launch.py` in the source and active overlay.
The RF2O sensor, lidar, physical board reader, drive controller and RTAB-Map
are untouched. The Python source module lives inside the vendor Python package
so it remains available after a workspace rebuild.

The **experimental** seeded handoff is not the supported field rollout.
On A it refused the last seed after the incumbent exited. Use the fresh-map
procedure in [the runbook](README.md) for a deliberate reference change.
In laboratory handoff tests, `--incumbent-pid` must be the original vendor
`base_node`, a direct child of the nominated SLAM launch in the same domain.
`HANDOFF_READY` means exactly one vendor odometry publisher is present and the
adapter has a fresh planar odometry sample, recent physical gyro and stationary
encoders. It is not proof the seed will remain valid. The adapter does not signal the
incumbent. Once the specific incumbent exits, the adapter refuses stale seeds
(older than 4 s), intervening wheel movement and alien publishers. It tolerates
a stale DDS discovery entry only for that dead, previously confirmed publisher.
`HANDOFF_ACTIVE` means it has emitted its first live sample. No loss of seed
silently falls back to a fresh origin in an existing map. Check source, number
of emitted samples, timestamps, and exact TF agreement after any handoff.

The health JSON includes timestamps/ages, estimator source, gyro/encoder rejection
counters and identical-feedback duration. Three seconds of identical full
six-axis IMU payload plus unchanged encoders suppresses odometry and TF until
fresh evidence returns. This catches a known board-reader failure that can
republish cached values with new ROS stamps. It is a conservative heuristic,
not proof that a stationary rover with noisy-looking input is physically healthy.

Rollback of *future* starts: restore the original source and installed
`bringup_lidar.launch.py` from the uniquely named backup (paths are relative to
the workspace). Additional Python modules can remain inert. Restoring launch
files does not change a live process. Never start the legacy publisher alongside
live replacement odometry or start a second mapping launch just to roll back.
