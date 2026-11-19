# Rover odometry: preventing false rotation while parked

This changes the odometry publisher used by **the lidar-based Humble SLAM
bringup on the two Jetsons**. The hardware bridge, drive controller, LD19,
OAK, RF2O, RTAB-Map, laptop coordinator, and existing map data are separate.
The normal `ros2 launch ugv_slam rtabmap_rgbd.launch.py use_rviz:=false`
includes the odometry publisher; **do not start a second publisher of
`/odom` or the `odom → base_footprint` transform**.

## What the heading represents

- The rover reports cumulative left/right encoder distances on `/odom/odom_raw`,
  in metres after the vendor bridge converts from centimetres. The first sample
  is a baseline, not a move. Each plausible wheel increment contributes to
  planar displacement and heading. The default track width is the vendor
  model's 0.175 m; effective track width on a specific floor needs measurement.
- The independent body gyro is `/imu/data_raw` (`base_imu_link`, radians/s).
  It provides evidence of rest or disagreement, including motion not supported
  by the wheels. **`/imu/data` is scan-matcher output on these robots, not a
  physical IMU.** RF2O remains available on `/odom_rf2o` for comparison; its
  stationary scan jitter is not fed back into the mapping heading.
- Both pose and twist carry uncertainty. RTAB-Map on these rovers normally
  uses the twist covariance; changing the pose field alone would leave the
  legacy overconfidence in the graph. Covariances here are engineering
  estimates, not measured sensor specifications.
- Publishes `/odom` and TF with the same stamp and pose when fresh telemetry
  is available. Wheel messages have no hardware stamp: the adapter uses the
  rover's ROS clock at receipt, with monotonic time for intervals. It does not
  restamp old odometry on a timer. Check `/crossmaps/odometry/status` for
  wheel resets, missing or frozen feedback, and disagreement.

This is **relative odometry, not absolute heading localization**. It does not
erase old drift or reweight existing map links. Wrong track width, track slip,
carrying a rover, and moving during a sensor outage can invalidate the pose.
If the system reports inconsistent inputs, inspect the rover before resuming
navigation. No software here sends motion commands.

## Tests and live checks

From this directory, the deterministic replay and unit tests run without ROS:

```sh
python3 validate.py replay
python3 -m unittest discover -p '*test*.py'
```

Inside the sourced container, select the correct domain (A: 0, B: 1).
With the patch live and the rover parked, compare the production output with
the independently running scan matcher:

```sh
python3 validate.py live --baseline /odom_rf2o --candidate /odom --seconds 60 --parked
ros2 topic echo --once /crossmaps/odometry/status
```

A zero change in yaw is only a pass when sensor data and pose timestamps
also advance. To validate while driving **in a separate, supervised trial**,
mark real start and end poses, run separate measurements using
`--turn-deg 90`, `--turn-deg -90`, and `--straight-m 0.5` (then reverse).
Use the signed physical measurement as the reference, not a dashboard or
another estimator. Try the actual operating floor; an unmeasured turn does not
calibrate sign, scale, wheelbase or slip. No forced stationary pose is needed.

### Field check, 2026-11-19

With the operator confirming both rovers parked, a 60-second shadow trial saw
peak heading excursions of 9.73° (A) and 5.99° (B) on stock odometry, versus
0° on wheel-led odometry. All 21 release tests (including isolated ROS input/TF
tests) passed inside each rover's container. We opened a new RTAB-Map database
without restarting either mapper. During the subsequent 60-second production check:

| Measurement | A (domain 0) | B (domain 1) |
|---|---:|---:|
| Fresh odometry messages | 1,183 | 1,179 |
| Maximum yaw excursion, odom → base | 0° | 0° |
| Maximum yaw excursion, map → odom | 0° | 0° |
| Diagnostic RF2O maximum yaw excursion | 7.57° | 4.93° |
| Fresh RTAB-Map status timestamps | 53 | 58 |

These are parked measurements, not estimates of moving or long-run heading
accuracy. No real turn was attempted. The synthetic motion and failure replay
exercises the code paths without establishing the rovers' effective wheelbases.

## Installation and rollback

Copy this directory to its own persistent staging directory under `/home/ws`
in the SLAM container (outside the ROS workspace). Source Humble and then
`/home/ws/ugv_ws/install/setup.bash`. Run `python3 install.py` for a read-only
plan. It checks the vendor launch and the active Python package. Before installing,
run a bounded `ros_node.py --mode shadow --owner-pid <actual-SLAM-launch-PID>`
and `validate.py live --parked` together. Shadow publishes on private diagnostic
topics, never production TF.

`python3 install.py --apply --backup-dir "$PWD/backup"` installs modules in
both source and active package locations and updates only the stock odometry
block in both copies of the lidar bringup launch. It does **not** restart a
process. Keep this backup. For a **parked, approved fresh-map transition** from
the vendor publisher, use `fresh_session.py --help`. Verify the SLAM owner,
mapper and incumbent PIDs in the **container PID namespace**. Give it a new
absolute DB path in an existing writable directory under `/home/ws` and an
unused log path. Preflight is read-only without `--apply`. The switch pauses
RTAB-Map, gracefully retires only the specified vendor odometry process, calls
`load_database` with `clear=false`, verifies the new SQLite file is actually
open in the original mapper, starts fresh-origin odometry and resumes mapping
only after verifying advancing odometry. The old database is saved and closed.
The new origin is a different coordinate reference: start a new laptop mapping
session before combining its observations with other rovers. The service
chooses the DB for this trial, not the default path of a later vendor relaunch.

Do not retire the only working odometry publisher to attempt a seamless reuse of
a live map. The experimental seeded handoff refused an unverified seed on A;
`HANDOFF_READY` alone does **not** guarantee continuity. Use a fresh session,
or plan a separate controlled migration if continuity is required.

To roll back **for a future SLAM session**, compare the launch files with the
two backup copies and restore the originals to their matching source/install
paths, retaining subsequent local edits. The added Python modules then aren't
invoked. Do not launch the old base node alongside the new one or casually
switch back mid-map: the stock base node has no pose-seed support. Plan a
controlled session boundary with the rover parked and the database backed up;
the vendor mapping launch passes **`-d`**, which deletes its previous database
on start.
