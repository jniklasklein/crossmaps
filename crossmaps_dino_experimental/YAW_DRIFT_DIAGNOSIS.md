# Stationary yaw drift: live diagnosis, 2026-09-26

Historical investigation notes follow, not current launch instructions. The
stationary test lock was subsequently disabled and the duplicate test node
stopped. The current default is `stationary_pose_lock=false`; this snapshot
does not fix the upstream yaw drift. Refer to README.md for current behavior.

No localization parameters, robot processes, TF publishers or movement commands
were changed during this investigation. The experimental stationary pose lock
remains enabled. Keep the robot stationary until that test mode is deliberately
ended with a fresh/aligned map.

## Measured evidence

Successful recording: `.runtime/diagnostics/pose_drift_20260926_b.json`, about
45 seconds. This contains sensor/TF samples and publisher discovery, not camera
images. The earlier `_a.json` is an incomplete diagnostic artifact from a JSON
serialization error; do not use it as a replay or complete recording.

| Signal | Heading change | Other evidence |
| --- | ---: | --- |
| `/odom_rf2o` | -7.85245 degrees | RF2O laser odometry publisher |
| `/imu/data` | -7.85245 degrees | Same publisher, exactly matching heading and timestamp |
| `/odom` | -7.85245 degrees | `base_node`; x/y unchanged; all reported velocities zero |
| `odom -> base_footprint` | -7.85245 degrees | Matches `/odom` |
| `map -> odom` | 0 degrees | Constant translation and rotation during this capture |

ROS endpoint discovery identified `rf2o_laser_odometry` as the publisher of both
`/imu/data` and `/odom_rf2o`, and `base_node` as a subscriber to `/imu/data` and
publisher of `/odom`. Discovery is intermittent across this network, so some
inventory entries report unknown names; positive named discoveries are retained
in the recording. The installed Python ROS callback API did not expose per-message
publisher GIDs: this probe cannot rule out competing TF publishers by GID.

All 446 `/imu/data` samples matched RF2O yaw exactly at the corresponding source
timestamp. `/odom` and TF match nearby RF2O headings within approximately
0.0000023 degrees at the 95th percentile, but stamp that same heading roughly
84 ms later. Thus their heading evidence is not independent. Restamping can also
misalign observations during movement; it does not by itself explain the measured
stationary rotation already present in RF2O output.

`/odom` yaw and yaw-rate covariance diagonals are both `1e-9`, despite changing
orientation and zero reported angular velocity. Those covariance values should
not be treated as evidence that this pose is accurate.

`/imu/data_raw` had no subscribers at the inventory check. Its mean reported
z angular velocity was -0.00302 rad/s with standard deviation 0.01116 rad/s;
the OAK IMU reported +0.00368 rad/s, in a different sensor frame. These values
must not be compared/fused without mounting transforms and bias calibration.
Constant identity orientations with zero covariance in these raw streams are
not independently verified heading measurements. OAK messages arrived about
1.1 s after their stamped time, subject also to inter-device clock error.

## Interpretation and next change

The observed drift reaches CrossMaps through its external odometry TF, not a
changing `map -> odom` correction. The earliest identified drifting output is
RF2O. Its implementation/inputs still require inspection to establish *why* it
drifts (scan quality/geometry, driver timing, algorithm or local modifications).
Do not assume a raw-gyro bias fix alone will correct this chain.

Recommended robot-side investigation, before changing the running stack:

1. Inspect the installed RF2O implementation, base-node heading callback, launch
   remappings, and RTAB-Map odometry/IMU inputs. The live RF2O node produces an
   IMU message, unlike the inspected upstream ROS2 RF2O node source.
2. Check scan angular ordering, timestamps, invalid ranges, mounting transform,
   and stationary scan consistency. Establish why scan-derived yaw moves.
3. Stop representing scan-derived heading as independent physical IMU evidence.
   Use explicit odometry inputs; preserve original measurement timestamps and
   publish consistent pose/twist with measured, realistic uncertainty.
4. If adding wheel/physical-IMU fusion, calibrate gyro bias, verify axes and
   timing, and avoid re-fusing correlated scan yaw through a second input.
   Exactly one estimator should own `odom -> base_footprint`.
5. Validate stationary stability, commanded turns, and return-to-start before
   removing the CrossMaps test override. Investigate RTAB-Map correction behaviour
   separately; a constant correction in this short window does not prove it is
   broken. Longer-term world-model storage should permit pose-graph corrections.

No robot-side patch was deployed: remote parameter services timed out after
discovery; SSH to the observed robot address rejected noninteractive login;
desktop control could not attach to the existing SSH terminal. Read-only access
to the installed RF2O/base-node sources and launch configuration is needed next.

## Follow-up: robot source inspected via authorized SSH

Password-authenticated SSH subsequently succeeded. Read-only inspection of the
running `ugv_jetson_ros_humble` container established the following. No password
is stored in this repository. All following paths are on the robot.

- `/home/ws/ugv_ws/src/ugv_else/rf2o_laser_odometry/src/CLaserOdometry2DNode.cpp`
  explicitly publishes the same scan-derived quaternion in `/odom_rf2o` and
  `/imu/data`. This is not a physical IMU measurement. Running parameters set
  `publish_tf=false`, `base_frame_id=base_footprint`, and `freq=20`.
- `/home/ws/ugv_ws/src/ugv_main/ugv_base_node/src/base_node.cpp` subscribes to
  `imu/data`, copies its quaternion directly into `/odom` and TF, and derives
  velocity from cumulative encoder differences. It republishes the quaternion
  with the current clock instead of preserving its measurement time.
- That base node sets stationary yaw/yaw-rate covariance to `1e-9`, has
  uninitialized `pre_odl`, `pre_odr`, `vx` and `vw`, and no first-encoder-sample
  early return before computing deltas. A nonzero-yaw sentinel is also used
  instead of explicit orientation validity. The published quaternion bypasses
  its computed wheel-yaw fallback entirely. These are defects irrespective of
  the precise cause of RF2O's scan drift.
- The lidar driver at
  `/home/ws/ugv_ws/src/ugv_else/ldlidar/src/demo.cpp` sets each scan's array width
  to `src.size()` and angle increment to `2*pi/(width-1)`.
- RF2O's `CLaserOdometry2D.cpp` caches width/FOV from its first scan. Its node
  then indexes subsequent `ranges[i]` up to that original width and the solver
  maps `ranges.data()` with the cached width, without checking later sizes.

A subsequent 30-second live check received 289 lidar scans: widths 500 (5),
501 (83), 502 (172), and 503 (29). The angle increment changed accordingly.
Only about 55–62% of bins had valid returns; a configured 225–315 degree crop
explains part, but not necessarily all, of the missing data. All 574 raw wheel
messages had identical encoder values.

**Confirmed incompatibility:** varying scan geometry is being fed to a solver
that assumes fixed geometry. Longer arrays are interpreted at the wrong angles;
arrays shorter than the cached width can cause out-of-bounds reads. This is a
strong drift candidate, not yet an experimentally proven sole cause.

Prepared `fixed_scan_probe.py` for an isolated validation. It re-bins returns by
their actual angles onto a fixed grid, keeps missing returns missing, retains
the nearest obstacle on bin collisions, preserves acquisition stamps, and
rejects changed frames/FOV or nonadvancing stamps. It only publishes
`/crossmaps_diagnostics/scan_fixed`, never TF or movement commands, and runs for
a bounded duration. Four numerical regression tests pass. It is not connected
to the production RF2O input. A fresh isolated RF2O process with diagnostic-only
output and `publish_tf=false` is required to evaluate whether this addresses
the observed drift. Do not change the scan width beneath a running RF2O process.

The RTAB-Map launch also passes `-d` in mapping mode, which deletes its previous
database at startup. Do not restart that launch casually; preserve its database
before any approved deployment affecting SLAM.

## Isolated A/B result: fixed bins are not a yaw-drift solution

The user approved temporary diagnostic RF2O processes, never production TF or
odometry changes. The valid capture is
`.runtime/diagnostics/pose_drift_fixed_ab_20260926_d.json`. RF2O was started
before the relay, allowing static mounting transforms to arrive before its first
scan. This run had no transform-lookup initialization error.

On their overlapping source-time interval (~55 seconds):

| Measure | Production RF2O | Diagnostic fixed 502-bin RF2O |
| --- | ---: | ---: |
| Yaw endpoint change | -7.689 degrees | -6.954 degrees |
| Fitted yaw slope | -0.146 degrees/s | -0.155 degrees/s |
| Reported yaw-rate standard deviation | 0.0838 rad/s | 0.0354 rad/s |
| Received samples | 546 | 528 |

Interpretation: short-term angular jitter decreased, but systematic drift did
not meaningfully improve. This does **not** justify deploying the fixed-grid
relay as a drift fix. The cached-width incompatibility remains a real safety
defect to correct eventually. Scene-dependent scan-matching error, driver
sampling effects and RF2O behaviour need further work; no sole cause was proved.
The shadow instance was freshly initialized and received its data via the Mac
relay, so this is a bounded diagnostic comparison, not a controlled benchmark.

Earlier test artifacts are not the accepted validation: the first attempted
shadow command rejected an empty ROS string argument; the `_b.json` trace had
only seven fixed-grid samples after relay expiry; `_c.json` had a failed initial
lidar-to-base lookup. That failure exposed another RF2O defect: its first-scan
callback ignores the mounting-transform lookup result and proceeds to initialize.
Production hardening should require valid TF before initialization and validate
every incoming scan's dimensions and angular geometry before accessing arrays.

All temporary RF2O instances had `publish_tf=false`, diagnostic-only odometry
and synthetic-heading topics, and automatic timeouts. No movement commands were
sent. The relays were stopped. No robot source/configuration files or production
processes were changed, and the user's SLAM database was not reset.

At the user's request this investigation is now time-boxed. Keep the stationary
CrossMaps pose lock for stationary perception development. Do not use it while
moving. Before mobile testing, revisit the base-node pose/twist consistency,
initialized encoder integration, honest covariance/timestamps, and wheel/IMU
fusion with scan matching as explicitly correlated evidence—not a synthetic
independent IMU heading. A fixed yaw-rate subtraction or blindly swapping raw
gyro input is not a validated remedy.

## Repeatable probe

From the experimental directory, in the ROS Python environment:

```sh
ROS_LOG_DIR="$PWD/.runtime/diagnostics" python3 -B pose_drift_probe.py \
  --seconds 60 --output .runtime/diagnostics/pose_drift_NEW.json
```

The output path must be new and inside this directory. The probe only subscribes
and discovers publishers. Duration is capped at 120 seconds and each signal at
30,000 samples. Yaw is unwrapped in degrees; angular velocity remains rad/s.
Rates describe received estimates, not ground-truth motion. Header-to-receipt
delays include possible inter-device clock offsets.
