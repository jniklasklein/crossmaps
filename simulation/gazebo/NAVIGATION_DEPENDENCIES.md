# Native macOS navigation dependencies

The simulator uses native Apple Silicon ROS 2 Humble packages from
[RoboStack](https://robostack.github.io/humble.html). Docker or an Ubuntu VM is
not needed for this installation. Gazebo remains the separate Homebrew
Harmonic installation; do not load its libraries into the conda ROS processes.

The existing `~/miniforge3/envs/ros_humble_env` uses RoboStack's Python 3.11
generation. Its compatible packages are:

| Package | Version | Build |
| --- | --- | --- |
| Python | 3.11.14 | `h18782d2_2_cpython` |
| NumPy | 1.26.4 | `py311h7125741_0` |
| ROS distro mutex | 0.7.0 | `humble_13` |
| Nav2 / nav2_bringup | 1.1.18 | `np126py311h2a51a2c_13` |
| SLAM Toolbox | 2.6.10 | `np126py311h2a51a2c_13` |
| RViz2 | 11.2.18 | `np126py311h2a51a2c_13` |

These exact historical Apple Silicon binaries remain available in the
official `robostack-humble` channel. Newer RoboStack generations use different
Python/NumPy/ABI constraints; do not install an unpinned latest Nav2 into this
environment. Official package records:
[Nav2 bringup](https://api.anaconda.org/package/robostack-humble/ros-humble-nav2-bringup),
[Navigation2](https://api.anaconda.org/package/robostack-humble/ros-humble-navigation2),
[SLAM Toolbox](https://api.anaconda.org/package/robostack-humble/ros-humble-slam-toolbox).

## Check or reproduce

From the repository root:

```sh
python3 simulation/gazebo/setup_nav2.py --check
python3 simulation/gazebo/setup_nav2.py --install
```

The default/check mode only reads installed resources and checks ROS/vision
imports. The explicit install mode saves conda revision and exact package
inventories under the ignored `.runtime/` directory, performs a dry-run, and
refuses replacement of existing packages other than the same-version
`libboost-python` 1.86.0 build adjustment required by SLAM Toolbox. It pins the
successfully solved additions before installing. A different Python/NumPy/ROS
generation is refused rather than migrated.

On the development Mac, the validated installation advanced conda revision
24 to 25. The only replacement was `libboost-python` 1.86.0 from
`py311h08cb3c8_5` to `py311h9fe4b7f_4`, matching
`libboost-python-devel` 1.86.0. Python, NumPy, ROS and RViz versions were
unchanged. Before/after audit inventories are in
`.runtime/nav2-install-audit-20261123/` on that machine. This revision number
is machine-specific, not a rollback instruction for other installations.

## Runtime architecture

Use separate Nav2, SLAM Toolbox and RViz processes in domains 80 (`rover_a`)
and 81 (`rover_b`), with `ROS_LOCALHOST_ONLY=1`, Cyclone DDS, and
`use_sim_time=true`. The Gazebo transport uses the simulator's private
`crossmaps_sim_*` partition and loopback address. These are intentionally not
the physical rover domains 0 and 1.

Each rover owns its own map/odometry frame tree. SLAM Toolbox supplies `map`
to `odom`; the bridge supplies wheel-integrated `odom` to `base_footprint`
and sensor extrinsics. Do not publish a second competing `map` transform.
Nav2 needs an obstacle/map input and complete transforms, not just RGB-D
previews or CrossMaps semantic point clouds. Configure its robot base frame
as `base_footprint` to match this simulator.

The installed RViz tool is `nav2_rviz_plugins/GoalTool`, with the
`nav2_rviz_plugins/Navigation 2` panel for action state/cancel controls.
Selecting a goal in rover A's domain must not send commands to rover B.
Independent Nav2 stacks provide local obstacle avoidance, not coordinated
multi-robot traffic scheduling or a shared fleet planner.
