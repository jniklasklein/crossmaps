# CrossMaps
## Confidence-Aware Open-Vocabulary Semantic Mapping for Rover Navigation

This repository contains the CrossMaps ROS 2 node, a pre-saved semantic map state, and the RViz configuration used to inspect the map. It is intended both as a reproducibility artifact and as a practical starting point for running CrossMaps on a Jetson-based rover.

This repository supports two use cases:
- `Quick start`: load and inspect the provided saved semantic map.
- `Live system`: run CrossMaps on the rover setup with RGB, depth, TF, and occupancy-grid inputs.

## Demo Video

```md
https://youtu.be/vMQndtoBYTU
```


## Repository Contents

- `crossmaps_3_node.py` - main ROS 2 CrossMaps node.
- `data/crossmaps_3_node_state.pkl` - saved semantic map state for quick inspection.
- `crossmaps_config.rviz` - RViz configuration for visualization.

## System Requirements

### Minimum for inspecting the saved map

- A machine with ROS 2 installed.
- Python with the dependencies required by `crossmaps_3_node.py`.
- RViz2 for visualization.
- Enough compute to load the OpenCLIP model used by the node.

### Full rover replication

- A Jetson-based UGV rover setup.
- ROS 2 communication between the rover and the visualization machine.
- RGB image, depth image, camera info, TF, and occupancy grid topics available at runtime.
- The rover/network setup described below and in the external rover documentation.

## Dependencies

The main node imports and depends on at least the following packages:

### Python packages

- `numpy`
- `torch`
- `open_clip`
- `Pillow`

### ROS 2 Python packages

- `rclpy`
- `tf2_ros`
- `cv_bridge`
- `message_filters`

### ROS message packages

- `sensor_msgs`
- `nav_msgs`
- `std_msgs`

## Installation

### Quick install for reviewing the saved map

Install the software required to:
- run `crossmaps_3_node.py`
- open `rviz2`
- load the provided map state
- set semantic text queries through ROS 2 parameters

If you only want to inspect the provided saved map, you do not need the full rover setup. You only need a working ROS 2 environment, RViz2, and the Python dependencies listed above.

### Full installation for rover replication

For a full live deployment on the rover, use the steps below together with the external rover setup documentation. In particular, ensure the following are configured correctly before launching CrossMaps:

- rover/network configuration
- ROS 2 communication between rover and workstation
- camera topics
- occupancy grid publication
- TF between the camera, `odom`, and `map`

This setup was tested with the following external references:
- Waveshare UGV Rover Jetson Orin ROS 2 setup documentation
- a UGV rover manual used during the project

## Quick Start: Load the Included Saved Map

This is the fastest way to inspect CrossMaps without reproducing the full rover experiment.

### 1. Prepare the saved state

Create the CrossMaps state directory:

```bash
mkdir -p ~/.crossmaps
```

Copy the provided map state into place:

```bash
cp data/crossmaps_3_node_state.pkl ~/.crossmaps/
```

### 2. Run the node

Launch the node with the saved map enabled:

```bash
python crossmaps_3_node.py --ros-args \
  -p internal_frame:=odom \
  -p publish_frame:=map \
  -p use_ray_consistency:=true \
  -p occ_grid_topic:=/map \
  -p ray_check_stride:=6 \
  -p resolution:=0.10 \
  -p embed_every_n:=8 \
  -p stride:=8 \
  -p gating_cos_min:=0.20 \
  -p coherence_floor:=0.20 \
  -p coherence_power:=2.0 \
  -p use_visibility_boost:=true \
  -p view_bins:=8 \
  -p view_boost_strength:=0.7 \
  -p decay_half_life_s:=12000000 \
  -p cell_weight_max:=50.0 \
  -p ltm_weight_max:=200.0 \
  -p ltm_update_gain:=1.2 \
  -p ltm_replace_cos_thresh:=0.25 \
  -p ltm_replace_gain:=3.0 \
  -p promote_min_conf:=0.55 \
  -p promote_min_coh01:=0.40 \
  -p promote_min_views:=2 \
  -p autoload_map_state:=true
```

### 3. Open RViz2

In a second terminal:

```bash
rviz2
```

Load the configuration file:

`crossmaps_config.rviz`

With the preset, you may still need to choose the correct displayed map topics, for example the STM or LTM maps.

### 4. Set a semantic query

In a third terminal, update the current text query:

```bash
ros2 param set /crossmaps_3_node query_text plant
```

You can replace `plant` with any other query term of interest.

## Full Replication on the Rover

This setup assumes that CrossMaps runs locally on your computer while subscribing to ROS 2 topics that are published by the Jetson rover. This setup assumes a Waveshare UGV Rover Jetson Orin.

### Preparations on the rover

#### 1. Configure the rover network

Connect the rover to your own Wi-Fi network instead of the default hotspot setup. Follow the Waveshare network-configuration instructions here:

https://www.waveshare.com/wiki/UGV_Rover_Jetson_Orin_ROS2#Network_Configuration

For the initial setup, connect the rover to your computer via Ethernet or USB-C.

#### 2. Connect to the rover over SSH

Connect using:

```bash
ssh -Y jetson@<IP-ADDRESS>
```

The default username is `jetson` and the password is `jetson`. The rover IP address is shown on the rover display.

#### 3. Kill the default Python process on the rover

After each rover restart, the main Python process must be stopped before using ROS 2 manually.

The reference preparation instructions are here:

https://www.waveshare.com/wiki/UGV_Rover_Jetson_Orin_ROS2_1._Preparation

Run:

```bash
top
```

Find the main Jetson Python process and note its PID, then exit `top` with `Ctrl+C` and kill the process:

```bash
sudo kill -9 <PID>
```

#### 4. Start the ROS 2 Docker container on the rover

```bash
cd /home/ws/ugv_ws
sudo chmod +x ros2_humble.sh remotessh.sh
./ros2_humble.sh
```

#### 5. Enter the ROS 2 Docker container

```bash
docker exec -it ugv_jetson_ros_humble bash
```

Use `exit` when you want to leave the container.

### Preparations on your computer

#### 1. Install the local ROS 2 environment

Set up ROS 2 on your computer following the rover manual installation instructions. The original workflow used RoboStack and RViz2 on an Apple Silicon Mac.

You need:
- a working ROS 2 environment on your computer
- RViz2
- the dependencies required by `crossmaps_3_node.py`

#### 2. Match the ROS domain ID

CrossMaps runs on your computer and subscribes to ROS topics from the rover, so the `ROS_DOMAIN_ID` must match on both machines.

If you are using the same mamba environment as in the original setup:

```bash
mamba activate ros_humble_env
export ROS_DOMAIN_ID=0
```

#### 3. Optionally copy the rover workspace locally

For better compatibility, especially for RViz assets such as the rover model, it is recommended to copy the `ugv_ws` workspace from the rover to your computer and build it locally as well.

If you do this, source the workspace in each new terminal:

```bash
source ~/ugv_ws/install/setup.zsh
```

### On the rover inside the ROS 2 Docker container

#### 1. Source ROS 2 and set the ROS domain

```bash
source /opt/ros/humble/setup.bash
source /home/ws/ugv_ws/install/setup.bash
export ROS_DOMAIN_ID=0
```

#### 2. Start RTAB-Map SLAM first

Always start the SLAM system before the controller node and before CrossMaps. The original workflow notes that starting other nodes first can lead to unstable setups.

```bash
ros2 launch ugv_slam rtabmap_rgbd.launch.py use_rviz:=false
```

#### 3. Start joystick teleoperation if needed

If you want to drive the rover with the controller:

- plug in the USB dongle first
- leave the controller powered off initially
- start the teleop node
- then turn the controller on
- if it does not connect cleanly, turn the controller off and on again

Run in a second terminal connected to the rover and the ROS 2 container:

```bash
ros2 launch ugv_tools teleop_twist_joy.launch.py
```

### On your computer

#### 1. Check whether rover topics are visible

After starting the ROS 2 nodes on the rover, verify on your computer that you can see the published topics:

```bash
ros2 topic list
```

If you cannot see the rover topics on your computer, the DDS configuration between the Jetson container and your computer may not be aligned yet.

#### 2. Launch CrossMaps locally

For the live system, the node expects the following input topics by default:

- RGB image: `/oak/rgb/image_rect`
- depth image: `/oak/stereo/image_raw`
- camera info: `/oak/rgb/camera_info`
- occupancy grid: `/map`

The default frames used by the node are:

- internal frame: `odom`
- publish frame: `map`

Launch CrossMaps on your computer:

```bash
python crossmaps_3_node.py --ros-args \
  -p internal_frame:=odom \
  -p publish_frame:=map \
  -p use_ray_consistency:=true \
  -p occ_grid_topic:=/map \
  -p ray_check_stride:=6 \
  -p resolution:=0.10 \
  -p embed_every_n:=8 \
  -p stride:=8 \
  -p gating_cos_min:=0.20 \
  -p coherence_floor:=0.20 \
  -p coherence_power:=2.0 \
  -p use_visibility_boost:=true \
  -p view_bins:=8 \
  -p view_boost_strength:=0.7 \
  -p decay_half_life_s:=120.0 \
  -p cell_weight_max:=50.0 \
  -p ltm_weight_max:=200.0 \
  -p ltm_update_gain:=1.2 \
  -p ltm_replace_cos_thresh:=0.25 \
  -p ltm_replace_gain:=3.0 \
  -p promote_min_conf:=0.55 \
  -p promote_min_coh01:=0.40 \
  -p promote_min_views:=2
```

#### 3. Open RViz2

In a second terminal on your computer, inside the same ROS environment:

```bash
rviz2
```

Set the fixed frame to `map`.

Then either:
- add the CrossMaps map topics manually in RViz
- or load the preset RViz configuration `crossmaps_config.rviz`

With the preset, you may still need to choose the correct displayed map topics, for example the STM or LTM maps.

#### 4. Change the semantic query at runtime

In another terminal on your computer:

```bash
ros2 param set /crossmaps_3_node query_text door
```

The default query in the script is `ring`.

### Quick summary once everything is set up

#### On the Jetson

```bash
ssh -Y jetson@<IP-ADDRESS>
top
sudo kill -9 <PID>
cd ~/ugv_ws
./ros2_humble.sh
docker exec -it ugv_jetson_ros_humble bash
source /opt/ros/humble/setup.bash
source /home/ws/ugv_ws/install/setup.bash
export ROS_DOMAIN_ID=0
ros2 launch ugv_slam rtabmap_rgbd.launch.py use_rviz:=false
ros2 launch ugv_tools teleop_twist_joy.launch.py
```

#### On your computer

```bash
python crossmaps_3_node.py --ros-args \
  -p internal_frame:=odom \
  -p publish_frame:=map \
  -p use_ray_consistency:=true \
  -p occ_grid_topic:=/map \
  -p ray_check_stride:=6 \
  -p resolution:=0.10 \
  -p embed_every_n:=8 \
  -p stride:=8 \
  -p gating_cos_min:=0.20 \
  -p coherence_floor:=0.20 \
  -p coherence_power:=2.0 \
  -p use_visibility_boost:=true \
  -p view_bins:=8 \
  -p view_boost_strength:=0.7 \
  -p decay_half_life_s:=120.0 \
  -p cell_weight_max:=50.0 \
  -p ltm_weight_max:=200.0 \
  -p ltm_update_gain:=1.2 \
  -p ltm_replace_cos_thresh:=0.25 \
  -p ltm_replace_gain:=3.0 \
  -p promote_min_conf:=0.55 \
  -p promote_min_coh01:=0.40 \
  -p promote_min_views:=2
rviz2
ros2 param set /crossmaps_3_node query_text door
```

## Outputs

The node publishes the following outputs:

- `/vlmaps/stm/semantic_grid`
- `/vlmaps/stm/confidence_grid`
- `/vlmaps/stm/top1_grid`
- `/vlmaps/ltm/semantic_grid`
- `/vlmaps/ltm/confidence_grid`
- `/vlmaps/ltm/top1_grid`
- `/vlmaps/debug_cloud` default: disabled
- `/vlmaps/semantic_cloud` default: disabled
- `/vlmaps/history_cloud` default: disabled

## Expected Results

If the saved state is loaded successfully:

- the LTM grids should show persistent semantic structure in RViz
- changing `query_text` should update the semantic response
- the node should publish the STM and LTM grid topics listed above

## Troubleshooting

### STM appears empty after loading the saved map

This is expected with the paper configuration because STM uses time decay. If you load an old snapshot with a short half-life, STM may fade immediately.

Two common fixes are:
- set `decay_half_life_s` to a much larger value such as `1200000`
- comment out the STM decay update in [`crossmaps_3_node.py`](/Users/niklasklein/Documents/Projekte/HPI/crossmaps-ws/crossmaps_3_node.py#L770) if you are explicitly inspecting old STM state

The LTM should remain the more stable artifact for inspection.

### No data appears in RViz

Check:
- the node is running
- RViz is using the intended fixed frame
- the required input topics and TF are available
- the saved map file is located at `~/.crossmaps/crossmaps_3_node_state.pkl`

### Query changes do not update the visualization

Check that the node name is `/crossmaps_3_node` and that the parameter update succeeds:

```bash
ros2 param set /crossmaps_3_node query_text plant
```

### Rover startup is unstable

The original workflow recommends:
- always start the rover nodes first
- start RTAB-Map before the controller node and before CrossMaps
- restart the rover if the setup becomes inconsistent
- after each restart, kill the main Python process again and set the same `ROS_DOMAIN_ID` on rover and computer

## Limitations

- This repository is currently easiest to reproduce for saved-map inspection rather than full rover deployment.
- Full live replication depends on external rover hardware, ROS 2 setup, and sensor availability.
- Topic names and frames may need adaptation for other robots or sensor stacks.
- The environment is documented here but not yet frozen in a pinned environment file.