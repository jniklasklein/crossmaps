#!/bin/zsh
# Both local navigation stacks and goal-click windows, plus Gazebo/CrossMaps.
set -eu
cd -- "${0:A:h}/../.."
task_python="${CROSSMAPS_ROS_PYTHON:-$HOME/miniforge3/envs/ros_humble_env/bin/python3}"
if [[ ! -x "$task_python" ]]; then
  print -u2 "ROS Python not found. See simulation/gazebo/README.md."
  exit 1
fi
exec "$task_python" simulation/gazebo/run.py --nav2 --rviz both "$@"
