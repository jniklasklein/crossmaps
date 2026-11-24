#!/bin/zsh
# Double-clickable entry point; all configuration remains in run.py.
set -eu
cd -- "${0:A:h}/../.."
task_python="${CROSSMAPS_ROS_PYTHON:-$HOME/miniforge3/envs/ros_humble_env/bin/python3}"
if [[ ! -x "$task_python" ]]; then
  print -u2 "ROS Python not found. Set CROSSMAPS_ROS_PYTHON or see simulation/gazebo/README.md."
  exit 1
fi
exec "$task_python" simulation/gazebo/run.py "$@"
