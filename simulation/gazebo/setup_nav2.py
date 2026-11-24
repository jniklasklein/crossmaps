#!/usr/bin/env python3
"""Check or explicitly install the native Apple Silicon Nav2 dependencies.

Default is read-only. --install snapshots the environment, solves first, and
refuses any replacement of installed packages except the compatible Boost
Python 1.86 build needed by this RoboStack generation.
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
DEFAULT_PREFIX = Path.home() / "miniforge3/envs/ros_humble_env"
SPECS = [
    "python=3.11.14",
    "numpy=1.26.4",
    "ros2-distro-mutex=0.7.0=humble_13",
    "ros-humble-nav2-bringup=1.1.18=np126py311h2a51a2c_13",
    "ros-humble-slam-toolbox=2.6.10=np126py311h2a51a2c_13",
    "libboost-python=1.86.0=py311h9fe4b7f_4",
]
REQUIRED = [
    "bin/rviz2",
    "lib/slam_toolbox/async_slam_toolbox_node",
    "lib/nav2_planner/planner_server",
    "lib/nav2_controller/controller_server",
    "lib/nav2_bt_navigator/bt_navigator",
    "lib/nav2_behaviors/behavior_server",
    "lib/nav2_lifecycle_manager/lifecycle_manager",
    "share/nav2_bringup/launch/navigation_launch.py",
    "share/nav2_rviz_plugins/plugins_description.xml",
]


def inventory(prefix):
    return {d["name"]: d for path in (prefix / "conda-meta").glob("*.json")
            for d in [json.loads(path.read_text())]}


def validate_plan(plan, installed):
    """Protect the existing perception/ROS ABI from broad solver changes."""
    if plan.get("success") is not True:
        raise RuntimeError("Dependency solve failed: " + json.dumps(plan.get("solver_problems", plan)))
    actions = plan.get("actions", {})
    replacements = {p["name"]: p for p in actions.get("LINK", [])}
    for old in actions.get("UNLINK", []):
        new = replacements.get(old["name"])
        allowed = (old["name"] == "libboost-python" and old["version"] == "1.86.0"
                   and new and new["version"] == "1.86.0"
                   and new.get("build_string", new.get("build")) == "py311h9fe4b7f_4")
        if not allowed:
            raise RuntimeError(f"Refusing to replace installed {old['name']}; use a separate native environment.")
    for name, version in (("python", "3.11.14"), ("numpy", "1.26.4"), ("ros2-distro-mutex", "0.7.0")):
        if installed.get(name, {}).get("version") != version:
            raise RuntimeError(f"This pinned installer expects existing {name} {version}; refusing a migration.")
    return replacements


def check(prefix):
    missing = [str(prefix / path) for path in REQUIRED if not (prefix / path).is_file()]
    for path in missing:
        print("Missing: " + path, file=sys.stderr)
    if missing:
        print("Run this script with --install to install the pinned compatible native dependencies.", file=sys.stderr)
        return 1
    installed = inventory(prefix)
    for name in ("python", "numpy", "ros2-distro-mutex", "ros-humble-nav2-bringup",
                 "ros-humble-slam-toolbox", "ros-humble-rviz2"):
        package = installed.get(name, {})
        print(f"{name}: {package.get('version', 'unknown')} {package.get('build', '')}")
    subprocess.run([str(prefix / "bin/python3"), "-c", "import rclpy, cv2, numpy, tf2_ros; print('ROS/vision imports OK')"],
                   check=True, timeout=45)
    print("Native Nav2, SLAM Toolbox and RViz resources are available. This check does not start or drive any robot.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Read-only dependency check (default)")
    mode.add_argument("--install", action="store_true", help="Install only after a validated dry-run and inventory snapshot")
    args = parser.parse_args(argv)
    prefix = args.prefix.expanduser().resolve()
    if not args.install:
        return check(prefix)
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("These exact native package builds are for Apple Silicon macOS only.")
    if not (prefix / "conda-meta/history").is_file():
        parser.error("--prefix must point to an existing RoboStack environment, not a new environment.")
    mamba, conda = prefix.parents[1] / "bin/mamba", prefix.parents[1] / "bin/conda"
    if not mamba.is_file() or not conda.is_file():
        parser.error("Expected mamba and conda beside the environment's envs directory.")
    installed = inventory(prefix)
    # Check the generation before downloading metadata or invoking the solver.
    validate_plan({"success": True}, installed)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    audit = HERE / ".runtime" / ("nav2-install-" + stamp)
    audit.mkdir(parents=True, exist_ok=False)
    for name, flags in (("before-explicit.txt", ["--explicit"]), ("before-revisions.txt", ["--revisions"])):
        captured = subprocess.run([str(conda), "list", "--prefix", str(prefix), *flags],
                                  check=True, capture_output=True, text=True, timeout=45)
        (audit / name).write_text(captured.stdout)
    common = [str(mamba), "install", "--strict-channel-priority", "--override-channels",
              "-c", "robostack-humble", "-c", "conda-forge", "--prefix", str(prefix)]
    print("Validating pinned native dependency plan…", flush=True)
    solved = subprocess.run([*common, *SPECS, "--dry-run", "--json"], capture_output=True, text=True, timeout=600)
    (audit / "dry-run.json").write_text(solved.stdout)
    plan = json.loads(solved.stdout)
    replacements = validate_plan(plan, installed)
    # Lock every added/replaced package to the exact successfully checked plan.
    locked = [f"{p['name']}={p['version']}={p.get('build_string', p.get('build'))}" for p in replacements.values()]
    command = [*common, *SPECS, *locked, "--yes"]
    (audit / "install-command.txt").write_text(shlex.join(command) + "\n")
    print(f"Audit: {audit}\nInstalling validated plan ({len(replacements)} packages)…", flush=True)
    subprocess.run(command, check=True, timeout=1200)
    return check(prefix)


if __name__ == "__main__":
    raise SystemExit(main())
