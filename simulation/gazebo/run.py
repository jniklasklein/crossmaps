#!/usr/bin/env python3
"""Supervise the isolated Gazebo + two ROS bridges + CrossMaps experiment."""
from __future__ import annotations
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
APP = ROOT / "crossmaps_dino_experimental"
CONFIG = APP / "config/collaborative_gazebo.json"
if str(APP) not in sys.path:
    sys.path.append(str(APP))
from collab_log_budget import ScopedLogBudget


def ros_python():
    override = os.environ.get("CROSSMAPS_ROS_PYTHON")
    candidates = [override] if override else [str(Path.home() / "miniforge3/envs/ros_humble_env/bin/python3"), sys.executable]
    for executable in candidates:
        if executable and Path(executable).is_file():
            try:
                result = subprocess.run([executable, "-c", "import rclpy, cv2, numpy, tf2_ros"], capture_output=True, timeout=20)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if result.returncode == 0:
                return executable
    raise RuntimeError("ROS Python not found. Activate ros_humble_env or set CROSSMAPS_ROS_PYTHON.")


def environments(python, logs):
    common = dict(os.environ)
    for name in ("ROS_DOMAIN_ID", "CYCLONEDDS_URI", "ROS_DISCOVERY_SERVER", "FASTRTPS_DEFAULT_PROFILES_FILE",
                 "FASTDDS_DEFAULT_PROFILES_FILE", "GZ_DISCOVERY_SERVER", "IGN_PARTITION", "GZ_PARTITION",
                 "PYTHONPATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "QT_PLUGIN_PATH",
                 "ROS_STATIC_PEERS", "ROS_SUPER_CLIENT"):
        common.pop(name, None)
    common.update(ROS_LOCALHOST_ONLY="1", ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST",
                  RMW_IMPLEMENTATION="rmw_cyclonedds_cpp", GZ_IP="127.0.0.1",
                  GZ_PARTITION=f"crossmaps_sim_{os.getuid()}", ROS_LOG_DIR=str(logs / "ros"))
    native = dict(common)
    native["PATH"] = "/opt/homebrew/bin:/opt/homebrew/opt/ruby/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    ros = dict(common)
    ros["PATH"] = str(Path(python).parent) + ":" + native["PATH"]
    return native, ros


def apply_environment(environment):
    """Replace, not merge: removed DDS overrides must not survive in CLI tools."""
    os.environ.clear()
    os.environ.update(environment)


def validate_simulation_config(path=CONFIG):
    """Fail closed if this fixed-purpose launcher is pointed at physical domains."""
    config = json.loads(Path(path).read_text())
    robots = config.get('robots', [])
    if (len(robots) != 2 or {r.get('id'): r.get('domain') for r in robots} != {'rover_a': 80, 'rover_b': 81}
            or config.get('central_domain') != 82
            or config.get('dashboard') != {'host': '127.0.0.1', 'port': 8767}
            or config.get('input', {}).get('use_sim_time') is not True
            or any(r.get('camera_relay', {}).get('mode', 'off') != 'off' or
                   r.get('rgb_compressed', True) is not True or r.get('depth_compressed', True) is not True for r in robots)):
        raise RuntimeError('Simulation config must use rover_a/80, rover_b/81, central/82, localhost:8767, '
                           'simulation time, compressed cameras and no SSH relay; refusing to launch.')
    return config


def stop_owned(children, grace=12.):
    """Terminate owned process groups, including helpers of an already-dead leader."""
    inaccessible = set()

    def signal_group(process, sig):
        if process.pid in inaccessible:
            return False
        try:
            os.killpg(process.pid, sig)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # macOS may retain protected helpers in a process group. Never let
            # one inaccessible group prevent cleanup of all other owned groups,
            # nor broaden the target to matching names or unrelated processes.
            inaccessible.add(process.pid)
            print(f'Warning: cannot signal process group {process.pid} (permission denied); '
                  'its remaining members could not be verified. Continuing other cleanup.', file=sys.stderr)
            return False

    def alive():
        live = []
        for name, process in children:
            process.poll()  # reap exited group leaders before checking descendants
            if signal_group(process, 0):
                live.append((name, process))
        return live

    for sig, timeout in ((signal.SIGINT, grace), (signal.SIGTERM, 3.), (signal.SIGKILL, 3.)):
        remaining = alive()
        if not remaining:
            break
        for _, process in reversed(remaining):
            signal_group(process, sig)  # ProcessLookupError is a normal exit race.
        deadline = time.monotonic() + timeout
        while alive() and time.monotonic() < deadline:
            time.sleep(.05)
    for name, process in children:
        try:
            process.wait(timeout=.1)
        except subprocess.TimeoutExpired:
            print(f'Warning: {name} is still running after cleanup; inspect PID {process.pid}.', file=sys.stderr)


def run_bounded(command, environment, timeout):
    """Bound setup commands and clean up compiler descendants on timeout/Ctrl-C."""
    process = subprocess.Popen(command, env=environment, start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        stop_owned([(str(command[0]), process)], grace=1.)


def source_ready(path, robot, domain, pid):
    """Read current bridge health, not merely a file left by an earlier frame."""
    try:
        health = json.loads(Path(path).read_text())
        if not isinstance(health, dict) or not isinstance(health.get('advancing_age_s'), dict):
            return False
        age = time.time() - float(health['updated_unix'])
        advances = health['advancing_age_s']
        return (health.get('ready') is True and health.get('robot') == robot
                and health.get('domain') == domain and health.get('pid') == pid
                and math.isfinite(age) and -.5 <= age <= 10.
                and all(isinstance(advances.get(key), (int, float)) and 0 <= advances[key] < 5.
                        for key in ('clock', 'odom', 'rgb', 'depth', 'scan')))
    except (OSError, ValueError, TypeError, KeyError):
        return False


def navigation_ready(path, robot, domain, pid):
    try:
        health = json.loads(Path(path).read_text())
        age = time.time() - float(health['updated_unix'])
        return (health.get('ready') is True and health.get('robot') == robot
                and health.get('domain') == domain and health.get('pid') == pid
                and math.isfinite(age) and -.5 <= age <= 10.)
    except (OSError, ValueError, TypeError, KeyError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gui", action="store_true", help="Camera rendering stays enabled; omit Gazebo's editor window")
    parser.add_argument("--sensors-only", action="store_true", help="Run cameras/physics/ROS without learned perception")
    parser.add_argument("--render-engine", choices=("ogre", "ogre2"), default="ogre2", help="Native macOS uses Ogre 2 with Metal")
    parser.add_argument("--duration", type=float, help="Stop all owned processes after N seconds of running")
    parser.add_argument("--check", action="store_true", help="Check dependencies without launching")
    parser.add_argument("--skip-build", action="store_true", help="Use the existing compiled Gazebo helper")
    parser.add_argument("--sensor-profile", choices=("clean", "noisy"), default="clean",
                        help="Clean baseline or deterministic uncalibrated sensor-noise stress test")
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--camera-width", type=int, choices=(640, 960, 1280), default=960,
                        help="16:9 camera render width; 640 is lighter, 1280 shows more detail")
    parser.add_argument("--nav2", action="store_true", help="Start laser SLAM and Nav2 for both simulated rovers")
    parser.add_argument("--rviz", choices=("none", "rover_a", "rover_b", "both"), default="none",
                        help="Open Nav2 click-to-drive windows (requires --nav2)")
    args = parser.parse_args()
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be finite and positive")
    if args.rviz != 'none' and not args.nav2:
        parser.error('--rviz requires --nav2')
    validate_simulation_config()
    python = ros_python()
    tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    logs = HERE / ".runtime" / tag
    native, ros = environments(python, logs)
    gz = shutil.which("gz", path=native["PATH"])
    cmake = shutil.which("cmake", path=native["PATH"])
    if not gz or not cmake:
        raise RuntimeError("Install Gazebo first: brew tap osrf/simulation && brew trust osrf/simulation && brew install gz-harmonic")
    print(f"ROS Python: {python}\nGazebo: {gz}\nDomains: 80 / 81 (loopback only); central: 82", flush=True)
    run_bounded([python, str(APP / "crossmaps_collaborative.py"), "--config", str(CONFIG), "--check"], ros, 120.)
    if args.nav2:
        run_bounded([python, str(HERE / 'navigation.py'), '--check'], ros, 60.)
    if args.check:
        return 0
    import fcntl
    HERE.joinpath(".runtime").mkdir(exist_ok=True)
    lock = (HERE / ".runtime/simulator.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("A simulator launched by run.py is already running; stop its terminal with Ctrl-C.")
    try:
        return launch(args, python, logs, native, ros, gz, cmake)
    finally:
        lock.close()


def launch(args, python, logs, native, ros, gz, cmake):
    if not args.sensors_only:
        with socket.socket() as probe:
            # Match the HTTP server: permit an immediate restart after old
            # connections enter TIME_WAIT, while an active listener still fails.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", 8767))
    logs.mkdir()
    print(f"Logs: {logs}", flush=True)
    helper = HERE / ".build/gz_capture"
    if not args.skip_build:
        for cmd in ([cmake, "-S", str(HERE), "-B", str(HERE / ".build"), "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_PREFIX_PATH=/opt/homebrew"],
                    [cmake, "--build", str(HERE / ".build"), "-j", "4"]):
            run_bounded(cmd, native, 600.)
    if not helper.is_file():
        raise RuntimeError("Gazebo bridge helper is missing; run without --skip-build")
    from generate_world import build
    world = build(logs / "world", render_engine=args.render_engine, camera_width=args.camera_width)
    children, files = [], []
    log_budget = ScopedLogBudget(logs)
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    def start(name, command, environment):
        # O_APPEND is required for in-place bounded diagnostic tail retention.
        stream = (logs / f"{name}.log").open("ab")
        files.append(stream)
        process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        children.append((name, process))
        return process

    def check_children():
        log_budget.tick()
        for name, process in children:
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited ({process.returncode}); inspect {logs / (name + '.log')}")

    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        start("gazebo", [gz, "sim", "--force-version", "8", "-s", "-r", "-v", "3", str(world)], native)
        bridges = {}
        for robot, domain in (("rover_a", 80), ("rover_b", 81)):
            bridges[robot] = start(robot, [python, str(HERE / "ros_bridge.py"), "--robot", robot, "--domain", str(domain),
                                          "--helper", str(helper), "--scene", str(world.parent / "scene.json"),
                                          "--sensor-profile", args.sensor_profile, "--noise-seed", str(args.noise_seed),
                                          "--ready-file", str(logs / f"{robot}.ready.json")], ros)
        deadline = time.monotonic() + 120
        print("Waiting for advancing RGB-D, laser scans, wheel odometry and simulation clock…", flush=True)
        while not stopping:
            check_children()
            if all(source_ready(logs / f"{robot}.ready.json", robot, domain, bridges[robot].pid)
                   for robot, domain in (("rover_a", 80), ("rover_b", 81))):
                break
            if time.monotonic() > deadline:
                raise RuntimeError(f"Cameras did not become ready within 120s. Check {logs}; try the other --render-engine.")
            time.sleep(.25)
        if not stopping:
            if not args.no_gui:
                start("gui", [gz, "sim", "--force-version", "8", "-g", "-v", "2"], native)
            if not args.sensors_only:
                truth_path = logs / 'truth.json'
                start('ground_truth', [python, str(HERE / 'ground_truth.py'), '--helper', str(helper),
                                       '--output', str(truth_path)], native)
                start("crossmaps", [python, str(APP / "crossmaps_collaborative.py"), "--config", str(CONFIG),
                                     '--simulation-truth', str(truth_path),
                                     '--simulation-scene', str(world.parent / 'scene.json')], ros)
            if args.nav2:
                navigation = {}
                for robot, domain in (("rover_a", 80), ("rover_b", 81)):
                    command = [python, str(HERE / 'navigation.py'), '--robot', robot,
                               '--ready-file', str(logs / f'navigation_{robot}.ready.json')]
                    if args.rviz not in (robot, 'both'):
                        command.append('--no-rviz')
                    navigation[robot] = start('navigation_' + robot, command, ros)
                print('Waiting for laser maps and active Nav2 stacks…', flush=True)
                navigation_deadline = time.monotonic() + 180.
                while not stopping:
                    check_children()
                    if all(navigation_ready(logs / f'navigation_{robot}.ready.json', robot, domain, navigation[robot].pid)
                           for robot, domain in (("rover_a", 80), ("rover_b", 81))):
                        print('Both Nav2 stacks ready. In RViz choose Nav2 Goal, click and drag a heading.', flush=True)
                        break
                    if time.monotonic() > navigation_deadline:
                        raise RuntimeError(f'Navigation did not become ready within 180s; inspect navigation logs in {logs}.')
                    time.sleep(.25)
            print("Sensors ready. " + ("CrossMaps: http://127.0.0.1:8767" if not args.sensors_only else "Sensors-only mode."), flush=True)
            print("Ctrl-C stops this simulation and its coordinator, not the real robots.\n"
                  + ("Drive: RViz Nav2 Goal; Cancel in Navigation 2 stops a goal." if args.nav2 else
                     "Drive: python simulation/gazebo/teleop.py --robot rover_a"), flush=True)
        started = time.monotonic()
        while not stopping and (args.duration is None or time.monotonic() - started < args.duration):
            check_children()
            time.sleep(.25)
    finally:
        print("Stopping owned simulation processes…", flush=True)
        try:
            stop_owned(children)
        finally:
            for stream in files:
                stream.close()
            for sig, handler in old.items():
                signal.signal(sig, handler)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f"Simulation: {error}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise SystemExit(130)
