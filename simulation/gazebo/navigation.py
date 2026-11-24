#!/usr/bin/env python3
"""Own one simulation-only laser-SLAM/Nav2 stack, optionally with click-to-drive RViz.

No goals are submitted by this launcher. Use RViz's Nav2 Goal tool to click a
mapped free-space destination and drag its heading; the Navigation 2 panel can
cancel it. The two rovers have independent local maps and navigation domains.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run import (ScopedLogBudget, apply_environment, environments, ros_python,
                 stop_owned, validate_simulation_config)

HERE = Path(__file__).resolve().parent
DOMAINS = {"rover_a": 80, "rover_b": 81}
LIFECYCLE_NODES = ("controller_server", "smoother_server", "planner_server", "behavior_server",
                   "bt_navigator", "waypoint_follower", "velocity_smoother")


def domain_for(robot):
    if robot not in DOMAINS:
        raise ValueError("Navigation is restricted to simulated rover_a/80 or rover_b/81")
    return DOMAINS[robot]


def isolated_environment(python, logs, robot):
    _, environment = environments(python, logs)
    environment["ROS_DOMAIN_ID"] = str(domain_for(robot))
    # Python launch discovers native conda ROS packages without a sourced shell.
    environment["AMENT_PREFIX_PATH"] = str(Path(python).parent.parent)
    return environment


def commands(python, robot, *, rviz_only=False, no_rviz=False, rviz_config=None, rviz_executable=None):
    """Pure command construction, also used by dependency/unit checks."""
    domain_for(robot)
    ros2 = str(Path(python).parent / "ros2")
    result = []
    if not rviz_only:
        result.extend([
            ("slam", [ros2, "launch", "slam_toolbox", "online_async_launch.py",
                      f"slam_params_file:={HERE / 'slam_params.yaml'}", "use_sim_time:=true"]),
            ("nav2", [ros2, "launch", "nav2_bringup", "navigation_launch.py",
                      f"params_file:={HERE / 'nav2_params.yaml'}", "use_sim_time:=true",
                      "autostart:=true", "use_composition:=False", "use_respawn:=False"]),
        ])
    if not no_rviz:
        result.append(("rviz", [str(rviz_executable or Path(python).parent / "rviz2"), "-d", str(rviz_config or HERE / "navigation.rviz"),
                                 "--ros-args", "-p", "use_sim_time:=true", "-r", f"__node:=rviz_{robot}"]))
    return result


def write_rviz_config(logs, robot):
    """Give each native window its own saved filename and visible rover label."""
    import yaml
    domain = domain_for(robot)
    config = yaml.safe_load((HERE / "navigation.rviz").read_text())
    label = f"SIMULATED {robot.upper()} — domain {domain}"
    config["Panels"][0]["Name"] = label
    config["Visualization Manager"]["Displays"][0]["Name"] = label + " · metre grid"
    # RViz uses the configuration filename in its window title. Keep the robot
    # explicit in that filename as well as in the persistent Displays dock.
    path = Path(logs) / f"{robot}-navigation.rviz"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    return path


def dependency_check(python):
    from ament_index_python.packages import get_package_prefix, get_package_share_directory
    expected = {
        "slam_toolbox": ("launch/online_async_launch.py",),
        "nav2_bringup": ("launch/navigation_launch.py",),
        "nav2_rviz_plugins": ("plugins_description.xml",),
    }
    found = {}
    for package, relative_paths in expected.items():
        share = Path(get_package_share_directory(package))
        for relative in relative_paths:
            path = share / relative
            if not path.is_file():
                raise RuntimeError(f"Missing native navigation dependency: {path}")
        found[package] = str(share)
    for package, executable in (("slam_toolbox", "async_slam_toolbox_node"),
                                ("nav2_controller", "controller_server"),
                                ("nav2_planner", "planner_server"),
                                ("nav2_bt_navigator", "bt_navigator")):
        path = Path(get_package_prefix(package)) / "lib" / package / executable
        if not os.access(path, os.X_OK):
            raise RuntimeError(f"Missing navigation executable: {path}")
    for executable in ("ros2", "rviz2"):
        if not os.access(Path(python).parent / executable, os.X_OK):
            raise RuntimeError(f"Missing {executable} in the ROS environment")
    for name in ("nav2_params.yaml", "slam_params.yaml", "navigation.rviz"):
        if not (HERE / name).is_file():
            raise RuntimeError(f"Missing navigation configuration: {name}")
    import yaml
    nav = yaml.safe_load((HERE / "nav2_params.yaml").read_text())
    slam = yaml.safe_load((HERE / "slam_params.yaml").read_text())
    for name in LIFECYCLE_NODES:
        if nav[name]["ros__parameters"].get("use_sim_time") is not True:
            raise RuntimeError(f"{name} must use simulation time")
    if slam["slam_toolbox"]["ros__parameters"].get("use_sim_time") is not True:
        raise RuntimeError("slam_toolbox must use simulation time")
    return found


class InputHealth:
    """Advancing timestamps, not repeated cached messages, establish liveness."""
    def __init__(self):
        self.stamps = {}
        self.advanced = {}
        self.counts = {}
        self.clock_reset = False

    def update(self, kind, stamp, now):
        previous = self.stamps.get(kind)
        if kind == "clock" and previous is not None and stamp < previous:
            self.clock_reset = True
        if previous is None or stamp > previous:
            self.stamps[kind] = stamp
            self.advanced[kind] = now
            self.counts[kind] = self.counts.get(kind, 0) + 1

    def fresh(self, now, limit=3.0):
        return (not self.clock_reset and all(self.counts.get(k, 0) >= 2 and
                0 <= now - self.advanced.get(k, -math.inf) < limit for k in ("clock", "scan", "odom")))

    def ages(self, now):
        return {kind: max(0.0, now - self.advanced[kind]) if kind in self.advanced else None
                for kind in ("clock", "scan", "odom")}


class LifecyclePolicy:
    """RViz Pause/Reset are valid states, not failed child processes."""
    def __init__(self, transition_grace=5.0):
        self.transition_grace = transition_grace
        self.transition_started = None

    def evaluate(self, states, now):
        values = {states.get(name, 0) for name in LIFECYCLE_NODES}
        stable = {frozenset({3}): ("ready", True), frozenset({2}): ("paused", False),
                  frozenset({1}): ("reset", False)}.get(frozenset(values))
        if stable is not None:
            self.transition_started = None
            return stable
        # Primary states 1/2/3 and normal configure/cleanup/activate/deactivate
        # transitions may overlap while the manager processes servers in order.
        # Unknown (including expired state replies), finalized and error states
        # are never silently treated as a deliberate user pause.
        if not values <= {1, 2, 3, 10, 11, 13, 14}:
            raise RuntimeError(f"Navigation server is unavailable or failed: {states}")
        if self.transition_started is None:
            self.transition_started = now
        if now - self.transition_started > self.transition_grace:
            raise RuntimeError(f"Navigation lifecycle transition exceeded {self.transition_grace:g}s: {states}")
        return "lifecycle_transition", False


def publish_health(path, robot, ready, phase, probe=None):
    if path is None:
        return
    data = dict(robot=robot, domain=domain_for(robot), pid=os.getpid(), ready=ready,
                phase=phase, updated_unix=time.time())
    if probe is not None:
        data.update(advancing_age_s=probe.health.ages(time.monotonic()),
                    lifecycle_states=dict(probe.states), map_received=probe.map_received)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, sort_keys=True) + "\n")
    temporary.replace(path)


class NavigationProbe:
    """ROS imports are delayed so pure configuration tests require no ROS runtime."""
    def __init__(self, robot):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
        from rosgraph_msgs.msg import Clock
        from nav_msgs.msg import Odometry, OccupancyGrid
        from sensor_msgs.msg import LaserScan
        from lifecycle_msgs.srv import GetState
        import tf2_ros
        self.robot, self.health = robot, InputHealth()
        self.context = Context()
        rclpy.init(context=self.context, domain_id=domain_for(robot))
        self.node = Node(f"crossmaps_sim_navigation_{robot}", context=self.context,
                         parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.node.create_subscription(Clock, "/clock", lambda m: self.record("clock", m.clock), qos_profile_sensor_data)
        self.node.create_subscription(Odometry, "/odom", lambda m: self.record("odom", m.header.stamp), qos_profile_sensor_data)
        self.node.create_subscription(LaserScan, "/scan", self.scan, qos_profile_sensor_data)
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.node.create_subscription(OccupancyGrid, "/map", self.mapping, map_qos)
        self.buffer = tf2_ros.Buffer(node=self.node)
        self.listener = tf2_ros.TransformListener(self.buffer, self.node)
        self.map_received = False
        self.clients = {name: self.node.create_client(GetState, f"/{name}/get_state") for name in LIFECYCLE_NODES}
        self.states, self.pending = {}, {}
        self.state_replies = {}
        self.last_query = -math.inf

    def record(self, kind, stamp):
        self.health.update(kind, stamp.sec * 1_000_000_000 + stamp.nanosec, time.monotonic())

    def scan(self, msg):
        # Reject an absent sensor, empty scan, or an impossible frame contract.
        if len(msg.ranges) >= 90 and msg.header.frame_id and msg.range_max > msg.range_min:
            self.record("scan", msg.header.stamp)

    def mapping(self, msg):
        self.map_received = (msg.header.frame_id == "map" and msg.info.width > 0 and
                             msg.info.height > 0 and len(msg.data) == msg.info.width * msg.info.height)

    def spin(self, timeout=.05):
        self.executor.spin_once(timeout_sec=timeout)

    def check_exclusive(self):
        # These checks are performed before any Nav2 child exists. A lock handles
        # other launcher instances; ROS graph checks also catch keyboard/CLI input.
        writers = self.node.get_publishers_info_by_topic("/cmd_vel")
        if writers:
            names = sorted({writer.node_name for writer in writers})
            raise RuntimeError(f"{self.robot} already has command publishers {names}; stop teleop/other navigation first")
        existing = set(self.node.get_node_names()) & (set(LIFECYCLE_NODES) | {"slam_toolbox"})
        if existing:
            raise RuntimeError(f"Navigation nodes already exist in domain {domain_for(self.robot)}: {sorted(existing)}")

    def correct_source(self):
        expected = "crossmaps_gazebo_" + self.robot
        return all(any(pub.node_name == expected for pub in self.node.get_publishers_info_by_topic(topic))
                   for topic in ("/clock", "/odom", "/scan"))

    def lifecycle_ready(self):
        from lifecycle_msgs.srv import GetState
        now = time.monotonic()
        for name, (future, requested_at) in list(self.pending.items()):
            if future.done():
                try:
                    self.states[name] = future.result().current_state.id
                    self.state_replies[name] = now
                except Exception:
                    self.states[name] = 0
                del self.pending[name]
            elif now - requested_at > 2.5:
                # A hung call must not preserve an ACTIVE reply indefinitely or
                # prevent fresh state requests after a transient discovery loss.
                self.clients[name].remove_pending_request(future)
                future.cancel()
                del self.pending[name]
        for name in list(self.states):
            if now - self.state_replies.get(name, -math.inf) > 5.0:
                self.states[name] = 0
        if now - self.last_query >= 1.0:
            self.last_query = now
            for name, client in self.clients.items():
                if name not in self.pending and client.service_is_ready():
                    self.pending[name] = (client.call_async(GetState.Request()), now)
        return all(self.states.get(name) == 3 for name in LIFECYCLE_NODES)

    def geometry_ready(self):
        from rclpy.time import Time
        if not self.map_received:
            return False
        try:
            transform = self.buffer.lookup_transform("map", "base_footprint", Time())
        except Exception:
            return False
        stamp = transform.header.stamp
        transform_time = stamp.sec * 1_000_000_000 + stamp.nanosec
        clock_time = self.health.stamps.get("clock")
        # Latest TF is still timestamped data: a cached transform from before a
        # crashed SLAM process must not make an apparently healthy navigator.
        return clock_time is not None and -500_000_000 <= clock_time - transform_time <= 1_000_000_000

    def close(self, *, stop_robot):
        if stop_robot:
            from geometry_msgs.msg import Twist
            publisher = self.node.create_publisher(Twist, "/cmd_vel", 1)
            # Repeated zero after controller teardown; the native helper also has
            # an independent 0.5s wall-clock deadman if DDS delivery is unavailable.
            for _ in range(8):
                publisher.publish(Twist())
                self.spin(.03)
        self.node.destroy_node()
        self.executor.shutdown()
        self.context.shutdown()


def cleanup_navigation(children, streams, probe, lock, handlers, ready_file, robot, phase, started_stack):
    """Every cleanup runs, including if status I/O or an earlier cleanup fails."""
    with ExitStack() as cleanup:
        # Callbacks run in reverse: stop controllers, publish zero, release files,
        # restore handlers, then release exclusive ownership. A full disk or
        # removed run directory must never strand a still-driving Nav2 process.
        if lock is not None:
            cleanup.callback(lock.close)
        for sig, previous in handlers.items():
            cleanup.callback(signal.signal, sig, previous)
        for stream in streams:
            cleanup.callback(stream.close)
        if probe is not None:
            cleanup.callback(probe.close, stop_robot=started_stack)
        cleanup.callback(stop_owned, children, grace=5.0)
        publish_health(ready_file, robot, False, phase, probe)


def supervise(args, python, logs, environment):
    import fcntl
    logs.mkdir(parents=True, exist_ok=True)
    lock = None
    if not args.rviz_only:
        lock = (HERE / ".runtime" / f"navigation_{args.robot}.lock").open("a+")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise RuntimeError(f"A navigation supervisor already owns {args.robot}; use --rviz-only to open its map")
    children, streams, probe = [], [], None
    log_budget = ScopedLogBudget(logs)
    stopping, started_stack = False, False
    phase = "starting"

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    def start(name, command):
        stream = (logs / f"{name}.log").open("ab")
        streams.append(stream)
        process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        children.append((name, process))

    def check_children():
        log_budget.tick()
        for name, process in children:
            if process.poll() is not None:
                if name == "rviz" and process.returncode == 0:
                    continue  # Closing a viewer must not kill its navigation stack.
                raise RuntimeError(f"{name} exited ({process.returncode}); see {logs / (name + '.log')}")

    old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        publish_health(args.ready_file, args.robot, False, phase)
        rviz_config, rviz_executable = None, None
        if not args.no_rviz:
            from macos_rviz import prepare_rviz_app
            rviz_config = write_rviz_config(logs, args.robot)
            rviz_executable = prepare_rviz_app(python, args.robot)
        plan = commands(python, args.robot, rviz_only=args.rviz_only, no_rviz=args.no_rviz,
                        rviz_config=rviz_config, rviz_executable=rviz_executable)
        if args.rviz_only:
            for name, command in plan:
                start(name, command)
        else:
            probe = NavigationProbe(args.robot)
            phase = "waiting_for_simulated_scan_odometry_clock"
            deadline = time.monotonic() + args.startup_timeout
            # Allow DDS discovery to settle before testing exclusive ownership.
            graph_settle = time.monotonic() + 1.0
            while not stopping:
                probe.spin()
                probe.check_exclusive()
                if time.monotonic() >= graph_settle and probe.health.fresh(time.monotonic()) and probe.correct_source():
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("No advancing local Gazebo /scan, /odom and /clock; start run.py first")
                publish_health(args.ready_file, args.robot, False, phase, probe)
            if stopping:
                return 0
            for name, command in plan:
                if name != "rviz":
                    start(name, command)
                    started_stack = True
            phase = "waiting_for_slam_map_and_nav2_lifecycle"
            deadline = time.monotonic() + args.startup_timeout
            while not stopping:
                probe.spin()
                check_children()
                if probe.health.clock_reset:
                    raise RuntimeError("Simulation clock reset; restart the navigation stack")
                if probe.health.fresh(time.monotonic()) and probe.lifecycle_ready() and probe.geometry_ready():
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError(f"SLAM/Nav2 did not become ready; states={probe.states}, map={probe.map_received}; see {logs}")
                publish_health(args.ready_file, args.robot, False, phase, probe)
            if stopping:
                return 0
            for name, command in plan:
                if name == "rviz":
                    start(name, command)
        phase = "ready"
        print(f"SIMULATION ONLY: {args.robot}, domain {domain_for(args.robot)}. "
              f"{'RViz viewer opened' if args.rviz_only else 'Laser SLAM and Nav2 active'}.\n"
              "Use Nav2 Goal: click mapped free space, drag the desired heading, release. "
              f"No goal is sent automatically. Logs: {logs}", flush=True)
        began, last_health = time.monotonic(), -math.inf
        lifecycle_policy = LifecyclePolicy()
        while not stopping:
            check_children()
            now = time.monotonic()
            if args.duration is not None and now - began >= args.duration:
                break
            if args.rviz_only:
                ready = True
                if children and all(process.poll() is not None for _, process in children):
                    break
                time.sleep(.1)
            else:
                probe.spin()
                now = time.monotonic()
                if not probe.health.fresh(now, limit=5.0):
                    raise RuntimeError("Simulated scan/odometry/clock stopped advancing or reset; navigation stopped safely. Restart after resuming Gazebo.")
                # Pause/Reset in the native Navigation 2 panel intentionally
                # changes lifecycle state. Keep those stacks alive for Resume or
                # Startup while still rejecting stale state replies and TF.
                probe.lifecycle_ready()
                phase, ready = lifecycle_policy.evaluate(probe.states, now)
                if not probe.geometry_ready():
                    raise RuntimeError("Laser SLAM map-to-rover transform is unavailable or stale; navigation stopped safely")
            if now - last_health >= .5:
                publish_health(args.ready_file, args.robot, ready, phase, probe)
                last_health = now
        return 0
    except Exception:
        phase = "failed"
        raise
    finally:
        cleanup_navigation(children, streams, probe, lock, old_handlers, args.ready_file, args.robot,
                           phase if phase == "failed" else "stopped", started_stack)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=tuple(DOMAINS), default="rover_a")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--no-rviz", action="store_true", help="Run laser SLAM/Nav2 without a viewer")
    mode.add_argument("--rviz-only", action="store_true", help="Open another viewer for an already-running stack")
    parser.add_argument("--duration", type=float, help="Stop owned children N seconds after readiness")
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--ready-file", type=Path, help="Periodically written lifecycle/input health JSON for the parent launcher")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--check", action="store_true", help="Validate packages/configuration without ROS nodes or motion")
    args = parser.parse_args(argv)
    for key in ("duration", "startup_timeout"):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    validate_simulation_config()
    python = ros_python()
    if Path(sys.executable).resolve() != Path(python).resolve():
        os.execv(python, [python, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)])
    tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    logs = args.log_dir or HERE / ".runtime" / f"navigation-{args.robot}-{tag}"
    environment = isolated_environment(python, logs, args.robot)
    apply_environment(environment)
    found = dependency_check(python)
    if args.check:
        print(json.dumps(dict(robot=args.robot, domain=domain_for(args.robot), packages=found,
                              simulation_only=True, goal_sent=False), indent=2))
        return 0
    (HERE / ".runtime").mkdir(exist_ok=True)
    return supervise(args, python, logs, environment)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError) as error:
        print(f"Navigation error: {error}", file=sys.stderr)
        raise SystemExit(1)
