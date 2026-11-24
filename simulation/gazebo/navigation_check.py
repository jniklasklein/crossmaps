#!/usr/bin/env python3
"""Opt-in, bounded NavigateToPose/cancel test for the LOCAL Gazebo rovers.

Without --execute no ROS node, action request, or velocity publisher is made.
The checker observes /cmd_vel; it never manufactures a zero-velocity result.
Only the checker-owned navigation goal is canceled on timeout/interruption.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import uuid

from run import apply_environment, environments, ros_python, validate_simulation_config

DOMAINS = {"rover_a": 80, "rover_b": 81}
ALLOWED_CONTROLLERS = {"velocity_smoother", "controller_server", "behavior_server"}


def same_goal_id(left, right):
    # Humble's generated fixed-size UUID field may be a NumPy uint8 array.
    # Direct equality then yields 16 booleans rather than a scalar truth value.
    return bytes(left) == bytes(right)


def relative_goal(pose, forward, lateral):
    x, y, heading = pose
    return [x + math.cos(heading) * forward - math.sin(heading) * lateral,
            y + math.sin(heading) * forward + math.cos(heading) * lateral, heading]


def pose_errors(actual, goal):
    if actual is None or goal is None:
        return None, None
    return (math.hypot(actual[0] - goal[0], actual[1] - goal[1]),
            abs(math.atan2(math.sin(actual[2] - goal[2]), math.cos(actual[2] - goal[2]))))


def evaluate_navigation(report):
    """Pass requires observed planning, fresh scans, real action state and stop."""
    position, heading = pose_errors(report.get("final_pose"), report.get("goal_pose"))
    start, finish = report.get("start_pose"), report.get("final_pose")
    travel = math.hypot(finish[0] - start[0], finish[1] - start[1]) if start and finish else None
    basic = (not report.get("error") and report.get("goal_accepted") is True
             and report.get("planned_path_count", 0) > 0 and report.get("max_path_poses", 0) >= 2
             and report.get("scan_advances_during_test", 0) > 0 and report.get("scan_fresh") is True
             and report.get("last_command_zero") is True and report.get("command_fresh") is True
             and report.get("nonzero_command_seen") is True)
    if report.get("cancel_test"):
        ok = basic and report.get("cancel_acknowledged") is True and report.get("action_status") == 5
    else:
        ok = (basic and report.get("action_status") == 4 and position is not None
              and position <= .20 and heading <= .25 and travel is not None and travel >= .05)
    return dict(ok=bool(ok), position_error_m=position, heading_error_rad=heading,
                displacement_m=travel, position_tolerance_m=.20, heading_tolerance_rad=.25)


def validate_control_graph(receivers, publishers, scan_sources, robot, observer):
    """The observer is not a command receiver; only the private bridge may drive."""
    expected = "crossmaps_gazebo_" + robot
    actual = [name for name in receivers if name != observer]
    if actual != [expected]:
        raise RuntimeError("Expected the isolated Gazebo bridge as the only /cmd_vel receiver.")
    if not publishers or any(name not in ALLOWED_CONTROLLERS for name in publishers):
        raise RuntimeError("Unexpected /cmd_vel publisher; stop teleop/other motion tests before Nav2 testing.")
    if scan_sources != [expected]:
        raise RuntimeError("Expected /scan to originate only from the isolated Gazebo bridge.")


def save_report(report, output):
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    print(serialized, flush=True)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=DOMAINS, default="rover_a")
    parser.add_argument("--execute", action="store_true", help="Explicitly authorize this bounded simulated navigation goal")
    parser.add_argument("--cancel", action="store_true", help="Cancel this test's goal shortly after motion starts and verify stopping")
    parser.add_argument("--timeout", type=float, default=90., help="Total wall-time budget, including cleanup (15–180 seconds)")
    parser.add_argument("--forward", type=float, default=.5, help="Goal offset along current heading (0.25–1.0 metres)")
    parser.add_argument("--lateral", type=float, default=.25, help="Goal offset to the left (-0.5–0.5 metres)")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or not 15 <= args.timeout <= 180:
        parser.error("--timeout must be finite and between 15 and 180 seconds")
    if (not math.isfinite(args.forward) or not .25 <= args.forward <= 1.
            or not math.isfinite(args.lateral) or not -.5 <= args.lateral <= .5
            or math.hypot(args.forward, args.lateral) > 1.):
        parser.error("Goal offsets must be finite, forward 0.25–1.0m, lateral ±0.5m, total distance at most 1m")
    report = dict(robot=args.robot, domain=DOMAINS[args.robot], simulation_only=True,
                  execute=args.execute, cancel_test=args.cancel, max_wall_seconds=args.timeout,
                  relative_goal=dict(forward_m=args.forward, lateral_m=args.lateral, yaw_offset_rad=0.),
                  goal_accepted=False, action_status=None, cancel_acknowledged=False,
                  start_pose=None, goal_pose=None, final_pose=None, error=None)
    if not args.execute:
        report["note"] = "No ROS node or motion request created. Run with --execute only after Nav2 is ready and the chosen short path is clear."
        save_report(report, args.output)
        return 0
    validate_simulation_config()
    python = ros_python()
    if Path(sys.executable).resolve() != Path(python).resolve():
        command_args = list(argv) if argv is not None else sys.argv[1:]
        os.execv(python, [python, str(Path(__file__).resolve()), *command_args])
    _, env = environments(python, Path(__file__).parent / ".runtime/navigation-check")
    env["ROS_DOMAIN_ID"] = str(DOMAINS[args.robot])
    apply_environment(env)
    import rclpy
    from action_msgs.msg import GoalStatusArray
    from action_msgs.srv import CancelGoal
    from geometry_msgs.msg import Twist
    from nav2_msgs.action import NavigateToPose
    from nav_msgs.msg import Path as NavPath
    from rclpy.action import ActionClient
    from rclpy.parameter import Parameter
    from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
    from rclpy.time import Time
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer, TransformException, TransformListener
    from unique_identifier_msgs.msg import UUID

    started = time.monotonic()
    deadline, work_deadline = started + args.timeout, started + args.timeout - 5.
    rclpy.init(args=[], domain_id=DOMAINS[args.robot])
    name = "crossmaps_sim_navigation_check_" + args.robot
    node = rclpy.create_node(name, parameter_overrides=[Parameter("use_sim_time", value=True)])
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    client = ActionClient(node, NavigateToPose, "/navigate_to_pose")
    cancel_client = node.create_client(CancelGoal, "/navigate_to_pose/_action/cancel_goal")
    owned_uuid = UUID(uuid=list(uuid.uuid4().bytes))
    state = dict(clock=0, clock_wall=started, clock_reset=False, scan_stamp=-1,
                 scan_wall=0., scan_count=0, valid_scan=False, path_count=0,
                 max_path_poses=0, command_wall=0., command_zero=False,
                 nonzero_wall=None, active_goals=False)
    interrupted, goal_future, handle, result_future, cancel_future = False, None, None, None, None
    goal_sent_wall = None
    first_scan_count = 0
    last_graph_check = 0.

    def stop_requested(*_):
        nonlocal interrupted
        interrupted = True

    previous = {sig: signal.signal(sig, stop_requested) for sig in (signal.SIGINT, signal.SIGTERM)}

    def on_clock(message):
        tick = message.clock.sec * 1_000_000_000 + message.clock.nanosec
        if tick < state["clock"]:
            state["clock_reset"] = True
        if tick > state["clock"]:
            state["clock_wall"] = time.monotonic()
        state["clock"] = tick

    def on_scan(message):
        tick = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        if tick > state["scan_stamp"]:
            state["scan_stamp"], state["scan_wall"] = tick, time.monotonic()
            state["scan_count"] += 1
        state["valid_scan"] = bool(message.ranges) and any(
            math.isfinite(value) and message.range_min <= value <= message.range_max for value in message.ranges)

    def on_path(message):
        if goal_sent_wall is not None and len(message.poses) >= 2:
            state["path_count"] += 1
            state["max_path_poses"] = max(state["max_path_poses"], len(message.poses))

    def on_command(message):
        values = [getattr(vector, axis) for vector in (message.linear, message.angular) for axis in ("x", "y", "z")]
        state["command_wall"] = time.monotonic()
        state["command_zero"] = all(math.isfinite(value) and abs(value) <= .001 for value in values)
        if goal_sent_wall is not None and not state["command_zero"] and state["nonzero_wall"] is None:
            state["nonzero_wall"] = time.monotonic()

    node.create_subscription(Clock, "/clock", on_clock, qos_profile_sensor_data)
    node.create_subscription(LaserScan, "/scan", on_scan, qos_profile_sensor_data)
    node.create_subscription(NavPath, "/plan", on_path, qos_profile_sensor_data)
    node.create_subscription(Twist, "/cmd_vel", on_command, qos_profile_sensor_data)
    node.create_subscription(GoalStatusArray, "/navigate_to_pose/_action/status",
                             lambda msg: state.update(active_goals=any(item.status in (1, 2, 3) for item in msg.status_list)),
                             QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def check_graph():
        validate_control_graph([item.node_name for item in node.get_subscriptions_info_by_topic("/cmd_vel")],
                               [item.node_name for item in node.get_publishers_info_by_topic("/cmd_vel")],
                               [item.node_name for item in node.get_publishers_info_by_topic("/scan")], args.robot, name)

    def step(enforce_graph=False):
        nonlocal last_graph_check
        if interrupted or not rclpy.ok():
            raise RuntimeError("Navigation check interrupted; canceling the checker-owned simulated goal.")
        if time.monotonic() >= work_deadline:
            raise TimeoutError("Navigation test timed out; reserving five seconds for owned-goal cancellation.")
        if state["clock_reset"]:
            raise RuntimeError("Simulation clock reset during navigation check.")
        if enforce_graph and (time.monotonic() - state["clock_wall"] > 5. or time.monotonic() - state["scan_wall"] > 5.):
            raise RuntimeError("Simulation clock or advancing lidar stream stalled during navigation.")
        rclpy.spin_once(node, timeout_sec=.02)
        if enforce_graph and time.monotonic() - last_graph_check >= 1.:
            check_graph()
            last_graph_check = time.monotonic()

    def current_pose():
        try:
            transform = buffer.lookup_transform("map", "base_footprint", Time())
        except TransformException:
            return None
        stamp = transform.header.stamp.sec * 1_000_000_000 + transform.header.stamp.nanosec
        if not 0 <= state["clock"] - stamp <= 2_000_000_000:
            return None
        point, q = transform.transform.translation, transform.transform.rotation
        values = [point.x, point.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))]
        return values if all(math.isfinite(value) for value in values) else None

    def request_cancel():
        nonlocal cancel_future
        if handle is not None and handle.accepted and cancel_future is None:
            cancel_future = handle.cancel_goal_async()

    def record_cancel_reply():
        if cancel_future is not None and cancel_future.done():
            response = cancel_future.result()
            report["cancel_acknowledged"] = bool(response and response.return_code == 0
                                                  and any(same_goal_id(item.goal_id.uuid, handle.goal_id.uuid)
                                                          for item in response.goals_canceling))

    try:
        while True:
            step()
            pose = current_pose()
            if (client.server_is_ready() and state["clock"] > 0 and state["scan_count"] >= 2
                    and state["valid_scan"] and time.monotonic() - state["scan_wall"] <= 3.
                    and time.monotonic() - started >= 1. and pose is not None):
                check_graph()
                if state["active_goals"]:
                    raise RuntimeError("A navigation goal is already active; refusing to preempt it.")
                report["start_pose"] = pose
                break
        report["goal_pose"] = relative_goal(report["start_pose"], args.forward, args.lateral)
        request = NavigateToPose.Goal()
        request.pose.header.frame_id = "map"
        request.pose.header.stamp.sec, request.pose.header.stamp.nanosec = divmod(state["clock"], 1_000_000_000)
        request.pose.pose.position.x, request.pose.pose.position.y, heading = report["goal_pose"]
        request.pose.pose.orientation.z, request.pose.pose.orientation.w = math.sin(heading / 2.), math.cos(heading / 2.)
        first_scan_count, goal_sent_wall = state["scan_count"], time.monotonic()
        goal_future = client.send_goal_async(request, goal_uuid=owned_uuid)
        while not goal_future.done():
            step(enforce_graph=True)
        handle = goal_future.result()
        report["goal_accepted"] = bool(handle and handle.accepted)
        if not report["goal_accepted"]:
            raise RuntimeError("NavigateToPose rejected the test goal.")
        report["goal_uuid"] = bytes(handle.goal_id.uuid).hex()
        result_future = handle.get_result_async()
        while not result_future.done():
            step(enforce_graph=True)
            if (args.cancel and state["path_count"] > 0 and state["nonzero_wall"] is not None
                    and time.monotonic() - state["nonzero_wall"] >= .8):
                request_cancel()
            record_cancel_reply()
        report["action_status"] = result_future.result().status
        # Observe the actual controller stop, without publishing a fake zero.
        settled = time.monotonic()
        while time.monotonic() - settled < .5 or not state["command_zero"]:
            step(enforce_graph=True)
        record_cancel_reply()
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            # If acceptance is still in flight, cancel by our exact UUID as
            # well. An all-zero UUID would cancel unrelated goals: never use it.
            pending_cancel, next_cancel = None, 0.
            while goal_future is not None and not goal_future.done() and time.monotonic() < deadline - 1.:
                if time.monotonic() >= next_cancel and (pending_cancel is None or pending_cancel.done()):
                    cancellation = CancelGoal.Request()
                    cancellation.goal_info.goal_id = owned_uuid
                    pending_cancel = cancel_client.call_async(cancellation)
                    next_cancel = time.monotonic() + .2
                rclpy.spin_once(node, timeout_sec=.02)
            if handle is None and goal_future is not None and goal_future.done():
                handle = goal_future.result()
                report["goal_accepted"] = bool(handle and handle.accepted)
            if handle is not None and handle.accepted:
                if result_future is None:
                    result_future = handle.get_result_async()
                if not result_future.done():
                    request_cancel()
                    while time.monotonic() < deadline and (not result_future.done() or not state["command_zero"]):
                        rclpy.spin_once(node, timeout_sec=.02)
                record_cancel_reply()
                if result_future.done():
                    report["action_status"] = result_future.result().status
                else:
                    report["cleanup_warning"] = "Owned-goal cancellation was requested but terminal state was not confirmed within the wall-time budget."
            elif goal_future is not None and not goal_future.done():
                report["cleanup_warning"] = "Goal acceptance is still unresolved; cannot confirm that the goal was canceled."
        except Exception as exc:
            report["cleanup_warning"] = str(exc)
        report["final_pose"] = current_pose()
        now = time.monotonic()
        report.update(elapsed_wall_seconds=now - started,
                      planned_path_count=state["path_count"], max_path_poses=state["max_path_poses"],
                      scan_advances_during_test=max(0, state["scan_count"] - first_scan_count),
                      scan_age_wall_seconds=now - state["scan_wall"] if state["scan_wall"] else None,
                      scan_fresh=state["valid_scan"] and 0 <= now - state["scan_wall"] <= 3.,
                      last_command_zero=state["command_zero"],
                      command_fresh=goal_sent_wall is not None and state["command_wall"] >= goal_sent_wall
                                    and 0 <= now - state["command_wall"] <= 3.,
                      nonzero_command_seen=state["nonzero_wall"] is not None)
        report["result"] = evaluate_navigation(report)
        if report.get("cleanup_warning"):
            report["result"]["ok"] = False
        listener.unregister()
        client.destroy()
        node.destroy_client(cancel_client)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for sig, old in previous.items():
            signal.signal(sig, old)
        save_report(report, args.output)
    return 0 if report["result"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
