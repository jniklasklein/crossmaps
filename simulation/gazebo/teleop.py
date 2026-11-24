#!/usr/bin/env python3
"""Keyboard control restricted to the two local simulated rovers."""
import argparse
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty
from run import apply_environment, environments, ros_python


def check_control_graph(node, robot, own_publisher=False):
    """Refuse to compete with Nav2 or an unexpected control endpoint."""
    expected = 'crossmaps_gazebo_' + robot
    subscribers = node.get_subscriptions_info_by_topic('/cmd_vel')
    if len(subscribers) != 1 or subscribers[0].node_name != expected:
        raise RuntimeError('Expected the isolated Gazebo bridge as the only command receiver.')
    if node.count_publishers('/cmd_vel') != int(own_publisher):
        raise RuntimeError('Another controller (such as Nav2) is active. Stop its navigation stack before keyboard driving.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("rover_a", "rover_b"), default="rover_a")
    args = parser.parse_args()
    python = ros_python()
    if Path(sys.executable).resolve() != Path(python).resolve():
        os.execv(python, [python, str(Path(__file__).resolve()), *sys.argv[1:]])
    _, env = environments(python, Path(__file__).parent / ".runtime/teleop")
    apply_environment(env)
    if not sys.stdin.isatty():
        raise RuntimeError("Use an interactive terminal for keyboard driving.")
    import rclpy
    from geometry_msgs.msg import Twist
    rclpy.init(domain_id=80 if args.robot == "rover_a" else 81)
    node = rclpy.create_node("crossmaps_sim_keyboard")
    deadline = time.monotonic() + 8.
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=.1)
        if node.get_subscriptions_info_by_topic('/cmd_vel'):
            break
    try:
        check_control_graph(node, args.robot)
    except RuntimeError:
        node.destroy_node()
        rclpy.shutdown()
        raise
    publisher = node.create_publisher(Twist, "/cmd_vel", 1)
    print(f"SIMULATION ONLY: {args.robot}. Hold/repeat W/S forward/back, A/D turn. Space stop, Q quit.\n"
          "Commands expire after 0.3 seconds without a key repeat.", flush=True)
    previous = termios.tcgetattr(sys.stdin)
    velocity, yaw, last = 0., 0., 0.
    last_graph_check = time.monotonic()
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            if time.monotonic() - last_graph_check >= 1.:
                check_control_graph(node, args.robot, own_publisher=True)
                last_graph_check = time.monotonic()
            if select.select([sys.stdin], [], [], .05)[0]:
                key = sys.stdin.read(1).lower()
                if key in ("q", "\x03"):
                    break
                velocity, yaw = {"w": (.2, 0.), "s": (-.2, 0.), "a": (0., .4), "d": (0., -.4)}.get(key, (0., 0.))
                last = time.monotonic()
            if time.monotonic() - last > .3:
                velocity, yaw = 0., 0.
            msg = Twist()
            msg.linear.x, msg.angular.z = velocity, yaw
            publisher.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.)
    finally:
        for _ in range(3):
            publisher.publish(Twist())
            time.sleep(.02)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, previous)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
