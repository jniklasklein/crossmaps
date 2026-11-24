#!/usr/bin/env python3
"""Opt-in bounded motion check for the PRIVATE LOCAL Gazebo simulation only.

Without --execute this prints the planned test and creates no ROS node.
It never accepts a domain or topic override. Do not use alongside teleop.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

from run import apply_environment, environments, ros_python, validate_simulation_config

DOMAINS = {'rover_a': 80, 'rover_b': 81}


def angle_delta(a, b):
    return math.atan2(math.sin(b-a), math.cos(b-a))


def yaw(quaternion):
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    return math.atan2(2 * (w*z+x*y), 1 - 2 * (y*y+z*z))


def evaluate_motion(before, after_forward, after_turn):
    def distance(a, b, key):
        return math.hypot(b[key][0]-a[key][0], b[key][1]-a[key][1])
    result = dict(
        forward_odometry_m=distance(before, after_forward, 'odom_xy'),
        forward_capture_tf_m=distance(before, after_forward, 'capture_xy'),
        turn_odometry_deg=math.degrees(angle_delta(after_forward['odom_yaw'], after_turn['odom_yaw'])),
        turn_capture_tf_deg=math.degrees(angle_delta(after_forward['capture_yaw'], after_turn['capture_yaw'])),
        image_mean_absolute_change=after_turn['image_change'],
        stamps_advancing=before['stamp'] < after_forward['stamp'] < after_turn['stamp'])
    result['ok'] = bool(result['forward_odometry_m'] > .01 and result['forward_capture_tf_m'] > .01
                        and 3. < result['turn_odometry_deg'] < 60. and 3. < result['turn_capture_tf_deg'] < 60.
                        and result['image_mean_absolute_change'] > .5 and result['stamps_advancing'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', choices=DOMAINS, default='rover_a')
    parser.add_argument('--execute', action='store_true', help='Explicitly permit this small simulated forward move and turn')
    parser.add_argument('--timeout', type=float, default=90., help='Total wall-time bound, including discovery (5–180 s)')
    parser.add_argument('--output', type=Path, help='Optional directory for before/after images and JSON report')
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or not 5 <= args.timeout <= 180:
        parser.error('--timeout must be finite and between 5 and 180 seconds')
    plan = dict(robot=args.robot, domain=DOMAINS[args.robot], simulation_only=True,
                forward_speed_m_s=.1, forward_sim_seconds=.75, turn_rate_rad_s=.3, turn_sim_seconds=.75,
                max_wall_seconds=args.timeout)
    print(json.dumps({'planned_test': plan, 'execute': args.execute}), flush=True)
    if not args.execute:
        print('No motion sent. Use --execute only with run.py running and no other teleop controller.')
        return 0
    validate_simulation_config()
    python = ros_python()
    if Path(sys.executable).resolve() != Path(python).resolve():
        os.execv(python, [python, str(Path(__file__).resolve()), *sys.argv[1:]])
    _, env = environments(python, Path(__file__).parent / '.runtime/motion-check')
    apply_environment(env)
    import cv2
    import numpy as np
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CompressedImage, CameraInfo
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener, TransformException

    rclpy.init(args=[], domain_id=DOMAINS[args.robot])
    node = rclpy.create_node('crossmaps_sim_motion_check_' + args.robot)
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    publisher = node.create_publisher(Twist, '/cmd_vel', 1)
    started = time.monotonic()
    deadline = started + args.timeout
    state = dict(clock=0, last_progress=started, odom=None, rgb={}, depth={}, info=None, reset=False)
    baseline_image = None
    cancelled = False

    def on_signal(*_):
        nonlocal cancelled
        cancelled = True

    previous = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGINT, signal.SIGTERM)}

    def clock(msg):
        tick = msg.clock.sec * 1_000_000_000 + msg.clock.nanosec
        if tick < state['clock']:
            state['reset'] = True
        if tick > state['clock']:
            state['last_progress'] = time.monotonic()
        state['clock'] = tick

    node.create_subscription(Clock, '/clock', clock, qos_profile_sensor_data)
    node.create_subscription(Odometry, '/odom', lambda msg: state.update(odom=msg), qos_profile_sensor_data)
    def image(kind, msg):
        tick = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        state[kind][tick] = msg
        while len(state[kind]) > 8:
            del state[kind][min(state[kind])]
    node.create_subscription(CompressedImage, '/oak/rgb/image_rect/compressed', lambda msg: image('rgb', msg), qos_profile_sensor_data)
    node.create_subscription(CompressedImage, '/oak/stereo/image_raw/compressedDepth', lambda msg: image('depth', msg), qos_profile_sensor_data)
    node.create_subscription(CameraInfo, '/oak/rgb/camera_info', lambda msg: state.update(info=msg), qos_profile_sensor_data)

    def transform_matrix(transform):
        t, q = transform.transform.translation, transform.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        result = np.eye(4)
        result[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
        result[:3, 3] = [t.x, t.y, t.z]
        return result

    def command(speed=0., turn=0.):
        msg = Twist()
        msg.linear.x, msg.angular.z = float(speed), float(turn)
        publisher.publish(msg)

    def step(moving=False):
        if cancelled or not rclpy.ok():
            raise RuntimeError('Motion check interrupted; stopping simulated rover.')
        if time.monotonic() >= deadline:
            raise TimeoutError('Motion check exceeded its wall-time limit.')
        if state['reset']:
            raise RuntimeError('Simulation clock reset during motion check.')
        if moving and time.monotonic()-state['last_progress'] > 3.:
            raise RuntimeError('Simulation clock stalled during commanded motion.')
        rclpy.spin_once(node, timeout_sec=.01)

    def capture(label, minimum_stamp=0, minimum_clock=0):
        nonlocal baseline_image
        while True:
            step()
            common = state['rgb'].keys() & state['depth'].keys()
            odom, info = state['odom'], state['info']
            if not common or odom is None or info is None:
                continue
            tick = max(common)
            rgb, depth_msg = state['rgb'][tick], state['depth'][tick]
            if tick <= minimum_stamp or tick < minimum_clock:
                continue
            try:
                pose = buffer.lookup_transform('odom', 'base_footprint', Time.from_msg(rgb.header.stamp))
                camera_pose = buffer.lookup_transform('odom', 'oak_rgb_camera_optical_frame', Time.from_msg(rgb.header.stamp))
            except TransformException:
                continue
            bgr = cv2.imdecode(np.frombuffer(bytes(rgb.data), np.uint8), cv2.IMREAD_COLOR)
            depth = cv2.imdecode(np.frombuffer(bytes(depth_msg.data)[12:], np.uint8), cv2.IMREAD_UNCHANGED)
            if bgr is None or depth is None or depth.dtype != np.uint16 or depth.shape != bgr.shape[:2]:
                continue
            gray = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)
            if baseline_image is None:
                baseline_image = gray.copy()
            result = dict(stamp=tick, odom_xy=[odom.pose.pose.position.x, odom.pose.pose.position.y],
                          odom_yaw=yaw(odom.pose.pose.orientation),
                          capture_xy=[pose.transform.translation.x, pose.transform.translation.y],
                          capture_yaw=yaw(pose.transform.rotation),
                          image_change=float(np.mean(np.abs(gray-baseline_image))))
            if args.output:
                args.output.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.output / (label + '.jpg')), bgr)
                # Lossless decoded observations make later pose-guard debugging
                # reproducible; a re-encoded screenshot changes ORB matches.
                np.savez_compressed(args.output / (label + '.npz'), rgb=bgr,
                    depth=depth.astype(np.float32)*.001, K=np.asarray(info.k).reshape(3, 3),
                    T_local_camera=transform_matrix(camera_pose), T_local_base=transform_matrix(pose),
                    stamp_ns=np.asarray(tick))
            return result

    def move(speed, turn):
        target = state['clock'] + 750_000_000
        next_command = 0.
        while state['clock'] < target:
            step(moving=True)
            if time.monotonic() >= next_command:
                command(speed, turn)
                next_command = time.monotonic()+.05
        command()
        return state['clock']

    try:
        expected = 'crossmaps_gazebo_' + args.robot
        while True:
            step()
            odom_sources = node.get_publishers_info_by_topic('/odom')
            subscribers = node.get_subscriptions_info_by_topic('/cmd_vel')
            if (state['clock'] > 0 and any(item.node_name == expected for item in odom_sources)
                    and len(subscribers) == 1 and subscribers[0].node_name == expected):
                break
        # Refuse concurrent keyboard or other command publishers, including a
        # previous test still running. This node is the only permitted publisher.
        if node.count_publishers('/cmd_vel') != 1:
            raise RuntimeError('Another /cmd_vel publisher is present; stop simulation teleop before testing.')
        command()
        before = capture('before')
        forward_end = move(.1, 0.)
        after_forward = capture('after_forward', before['stamp'], forward_end+200_000_000)
        turn_end = move(0., .3)
        after_turn = capture('after_turn', after_forward['stamp'], turn_end+200_000_000)
        report = dict(scope='Simulator command, odometry, capture-time TF and rendered RGB only; CrossMaps pose guard is not evaluated here.',
                      plan=plan, result=evaluate_motion(before, after_forward, after_turn),
                      captures=dict(before=before, after_forward=after_forward, after_turn=after_turn))
        print(json.dumps(report, indent=2), flush=True)
        if args.output:
            (args.output / 'motion_report.json').write_text(json.dumps(report, indent=2)+'\n')
        return 0 if report['result']['ok'] else 1
    finally:
        # Repeat stop for DDS delivery; native helper also enforces its deadman.
        for _ in range(5):
            try:
                command()
                rclpy.spin_once(node, timeout_sec=.02)
            except Exception:
                pass
        listener.unregister()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, TimeoutError) as error:
        print(f'Motion check: {error}', file=sys.stderr)
        raise SystemExit(1)
