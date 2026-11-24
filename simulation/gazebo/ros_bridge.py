#!/usr/bin/env python3
"""Compressed-only localhost ROS bridge for the two Gazebo test rovers.

The helper uses Homebrew's Gazebo ABI; this process uses the EXISTING ROS
environment. It publishes simulation time but its own housekeeping uses wall
time, so stop/deadman and EOF handling also work while simulation is paused.
"""
import argparse
from collections import defaultdict, deque
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from bridge_codec import (CAMERA_FRAME, LIDAR_FRAME, SIM_DOMAINS, LatestRgbdPair, camera_calibration,
                          camera_extrinsic, encode_depth,
                          encode_rgb, laser_scan, lidar_extrinsic, safe_velocity, stamp_ns, stamp_parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=SIM_DOMAINS, required=True)
    parser.add_argument("--domain", type=int, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True,
                        help="Generated scene.json containing the actual model optical extrinsics")
    parser.add_argument("--sensor-profile", choices=("clean","noisy"), default="clean",
                        help="noisy is a deterministic stress model, not calibrated OAK-D noise")
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--clock-topic", default="/clock")
    args = parser.parse_args(argv)
    if SIM_DOMAINS[args.robot] != args.domain:
        parser.error("Only rover_a/domain80 and rover_b/domain81 are allowed; never physical rover domains")
    if not os.environ.get("GZ_PARTITION", "").startswith("crossmaps_sim_"):
        parser.error("GZ_PARTITION must start with crossmaps_sim_")
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    os.environ["ROS_DOMAIN_ID"] = str(args.domain)
    if not args.helper.is_file():
        parser.error("Build gz_capture first (see simulation/gazebo/CMakeLists.txt)")
    try:
        scene = json.loads(args.scene.read_text())
        extrinsic = camera_extrinsic(scene, args.robot)
        laser_mount = lidar_extrinsic(scene, args.robot)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    import rclpy
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import TransformStamped, Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CameraInfo, CompressedImage, LaserScan
    from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

    rclpy.init(args=[], domain_id=args.domain)
    node = Node("crossmaps_gazebo_" + args.robot)
    rgb_pub = node.create_publisher(CompressedImage, "/oak/rgb/image_rect/compressed", qos_profile_sensor_data)
    depth_pub = node.create_publisher(CompressedImage, "/oak/stereo/image_raw/compressedDepth", qos_profile_sensor_data)
    info_pubs = [node.create_publisher(CameraInfo, topic, qos_profile_sensor_data) for topic in
                 ("/oak/rgb/camera_info", "/oak/stereo/camera_info")]
    # Nav2 Humble's OdomSmoother requests reliable SystemDefaultsQoS. A reliable
    # writer also satisfies existing best-effort mapping readers; images and
    # scans remain sensor-data QoS to avoid accumulating acquisition backlogs.
    odom_pub = node.create_publisher(Odometry, "/odom",
                                     QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
    scan_pub = node.create_publisher(LaserScan, "/scan", qos_profile_sensor_data)
    clock_pub = node.create_publisher(Clock, "/clock", QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
    tf_pub, static_pub = TransformBroadcaster(node), StaticTransformBroadcaster(node)
    helper = subprocess.Popen([str(args.helper.resolve()), args.robot, args.clock_topic],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1024*1024)
    pending = defaultdict(lambda: deque(maxlen=4))
    pending["odom"] = deque(maxlen=32)
    pending["clock"] = deque(maxlen=1)
    pending["info"] = deque(maxlen=1)
    pending["scan"] = deque(maxlen=1)
    pairs = LatestRgbdPair(capacity=4)
    received_counts = defaultdict(int)
    published_pairs = 0
    lock = threading.Lock()
    ended = threading.Event()
    errors = []
    counts, last_stamps, advance = defaultdict(int), {}, defaultdict(int)
    last_advance_wall = {}
    calibration = None
    ready = False
    ever_ready = False
    last_health = time.monotonic()

    def read_packets():
        try:
            while True:
                line = helper.stdout.readline(40 * 1024 * 1024)
                if not line:
                    break
                if not line.endswith(b"\n"):
                    raise ValueError("oversized helper packet")
                packet = json.loads(line)
                kind = packet.get("kind")
                if kind not in ("clock", "odom", "info", "rgb", "depth", "scan"):
                    raise ValueError("unknown helper stream")
                with lock:
                    pending[kind].append(packet["message"])
                    received_counts[kind] += 1
        except Exception as exc:
            errors.append(str(exc))
        finally:
            ended.set()

    reader = threading.Thread(target=read_packets, daemon=True)
    reader.start()

    def command(msg):
        forward, yaw = safe_velocity(msg.linear.x, msg.angular.z)
        try:
            helper.stdin.write(f"{forward:.6f} {yaw:.6f}\n".encode())
            helper.stdin.flush()
        except (BrokenPipeError, OSError):
            ended.set()

    node.create_subscription(Twist, "/cmd_vel", command, 1)

    def ros_stamp(value):
        sec, nsec = stamp_parts(value)
        return Time(sec=sec, nanosec=nsec)

    static = TransformStamped()
    static.header.frame_id, static.child_frame_id = "base_footprint", CAMERA_FRAME
    (static.transform.translation.x, static.transform.translation.y,
     static.transform.translation.z) = extrinsic["translation"]
    (static.transform.rotation.x, static.transform.rotation.y,
     static.transform.rotation.z, static.transform.rotation.w) = extrinsic["quaternion_xyzw"]
    laser_static = TransformStamped()
    laser_static.header.frame_id, laser_static.child_frame_id = "base_footprint", LIDAR_FRAME
    (laser_static.transform.translation.x, laser_static.transform.translation.y,
     laser_static.transform.translation.z) = laser_mount["translation"]
    (laser_static.transform.rotation.x, laser_static.transform.rotation.y,
     laser_static.transform.rotation.z, laser_static.transform.rotation.w) = laser_mount["quaternion_xyzw"]
    static_pub.sendTransform([static, laser_static])

    def make_info(stamp):
        msg = CameraInfo()
        msg.header.frame_id, msg.header.stamp = CAMERA_FRAME, stamp
        for key, value in calibration.items():
            setattr(msg, key, value)
        msg.distortion_model = "plumb_bob"
        return msg

    def handle(kind, value):
        nonlocal calibration
        if kind == "clock":
            clock = Clock()
            clock.clock = ros_stamp(value.get("sim", {}))
            clock_pub.publish(clock)
            tick = clock.clock.sec*1_000_000_000+clock.clock.nanosec
        elif kind == "info":
            calibration = camera_calibration(value)
            tick = stamp_ns(value)
            for publisher in info_pubs:
                publisher.publish(make_info(ros_stamp(value.get("header", {}).get("stamp", {}))))
        elif kind == "scan":
            tick = stamp_ns(value)
            if tick <= last_stamps.get("scan", -1):
                return  # Never replay a delayed older scan into a live costmap.
            fields = laser_scan(value, laser_mount["update_rate_hz"])
            msg = LaserScan()
            msg.header.frame_id = LIDAR_FRAME
            msg.header.stamp = ros_stamp(value.get("header", {}).get("stamp", {}))
            for key, item in fields.items():
                setattr(msg, key, item)
            scan_pub.publish(msg)
        elif kind == "odom":
            msg = Odometry()
            msg.header.stamp = ros_stamp(value.get("header", {}).get("stamp", {}))
            msg.header.frame_id, msg.child_frame_id = "odom", "base_footprint"
            pose = value.get("pose", {})
            for key in ("x", "y", "z"):
                setattr(msg.pose.pose.position, key, float(pose.get("position", {}).get(key, 0.)))
                setattr(msg.twist.twist.linear, key, float(value.get("twist", {}).get("linear", {}).get(key, 0.)))
                setattr(msg.twist.twist.angular, key, float(value.get("twist", {}).get("angular", {}).get(key, 0.)))
            for key in ("x", "y", "z", "w"):
                setattr(msg.pose.pose.orientation, key, float(pose.get("orientation", {}).get(key, 0.)))
            odom_pub.publish(msg)
            tf = TransformStamped()
            tf.header, tf.child_frame_id = msg.header, msg.child_frame_id
            for key in ("x", "y", "z"):
                setattr(tf.transform.translation, key, getattr(msg.pose.pose.position, key))
            tf.transform.rotation = msg.pose.pose.orientation
            tf_pub.sendTransform(tf)
            tick = stamp_ns(value)
        else:
            if calibration is None:
                return
            if (value.get("width"), value.get("height")) != (calibration["width"], calibration["height"]):
                raise ValueError("image dimensions differ from CameraInfo")
            msg = CompressedImage()
            msg.header.frame_id = CAMERA_FRAME
            msg.header.stamp = ros_stamp(value.get("header", {}).get("stamp", {}))
            msg.format, msg.data = (encode_rgb if kind == "rgb" else encode_depth)(
                value,args.sensor_profile,seed=args.noise_seed,robot=args.robot)
            # Republish calibration with the capture timestamp; a late adapter
            # does not depend on having caught the first CameraInfo sample.
            for publisher in info_pubs:
                publisher.publish(make_info(msg.header.stamp))
            (rgb_pub if kind == "rgb" else depth_pub).publish(msg)
            tick = stamp_ns(value)
        counts[kind] += 1
        if tick > last_stamps.get(kind, -1):
            advance[kind] += 1
            last_advance_wall[kind] = time.monotonic()
        last_stamps[kind] = tick

    def health():
        return dict(robot=args.robot, domain=args.domain, ready=ready,
                    counts=dict(counts), stamps_ns=dict(last_stamps), errors=errors[-3:],
                    received_counts=dict(received_counts), published_pairs=published_pairs,
                    pair_dropped_images=pairs.dropped,
                    advancing_age_s={key: round(time.monotonic()-value,3) for key,value in last_advance_wall.items()},
                    pid=os.getpid(), helper_pid=helper.pid, transport="compressed ROS localhost",
                    sensor_profile=args.sensor_profile, noise_seed=args.noise_seed,
                    optical_calibration=extrinsic,
                    lidar_calibration=laser_mount,
                    updated_unix=time.time())

    stop_requested = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.set())
    signal.signal(signal.SIGINT, lambda *_: stop_requested.set())
    try:
        while rclpy.ok() and not ended.is_set() and not stop_requested.is_set():
            with lock:
                batch = {key: list(values) for key, values in pending.items()}
                for values in pending.values():
                    values.clear()
            # /clock precedes the corresponding data; calibration precedes RGB.
            # Retain bounded TF history but never spend a loop encoding a stale
            # backlog of images. Pair by EXACT capture time before encoding.
            for kind in ("clock", "info", "odom", "scan"):
                for value in batch.get(kind, []):
                    try:
                        handle(kind, value)
                    except Exception as exc:
                        if len(errors) < 20:
                            errors.append(f"{kind}: {exc}")
                            print(errors[-1], file=sys.stderr, flush=True)
            for kind in ("rgb", "depth"):
                for value in batch.get(kind, []):
                    pairs.add(kind,value)
            pair = pairs.take_latest() if calibration is not None else None
            if pair is not None:
                try:
                    for kind,value in zip(("rgb","depth"),pair):
                        handle(kind,value)
                        rclpy.spin_once(node,timeout_sec=0.)
                    published_pairs += 1
                except Exception as exc:
                    if len(errors) < 20:
                        errors.append(f"rgbd: {exc}")
                        print(errors[-1],file=sys.stderr,flush=True)
            was_ready = ready
            ready = calibration is not None and all(advance[key] >= 2 and
                time.monotonic()-last_advance_wall.get(key, -1e9) < 5.
                for key in ("clock", "odom", "rgb", "depth", "scan"))
            ever_ready = ever_ready or ready
            if ever_ready and (ready != was_ready or time.monotonic()-last_health >= 5.):
                summary = health()
                if args.ready_file:
                    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
                    temporary = args.ready_file.with_suffix(".tmp")
                    temporary.write_text(json.dumps(summary, indent=2))
                    temporary.replace(args.ready_file)
                if not was_ready:
                    print(json.dumps(summary), flush=True)
                last_health = time.monotonic()
            rclpy.spin_once(node, timeout_sec=.005)
        if ended.is_set() and not stop_requested.is_set():
            raise RuntimeError(f"Gazebo helper stream ended (exit={helper.poll()}, errors={errors[-3:]})")
    finally:
        try:
            helper.stdin.write(b"0 0\n")
            helper.stdin.close()
        except (OSError, BrokenPipeError):
            pass
        try:
            helper.wait(timeout=2.)
        except subprocess.TimeoutExpired:
            helper.terminate()
            try:
                helper.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                helper.kill()
                helper.wait()
        reader.join(timeout=1.)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
