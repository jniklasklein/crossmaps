#!/usr/bin/env python3
"""Read-only end-to-end check of simulated compressed sensors, time and TF."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from run import apply_environment, environments, ros_python


def rate_sample(rate, stamp_ns, wall):
    """Count advancing acquisitions, not duplicate ROS callbacks."""
    if not rate:
        rate.update(first_stamp=stamp_ns, last_stamp=stamp_ns, first_wall=wall, last_wall=wall, frames=1)
    elif stamp_ns > rate['last_stamp']:
        rate.update(last_stamp=stamp_ns, last_wall=wall, frames=rate['frames']+1)


def rate_report(rate):
    wall = rate.get('last_wall', 0.) - rate.get('first_wall', 0.)
    sim = (rate.get('last_stamp', 0) - rate.get('first_stamp', 0)) / 1e9
    intervals = max(0, rate.get('frames', 0)-1)
    return dict(advancing_frames=rate.get('frames', 0),
                hz_wall=intervals / wall if wall > 0 else None,
                hz_simulation=intervals / sim if sim > 0 else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=20.)
    parser.add_argument("--output", type=Path, help="Optional report and sample RGB images directory")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or not 0 < args.duration <= 600:
        parser.error('--duration must be finite, positive and at most 600 seconds')
    python = ros_python()
    if Path(sys.executable).resolve() != Path(python).resolve():
        os.execv(python, [python, str(Path(__file__).resolve()), *sys.argv[1:]])
    _, env = environments(python, Path(__file__).parent / ".runtime/smoke")
    apply_environment(env)
    import cv2
    import numpy as np
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import CompressedImage, CameraInfo
    from rosgraph_msgs.msg import Clock
    from tf2_ros import Buffer, TransformListener
    channels = []

    def ns(stamp):
        return stamp.sec * 1000000000 + stamp.nanosec

    def image(source, kind, msg):
        rate_sample(source['rates'][kind], ns(msg.header.stamp), time.monotonic())
        source[kind][ns(msg.header.stamp)] = msg
        if len(source[kind]) > 12:
            del source[kind][min(source[kind])]
        source[kind + "_count"] += 1

    try:
        for robot, domain in (("rover_a", 80), ("rover_b", 81)):
            context = Context()
            rclpy.init(context=context, domain_id=domain)
            node = Node("crossmaps_sim_probe_" + robot, context=context)
            executor = SingleThreadedExecutor(context=context)
            executor.add_node(node)
            source = dict(robot=robot, context=context, node=node, executor=executor, rgb={}, depth={},
                          rgb_count=0, depth_count=0, clock=0, clock_first=None, clock_first_wall=None, clock_last_wall=None, info=None,
                          rates={'rgb': {}, 'depth': {}}, checked=set(), tf_pairs=0, depth_valid=0., rgb_std=0., sample={})
            source["buffer"] = Buffer()
            source["listener"] = TransformListener(source["buffer"], node)
            for kind, topic in (("rgb", "/oak/rgb/image_rect/compressed"), ("depth", "/oak/stereo/image_raw/compressedDepth")):
                node.create_subscription(CompressedImage, topic, lambda msg, s=source, k=kind: image(s, k, msg), qos_profile_sensor_data)
            node.create_subscription(CameraInfo, "/oak/rgb/camera_info", lambda msg, s=source: s.update(info=msg), qos_profile_sensor_data)

            def clock(msg, s=source):
                s["clock"] = ns(msg.clock)
                s['clock_last_wall'] = time.monotonic()
                if s["clock_first"] is None:
                    s["clock_first"] = s["clock"]
                    s['clock_first_wall'] = s['clock_last_wall']

            node.create_subscription(Clock, "/clock", clock, qos_profile_sensor_data)
            channels.append(source)
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            for source in channels:
                source["executor"].spin_once(timeout_sec=.01)
                for stamp in set(source["rgb"]) & set(source["depth"]) - source["checked"]:
                    try:
                        pose = source["buffer"].lookup_transform("odom", "oak_rgb_camera_optical_frame", Time.from_msg(source['rgb'][stamp].header.stamp))
                    except Exception:
                        continue
                    rgb = cv2.imdecode(np.frombuffer(bytes(source["rgb"][stamp].data), dtype=np.uint8), cv2.IMREAD_COLOR)
                    depth_data = bytes(source["depth"][stamp].data)
                    depth = cv2.imdecode(np.frombuffer(depth_data[12:], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    info = source["info"]
                    if rgb is None or depth is None or info is None:
                        continue
                    if (depth.dtype != np.uint16 or rgb.shape[:2] != depth.shape or rgb.shape[:2] != (info.height, info.width)
                            or not all(math.isfinite(value) for value in info.k) or info.k[0] <= 0 or info.k[4] <= 0):
                        raise RuntimeError(f"{source['robot']}: calibration/image size mismatch")
                    source["checked"].add(stamp)
                    source["tf_pairs"] += 1
                    source["depth_valid"] = float(np.mean(depth > 0))
                    source["rgb_std"] = float(rgb.std())
                    source["last_tf"] = [pose.transform.translation.x, pose.transform.translation.y, pose.transform.translation.z]
                    valid_depth = depth[depth > 0].astype(np.float32) * .001
                    source['sample'] = dict(
                        image_width=int(info.width), image_height=int(info.height),
                        camera_matrix=list(info.k), camera_frame=info.header.frame_id,
                        horizontal_fov_deg=math.degrees(2 * math.atan(info.width / (2 * info.k[0]))),
                        depth_encoding=source['depth'][stamp].format,
                        depth_range_m=({key: float(value) for key, value in zip(('min', 'median', 'p95', 'max'),
                                        np.percentile(valid_depth, [0, 50, 95, 100]))} if valid_depth.size else None),
                        capture_stamp_ns=stamp, capture_time_tf_xyz=source['last_tf'])
                    if args.output and source["tf_pairs"] == 1:
                        args.output.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(args.output / (source["robot"] + ".jpg")), rgb)
        report = {}
        for source in channels:
            elapsed = (source["clock"] - (source["clock_first"] or 0)) / 1e9
            wall_elapsed = (source['clock_last_wall'] or 0) - (source['clock_first_wall'] or 0)
            report[source["robot"]] = {key: source[key] for key in ("rgb_count", "depth_count", "tf_pairs", "depth_valid", "rgb_std")}
            report[source["robot"]].update(sim_elapsed_s=elapsed, wall_elapsed_s=wall_elapsed,
                real_time_factor=elapsed / wall_elapsed if wall_elapsed > 0 else None,
                rgb_rate=rate_report(source['rates']['rgb']), depth_rate=rate_report(source['rates']['depth']),
                sample=source['sample'],
                ok=bool(source["tf_pairs"] >= 3 and source["depth_valid"] > .05 and source["rgb_std"] > 5 and elapsed > 1))
        print(json.dumps(report, indent=2))
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return 0 if all(item["ok"] for item in report.values()) else 1
    finally:
        for source in channels:
            source["executor"].shutdown()
            source["node"].destroy_node()
            source["context"].try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
