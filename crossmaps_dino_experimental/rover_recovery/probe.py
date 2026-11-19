#!/usr/bin/env python3
"""Read-only, rover-local ROS 2 sensor and SLAM probe. Prints a JSON report.

Run in a sourced SLAM container with its ROS_DOMAIN_ID. No raw camera, TF, map
cloud, or graph subscriptions, and no motion commands or ROS services invoked.
Default exit status checks the three primary sensor streams; to check the full
pipeline, pass --require scan rgb depth odom rf2o info.
"""

import argparse
import json
import math
import os
import time


BASE = {
    "scan": ("/scan", "sensor_msgs/msg/LaserScan"),
    "rgb": ("/oak/rgb/image_rect/compressed", "sensor_msgs/msg/CompressedImage"),
    "depth": ("/oak/stereo/image_raw/compressedDepth", "sensor_msgs/msg/CompressedImage"),
    "odom": ("/odom", "nav_msgs/msg/Odometry"),
    "rf2o": ("/odom_rf2o", "nav_msgs/msg/Odometry"),
}
INFO = "rtabmap_msgs/msg/Info"
TWIST = "geometry_msgs/msg/Twist"
STAMPED_TWIST = "geometry_msgs/msg/TwistStamped"


class Stream:
    def __init__(self):
        self.stamps = set()
        self.received = self.advanced = self.regressions = self.nonzero = 0
        self.previous_ns = self.maximum_ns = self.last_advance = None

    def observe(self, message, now, cmd=False):
        self.received += 1
        if cmd:
            vel = getattr(message, "twist", message)
            entries = [getattr(v, axis) for v in (vel.linear, vel.angular) for axis in "xyz"]
            if any(abs(v) > 1e-5 for v in entries):
                self.nonzero += 1
        header = getattr(message, "header", None)
        if header is None:
            return
        ns = header.stamp.sec * 10**9 + header.stamp.nanosec
        if ns <= 0:
            return
        self.regressions += int(self.previous_ns is not None and ns < self.previous_ns)
        self.previous_ns = ns
        self.stamps.add(ns)
        if self.maximum_ns is None or ns > self.maximum_ns:
            self.maximum_ns = ns
            self.advanced += 1
            self.last_advance = now

    def report(self, now, age_limit):
        age = None if self.last_advance is None else now - self.last_advance
        return dict(
            messages=self.received, distinct_stamps=len(self.stamps),
            advancing_stamps=self.advanced, backwards_steps=self.regressions,
            last_source_stamp_ns=self.maximum_ns,
            last_advance_age_s=None if age is None else round(age, 2),
            fresh=self.advanced >= 2 and age is not None and age <= age_limit,
        )


def run(args):
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rosidl_runtime_py.utilities import get_message

    rclpy.init(args=[])
    node = rclpy.create_node(
        "crossmaps_health_probe_%d" % os.getpid(), start_parameter_services=False, enable_rosout=False
    )
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    streams, specs, subs = {}, {}, []
    errors = {}

    def subscribe(key, topic, msgtype):
        if key in streams or key in errors:
            return
        s = Stream()
        try:
            msgclass = get_message(msgtype)
            sub = node.create_subscription(
                msgclass, topic, lambda msg: s.observe(msg, time.monotonic(), key == "cmd_vel"), qos
            )
        except (ImportError, AttributeError, ValueError, RuntimeError) as exc:
            errors[key] = str(exc)
            return
        streams[key] = s
        specs[key] = (topic, msgtype)
        subs.append(sub)

    try:
        for key, (topic, msgtype) in BASE.items():
            subscribe(key, topic, msgtype)
        # /info is the usual un-namespaced RTAB-Map status output. Discover
        # alternate namespaces, but don't mistake another message type for Info.
        if args.info_topic:
            subscribe("info", args.info_topic, INFO)
        start = time.monotonic()
        end = start + args.seconds
        next_discovery = start
        candidates = []
        while rclpy.ok() and (now := time.monotonic()) < end:
            if now >= next_discovery:
                topics = dict(node.get_topic_names_and_types())
                candidates = sorted(t for t, types in topics.items() if INFO in types)
                if not args.info_topic and candidates:
                    if "/info" in candidates:
                        subscribe("info", "/info", INFO)
                    elif len(candidates) == 1:
                        subscribe("info", candidates[0], INFO)
                types = topics.get(args.cmd_topic, [])
                for cls in [TWIST, STAMPED_TWIST]:
                    if cls in types:
                        subscribe("cmd_vel", args.cmd_topic, cls)
                        break
                next_discovery = now + 1.0
            rclpy.spin_once(node, timeout_sec=min(.15, max(0., end - time.monotonic())))
        now = time.monotonic()
        result = {}
        for key, stream in streams.items():
            topic, msgtype = specs[key]
            row = stream.report(now, args.max_advance_age)
            row.update(topic=topic, msgtype=msgtype, publishers=node.count_publishers(topic))
            if key == "cmd_vel":
                row.pop("fresh")
                row["nonzero_commands_observed"] = stream.nonzero
            result[key] = row
        missing = [key for key in args.require if not result.get(key, {}).get("fresh", False)]
        print(json.dumps({
            "domain": os.environ.get("ROS_DOMAIN_ID", "0"), "sample_s": round(now - start, 2),
            "streams": result, "discovered_info_topics": candidates, "subscribe_errors": errors,
            "unhealthy_required": missing, "ok": not missing,
            "notes": [
                "A fresh stream advanced its source timestamp at least twice; age is measured locally since last advance.",
                "RTAB-Map Info is update-driven; no Info is not proof the process exited.",
                "No observed velocity commands does not establish that the rover is stationary.",
            ],
        }, indent=2))
        return 1 if missing else 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=float, default=12., help="sample window including ROS discovery")
    p.add_argument("--max-advance-age", type=float, default=6., help="maximum local seconds since last advancing stamp")
    p.add_argument("--info-topic", help="explicit RTAB-Map Info topic (normally discovered)")
    p.add_argument("--cmd-topic", default="/cmd_vel")
    p.add_argument("--require", nargs="+", choices=list(BASE) + ["info"], default=["scan", "rgb", "depth"])
    args = p.parse_args()
    if not all(math.isfinite(x) and x > 0 for x in (args.seconds, args.max_advance_age)):
        p.error("timing arguments must be positive and finite")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
