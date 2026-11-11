"""Bounded, read-only ROS camera transport probe; no models, files or settings.

Run from the ROS environment: python3 -B camera_transport_probe.py --seconds 30
Uses compressed topics by default; --raw compares the raw stream instead.
"""
import argparse
import json
import time


def main():
    import cv2
    import numpy as np
    import rclpy
    import message_filters
    from sensor_msgs.msg import Image, CompressedImage, CameraInfo
    from rclpy.qos import qos_profile_sensor_data

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--raw', action='store_true')
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node('crossmaps_camera_transport_probe')
    counts = dict(rgb=0, depth=0, camera_info=0, pairs=0)
    sizes = dict(rgb=0, depth=0)
    formats = {}
    received = {}
    decoded = {}

    def note(msg, kind):
        counts[kind] += 1
        received[kind] = time.monotonic()
        if kind in sizes:
            sizes[kind] += len(msg.data)
            formats[kind] = getattr(msg, 'format', getattr(msg, 'encoding', ''))
            if not args.raw and kind not in decoded:
                data = np.frombuffer(msg.data, dtype=np.uint8)
                for offset in ((0,) if kind == 'rgb' else (12, 0, 16)):
                    image = cv2.imdecode(data[offset:], cv2.IMREAD_UNCHANGED)
                    if image is not None:
                        decoded[kind] = dict(shape=list(image.shape), dtype=str(image.dtype))
                        break

    rgb = '/oak/rgb/image_rect' + ('' if args.raw else '/compressed')
    depth = '/oak/stereo/image_raw' + ('' if args.raw else '/compressedDepth')
    cls = Image if args.raw else CompressedImage
    rgb_sub = message_filters.Subscriber(node, cls, rgb, qos_profile=qos_profile_sensor_data)
    depth_sub = message_filters.Subscriber(node, cls, depth, qos_profile=qos_profile_sensor_data)
    rgb_sub.registerCallback(note, 'rgb')
    depth_sub.registerCallback(note, 'depth')
    info_sub = node.create_subscription(CameraInfo, '/oak/rgb/camera_info',
        lambda msg: note(msg, 'camera_info'), qos_profile_sensor_data)
    sync = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 20, .35)
    def pair(*_):
        counts['pairs'] += 1
        received['pairs'] = time.monotonic()
    sync.registerCallback(pair)
    started = time.monotonic()
    next_report = started + 5.
    try:
        while time.monotonic()-started < max(1., min(args.seconds, 120.)):
            rclpy.spin_once(node, timeout_sec=.1)
            now = time.monotonic()
            if now >= next_report:
                print(json.dumps(dict(elapsed_s=round(now-started, 1), counts=counts,
                    age_s={k:round(now-v, 2) for k,v in received.items()},
                    bytes=sizes, formats=formats, decoded=decoded)), flush=True)
                next_report = now + 5.
        print(json.dumps(dict(final_counts=counts, topics=[rgb, depth], decoded=decoded)), flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
