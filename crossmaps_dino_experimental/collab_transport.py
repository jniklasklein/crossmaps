"""Read-only ROS domain adapters and strict RGB-D observation preparation.

Only run_adapter imports ROS. The parent never combines raw robot TF trees.
Each spawned process owns its own ROS context and one bounded output queue.
"""
import hashlib
import json
import math
import queue
import time
import uuid

import cv2
import numpy as np


class PendingPair:
    """Keep one TF-waiting pair and one newest replacement, never a FIFO.

    Replacing the waiting pair at camera rate can starve every timestamped TF
    lookup when transforms consistently arrive slightly after their images.
    """
    def __init__(self, max_age_s, clock=time.monotonic):
        self.max_age_s, self.clock = max_age_s, clock
        self.active = self.latest = None

    def offer(self, rgb, depth):
        self.latest = (rgb, depth, self.clock())

    def take(self):
        if self.active is not None and self.clock()-self.active[2] > self.max_age_s:
            self.active = None
        if self.active is None:
            self.active, self.latest = self.latest, None
        if self.active is not None and self.clock()-self.active[2] > self.max_age_s:
            self.active = None
        return self.active

    def done(self):
        self.active = None


class SensorSequencer:
    """Separate ordinary reordering from a corroborated sensor-clock rollback.

    Inspect only pairs that already pass clock-age/skew checks. A single older
    pair never changes source identity. At least three advancing old-epoch
    stamps over 200 ms are needed for a rollback of at least two seconds.
    Commit occurs only after calibration, depth and timestamped TF validation.
    Rechecking the same TF-waiting pair is not independent corroboration.
    """
    def __init__(self, clock=time.monotonic, rollback_s=2., confirmations=3,
                 window_s=5., min_duration_s=.2):
        self.clock = clock
        self.rollback_ns = int(rollback_s*1e9)
        self.confirmations = int(confirmations)
        self.window_s, self.min_duration_s = float(window_s), float(min_duration_s)
        self.last_stamp = 0
        self.pending = None

    def inspect(self, stamp):
        stamp, now = int(stamp), self.clock()
        if stamp <= 0:
            return dict(accept=False, reset=False, reason='missing_sensor_timestamp')
        if stamp > self.last_stamp:
            return dict(accept=True, reset=False, reason='advancing_timestamp')
        if stamp == self.last_stamp:
            return dict(accept=False, reset=False, reason='duplicate_depth_timestamp')
        if self.last_stamp-stamp < self.rollback_ns:
            return dict(accept=False, reset=False, reason='out_of_order_depth_timestamp')
        pending = self.pending
        if pending is None or now-pending['started'] > self.window_s or stamp < pending['last_stamp']:
            pending = self.pending = dict(started=now, last_stamp=stamp, count=1)
        elif stamp > pending['last_stamp']:
            pending.update(last_stamp=stamp, count=pending['count']+1)
        confirmed = pending['count'] >= self.confirmations and now-pending['started'] >= self.min_duration_s
        return dict(accept=confirmed, reset=confirmed,
                    reason='corroborated_sensor_clock_rollback' if confirmed else 'sensor_clock_rollback_pending',
                    confirmations=pending['count'])

    def commit(self, stamp):
        self.last_stamp = int(stamp)
        self.pending = None


def stamp_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def latest_put(channel, value):
    try:
        channel.put_nowait(value)
        return True
    except queue.Full:
        try:
            channel.get_nowait()
        except queue.Empty:
            return False  # multiprocessing feeder has not exposed its previous item yet
        try:
            channel.put_nowait(value)
            return True
        except queue.Full:
            return False


def decode_rgb(data, compressed=True, encoding="bgr8", width=0, height=0, step=0):
    if compressed:
        rgb = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError("invalid compressed RGB image")
        return rgb
    if encoding not in ("rgb8", "bgr8") or width <= 0 or height <= 0:
        raise ValueError("raw RGB requires rgb8 or bgr8")
    step = int(step or width * 3)
    if step < width * 3 or len(data) != step * height:
        raise ValueError("invalid raw RGB stride or payload")
    rgb = np.frombuffer(data, np.uint8).reshape(height, step)[:, :width*3].reshape(height, width, 3)
    return rgb.copy() if encoding == "bgr8" else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def decode_depth(data, compressed=True, encoding="16UC1", width=0, height=0,
                 step=0, bigendian=False, scale=.001):
    if compressed:
        if "32FC1" in encoding:
            raise ValueError("inverse-quantized 32FC1 compressedDepth is unsupported; use 16UC1 PNG or raw float")
        offset = bytes(data[:32]).find(b"\x89PNG\r\n\x1a\n")
        if offset < 0:
            raise ValueError("compressed depth is not PNG (RVL is unsupported)")
        depth = cv2.imdecode(np.frombuffer(data, np.uint8)[offset:], cv2.IMREAD_UNCHANGED)
        if depth is None or depth.ndim != 2 or depth.dtype != np.uint16:
            raise ValueError("compressed depth must decode to uint16")
        depth = depth.astype(np.float32) * scale
    else:
        types = {"16UC1": "u2", "mono16": "u2", "32FC1": "f4"}
        if encoding not in types or width <= 0 or height <= 0:
            raise ValueError("unsupported raw depth encoding")
        dtype = np.dtype((">" if bigendian else "<") + types[encoding])
        step = int(step or width * dtype.itemsize)
        if step < width*dtype.itemsize or len(data) != step*height:
            raise ValueError("invalid depth stride or payload")
        rows = np.frombuffer(data, np.uint8).reshape(height, step)[:, :width*dtype.itemsize].copy()
        depth = rows.view(dtype).reshape(height, width).astype(np.float32)
        if encoding != "32FC1":
            depth *= scale
    depth[~np.isfinite(depth) | (depth <= 0)] = 0
    return depth


def camera_matrix(info, shape):
    """Rectified topics use P; K is a fallback for zero-P camera info."""
    h, w = shape
    if info.width != w or info.height != h:
        raise ValueError(f"camera calibration {info.width}x{info.height} differs from image {w}x{h}")
    p = np.asarray(getattr(info, "p", []), dtype=np.float64)
    k = p.reshape(3, 4)[:, :3].copy() if p.size == 12 and p[0] > 0 and p[5] > 0 else np.asarray(info.k, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError("invalid camera intrinsics")
    return k


def tf_matrix(message):
    tf = message.transform
    q = np.array([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w], np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-6:
        raise ValueError("invalid TF quaternion")
    x, y, z, w = q / norm
    result = np.eye(4)
    result[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                       [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                       [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    result[:3, 3] = [tf.translation.x, tf.translation.y, tf.translation.z]
    if not np.isfinite(result).all():
        raise ValueError("invalid TF translation")
    return result


def register_depth(depth, k_depth, k_rgb, t_rgb_depth, rgb_shape):
    """Forward projection with a nearest-surface z-buffer; never resize depth."""
    h, w = rgb_shape
    v, u = np.nonzero(np.isfinite(depth) & (depth > 0))
    z = depth[v, u]
    xyz = np.stack(((u-k_depth[0, 2])*z/k_depth[0, 0],
                    (v-k_depth[1, 2])*z/k_depth[1, 1], z), axis=1)
    xyz = xyz @ t_rgb_depth[:3, :3].T + t_rgb_depth[:3, 3]
    valid = xyz[:, 2] > .01
    xyz = xyz[valid]
    u = np.rint(xyz[:, 0]*k_rgb[0, 0]/xyz[:, 2]+k_rgb[0, 2]).astype(int)
    v = np.rint(xyz[:, 1]*k_rgb[1, 1]/xyz[:, 2]+k_rgb[1, 2]).astype(int)
    valid = (u >= 0) & (v >= 0) & (u < w) & (v < h)
    flat = np.full(h*w, np.inf, np.float32)
    np.minimum.at(flat, v[valid]*w+u[valid], xyz[valid, 2])
    flat[~np.isfinite(flat)] = 0
    return flat.reshape(h, w)


def calibration_id(k, shape, rgb_frame, depth_frame, registration, k_depth=None, t_rgb_depth=None):
    fields = [np.asarray(k).round(9).tolist(), list(shape), rgb_frame, depth_frame, registration]
    if k_depth is not None:
        fields.append(np.asarray(k_depth).round(9).tolist())
    if t_rgb_depth is not None:
        fields.append(np.asarray(t_rgb_depth).round(9).tolist())
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:24]


def run_adapter(robot, settings, output, health_output, stop_event):
    """Spawn target. Subscribes only; never publishes to a robot domain."""
    import rclpy
    from rclpy.context import Context
    from rclpy.duration import Duration
    from rclpy.time import Time
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, CompressedImage, CameraInfo
    import message_filters
    import tf2_ros

    context = Context()
    rclpy.init(args=[], context=context, domain_id=int(robot["domain"]))
    node = Node(f"crossmaps_input_{robot['id']}", context=context)
    # Defer paired-frame preparation to a timer so TF can arrive before lookup.
    # Never block this executor in a TF wait or substitute a latest transform.
    buffer = tf2_ros.Buffer(cache_time=Duration(seconds=15))
    listener = tf2_ros.TransformListener(buffer, node, spin_thread=False)
    calibration = {}
    state = dict(robot_id=robot["id"], domain=robot["domain"], session_id=str(uuid.uuid4()),
                 status="waiting_for_camera", received=0, offered=0, rejected=0,
                 last_error=None, last_rgb_monotonic=None, last_depth_monotonic=None,
                 rgb_stamp_ns=None, depth_stamp_ns=None, tf_age_s=None)
    sequencing = SensorSequencer()
    last_offer = 0.
    last_preview = 0.
    pending = PendingPair(settings['max_age_s'])
    seq = 0
    qos = qos_profile_sensor_data
    qos.depth = int(settings["qos_depth"])

    def image_seen(msg, kind):
        nonlocal last_preview
        now = time.monotonic()
        stamp = stamp_ns(msg.header.stamp)
        state[f"last_{kind}_monotonic"] = now
        if state.get(f"{kind}_stamp_ns") != stamp:
            state[f"{kind}_advanced_monotonic"] = now
        state[f"{kind}_stamp_ns"] = stamp
        if kind == 'rgb' and now-last_preview >= .5:
            last_preview = now
            try:
                bgr = decode_rgb(msg.data, robot['rgb_compressed'], getattr(msg, 'encoding', 'bgr8'),
                    getattr(msg, 'width', 0), getattr(msg, 'height', 0), getattr(msg, 'step', 0))
                scale = min(1., 960/max(bgr.shape[:2]))
                if scale < 1:
                    bgr = cv2.resize(bgr, None, fx=scale, fy=scale)
                ok, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 78])
                if ok:
                    state['preview_jpeg'], state['preview_monotonic'] = jpeg.tobytes(), now
            except Exception:
                pass  # Preparation reports decoding failures; preview is optional.

    def prepare(rgb_msg, depth_msg, received):
        nonlocal last_offer, seq
        now = time.monotonic()
        if now-last_offer < 1./float(settings["max_hz"]):
            return None
        try:
            rgb_stamp, depth_stamp = stamp_ns(rgb_msg.header.stamp), stamp_ns(depth_msg.header.stamp)
            if min(rgb_stamp, depth_stamp) <= 0:
                raise ValueError("missing sensor timestamps")
            if abs(rgb_stamp-depth_stamp) > settings["sync_slop_s"]*1e9:
                raise ValueError("RGB/depth timestamp skew exceeds configured tolerance")
            age = (node.get_clock().now().nanoseconds-depth_stamp)/1e9
            state["sensor_clock_age_s"] = age
            if age > settings["max_age_s"] or age < -settings["max_age_s"]:
                raise ValueError("sensor clock skew/staleness: synchronize rover and laptop clocks")
            decision = sequencing.inspect(depth_stamp)
            state['timestamp_status'] = decision['reason']
            if not decision['accept']:
                raise ValueError(decision['reason'].replace('_', ' '))
            rgb = decode_rgb(rgb_msg.data, robot["rgb_compressed"], getattr(rgb_msg, "encoding", "bgr8"),
                             getattr(rgb_msg, "width", 0), getattr(rgb_msg, "height", 0), getattr(rgb_msg, "step", 0))
            depth = decode_depth(depth_msg.data, robot["depth_compressed"],
                                 getattr(depth_msg, "format", getattr(depth_msg, "encoding", "16UC1")),
                                 getattr(depth_msg, "width", 0), getattr(depth_msg, "height", 0),
                                 getattr(depth_msg, "step", 0), getattr(depth_msg, "is_bigendian", False), robot["depth_scale"])
            if max(rgb.shape[0]*rgb.shape[1], depth.size) > settings["max_pixels"]:
                raise ValueError("image exceeds configured pixel limit")
            if "rgb" not in calibration:
                raise ValueError("waiting for RGB CameraInfo")
            k = camera_matrix(calibration["rgb"], rgb.shape[:2])
            rgb_frame = rgb_msg.header.frame_id
            depth_frame = depth_msg.header.frame_id
            if not rgb_frame or not depth_frame or calibration["rgb"].header.frame_id != rgb_frame:
                raise ValueError("RGB image and calibration frame IDs must agree")
            mode = robot["depth_registration"]
            registration = "registered_explicit" if mode == "registered" else "same_optical_frame"
            kd = extrinsic = None
            if mode == "project" or (mode == "auto" and depth_frame != rgb_frame):
                if "depth" not in calibration:
                    raise ValueError("different optical frames: waiting for depth CameraInfo to register depth")
                if calibration["depth"].header.frame_id != depth_frame:
                    raise ValueError("depth image and calibration frame IDs must agree")
                kd = camera_matrix(calibration["depth"], depth.shape)
                extrinsic = tf_matrix(buffer.lookup_transform(rgb_frame, depth_frame,
                    Time.from_msg(depth_msg.header.stamp), timeout=Duration(seconds=0)))
                depth = register_depth(depth, kd, k, extrinsic, rgb.shape[:2])
                registration = "projected_with_calibration"
            if depth.shape != rgb.shape[:2]:
                raise ValueError("registered depth and RGB must share the same pixel grid")
            pose = tf_matrix(buffer.lookup_transform(robot["local_frame"], rgb_frame,
                Time.from_msg(depth_msg.header.stamp), timeout=Duration(seconds=0)))
            depth[(depth < settings["min_depth_m"]) | (depth > settings["max_depth_m"])] = 0
            if np.count_nonzero(depth) < 80:
                raise ValueError("insufficient valid registered depth")
            if decision['reset']:
                state['session_id'] = str(uuid.uuid4())
                state['session_reset_reason'] = decision['reason']
                state['session_resets'] = state.get('session_resets', 0)+1
                seq = 0
            seq += 1
            packet = dict(robot_id=robot["id"], session_id=state["session_id"], seq=seq,
                          stamp_ns=depth_stamp, rgb_stamp_ns=rgb_stamp, received_monotonic=received,
                          rgb=rgb, depth=depth, K=k, T_local_camera=pose,
                          calibration_id=calibration_id(k, depth.shape, rgb_frame, depth_frame, registration, kd, extrinsic),
                          camera_frame=rgb_frame, local_frame=robot["local_frame"],
                          pose_quality="external_tf_not_validated", depth_registration=registration)
            if latest_put(output, packet):
                state["offered"] += 1
            sequencing.commit(depth_stamp)
            state.update(status="receiving", last_error=None, last_packet_monotonic=now,
                         depth_registration=registration, calibration_id=packet["calibration_id"])
            last_offer = now
            return True
        except tf2_ros.TransformException as error:
            state.update(status='waiting_for_tf', last_error=f'Timestamped TF unavailable: {error}'[:350])
            return None
        except Exception as error:
            state.update(status="input_rejected", last_error=str(error)[:350])
            state["rejected"] += 1
            last_offer = now
            return False

    def receive_pair(rgb_msg, depth_msg):
        state['received'] += 1
        pending.offer(rgb_msg, depth_msg)

    def process_pending():
        # Don't pin a pair during rate limiting: choose the freshest pair when
        # capacity is available, then retain it only while TF catches up.
        if time.monotonic()-last_offer < 1./float(settings['max_hz']):
            return
        pair = pending.take()
        if pair is not None and prepare(*pair) is not None:
            pending.done()

    subscriptions = [node.create_subscription(CameraInfo, robot["camera_info_topic"],
        lambda msg: calibration.__setitem__("rgb", msg), qos),
        node.create_subscription(CameraInfo, robot["depth_camera_info_topic"],
        lambda msg: calibration.__setitem__("depth", msg), qos)]
    rgb = message_filters.Subscriber(node, CompressedImage if robot["rgb_compressed"] else Image, robot["rgb_topic"], qos_profile=qos)
    depth = message_filters.Subscriber(node, CompressedImage if robot["depth_compressed"] else Image, robot["depth_topic"], qos_profile=qos)
    rgb.registerCallback(image_seen, "rgb")
    depth.registerCallback(image_seen, "depth")
    sync = message_filters.ApproximateTimeSynchronizer([rgb, depth], 10, settings["sync_slop_s"], allow_headerless=False)
    sync.registerCallback(receive_pair)
    prepare_timer = node.create_timer(max(.01, min(.05, float(settings['tf_timeout_s']) or .05)), process_pending)
    timer = node.create_timer(1., lambda: latest_put(health_output, dict(state, reported_monotonic=time.monotonic())))
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    try:
        while not stop_event.is_set() and context.ok():
            executor.spin_once(timeout_sec=.1)
    except KeyboardInterrupt:
        pass
    finally:
        listener.unregister()
        executor.shutdown()
        node.destroy_node()
        context.try_shutdown()
        # Do not hang on a feeder flushing a frame after the parent has exited.
        output.cancel_join_thread()
        health_output.cancel_join_thread()
