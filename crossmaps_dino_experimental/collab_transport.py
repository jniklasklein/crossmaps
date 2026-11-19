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
        self.expired = 0
        self.last_expired = None

    def offer(self, rgb, depth, metadata=None):
        pair = (rgb, depth, self.clock())
        self.latest = pair if metadata is None else pair+(metadata,)

    def take(self):
        if self.active is not None and self.clock()-self.active[2] > self.max_age_s:
            self.expired += 1
            self.last_expired = self.active
            self.active = None
        if self.active is None:
            self.active, self.latest = self.latest, None
        if self.active is not None and self.clock()-self.active[2] > self.max_age_s:
            self.expired += 1
            self.last_expired = self.active
            self.active = None
        return self.active

    def done(self):
        self.active = None

    def clear(self):
        self.active = self.latest = self.last_expired = None


class SensorSequencer:
    """Separate ordinary reordering from a corroborated sensor-clock rollback.

    Inspect only pairs that already pass clock-age/skew checks. A single older
    pair never changes source identity. At least three advancing old-epoch
    stamps over 200 ms are needed for a rollback of at least two seconds.
    Commit occurs only after calibration and depth validation. Diagnostic-only
    previews can advance source sequencing but never become map evidence; their
    later timestamped-TF completion is explicitly tracked by the active pair.
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


class RGBReaderWatch:
    """Bound re-subscription to a single stalled RGB reader, not general robot outages."""
    def __init__(self, max_age_s, clock=time.monotonic, stale_s=12., retry_s=30.):
        self.clock, self.max_age_s = clock, float(max_age_s)
        self.stale_s, self.retry_s = stale_s, retry_s
        self.progress = clock()
        self.last_attempt = self.last_recovery = self.last_error = None
        self.attempts = self.recoveries = 0
        self.awaiting = False
        self.sequence = SensorSequencer(clock)

    def observe(self, stamp, arrival_age):
        """'advance', 'unchanged' or 'drop'; never refresh a cached JPEG on rebind."""
        if self.awaiting and (not math.isfinite(arrival_age) or abs(arrival_age) > self.max_age_s):
            return 'drop'
        if not self.sequence.inspect(stamp)['accept']:
            # Outside a rebind, preserve normal pair-level clock epoch handling.
            return 'drop' if self.awaiting else 'unchanged'
        self.sequence.commit(stamp)
        self.progress = self.clock()
        if self.awaiting:
            self.recoveries += 1
            self.last_recovery = self.progress
            self.awaiting = False
        return 'advance'

    def poll(self, depth_advanced, depth_header_age, rebind):
        now = self.clock()
        if (depth_advanced is None or now-depth_advanced > 3. or depth_header_age is None
                or abs(depth_header_age) > self.max_age_s or now-self.progress < self.stale_s
                or (self.last_attempt is not None and now-self.last_attempt < self.retry_s)):
            return
        self.attempts += 1
        self.awaiting = True
        self.last_attempt = now
        try:
            rebind()
            self.last_error = None
        except Exception as error:
            # Leave the node and other inputs running; retry after backoff.
            self.last_error = str(error)[:250]

    def health(self):
        result = dict(rgb_reader_attempts=self.attempts, rgb_reader_recoveries=self.recoveries,
                      rgb_reader_awaiting_frame=self.awaiting,
                      rgb_reader_last_attempt_monotonic=self.last_attempt,
                      rgb_reader_last_recovery_monotonic=self.last_recovery,
                      rgb_reader_last_error=self.last_error)
        if self.awaiting:
            result.update(status='rgb_stale', timing_status='awaiting_fresh_compressed_rgb',
                          last_error=('Compressed RGB reader rebind failed: '+self.last_error if self.last_error
                                      else 'Compressed RGB stopped advancing; waiting for a fresh source frame after reader rebind'))
        return result


class RebindableRGB:
    """Single-executor lifetime of a compressed RGB reader + per-generation RGB-D queues.

    The persistent depth subscription feeds only the current synchronizer.
    Retired RGB callbacks cannot update previews or enqueue old depth/RGB pairs.
    """
    def __init__(self, subscribe, destroy, make_filters, observe, clear_pending):
        self.subscribe, self.destroy = subscribe, destroy
        self.make_filters, self.observe, self.clear_pending = make_filters, observe, clear_pending
        self.generation = 0
        self.rgb_sub = self.depth_sink = None

    def close(self):
        self.generation += 1
        self.depth_sink = None
        self.clear_pending()
        if self.rgb_sub is not None:
            if self.destroy(self.rgb_sub) is False:
                raise RuntimeError('Could not retire previous compressed RGB subscription')
            self.rgb_sub = None

    def rebind(self):
        self.close()
        generation = self.generation
        rgb_sink, depth_sink = self.make_filters()

        def rgb(msg):
            if generation == self.generation and self.observe(msg, 'rgb'):
                rgb_sink(msg)
        self.rgb_sub = self.subscribe(rgb)
        self.depth_sink = depth_sink

    def depth(self, msg):
        if self.observe(msg, 'depth') and self.depth_sink is not None:
            self.depth_sink(msg)


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


def preview_put(channel, value):
    """Diagnostic work must never displace a queued mapping observation."""
    try:
        channel.put_nowait(value)
        return True
    except queue.Full:
        return False


def require_compressed_transport(robot):
    if robot.get('rgb_compressed', True) is not True or robot.get('depth_compressed', True) is not True:
        raise ValueError('Live collaborative adapters require compressed RGB and compressedDepth transport')


def pair_timing(depth_stamp_ns, received_wall_ns, received_monotonic, now_monotonic, max_age_s):
    """Distinguish acquisition/transport header age from adapter TF waiting.

    Header age at receipt includes transport latency AND possible clock offset;
    it is not a clock-offset measurement. Waiting is measured monotonically and
    cannot grow that arrival-age measurement or turn it into a clock error.
    """
    arrival = (int(received_wall_ns)-int(depth_stamp_ns))/1e9
    wait = max(0., float(now_monotonic)-float(received_monotonic))
    if arrival > max_age_s:
        status = 'frame_stale_on_arrival'
    elif arrival < -max_age_s:
        status = 'sensor_timestamp_ahead_of_receiver'
    elif max(0., arrival)+wait > max_age_s:
        status = 'timestamped_tf_wait_expired'
    else:
        status = 'within_latency_budget'
    return dict(timing_status=status, selected_frame_age_at_receipt_s=arrival,
                pending_wait_s=wait, selected_frame_age_s=arrival+wait)


def cached_rgbd(metadata, prepare):
    """At most one decoded/registered RGB-D payload per bounded active pair."""
    if metadata.get('rgbd') is None:
        metadata['rgbd'] = prepare()
    return metadata['rgbd']


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


def depth_preview_jpeg(depth, max_depth_m=8., max_side=960):
    """Acquisition diagnostic in the depth camera's own grid, not registered RGB."""
    valid = np.isfinite(depth) & (depth > 0)
    gray = np.clip(255.*(1.-np.where(valid, depth, max_depth_m)/max_depth_m), 0, 255).astype(np.uint8)
    image = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    image[~valid] = 0
    scale = min(1., max_side/max(image.shape[:2]))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale)
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        raise ValueError('Could not encode depth diagnostic')
    return encoded.tobytes()


def prepare_registered_rgbd(rgb_msg, depth_msg, robot, settings, calibration, lookup_extrinsic):
    """Prepare calibrated pixels independently of odometry; never guess extrinsics.

    lookup_extrinsic(target_frame, source_frame) must use this pair's depth stamp.
    Missing registration TF raises before expensive decoding and cannot yield a
    model-preview packet. The separate acquisition depth image remains available.
    """
    require_compressed_transport(robot)
    rgb_frame, depth_frame = rgb_msg.header.frame_id, depth_msg.header.frame_id
    if 'rgb' not in calibration:
        raise ValueError('waiting for RGB CameraInfo')
    if not rgb_frame or not depth_frame or calibration['rgb'].header.frame_id != rgb_frame:
        raise ValueError('RGB image and calibration frame IDs must agree')
    mode = robot['depth_registration']
    projecting = mode == 'project' or (mode == 'auto' and depth_frame != rgb_frame)
    extrinsic = None
    if projecting:
        if 'depth' not in calibration:
            raise ValueError('different optical frames: waiting for depth CameraInfo to register depth')
        if calibration['depth'].header.frame_id != depth_frame:
            raise ValueError('depth image and calibration frame IDs must agree')
        extrinsic = lookup_extrinsic(rgb_frame, depth_frame)
    rgb = decode_rgb(rgb_msg.data)
    depth = decode_depth(depth_msg.data, encoding=getattr(depth_msg, 'format', '16UC1'), scale=robot['depth_scale'])
    if max(rgb.shape[0]*rgb.shape[1], depth.size) > settings['max_pixels']:
        raise ValueError('image exceeds configured pixel limit')
    k = camera_matrix(calibration['rgb'], rgb.shape[:2])
    kd = None
    registration = 'registered_explicit' if mode == 'registered' else 'same_optical_frame'
    if projecting:
        kd = camera_matrix(calibration['depth'], depth.shape)
        depth = register_depth(depth, kd, k, extrinsic, rgb.shape[:2])
        registration = 'projected_with_calibration'
    if depth.shape != rgb.shape[:2]:
        raise ValueError('registered depth and RGB must share the same pixel grid')
    depth[(depth < settings['min_depth_m']) | (depth > settings['max_depth_m'])] = 0
    if np.count_nonzero(depth) < 80:
        raise ValueError('insufficient valid registered depth')
    return dict(rgb=rgb, depth=depth, K=k,
                calibration_id=calibration_id(k, depth.shape, rgb_frame, depth_frame, registration, kd, extrinsic),
                camera_frame=rgb_frame, local_frame=robot['local_frame'], depth_registration=registration)


def run_adapter(robot, settings, output, health_output, stop_event):
    """Spawn target. Subscribes only; never publishes to a robot domain."""
    require_compressed_transport(robot)
    import rclpy
    from rclpy.context import Context
    from rclpy.duration import Duration
    from rclpy.time import Time
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage, CameraInfo
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
    last_preview = dict(rgb=0., depth=0.)
    last_model_preview = 0.
    last_tf_diagnostic = 0.
    pending = PendingPair(settings['max_age_s'])
    rgb_watch = RGBReaderWatch(settings['max_age_s'])
    seq = 0
    qos = qos_profile_sensor_data
    qos.depth = int(settings["qos_depth"])

    def image_seen(msg, kind):
        now = time.monotonic()
        stamp = stamp_ns(msg.header.stamp)
        state[f"last_{kind}_monotonic"] = now
        arrival_age = (node.get_clock().now().nanoseconds-stamp)/1e9
        if kind == 'rgb':
            progress = rgb_watch.observe(stamp, arrival_age)
            if progress != 'advance':
                return progress != 'drop'
        if state.get(f"{kind}_stamp_ns") != stamp:
            state[f"{kind}_advanced_monotonic"] = now
        state[f"{kind}_stamp_ns"] = stamp
        state[f'{kind}_header_age_at_receipt_s'] = arrival_age
        if now-last_preview[kind] >= .5:
            last_preview[kind] = now
            try:
                if kind == 'rgb':
                    bgr = decode_rgb(msg.data)
                    scale = min(1., 960/max(bgr.shape[:2]))
                    if scale < 1:
                        bgr = cv2.resize(bgr, None, fx=scale, fy=scale)
                    ok, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 78])
                    if ok:
                        state['preview_jpeg'], state['preview_monotonic'] = jpeg.tobytes(), state['rgb_advanced_monotonic']
                        state['preview_stamp_ns'] = stamp
                else:
                    depth = decode_depth(msg.data, encoding=msg.format, scale=robot['depth_scale'])
                    state['depth_preview_jpeg'] = depth_preview_jpeg(depth, settings['max_depth_m'])
                    state['depth_preview_monotonic'], state['depth_preview_stamp_ns'] = state['depth_advanced_monotonic'], stamp
                    state['depth_preview_registration'] = 'depth_camera_native_grid'
                state.pop(f'{kind}_preview_error', None)
            except Exception as error:
                state[f'{kind}_preview_error'] = str(error)[:250]
        return True

    def tf_wait(error, rgb_frame, depth_stamp, wait):
        nonlocal last_tf_diagnostic
        state.update(status='waiting_for_tf', last_error=f'Timestamped TF unavailable: {error}'[:350],
                     pending_wait_s=wait)
        now = time.monotonic()
        if now-last_tf_diagnostic >= .5:
            last_tf_diagnostic = now
            try:
                # Latest TF is diagnostics ONLY, never a mapping/preview pose.
                latest = buffer.lookup_transform(robot['local_frame'], rgb_frame, Time(), timeout=Duration(seconds=0))
                latest_stamp = stamp_ns(latest.header.stamp)
                state['tf_latest_stamp_ns'] = latest_stamp or None
                state['tf_coverage_lag_s'] = (depth_stamp-latest_stamp)/1e9 if latest_stamp else None
                state['tf_age_s'] = (node.get_clock().now().nanoseconds-latest_stamp)/1e9 if latest_stamp else None
            except tf2_ros.TransformException:
                state['tf_latest_stamp_ns'] = state['tf_coverage_lag_s'] = state['tf_age_s'] = None

    def prepare(rgb_msg, depth_msg, received, metadata):
        nonlocal last_offer, last_model_preview, seq
        now = time.monotonic()
        if now-last_offer < 1./float(settings["max_hz"]):
            return None
        try:
            rgb_stamp, depth_stamp = stamp_ns(rgb_msg.header.stamp), stamp_ns(depth_msg.header.stamp)
            if min(rgb_stamp, depth_stamp) <= 0:
                raise ValueError("missing sensor timestamps")
            if abs(rgb_stamp-depth_stamp) > settings["sync_slop_s"]*1e9:
                raise ValueError("RGB/depth timestamp skew exceeds configured tolerance")
            timing = pair_timing(depth_stamp, metadata['received_wall_ns'], received, now, settings['max_age_s'])
            state.update(timing)
            # Backward-compatible field now correctly means immutable arrival age.
            state['sensor_clock_age_s'] = timing['selected_frame_age_at_receipt_s']
            if timing['timing_status'] != 'within_latency_budget':
                messages = dict(frame_stale_on_arrival='Frame was already stale on arrival; inspect transport latency and clock alignment',
                    sensor_timestamp_ahead_of_receiver='Sensor timestamp is ahead of receiver; inspect clock alignment',
                    timestamped_tf_wait_expired='Timestamped TF wait expired; selected frame exceeded the latency budget while awaiting pose coverage')
                if timing['timing_status'] == 'timestamped_tf_wait_expired':
                    state['tf_wait_expired'] = state.get('tf_wait_expired', 0)+1
                raise ValueError(messages[timing['timing_status']])
            # A diagnostics packet may already have consumed this sequence stamp;
            # its later real-TF completion is still new mapping evidence.
            decision = (dict(accept=True, reset=False, reason='preview_completed_with_timestamped_tf')
                        if metadata.get('preview_sent') else sequencing.inspect(depth_stamp))
            state['timestamp_status'] = decision['reason']
            if not decision['accept']:
                raise ValueError(decision['reason'].replace('_', ' '))
            rgb_frame = rgb_msg.header.frame_id
            if not rgb_frame:
                raise ValueError('RGB optical frame ID is missing')
            pose, pose_error = None, None
            try:
                pose = tf_matrix(buffer.lookup_transform(robot['local_frame'], rgb_frame,
                    Time.from_msg(depth_msg.header.stamp), timeout=Duration(seconds=0)))
            except tf2_ros.TransformException as error:
                pose_error = error
                tf_wait(error, rgb_frame, depth_stamp, timing['pending_wait_s'])
                grace = min(.35, settings['max_age_s']*.5)
                if metadata.get('preview_sent') or timing['pending_wait_s'] < grace or now-last_model_preview < 1.:
                    return None  # Cheap retry: do not repeatedly decode waiting frames.
            def lookup_extrinsic(target, source):
                return tf_matrix(buffer.lookup_transform(target, source,
                    Time.from_msg(depth_msg.header.stamp), timeout=Duration(seconds=0)))
            payload = cached_rgbd(metadata, lambda: prepare_registered_rgbd(
                rgb_msg, depth_msg, robot, settings, calibration, lookup_extrinsic))
            if decision['reset'] and not metadata.get('session_reset_applied'):
                state['session_id'] = str(uuid.uuid4())
                state['session_reset_reason'] = decision['reason']
                state['session_resets'] = state.get('session_resets', 0)+1
                seq = 0
                metadata['session_reset_applied'] = True
            seq += 1
            packet = dict(robot_id=robot["id"], session_id=state["session_id"], seq=seq,
                          stamp_ns=depth_stamp, rgb_stamp_ns=rgb_stamp, received_monotonic=received,
                          **payload, T_local_camera=pose, preview_only=pose is None,
                          integration_allowed=pose is not None,
                          pose_quality='external_tf_not_validated' if pose is not None else dict(
                              status='timestamped_tf_unavailable', integration_allowed=False, reason=str(pose_error)[:250]),
                          transport_timing=timing)
            if pose is None:
                if preview_put(output, packet):
                    metadata['preview_sent'] = True
                    sequencing.commit(depth_stamp)
                    last_model_preview = now
                    state['model_previews_offered'] = state.get('model_previews_offered', 0)+1
                    state['last_model_preview_monotonic'] = now
                return None  # Retain this pair for a possible timestamped-TF completion.
            if latest_put(output, packet):
                state["offered"] += 1
            sequencing.commit(depth_stamp)
            state.update(status="receiving", last_error=None, last_packet_monotonic=now,
                         depth_registration=payload['depth_registration'], calibration_id=packet["calibration_id"])
            last_offer = now
            return True
        except tf2_ros.TransformException as error:
            # Registration TF can also be missing: no calibrated model preview.
            tf_wait(error, rgb_msg.header.frame_id, stamp_ns(depth_msg.header.stamp), max(0., now-received))
            return None
        except Exception as error:
            state.update(status="input_rejected", last_error=str(error)[:350])
            state["rejected"] += 1
            last_offer = now
            return False

    def receive_pair(rgb_msg, depth_msg):
        state['received'] += 1
        pending.offer(rgb_msg, depth_msg, dict(received_wall_ns=node.get_clock().now().nanoseconds))

    def process_pending():
        # Don't pin a pair during rate limiting: choose the freshest pair when
        # capacity is available, then retain it only while TF catches up.
        if time.monotonic()-last_offer < 1./float(settings['max_hz']):
            return
        expired_before = pending.expired
        pair = pending.take()
        if pending.expired != expired_before:
            state.update(status='waiting_for_tf', timing_status='timestamped_tf_wait_expired',
                         last_error='Timestamped TF wait expired; discarded selected frame and retained only the newest pair')
            state['tf_wait_expired'] = state.get('tf_wait_expired', 0)+(pending.expired-expired_before)
            pending.last_expired = None  # Do not retain an extra decoded image payload.
        if pair is not None and prepare(*pair) is not None:
            pending.done()

    subscriptions = [node.create_subscription(CameraInfo, robot["camera_info_topic"],
        lambda msg: calibration.__setitem__("rgb", msg), qos),
        node.create_subscription(CameraInfo, robot["depth_camera_info_topic"],
        lambda msg: calibration.__setitem__("depth", msg), qos)]
    def make_filters():
        # Fresh filters own fresh queues. Never register additional callbacks on
        # a long-lived depth filter (Humble has no public unregisterCallback).
        rgb_filter, depth_filter = message_filters.SimpleFilter(), message_filters.SimpleFilter()
        sync = message_filters.ApproximateTimeSynchronizer([rgb_filter, depth_filter], 10,
                                                          settings["sync_slop_s"], allow_headerless=False)
        sync.registerCallback(receive_pair)
        return rgb_filter.signalMessage, depth_filter.signalMessage

    readers = RebindableRGB(lambda cb: node.create_subscription(CompressedImage, robot['rgb_topic'], cb, qos),
                            node.destroy_subscription, make_filters, image_seen, pending.clear)
    readers.rebind()
    subscriptions.append(node.create_subscription(CompressedImage, robot['depth_topic'], readers.depth, qos))

    def report_health():
        rgb_watch.poll(state.get('depth_advanced_monotonic'), state.get('depth_header_age_at_receipt_s'), readers.rebind)
        # Input health takes priority over a now-irrelevant cached TF error.
        latest_put(health_output, {**state, **rgb_watch.health(), 'reported_monotonic': time.monotonic()})
    prepare_timer = node.create_timer(max(.01, min(.05, float(settings['tf_timeout_s']) or .05)), process_pending)
    timer = node.create_timer(1., report_health)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    try:
        while not stop_event.is_set() and context.ok():
            executor.spin_once(timeout_sec=.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            readers.close()
        finally:
            listener.unregister()
            executor.shutdown()
            node.destroy_node()
            context.try_shutdown()
            # Do not hang on a feeder flushing a frame after the parent has exited.
            output.cancel_join_thread()
            health_output.cancel_join_thread()
