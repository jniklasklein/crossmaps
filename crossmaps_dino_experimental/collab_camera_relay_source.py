"""Ephemeral source-side compressed RGB-D sampler. Standalone on ROS Humble.

No ROS publishers, port listener, file output or dependency on the CrossMaps
checkout. Full resolution passes compressed payloads through; compact mode
uses Pillow and NumPy on selected pairs only. The owning SSH client reads
CMCP1 records from stdout and controls the helper's lifetime via stdin.
"""
import argparse
from bisect import bisect_left, insort
from collections import deque
from io import BytesIO
import json
import math
import os
import re
import select
import struct
import sys
import time


MAGIC = b'CMCP1'
LENGTHS = struct.Struct('>III')
MAX_HEADER = 16 * 1024
MAX_IMAGE = 8 * 1024 * 1024
PNG_SIG = b'\x89PNG\r\n\x1a\n'
# A late JPEG can arrive many depth frames after its exact mate. Bound
# retained *compressed* payloads by time, count and bytes per stream, not
# by a four-frame assumption tied to a particular camera publishing rate.
PAIR_HISTORY_S = 1.5
PAIR_HISTORY_COUNT = 48
PAIR_HISTORY_BYTES = 24 * 1024**2


def _header(msg):
    t = msg.header.stamp
    frame = msg.header.frame_id
    if not isinstance(frame, str) or not 0 < len(frame) <= 255 or not 0 <= t.nanosec < 10**9:
        raise ValueError('invalid camera frame or stamp')
    return dict(stamp_ns=int(t.sec) * 10**9 + int(t.nanosec), frame_id=frame)


def _camera_info(msg):
    info = _header(msg)
    if not (0 < msg.height <= 16384 and 0 < msg.width <= 16384 and len(msg.distortion_model) <= 128):
        raise ValueError('invalid calibration dimensions or distortion model')
    for key, count in [('k', 9), ('r', 9), ('p', 12), ('d', None)]:
        arr = getattr(msg, key)
        if (count is not None and len(arr) != count) or len(arr) > 32 or not all(map(math.isfinite, arr)):
            raise ValueError('invalid calibration matrix')
        info[key] = list(arr)
    if not ((info['p'][0] > 0 and info['p'][5] > 0) or (info['k'][0] > 0 and info['k'][4] > 0)):
        raise ValueError('missing focal length')
    roi = msg.roi
    info.update(height=int(msg.height), width=int(msg.width), distortion_model=msg.distortion_model,
                binning_x=int(msg.binning_x), binning_y=int(msg.binning_y),
                roi={k: getattr(roi, k) for k in ('width', 'height', 'x_offset', 'y_offset', 'do_rectify')})
    return info


def encode_parts(rgb, depth, rgb_info, depth_info, pose=None, mount=None):
    """Scatter/gather framing: preserve original payloads without JPEG copies."""
    a, b = rgb[0], depth[0]
    meta = dict(rgb=a, depth=b, rgb_info=rgb_info, depth_info=depth_info)
    if pose is not None:
        meta['pose'] = pose
    if mount is not None:
        meta['mount'] = mount
    header = json.dumps(meta,
                        separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(header) > MAX_HEADER or any(not 0 < len(v[1]) <= MAX_IMAGE for v in (rgb, depth)):
        raise ValueError('oversize camera record')
    return (MAGIC + LENGTHS.pack(len(header), len(rgb[1]), len(depth[1])) + header, rgb[1], depth[1])


def _jpeg_size(buf):
    """Check JPEG dimensions against calibration *before* expanding it."""
    if bytes(buf[:2]) != b'\xff\xd8':
        raise ValueError('compact mode requires a JPEG RGB payload')
    i = 2
    while i+4 < len(buf) and buf[i] == 255:
        while i < len(buf) and buf[i] == 255:
            i += 1
        if i+3 > len(buf) or buf[i] in (0xda,0xd9):
            break
        marker = buf[i]
        i += 1
        size = struct.unpack_from('>H',buf,i)[0]
        if size < 2 or size > len(buf)-i:
            break
        if marker in (0xc0,0xc1,0xc2) and size >= 8 and buf[i+2] == 8:
            h,w = struct.unpack_from('>HH',buf,i+3)
            return w,h
        i += size
    raise ValueError('unsupported JPEG header in compact mode')


def _scaled(info, dims, stride):
    w,h = dims
    if (info['width'],info['height']) != (w,h) or not 0 < w*h <= 12_000_000:
        raise ValueError('calibration and measured image dimensions differ or exceed limit')
    roi = info['roi']
    if (info['binning_x'] not in (0,1) or info['binning_y'] not in (0,1)
            or roi['do_rectify'] or roi['x_offset'] or roi['y_offset']
            or (roi['width'],roi['height']) not in ((0,0),(w,h))):
        raise ValueError('compact mode does not resample cropped/binned calibration')
    width,height = (w+stride-1)//stride,(h+stride-1)//stride
    # Output (u,v) is input (stride*u,stride*v), not half-pixel-centered
    # interpolation. Multiply K/P on the left by diag(1/s, 1/s, 1).
    return dict(info, width=width,height=height,
                k=[v/stride if j<6 else v for j,v in enumerate(info['k'])],
                p=[v/stride if j<8 else v for j,v in enumerate(info['p'])],
                roi=dict(roi,width=width if roi['width'] else 0,height=height if roi['height'] else 0))


def compact_record(rgb, depth, ri, di, stride, pose=None, mount=None):
    """Compact from the compressed messages only. Runs after credit + rate gate."""
    from PIL import Image
    import numpy as np
    rgb_format = rgb[0]['format'].lower()
    if rgb_format and not re.search(r'\bjpe?g\b',rgb_format):
        raise ValueError('compact mode supports JPEG color only')
    if not re.fullmatch(r'16uc1\s*;\s*compresseddepth(?:\s+png)?\s*', depth[0]['format'].lower()):
        raise ValueError('compact mode supports U16 PNG compressedDepth only')
    rdata,ddata=rgb[1],depth[1]
    rw,rh=_jpeg_size(rdata)
    off=bytes(ddata[:32]).find(PNG_SIG)
    if off not in (0,12) or len(ddata)<off+33 or bytes(ddata[off+8:off+16]) != b'\x00\x00\x00\rIHDR':
        raise ValueError('unsupported compressedDepth PNG header')
    dw,dh,bits,colour=struct.unpack_from('>IIBB', ddata, off+16)
    if (bits,colour)!=(16,0):
        raise ValueError('compact mode requires single-channel uint16 depth')
    ri,di=_scaled(ri,(rw,rh),stride),_scaled(di,(dw,dh),stride)
    # The Jetsons' user-installed cv2/typing module does not match the native
    # cv2 binary. Use the distro Pillow codec, not that broken Python binding.
    try:
        rc, dc = BytesIO(), BytesIO()
        with Image.open(BytesIO(rdata)) as im:
            if im.format != 'JPEG' or im.size != (rw,rh) or im.mode not in ('RGB','L'):
                raise ValueError('invalid encoded color image for compact relay')
            # Pillow does not implicitly rotate the image by EXIF orientation.
            px = np.asarray(im.convert('RGB'))[::stride,::stride]
            Image.fromarray(px).save(rc,format='JPEG',quality=75)
        with Image.open(BytesIO(ddata[off:])) as im:
            # Pillow 9 decodes a 16-bit PNG to I (int32); newer versions
            # may use I;16. IHDR already proved 16-bit unsigned grayscale.
            if im.format != 'PNG' or im.size != (dw,dh) or im.mode not in ('I','I;16','I;16B'):
                raise ValueError('invalid encoded depth image for compact relay')
            px = np.asarray(im)[::stride,::stride].astype('<u2',copy=False)
            Image.fromarray(px).save(dc,format='PNG',compress_level=2)
    except (OSError,SyntaxError) as exc:
        raise ValueError('compact camera encoding/decoding failed') from exc
    return encode_parts((rgb[0],rc.getvalue()),
                        (depth[0],bytes(ddata[:off])+dc.getvalue()),ri,di,pose,mount)


class CapturePose:
    """Read-only tf2 lookup, with a maximum interpolation span per dynamic edge.

    tf2 alone will interpolate across a multi-second hole when both endpoints
    exist. Check the actual brackets only on the path used by this camera.
    Static mounts need no bracketing. Never wait or query the latest pose.
    """
    MAX_GAP_NS = 500_000_000

    def __init__(self, target, buffer, make_time, tf_error):
        self.target, self.buffer, self.make_time, self.tf_error = target, buffer, make_time, tf_error
        self.edges = {}  # child -> (parent, stamps; None for static)

    def ingest(self, message, static):
        put = self.buffer.set_transform_static if static else self.buffer.set_transform
        for transform in message.transforms:
            try:
                h = _header(transform)
            except (ValueError, TypeError, OverflowError):
                continue
            parent, child, ns = h['frame_id'], transform.child_frame_id, h['stamp_ns']
            if not child or parent == child or (not static and ns <= 0):
                continue
            t, q = transform.transform.translation, transform.transform.rotation
            norm2 = q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z
            # tf2 can silently discard invalid input. Such a sample must
            # never contribute a false coverage bracket.
            if not all(map(math.isfinite,(norm2,t.x,t.y,t.z))) or abs(norm2-1.) >= .01:
                continue
            try:
                put(transform, 'local_relay')
            except (self.tf_error, ValueError):
                continue
            if static:
                self.edges[child] = (parent, None)
                continue
            prev = self.edges.get(child)
            if prev is None or prev[0] != parent or prev[1] is None:
                self.edges[child] = (parent, [])
            stamps = self.edges[child][1]
            if not stamps or stamps[-1] < ns:
                stamps.append(ns)
            elif stamps[bisect_left(stamps, ns)] != ns:
                insort(stamps, ns)
            del stamps[:max(len(stamps)-512, bisect_left(stamps, stamps[-1]-8_000_000_000))]

    def _ancestors(self, frame):
        chain = []
        while frame not in chain and len(chain) < 128:
            chain.append(frame)
            if frame not in self.edges:
                return chain
            frame = self.edges[frame][0]
        return []

    def lookup(self, source, stamp, target=None):
        target = self.target if target is None else target
        if stamp <= 0 or not source or target == source:
            return None
        a, b = self._ancestors(source), self._ancestors(target)
        common = next((f for f in a if f in b), None)
        if not common:
            return None
        for child in a[:a.index(common)] + b[:b.index(common)]:
            stamps = self.edges[child][1]
            if stamps is None:
                continue
            j = bisect_left(stamps, stamp)
            if j == len(stamps) or (stamps[j] != stamp and (not j or stamps[j]-stamps[j-1] > self.MAX_GAP_NS)):
                return None
        try:
            result = self.buffer.lookup_transform_core(target, source, self.make_time(stamp))
            h = _header(result)
            if (h['stamp_ns'], h['frame_id'], result.child_frame_id) != (stamp, target, source):
                return None
            t, q = result.transform.translation, result.transform.rotation
            x,y,z,w = q.x,q.y,q.z,q.w
            n = math.sqrt(x*x+y*y+z*z+w*w)
            if not all(map(math.isfinite, (t.x,t.y,t.z,n))) or abs(n-1.) > .01:
                return None
            x,y,z,w = x/n,y/n,z/n,w/n
            matrix = [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),t.x],
                      [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),t.y],
                      [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y),t.z],
                      [0.,0.,0.,1.]]
            return dict(target_frame=target, source_frame=source, stamp_ns=stamp, matrix=matrix)
        except (self.tf_error, ValueError, OverflowError):
            return None


def subscribe_capture_pose(node, target):
    import tf2_py
    from rclpy.duration import Duration
    from rclpy.time import Time
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from tf2_msgs.msg import TFMessage
    # No listener node, frame graph service, broadcaster or TF republishing.
    pose = CapturePose(target, tf2_py.BufferCore(Duration(seconds=8)),
                       lambda ns: Time(nanoseconds=ns), tf2_py.TransformException)
    for topic, static in (('/tf', False), ('/tf_static', True)):
        qos = QoSProfile(depth=100,
            reliability=ReliabilityPolicy.RELIABLE if static else ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL if static else DurabilityPolicy.VOLATILE)
        node.create_subscription(TFMessage, topic, lambda m,s=static: pose.ingest(m,s), qos)
    return pose.lookup


class PairedCapture:
    """Latest small per-channel rings; compare original acquisition stamps.

    Only the ROS spin thread accesses these queues. Output requires a new
    stamp on *each* stream; never replay a cached JPEG with new depth.
    """
    def __init__(self, slop_s, max_hz, clock=time.monotonic, image_stride=1, pose_lookup=None):
        self.slop_ns, self.period, self.clock = int(slop_s * 10**9), 1. / max_hz, clock
        self.images = {k: deque() for k in ('rgb', 'depth')}
        self.info = {k: deque(maxlen=64) for k in self.images}
        self.last = dict(rgb=-1, depth=-1)
        self.next_at = 0.
        self.reason = 'waiting for a calibrated compressed pair'
        self.image_stride = image_stride
        self.pose_lookup = pose_lookup
        self.mount_lookup = None

    def image(self, key, msg):
        try:
            h = _header(msg)
            if h['stamp_ns'] <= self.last[key]:
                return
            data = memoryview(msg.data).cast('B')
            if not 0 < len(data) <= MAX_IMAGE or not isinstance(msg.format, str) or len(msg.format) > 256:
                raise ValueError('invalid compressed payload size/format')
            h['format'] = msg.format
            now = self.clock()
            ring = self.images[key]
            ring.append((h, data, now))
            retained = sum(len(v[1]) for v in ring)
            while (len(ring) > PAIR_HISTORY_COUNT or retained > PAIR_HISTORY_BYTES
                   or now - ring[0][2] > PAIR_HISTORY_S):
                retained -= len(ring.popleft()[1])
        except (ValueError, TypeError, OverflowError) as exc:
            self.reason = str(exc)

    def calibration(self, key, msg):
        try:
            self.info[key].append(_camera_info(msg))
        except (ValueError, TypeError, OverflowError) as exc:
            self.reason = str(exc)
            self.info[key].clear()  # Do not reuse stale calibration after an invalid update.

    def _info(self, key, meta, allow_other_frame=False):
        stamp = meta['stamp_ns']
        # Calibration is often static; a later calibration must not be applied
        # to a materially older frame. Prefer matching optical frame, then age.
        choices = [i for i in self.info[key] if (allow_other_frame or i['frame_id'] == meta['frame_id'])
                   and i['stamp_ns'] <= stamp + self.slop_ns]
        return min(choices, key=lambda i: (i['frame_id'] != meta['frame_id'], abs(stamp-i['stamp_ns'])), default=None)

    def take(self):
        now = self.clock()
        if now < self.next_at:
            return None
        # Work in the newest capture window. Within one slop interval, choose
        # the best timestamp match rather than the new RGB's previous depth.
        choices = []
        for a in self.images['rgb']:
            for b in self.images['depth']:
                x, y = a[0]['stamp_ns'], b[0]['stamp_ns']
                if (x > self.last['rgb'] and y > self.last['depth'] and abs(x-y) <= self.slop_ns
                        and now - min(a[2], b[2]) <= PAIR_HISTORY_S):
                    # At 15 Hz the previous depth can be ~67 ms away and
                    # technically inside an 80-ms slop. Give the near-exact
                    # mate one slop interval to arrive before taking that pair.
                    if (abs(x-y) > min(15_000_000, self.slop_ns/4)
                            and now-max(a[2],b[2]) < self.slop_ns/1e9):
                        continue
                    choices.append((max(x,y), -abs(x-y), a, b))
        if not choices:
            return None
        newest = max(p[0] for p in choices)
        order = lambda p: (newest-p[0] <= self.slop_ns, p[1], p[0])
        selected = None
        poses = {}
        for capture_ns, neg_skew, a, b in sorted(choices, key=order, reverse=True):
            # At 1.5 Hz there are usually several recent synchronized captures
            # already buffered. Prefer one at most 350 ms behind the newest
            # with pose coverage, but never wait or withhold a camera packet.
            if selected is not None:
                first_skew = abs(selected[0][0]['stamp_ns']-selected[1][0]['stamp_ns'])
                if newest-capture_ns > 350_000_000 or -neg_skew > max(15_000_000,first_skew):
                    continue  # Never sacrifice a near-synchronous pair for an adjacent-frame match.
            rgb_info = self._info('rgb', a[0])
            # Registered depth can have the RGB image frame while the sensor
            # also publishes its native stereo info. Keep both originals; the
            # laptop's calibrated registration path decides which matrices apply.
            depth_info = self._info('depth', b[0], allow_other_frame=a[0]['frame_id']==b[0]['frame_id'])
            if rgb_info is None or depth_info is None:
                self.reason = 'waiting for matching camera calibration'
                continue
            key = (a[0]['frame_id'], b[0]['stamp_ns'])
            if self.pose_lookup and key not in poses:
                poses[key] = self.pose_lookup(*key)
            pose = poses.get(key)
            if selected is None or pose is not None:
                selected = a, b, rgb_info, depth_info, pose
            if not self.pose_lookup or pose is not None or len(poses) >= 12:
                break
        if selected is not None:
            a, b, rgb_info, depth_info, pose = selected
            mount = self.mount_lookup(a[0]['frame_id'], b[0]['stamp_ns']) if self.mount_lookup else None
            self.last.update(rgb=a[0]['stamp_ns'], depth=b[0]['stamp_ns'])
            for key, ring in self.images.items():
                # Only unconsumed original acquisitions can be sent next.
                while ring and ring[0][0]['stamp_ns'] <= self.last[key]:
                    ring.popleft()
            self.next_at = now + self.period
            try:
                record = (encode_parts(a,b,rgb_info,depth_info,pose,mount) if self.image_stride==1 else
                          compact_record(a,b,rgb_info,depth_info,self.image_stride,pose,mount))
            except ValueError as exc:
                self.reason = str(exc)
                return None
            self.reason = 'streaming'
            return record
        return None


class LatestWriter:
    """Nonblocking stdout, one partial record plus one replaceable latest record.

    Once the prefix has escaped, its payload cannot be replaced without
    corrupting the stream. A stuck partial record instead tears down this
    transport so the SSH owner can reconnect directly to a fresh capture.
    """
    def __init__(self, fd, clock=time.monotonic):
        self.fd, self.clock, self.current, self.latest, self.started = fd, clock, None, None, None
        os.set_blocking(fd, False)
        self.poll = select.poll()
        self.poll.register(fd, select.POLLHUP | select.POLLERR | select.POLLNVAL)

    def offer(self, record):
        if self.started is None:
            self.current = deque(memoryview(part) for part in record)
        else:
            self.latest = record

    def pump(self):
        if self.poll.poll(0):
            raise BrokenPipeError('relay output closed')
        if self.current is None:
            return
        now = self.clock()
        if self.started is not None and now-self.started > 8.:
            raise TimeoutError('partially written camera record expired')
        try:
            size = os.writev(self.fd, self.current)
        except BlockingIOError:
            return
        if size <= 0:
            raise BrokenPipeError('end of relay output')
        if self.started is None:
            self.started = now
        while size:
            head = self.current[0]
            if size < len(head):
                self.current[0] = head[size:]
                return
            size -= len(head)
            self.current.popleft()
        if not self.current:
            self.current, self.started = None, None
            if self.latest is not None:
                record, self.latest = self.latest, None
                self.offer(record)


def take_with_credit(sampler, writer, credit):
    if credit is not None and (writer.current is not None or not credit.is_set()):
        return None
    record = sampler.take()
    if record is not None and credit is not None:
        credit.clear()
    return record


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain', required=True, type=int)
    p.add_argument('--local-frame', help='Optionally attach a measured local-frame pose at depth capture')
    p.add_argument('--base-frame', default='base_footprint', help='Rover body frame for the map marker')
    p.add_argument('--rgb-topic', default='/oak/rgb/image_rect/compressed')
    p.add_argument('--depth-topic', default='/oak/stereo/image_raw/compressedDepth')
    p.add_argument('--camera-info-topic', default='/oak/rgb/camera_info')
    p.add_argument('--depth-camera-info-topic', default='/oak/stereo/camera_info')
    p.add_argument('--max-hz', type=float, default=1.5)
    p.add_argument('--slop', type=float, default=.08)
    p.add_argument('--image-stride', type=int, choices=(1,2,3,4), default=1)
    p.add_argument('--standalone-no-ack', action='store_true', help='local pipeline diagnostic without SSH credit')
    args = p.parse_args(argv)
    if not (0 <= args.domain <= 232 and .1 <= args.max_hz <= 10 and 0 <= args.slop <= .5):
        p.error('invalid ROS domain, rate or synchronization slop')
    if not args.rgb_topic.endswith('/compressed') or not args.depth_topic.endswith('/compressedDepth'):
        p.error('source relay accepts compressed RGB and compressedDepth topics only')
    return args


def main(argv=None):
    args = arguments(argv)
    credit = globals().get('CMCP_CREDIT')
    if credit is None and not args.standalone_no_ack:
        raise SystemExit('Relay requires an acknowledged SSH owner (or explicit --standalone-no-ack).')
    # Isolate *native* ROS logging as well as Python print from the wire.
    wire = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    os.environ['ROS_DOMAIN_ID'] = str(args.domain)
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import CameraInfo, CompressedImage
    rclpy.init(args=[])
    node = rclpy.create_node('crossmaps_compressed_tap_' + str(os.getpid()), enable_rosout=False,
                             start_parameter_services=False)
    sampler = PairedCapture(args.slop, args.max_hz,image_stride=args.image_stride)
    if args.local_frame:
        try:
            sampler.pose_lookup = subscribe_capture_pose(node, args.local_frame)
            # Reuse the same timestamp-checked TF buffer; no extra subscribers.
            sampler.mount_lookup = lambda optical, ns: sampler.pose_lookup(args.base_frame, ns, target=optical)
        except (ImportError, RuntimeError) as error:
            print(f'CrossMaps source: no optional local TF ({type(error).__name__}); sending images only',
                  file=sys.stderr, flush=True)
    qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
    try:
        for key, topic, calibration in [('rgb', args.rgb_topic, args.camera_info_topic),
                                        ('depth', args.depth_topic, args.depth_camera_info_topic)]:
            node.create_subscription(CompressedImage, topic, lambda m, k=key: sampler.image(k, m), qos)
            node.create_subscription(CameraInfo, calibration, lambda m, k=key: sampler.calibration(k, m), qos)
        writer = LatestWriter(wire)
        last_report = time.monotonic()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.02)
            # No pre-rendered record may wait behind an older frame in SSH.
            # Read/replace source observations normally while waiting for ACK.
            record = take_with_credit(sampler, writer, credit)
            if record is not None:
                writer.offer(record)
                last_report = time.monotonic()
            elif time.monotonic()-last_report > 12.:
                why = ('waiting for receiver acknowledgment' if credit is not None and not credit.is_set()
                       else sampler.reason)
                print('CrossMaps source: no new RGB-D packet; ' + why, file=sys.stderr, flush=True)
                last_report = time.monotonic()
            writer.pump()
    except (BrokenPipeError, TimeoutError, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        os.close(wire)


if __name__ == '__main__':
    main()
