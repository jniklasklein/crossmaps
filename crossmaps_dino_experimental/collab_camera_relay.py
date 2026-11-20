"""Laptop receiver for the optional authenticated compressed RGB-D relay.

No listener, ROS publisher, persistent remote file, or remote service. Only
camera bytes, calibration and optionally the same capture's measured ROS
transform come through this connection. Nothing publishes a transform.
"""
from array import array
import json
import math
import os
import re
import selectors
import shlex
import signal
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from collab_relay_auth import PASSWORD_ENV, launch_ssh, safe_environment, ssh_args

MAGIC = b'CMCP1'
HEADER = struct.Struct('>5sIII')
MAX_JSON, MAX_IMAGE = 16384, 8 * 1024**2

# A bare "python -" cannot execute before stdin closes. Separating the source
# from a tiny lease stream allows a detached docker exec to terminate even
# when the camera stops and there are no further stdout writes/SIGPIPE.
_BOOT = """import os, select, sys, threading
s = sys.stdin.buffer
n = int(s.readline(16))
if not 0 < n <= 256*1024: raise SystemExit(2)
code = s.read(n)
if len(code) != n: raise SystemExit(2)
credit = threading.Event()
credit.set()
def lease():
    try:
        while select.select([s], [], [], 15)[0]:
            token = os.read(s.fileno(), 1)
            if token == b'A': credit.set()
            elif token != b'.': break
    finally: os._exit(0)
threading.Thread(target=lease, daemon=True).start()
exec(compile(code, '<crossmaps-camera-relay>', 'exec'), {'__name__':'__main__', 'CMCP_CREDIT':credit})
"""


def ssh_command(robot, settings, environ=None, *, password_auth=False):
    relay = robot['camera_relay']
    host, container = relay['ssh_host'], relay['container']
    if not re.fullmatch(r'(?:[A-Za-z_][\w.-]{0,63}@)?[A-Za-z0-9][A-Za-z0-9.-]{0,252}', host, re.ASCII):
        raise ValueError('invalid camera relay SSH host')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', container, re.ASCII):
        raise ValueError('invalid camera relay container')
    stride = relay.get('image_stride', 1)
    if type(stride) is not int or stride not in (1, 2, 3, 4):
        raise ValueError('invalid camera relay image stride')
    py = ['python3', '-u', '-c', _BOOT, '--domain', str(int(robot['domain'])), '--image-stride', str(stride),
          '--max-hz', str(relay['max_hz']), '--slop', str(settings['sync_slop_s']),
          '--local-frame', robot.get('local_frame', 'odom'), '--base-frame', robot.get('base_frame','base_footprint')]
    for arg, key in (('--rgb-topic', 'rgb_topic'), ('--depth-topic', 'depth_topic'),
                     ('--camera-info-topic', 'camera_info_topic'), ('--depth-camera-info-topic', 'depth_camera_info_topic')):
        py += [arg, robot[key]]
    remote = shlex.join(['docker', 'exec', '-i', container, 'bash', '-c',
                         'source /opt/ros/humble/setup.bash >/dev/null && exec ' + shlex.join(py)])
    return ssh_args(host, remote, password_auth), safe_environment(environ)


def _integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('invalid numeric field in camera relay metadata')
    return value


def _text(value, limit=255):
    if not isinstance(value, str) or not value or len(value) > limit or not value.isprintable():
        raise ValueError('invalid string in camera relay metadata')
    return value


def _metadata(raw):
    m = json.loads(raw)
    for key in ('rgb', 'depth', 'rgb_info', 'depth_info'):
        v = m[key]
        _integer(v['stamp_ns'], 0 if key.endswith('_info') else 1, (2**31)*10**9-1)
        _text(v['frame_id'])
        if not key.endswith('_info'):
            if v['format'] != '':
                _text(v['format'], 256)
            continue
        for k in ('width', 'height'):
            _integer(v[k], 1, 32768)
        for k in ('binning_x', 'binning_y'):
            _integer(v[k], 0, 2**32-1)
        if v['distortion_model'] != '':
            _text(v['distortion_model'])
        for k, length in [('k', 9), ('p', 12), ('r', 9), ('d', None)]:
            vec = v[k]
            if (not isinstance(vec, list) or len(vec) > 32 or (length is not None and len(vec) != length)
                    or not all(type(x) in (float, int) and math.isfinite(x) for x in vec)):
                raise ValueError('invalid relay calibration vector')
        for k in ('width', 'height', 'x_offset', 'y_offset'):
            _integer(v['roi'][k], 0, 2**32-1)
        if type(v['roi']['do_rectify']) is not bool:
            raise ValueError('invalid relay calibration ROI')
    return m


class FrameReader:
    """Incremental bounded framing for a nonblocking pipe; never scans past corruption."""
    def __init__(self):
        self.data = bytearray()
        self.lengths = None

    def feed(self, data):
        if len(data) > 65536:
            raise ValueError('oversized relay read')
        self.data.extend(data)
        while True:
            if self.lengths is None:
                if len(self.data) < HEADER.size:
                    return
                magic, size, nr, nd = HEADER.unpack_from(self.data)
                if magic != MAGIC or not (0 < size <= MAX_JSON and 0 < nr <= MAX_IMAGE and 0 < nd <= MAX_IMAGE):
                    raise ValueError('invalid camera relay packet header')
                del self.data[:HEADER.size]
                self.lengths = size, nr, nd
            size, nr, nd = self.lengths
            total = size + nr + nd
            if len(self.data) < total:
                return
            meta = _metadata(self.data[:size])
            rgb, depth = bytes(self.data[size:size+nr]), bytes(self.data[size+nr:total])
            del self.data[:total]
            self.lengths = None
            yield meta, rgb, depth


@dataclass
class RelayPair:
    rgb: object
    depth: object
    rgb_info: object
    depth_info: object
    received_monotonic: float
    received_wall_ns: int
    generation: int
    pose: dict | None = None
    mount: dict | None = None


def checked_pose(value, target_frame, source_frame, stamp):
    """Only a rigid transform for this exact capture and configured frame pair.

    Camera data remains useful without an optional pose; never 'repair' a bad
    transform or infer it from neighboring relay records.
    """
    if (not isinstance(value, dict) or type(value.get('stamp_ns')) is not int
            or stamp <= 0 or value['stamp_ns'] != stamp or not source_frame or not target_frame
            or value.get('target_frame') != target_frame or value.get('source_frame') != source_frame):
        return None
    rows = value.get('matrix')
    if (type(rows) is not list or len(rows) != 4 or
            any(type(row) is not list or len(row) != 4 for row in rows) or
            any(type(x) not in (float, int) or abs(x)>1.e12 or not math.isfinite(x) for row in rows for x in row)):
        return None
    import numpy as np
    matrix = np.asarray(rows, dtype=np.float64)
    r = matrix[:3,:3]
    if (not np.allclose(matrix[3], [0,0,0,1], rtol=0., atol=1.e-7)
            or not np.allclose(r.T@r, np.eye(3), rtol=0., atol=1.e-5)
            or not np.isclose(np.linalg.det(r), 1., rtol=0., atol=1.e-5)):
        return None
    return dict(target_frame=target_frame, source_frame=source_frame, stamp_ns=stamp, matrix=rows)


def _messages(meta, rgb, depth, mono, wall_ns, generation):
    from sensor_msgs.msg import CameraInfo, CompressedImage
    def header(msg, v):
        msg.header.frame_id = v['frame_id']
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(v['stamp_ns'], 10**9)
        return msg
    def image(k, data):
        msg = header(CompressedImage(), meta[k])
        # Humble's generated setter validates generic sequences byte-by-byte;
        # a typed buffer avoids a Python pass through multi-MB JPEG/PNG data.
        msg.format, msg.data = meta[k]['format'], array('B', data)
        return msg
    def info(k):
        v = meta[k]
        msg = header(CameraInfo(), v)
        for field in ('width','height','distortion_model','binning_x','binning_y'):
            setattr(msg, field, v[field])
        for field in ('k','r','p','d'):
            setattr(msg, field, [float(x) for x in v[field]])
        for field in ('width','height','x_offset','y_offset','do_rectify'):
            setattr(msg.roi, field, v['roi'][field])
        return msg
    return RelayPair(image('rgb',rgb), image('depth',depth), info('rgb_info'), info('depth_info'),
                     mono, wall_ns, generation, meta.get('pose'), meta.get('mount'))


def _failure(stderr):
    text = bytes(stderr).lower()
    if b'host key verification failed' in text or b'remote host identification has changed' in text:
        return 'SSH host key is not trusted or changed; verify it before reconnecting'
    if b'permission denied' in text:
        if b'docker' in text:
            return 'SSH connected but Docker access in the rover account was denied'
        return 'SSH login denied; check credentials or run --setup-relay-auth before restarting the laptop coordinator'
    if b'no such container' in text or b'is not running' in text:
        return 'Configured camera relay container unavailable'
    if b"no module named 'pil'" in text or b"no module named 'numpy'" in text:
        return 'Compact camera relay needs Pillow and NumPy in the SLAM container; image_stride 1 uses neither'
    return 'SSH camera relay exited or failed to start; check SSH access, container and camera topics'


class CameraRelay:
    """One owned SSH client + one replaceable RGB-D pair. No ROS in the I/O thread."""
    def __init__(self, robot, settings, password=None):
        self.robot, self.settings = robot, settings
        self._password = password if password is not None else os.environ.get(PASSWORD_ENV)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = self._pending = self._last_packet = None
        self._attempts = self._generation = self._replaced = self._stale = 0
        self._arrival_age = None
        self._state, self._error = 'idle', None

    def start(self):
        with self._lock:
            if self._thread or self._stop.is_set() or self.robot['camera_relay'].get('mode') == 'off':
                return
            self._thread = threading.Thread(target=self._run, name='camera-relay-'+self.robot['id'], daemon=True)
            self._thread.start()

    def status(self):
        with self._lock:
            return dict(state=self._state, attempts=self._attempts, generation=self._generation, last_error=self._error,
                        last_packet_age_s=None if self._last_packet is None else round(time.monotonic()-self._last_packet, 2),
                        replaced_pairs=self._replaced, stale_pairs=self._stale,
                        source_arrival_age_s=None if self._arrival_age is None else round(self._arrival_age, 3))

    def poll(self):
        with self._lock:
            packet, self._pending = self._pending, None
        if packet is None or self._stop.is_set():
            return None
        waited = max(0., time.monotonic()-packet[3])
        age = max(0., (packet[4]-min(packet[0][k]['stamp_ns'] for k in ('rgb','depth')))/1.e9)
        if waited+age > self.settings['max_age_s']:
            return None
        return _messages(*packet)

    def close(self, wait=True):
        """Invalidate immediately; wait=False lets a ROS timer request cleanup
        without blocking the executor. Use wait=True before adapter exit."""
        self._stop.set()
        with self._lock:
            self._pending = None
            thread = self._thread
            self._state = 'stopped'
        if wait and thread and thread is not threading.current_thread():
            thread.join(3.)

    def _run(self):
        delay = 1.
        while not self._stop.is_set():
            with self._lock:
                self._generation += 1
                self._attempts += 1
                self._state, self._error = 'connecting', None
                self._last_packet = self._pending = self._arrival_age = None
            try:
                self._connect()
            except (OSError, ValueError, EOFError, KeyError, TypeError, OverflowError, RecursionError) as exc:
                with self._lock:
                    if self._state == 'streaming':
                        delay = 1.
                    self._error = str(exc)[:240]
            with self._lock:
                self._pending = None
                self._state = 'stopped' if self._stop.is_set() else 'retrying'
            if self._stop.wait(delay):
                break
            delay = min(30., delay*2)

    def _connect(self):
        source = Path(__file__).with_name('collab_camera_relay_source.py').read_bytes()
        if not 0 < len(source) <= 256*1024:
            raise ValueError('invalid relay source size')
        argv, env = ssh_command(self.robot, self.settings, password_auth=bool(self._password))
        proc = launch_ssh(argv, env, self._password, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, start_new_session=True, bufsize=0)
        try:
            self._read(proc, str(len(source)).encode('ascii') + b'\n' + source)
        finally:
            # EOF revokes the remote lease even for stalled cameras. Reap only
            # the private process group spawned here; never stop a ROS container.
            proc.stdin.close()
            for sig, wait in ((None, .3), (signal.SIGTERM, .5), (signal.SIGKILL, .5)):
                if sig is not None and proc.poll() is None:
                    try: os.killpg(proc.pid, sig)
                    except ProcessLookupError: pass
                try:
                    proc.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
            proc.stdout.close()
            proc.stderr.close()

    def _read(self, proc, source):
        parser, errors = FrameReader(), bytearray()
        start = heartbeat = time.monotonic()
        last = previous = None
        idle_limit = max(10., 4/float(self.robot['camera_relay']['max_hz']))
        pending_write = memoryview(source)
        ack_due = False
        with selectors.DefaultSelector() as sel:
            for stream, events in ((proc.stdout,selectors.EVENT_READ),(proc.stderr,selectors.EVENT_READ),
                                   (proc.stdin, selectors.EVENT_WRITE)):
                os.set_blocking(stream.fileno(), False)
                sel.register(stream, events)
            while not self._stop.is_set():
                now = time.monotonic()
                if now-(last or start) > (idle_limit if last is not None else max(22., idle_limit)):
                    raise TimeoutError('SSH camera relay timed out waiting for advancing paired images')
                if not pending_write and (ack_due or now-heartbeat >= 2.):
                    pending_write = memoryview(b'A' if ack_due else b'.')
                    ack_due = False
                    sel.register(proc.stdin, selectors.EVENT_WRITE)
                for key, _ in sel.select(.2):
                    if key.fileobj is proc.stdin:
                        try:
                            written = os.write(key.fd, pending_write)
                        except BlockingIOError:
                            continue
                        pending_write = pending_write[written:]
                        if not pending_write:
                            heartbeat = time.monotonic()
                            sel.unregister(proc.stdin)
                        continue
                    try: data = os.read(key.fd, 65536)
                    except BlockingIOError: continue
                    if key.fileobj is proc.stderr:
                        errors.extend(data)
                        del errors[:-4096]
                        if not data: sel.unregister(proc.stderr)
                        continue
                    if not data:
                        # select can report stdout EOF before pending stderr.
                        try: errors.extend(os.read(proc.stderr.fileno(), 4096))
                        except BlockingIOError: pass
                        raise EOFError(_failure(errors))
                    for meta, rgb, depth in parser.feed(data):
                        # Credit for consumed bytes, NOT map integration. Even
                        # stale/duplicate pairs must release the sender so it
                        # captures a fresh observation rather than replaying a queue.
                        ack_due = True
                        stamps = (meta['rgb']['stamp_ns'], meta['depth']['stamp_ns'])
                        if previous and (previous[0] == stamps[0] or previous[1] == stamps[1]):
                            continue
                        if abs(stamps[0]-stamps[1]) > self.settings['sync_slop_s']*1e9:
                            continue
                        if 'pose' in meta:
                            meta['pose'] = checked_pose(meta['pose'], self.robot.get('local_frame','odom'),
                                                        meta['rgb']['frame_id'], stamps[1])
                        if 'mount' in meta:
                            meta['mount'] = checked_pose(meta['mount'], meta['rgb']['frame_id'],
                                                         self.robot.get('base_frame','base_footprint'), stamps[1])
                        previous, last = stamps, time.monotonic()
                        wall = time.time_ns()
                        age = (wall-min(stamps))/1.e9
                        with self._lock:
                            self._arrival_age = age
                            if age > self.settings['max_age_s'] or (max(stamps)-wall)/1.e9 > self.settings['max_age_s']:
                                self._state, self._error = 'stale', 'Relay pair outside the acquisition-age budget'
                                self._stale += 1
                                continue
                            self._state, self._error, self._last_packet = 'streaming', None, last
                            self._replaced += self._pending is not None
                            self._pending = (meta, rgb, depth, last, wall, self._generation)
