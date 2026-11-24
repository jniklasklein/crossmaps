#!/usr/bin/env python3
"""Read-only Gazebo model poses for evaluation, never a ROS/TF input.

One atomic bounded snapshot holds at most 300 samples per rover (~30s at10Hz).
The generated model origin coincides with base_footprint: its frame cancels
the URDF base_link translation. No spawn/odometry pose is used as truth.
"""
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import time

ROBOTS = ('rover_a', 'rover_b')


def timestamp(value):
    if not isinstance(value, dict):
        raise ValueError('Missing simulator timestamp')
    numbers = []
    for field in ('sec', 'nsec'):
        raw = value.get(field, 0)
        if isinstance(raw, bool) or not isinstance(raw, (int, str)):
            raise ValueError('Invalid simulator timestamp')
        numbers.append(int(raw))
    sec, nsec = numbers
    if sec < 0 or sec > 10**10 or not 0 <= nsec < 10**9:
        raise ValueError('Invalid simulator timestamp')
    return sec*10**9+nsec


def matrix(pose):
    position, orientation = pose.get('position', {}), pose.get('orientation', {})
    if not isinstance(position, dict) or not isinstance(orientation, dict):
        raise ValueError('Invalid pose')
    # Protobuf JSON omits zero components, but a valid quaternion must have
    # at least one nonzero component; missing orientation is never identity.
    p = [float(position.get(k, 0)) for k in ('x', 'y', 'z')]
    q = [float(orientation.get(k, 0)) for k in ('x', 'y', 'z', 'w')]
    norm = math.sqrt(sum(v*v for v in q))
    if not all(math.isfinite(v) for v in p+q) or abs(norm-1.) > .01:
        raise ValueError('Invalid finite unit quaternion')
    x, y, z, w = [v/norm for v in q]
    return [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), p[0]],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), p[1]],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y), p[2]], [0., 0., 0., 1.]]


class TruthBuffer:
    def __init__(self, run_id):
        self.run_id = run_id
        self.robots = {name: deque(maxlen=300) for name in ROBOTS}
        self.last = {}
        self.last_received_unix = {}
        self.valid, self.error, self.rejected = True, None, 0

    def consume(self, packet):
        if not self.valid:
            return False
        try:
            kind, value = packet['kind'], packet['message']
            if kind == 'clock':
                stamp = timestamp(value['sim'])
            elif kind in ROBOTS:
                poses = value.get('pose', [])
                if not isinstance(poses, list) or len(poses) > 8:
                    raise ValueError('Invalid pose vector')
                matching = [p for p in poses if isinstance(p, dict) and p.get('name') == kind]
                if len(matching) != 1:
                    raise ValueError('Expected exactly one named model pose')
                selected = matching[0]
                header = selected.get('header', {})
                vector_stamp = value.get('header', {}).get('stamp')
                pose_stamp = header.get('stamp')
                if pose_stamp is None and vector_stamp is None:
                    raise ValueError('Missing model pose timestamp')
                stamp = timestamp(pose_stamp if pose_stamp is not None else vector_stamp)
                if vector_stamp is not None and timestamp(vector_stamp) != stamp:
                    raise ValueError('Conflicting pose/vector timestamps')
                labels = {}
                for item in header.get('data', []):
                    key = item.get('key')
                    if key in ('frame_id', 'child_frame_id'):
                        if key in labels:
                            raise ValueError('Duplicate model frame labels')
                        labels[key] = item.get('value')
                if (('frame_id' in labels and labels['frame_id'] != ['crossmaps_lab']) or
                        ('child_frame_id' in labels and labels['child_frame_id'] != [kind])):
                    raise ValueError('Unexpected model frame labels')
                transform = matrix(selected)
            else:
                raise ValueError('Unexpected ground-truth stream')
            previous = self.last.get(kind)
            if previous is not None and stamp < previous:
                self.valid, self.error = False, 'Simulation time regressed; restart required'
                for samples in self.robots.values():
                    samples.clear()
                return False
            self.last[kind] = stamp
            if kind == 'clock' or stamp == previous:
                return False
            self.robots[kind].append(dict(stamp_ns=stamp, T_world_base=transform))
            self.last_received_unix[kind] = time.time()
            return True
        except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
            self.rejected += 1
            return False

    def snapshot(self):
        now = time.time()
        return dict(schema=1, run_id=self.run_id, updated_unix=now, valid=self.valid,
                    error=self.error, rejected_packets=self.rejected,
                    last_received_unix=dict(self.last_received_unix),
                    sample_age_s={name: None if name not in self.last_received_unix else
                                  max(0., now-self.last_received_unix[name]) for name in ROBOTS},
                    robots={name: list(samples) for name, samples in self.robots.items()})

    def write(self, path):
        temporary = path.with_suffix('.json.tmp')
        data = json.dumps(self.snapshot(), separators=(',', ':'), allow_nan=False)
        with temporary.open('w') as stream:
            stream.write(data)
        os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--helper', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get('GZ_PARTITION', '').startswith('crossmaps_sim_'):
        parser.error('Ground truth requires an isolated crossmaps_sim_ partition')
    if not args.helper.is_file() or not args.output.parent.is_dir():
        parser.error('Existing native helper and owned output directory required')
    truth = TruthBuffer(args.output.parent.name)
    stopping = False
    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True
    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    helper = None
    try:
        truth.write(args.output)
        helper = subprocess.Popen([str(args.helper.resolve()), 'evaluation'], stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE)
        buffer, next_write = b'', time.monotonic()+.5
        while not stopping:
            if helper.poll() is not None:
                truth.valid, truth.error = False, 'Ground-truth helper exited'
                truth.write(args.output)
                return 1
            if select.select([helper.stdout], [], [], .1)[0]:
                chunk = os.read(helper.stdout.fileno(), 65536)
                if not chunk:
                    continue
                buffer += chunk
                if len(buffer) > 1024*1024:
                    raise ValueError('Oversized ground-truth packet')
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    try:
                        truth.consume(json.loads(line))
                    except (ValueError, TypeError):
                        truth.rejected += 1
            if time.monotonic() >= next_write:
                truth.write(args.output)
                next_write = time.monotonic()+.5
    finally:
        if helper is not None:
            helper.terminate()
            try:
                helper.wait(timeout=3.)
            except subprocess.TimeoutExpired:
                helper.kill(); helper.wait(timeout=3.)
            helper.stdout.close()
        truth.valid = False
        truth.error = truth.error or 'Ground-truth collector stopped'
        truth.write(args.output)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
