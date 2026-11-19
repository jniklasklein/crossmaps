"""Fair bounded scheduling and end-to-end timing, independent of ROS/models."""
from collections import deque
import threading
import time
import math

import numpy as np

class FairLatestScheduler:
    """One pending observation per source, never an unbounded inference backlog."""
    def __init__(self, robot_ids, max_age_s=2., clock=time.monotonic):
        self.order = list(robot_ids)
        self.max_age_s, self.clock = float(max_age_s), clock
        self.pending = {}
        self.latest = {}
        self.retired_sessions = {r: set() for r in self.order}
        self.last = -1
        self.lock = threading.RLock()
        self.counters = {r: dict(offered=0, replaced=0, expired=0, duplicate=0, taken=0) for r in self.order}

    def offer(self, packet):
        robot = packet['robot_id']
        if robot not in self.counters:
            raise ValueError('unknown robot source')
        session, sequence = packet['session_id'], int(packet['seq'])
        if not session or sequence < 0:
            raise ValueError('invalid source identity')
        with self.lock:
            last = self.latest.get(robot)
            if session in self.retired_sessions[robot] or (last and session == last[0] and sequence <= last[1]):
                self.counters[robot]['duplicate'] += 1
                return False
            if last and session != last[0]:
                self.retired_sessions[robot].add(last[0])
            self.latest[robot] = (session, sequence)
            queued = self.pending.get(robot)
            if (packet.get('preview_only') and queued and queued['session_id'] == session
                    and not queued.get('preview_only')
                    and 0 <= self.clock()-queued['received_monotonic'] <= self.max_age_s):
                # A pending frame with a measured pose can also produce previews.
                return False
            self.counters[robot]['offered'] += 1
            if robot in self.pending:
                self.counters[robot]['replaced'] += 1
            self.pending[robot] = packet
            return True

    def take(self):
        with self.lock:
            for offset in range(1, len(self.order)+1):
                index = (self.last+offset) % len(self.order)
                robot = self.order[index]
                packet = self.pending.pop(robot, None)
                if packet is None:
                    continue
                age = self.clock()-float(packet['received_monotonic'])
                if not math.isfinite(age) or age < -.1 or age > self.max_age_s:
                    self.counters[robot]['expired'] += 1
                    continue
                self.last = index
                self.counters[robot]['taken'] += 1
                return packet
        return None

    def snapshot(self):
        with self.lock:
            return {r: dict(c, pending=r in self.pending) for r, c in self.counters.items()}


class CycleTiming:
    def __init__(self, window=30):
        self.values = deque(maxlen=window)

    def add(self, elapsed_s, inference_s, input_age_s):
        self.values.append((float(elapsed_s)*1000, float(inference_s)*1000, float(input_age_s)*1000))

    def snapshot(self):
        if not self.values:
            return dict(count=0, cycle_ms=None, average_ms=None, p95_ms=None, inference_ms=None, input_to_map_ms=None)
        arr = np.asarray(self.values)
        return dict(count=len(arr), cycle_ms=round(arr[-1, 0], 1), average_ms=round(float(arr[:, 0].mean()), 1),
                    p95_ms=round(float(np.percentile(arr[:, 0], 95)), 1), inference_ms=round(arr[-1, 1], 1),
                    input_to_map_ms=round(arr[-1, 2], 1))
