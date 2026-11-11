"""Bounded camera scheduling and thread-safe, monotonic pipeline diagnostics.

No model/ROS dependencies. Stalls are reported, never 'recovered' by launching
another GPU worker or reusing old observations as new evidence.
"""
from collections import deque
import threading
import time


class LatestFrame:
    def __init__(self):
        self.pending = None
        self.last_taken = None
        self.replaced = 0
        self.expired = 0

    def offer(self, rgb, depth, now):
        if self.pending is not None:
            self.replaced += 1
        self.pending = (rgb, depth, float(now))

    def take(self, now, max_hz, max_age, busy=False):
        if self.pending is None:
            return None
        if now-self.pending[2] > max_age:
            self.pending = None
            self.expired += 1
            return None
        if busy or (self.last_taken is not None and now-self.last_taken < 1/max(.01, max_hz)):
            return None
        frame, self.pending = self.pending, None
        self.last_taken = float(now)
        return frame


class PipelineHealth:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.RLock()
        self.started = clock()
        self.streams = {}
        self.stage = 'idle'
        self.stage_since = self.started
        self.active_since = None
        self.active_seq = None
        self.completed = 0
        self.last_complete = None
        self.last_error = None
        self.last_rejection = None
        self.errors = 0
        self.rejections = 0

    def input(self, name, stamp_ns=None):
        now = self.clock()
        with self.lock:
            state = self.streams.setdefault(name, dict(last=None, advanced=None, stamp=None, count=0, times=deque(maxlen=120)))
            state['last'] = now
            state['count'] += 1
            state['times'].append(now)
            if stamp_ns is None or stamp_ns != state['stamp']:
                state['advanced'] = now
            state['stamp'] = stamp_ns

    def working(self, stage, seq=None):
        with self.lock:
            now = self.clock()
            if self.active_since is None:
                self.active_since = now
            self.stage, self.stage_since = stage, now
            if seq is not None:
                self.active_seq = int(seq)

    def idle(self, error=None):
        with self.lock:
            self.stage, self.stage_since = 'idle', self.clock()
            self.active_since = self.active_seq = None
            if error:
                self.errors += 1
                self.last_error = str(error)[:500]

    def reject(self, reason):
        with self.lock:
            self.rejections += 1
            self.last_rejection = str(reason)[:500]

    def complete(self):
        with self.lock:
            self.completed += 1
            self.last_complete = self.clock()
            self.last_error = self.last_rejection = None
            self.idle()

    def snapshot(self, stale_s=10., inference_timeout_s=30.):
        now = self.clock()
        with self.lock:
            inputs = {}
            for name in ('rgb', 'depth', 'camera_info', 'sync'):
                s = self.streams.get(name)
                recent = [] if s is None else [t for t in s['times'] if now-t <= 10.]
                inputs[name] = dict(age_s=None if s is None else round(now-s['last'], 3),
                    advancing_age_s=None if s is None else round(now-s['advanced'], 3),
                    count=0 if s is None else s['count'],
                    hz=round((len(recent)-1)/max(now-recent[0], .001), 2) if len(recent)>1 else 0.)
            age = None if self.last_complete is None else round(now-self.last_complete, 3)
            active_age = None if self.active_since is None else round(now-self.active_since, 3)
            missing = [n for n in ('rgb', 'depth') if inputs[n]['advancing_age_s'] is None or inputs[n]['advancing_age_s'] > stale_s]
            if active_age is not None and active_age > inference_timeout_s:
                state = 'processing_stalled'
            elif missing:
                state = 'waiting_for_' + '_and_'.join(missing)
            elif inputs['sync']['age_s'] is None or inputs['sync']['age_s'] > stale_s:
                state = 'waiting_for_synchronization'
            elif self.last_error:
                state = 'processing_error'
            elif self.active_since is not None:
                state = 'processing'
            elif age is None or age > stale_s:
                state = 'no_completed_observations'
            else:
                state = 'healthy'
            return dict(schema='crossmaps_pipeline_health_v1', state=state, inputs=inputs,
                stage=self.stage, stage_age_s=round(now-self.stage_since, 3), processing_age_s=active_age,
                active_sequence=self.active_seq, completed=self.completed, completion_age_s=age,
                errors=self.errors, last_error=self.last_error, rejections=self.rejections,
                last_rejection=self.last_rejection)
