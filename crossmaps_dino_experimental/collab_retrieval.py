"""Bounded scheduling for geometric checks. A priority is never pose evidence."""
from collections import Counter
import math


def pair_key(a, b):
    return tuple(sorted((a, b)))


def cue_strength(cues):
    # One nearly perfect object-level CLIP cue is a good retrieval lead, not
    # a reason to monopolize every verification if the other cues disagree.
    return (.5 * max(cues.values()) + .5 * sum(cues.values()) / len(cues))


class VerificationSchedule:
    # Separate the operator's last 300 review cards from the work queue.
    # Keep bounded alternatives even when one rover stops admitting new frames.
    MAX_PENDING = 512
    PER_SUBMAP_PAIR = 8

    def __init__(self):
        self.checked = set()
        self.frame_checks = Counter()
        self.cell_checks = Counter()
        self.pending = {}
        self.turn = self.backfilled = self.cursor = 0

    def mark(self, source, target):
        key = pair_key(source['id'], target['id'])
        self.pending.pop(key, None)
        if key not in self.checked:
            self.checked.add(key)
            self.frame_checks.update(key)
            self.cell_checks[pair_key(source['submap_id'], target['submap_id'])] += 1

    def live_priority(self, source, target, cues):
        cell = pair_key(source['submap_id'], target['submap_id'])
        return (cue_strength(cues) - .055 * math.log1p(self.frame_checks[target['id']])
                - .025 * math.log1p(self.cell_checks[cell]))

    @staticmethod
    def _cell(candidate):
        return pair_key(candidate['source_submap'], candidate['target_submap'])

    def add(self, c):
        key = pair_key(c['source'], c['target'])
        if key in self.checked or key in self.pending:
            return
        cell = self._cell(c)
        self.pending[key] = c
        bucket = [k for k, item in self.pending.items() if self._cell(item) == cell]
        if len(bucket) > self.PER_SUBMAP_PAIR:
            # Preserve the two oldest leads and replace the oldest of the
            # rotating leads. A run of near-identical frames cannot flood us.
            self.pending.pop(bucket[2])
        if len(self.pending) > self.MAX_PENDING:
            # Evict from the most heavily represented cell, never a singleton
            # first just because it belongs to the other rover's older scene.
            counts = Counter(self._cell(item) for item in self.pending.values())
            richest = max(counts, key=lambda v: (counts[v], self.cell_checks[v]))
            self.pending.pop(next(k for k, item in self.pending.items() if self._cell(item) == richest))

    def historical(self, frames, current_sessions, *, exclude_frame=None):
        # Runs under the same per-keyframe verifier budget as live queries.
        # Receipts come from the laptop clock, not two potentially skewed rover
        # clocks. A near receipt earns a small preference, never acceptance.
        best, best_rank = None, None
        for k, c in list(self.pending.items()):
            a, b = frames.get(c['source']), frames.get(c['target'])
            if (not a or not b or a['source_id'] not in current_sessions or b['source_id'] not in current_sessions
                    or a['feature_space_id'] != b['feature_space_id'] or k in self.checked):
                self.pending.pop(k)
                continue
            if exclude_frame in k:
                continue
            gap = c.get('receipt_gap_s')
            rank = (self.cell_checks[self._cell(c)],
                    sum(self.frame_checks[i] for i in k),
                    -(cue_strength(c['cues']) + (.05 / (1 + gap / 20.) if gap is not None else 0)))
            if best_rank is None or rank < best_rank:
                best, best_rank = c, rank
        return best

    def explore(self, sessions, frames, *, exclude_frame=None):
        """Sample at most 24 indices, visiting the whole finite grid over time.

        Bit reversal spreads short prefixes across trajectory time rather than
        scanning hundreds of frames against the first old reference. A diagonal
        offset visits all counterparts without forcing index synchronization.
        No Cartesian pair matrix or expensive full-image verification here.
        """
        groups = []
        total = 0
        for i, a in enumerate(sessions):
            for b in sessions[i+1:]:
                n, m = len(a['keyframe_ids']), len(b['keyframe_ids'])
                if n and m:
                    groups.append((total, total+n*m, a['keyframe_ids'], b['keyframe_ids']))
                    total += n*m
        if not total:
            return
        bits = (total-1).bit_length()
        period = 1 << bits
        for _ in range(min(24, period)):
            ordinal = self.cursor % period
            self.cursor += 1
            ix = int(f'{ordinal:0{bits}b}'[::-1], 2)
            if ix >= total:
                continue
            for lo, hi, left, right in groups:
                if lo <= ix < hi:
                    col, row = divmod(ix-lo, len(left))
                    aid, bid = left[row], right[(col+row*len(right)//len(left)) % len(right)]
                    pair = pair_key(aid, bid)
                    if (exclude_frame not in pair and pair not in self.checked and pair not in self.pending
                            and frames[aid]['feature_space_id'] == frames[bid]['feature_space_id']):
                        yield frames[aid], frames[bid]
                    break

    def report(self):
        return dict(checked_unique_pairs=len(self.checked), distinct_checked_submap_pairs=len(self.cell_checks),
                    pending_pairs=len(self.pending), backfill_checks=self.backfilled)
