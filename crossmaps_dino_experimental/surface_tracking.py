"""Mask-level identity consensus and bounded boundary-ownership corrections.

No screen pixels are propagated to a new frame. Temporal proposals require
current depth, world-position agreement, mask overlap and appearance approval.
Geometry ownership is separate from the mask's display/association hypothesis.
"""
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SurfaceConfig:
    min_samples: int = 3
    interior_px: int = 3
    dominance: float = .75
    mixed_fraction: float = .20
    switch_observations: int = 3
    max_age_s: float = 15.
    max_tracks: int = 512


def compact_mask(mask):
    return cv2.resize(np.asarray(mask, dtype=np.uint8), (96, 64), interpolation=cv2.INTER_NEAREST) > 0


class SurfaceConsensus:
    def __init__(self, config=None):
        self.config = config or SurfaceConfig()
        self.records = {}
        self.last_seq = -1
        self.switches = 0
        self.holds = 0

    def propose(self, masks, owners, points, uu, vv, now, voxel_size, alive, compatible):
        """Read-only proposals. commit() is called once after successful fusion."""
        decisions = []
        cfg = self.config
        for index, mask in enumerate(masks):
            item = dict(mask=index, winner=None, segments=[], confidence=0., reason='no_valid_depth',
                        samples=0, interior=False, previous=None, switch_hits=0)
            if mask is None or np.asarray(mask).ndim != 2:
                item['reason'] = 'bad_shape'
                decisions.append(item)
                continue
            mask = np.asarray(mask, dtype=bool)
            inside = mask[vv, uu]
            valid = inside & np.all(np.isfinite(points), axis=1)
            selected = np.flatnonzero(valid)
            item['samples'] = int(selected.size)
            if selected.size < cfg.min_samples:
                item['reason'] = 'no_valid_depth' if not selected.size else 'insufficient_depth'
                decisions.append(item)
                continue
            radius = max(0, cfg.interior_px)
            core = cv2.erode(mask.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0 if radius else mask
            core_indices = np.flatnonzero(valid & core[vv, uu])
            if core_indices.size >= cfg.min_samples:
                selected = core_indices
                item['interior'] = True
            values, counts = np.unique(owners[selected], return_counts=True)
            votes = {int(s): int(n) for s, n in zip(values, counts) if s > 0 and int(s) in alive}
            total = max(1, sum(votes.values()))
            ranked = sorted(votes, key=lambda s: (-votes[s], s))
            significant = [s for s in ranked if votes[s] >= cfg.min_samples and votes[s]/total >= cfg.mixed_fraction]
            item.update(segments=ranked, votes=votes, core_indices=selected,
                        centroid=np.median(points[selected], axis=0), thumbnail=compact_mask(mask))
            # Two well-supported current owners are not a majority-vote contest.
            if len(significant) > 1:
                item.update(reason='multiple_supported_owners', segments=significant)
                decisions.append(item)
                continue
            winner = ranked[0] if (ranked and votes[ranked[0]] >= cfg.min_samples and
                                    votes[ranked[0]]/selected.size >= .25 and
                                    votes[ranked[0]]/total >= cfg.dominance) else None
            if winner is not None and not compatible(winner, index):
                winner = None
                item['reason'] = 'appearance_conflict'
            if winner is not None:
                item.update(winner=winner, confidence=votes[winner]/max(selected.size, 1), reason='depth_consensus')
            # Compare to compact masks, but gate on world geometry. Image IoU
            # alone must never reidentify a different object after robot motion.
            candidates = []
            for sid, old in self.records.items():
                if sid not in alive or now-old['time'] > cfg.max_age_s or not compatible(sid, index):
                    continue
                distance = float(np.linalg.norm(item['centroid']-old['centroid']))
                if distance > max(.08, 1.5*voxel_size):
                    continue
                intersection = int(np.count_nonzero(item['thumbnail'] & old['thumbnail']))
                union = int(np.count_nonzero(item['thumbnail'] | old['thumbnail']))
                iou = intersection/max(1, union)
                if iou >= .45:
                    candidates.append((iou, sid, old))
            candidates.sort(key=lambda value: (-value[0], value[1]))
            if candidates and (len(candidates) == 1 or candidates[0][0]-candidates[1][0] >= .15):
                _, previous, old = candidates[0]
                item['previous'] = previous
                if winner is None and item['reason'] != 'appearance_conflict':
                    item.update(winner=previous, confidence=.5, reason='temporal_depth_hold')
                elif winner is not None and winner != previous:
                    hits = old.get('switch_hits', 0)+1 if old.get('challenger') == winner else 1
                    item.update(challenger=winner, switch_hits=hits)
                    if hits < cfg.switch_observations:
                        item.update(winner=previous, confidence=.5, reason='pending_owner_switch')
            decisions.append(item)
        return decisions

    def commit(self, seq, now, decisions, masks, alive):
        if seq <= self.last_seq:
            return False
        self.last_seq = seq
        self.records = {s: r for s, r in self.records.items() if s in alive and now-r['time'] <= self.config.max_age_s}
        updates = {}
        replaced = set()
        for item in decisions:
            sid = item.get('winner')
            if sid not in alive or 'centroid' not in item:
                continue
            if item['reason'] in ('temporal_depth_hold', 'pending_owner_switch'):
                self.holds += 1
            if item.get('previous') not in (None, sid):
                self.switches += 1
                replaced.add(item['previous'])
            update = updates.setdefault(sid, dict(thumbnail=np.zeros((64, 96), dtype=bool), centers=[],
                                                  challenger=None, switch_hits=0, time=now))
            update['thumbnail'] |= item['thumbnail']
            update['centers'].append(item['centroid'])
            update['challenger'] = item.get('challenger')
            update['switch_hits'] = item.get('switch_hits', 0)
        for sid in replaced - set(updates):
            self.records.pop(sid, None)
        # A missing/ambiguous observation may preserve a track, but cannot
        # preserve a run of confirmation votes for changing its owner.
        for sid, record in self.records.items():
            if sid not in updates:
                record.update(challenger=None, switch_hits=0)
        for sid, update in updates.items():
            update['centroid'] = np.mean(update.pop('centers'), axis=0)
            self.records[sid] = update
        if len(self.records) > self.config.max_tracks:
            keep = sorted(self.records, key=lambda s: (-self.records[s]['time'], s))[:self.config.max_tracks]
            self.records = {s: self.records[s] for s in keep}
        return True

    def diagnostics(self):
        return dict(active_mask_tracks=len(self.records), mask_owner_switches=self.switches,
                    temporal_holds=self.holds, observation_seq=self.last_seq)


class BoundaryOwnershipVotes:
    """Only repeated, small boundary corrections; never collapse whole tracks.

    The caller supplies eligible (voxel, old, new) votes after interior/mask,
    geometry, appearance and competing-mask checks. Missing frames reset runs.
    """
    def __init__(self, required=4, max_pending=4096):
        self.required = required
        self.max_pending = max_pending
        self.last_seq = -1
        self.pending = {}
        self.losses = Counter()
        self.budgets = {}

    def update(self, seq, proposals, support_sizes):
        if seq <= self.last_seq:
            return []
        self.last_seq = seq
        alive = set(support_sizes)
        self.losses = Counter({s: n for s, n in self.losses.items() if s in alive})
        self.budgets = {s: n for s, n in self.budgets.items() if s in alive}
        current, accepted = {}, []
        for key, old, new in sorted(set(proposals))[:self.max_pending]:
            if old == new:
                continue
            token = (tuple(key), int(old), int(new))
            hits = self.pending.get(token, 0)+1
            self.budgets.setdefault(old, int(.10*support_sizes.get(old, 0)))
            if hits >= self.required and self.losses[old] < self.budgets[old] and len(accepted) < 64:
                accepted.append(token)
                self.losses[old] += 1
            else:
                current[token] = min(hits, self.required)
        self.pending = current
        return accepted

    def to_dict(self):
        # Unconfirmed votes restart after loading; correction budgets persist.
        return dict(losses=dict(self.losses), budgets=dict(self.budgets))

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        obj.losses.update({int(k): int(v) for k,v in data.get('losses', {}).items()})
        obj.budgets = {int(k): int(v) for k,v in data.get('budgets', {}).items()}
        return obj
