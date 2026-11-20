"""Source-owned collaborative evidence, registration and reversible identity graph.

This module has no ROS, GPU, network or robot-control dependencies. An observation
is never fused into another robot's local map. The global map is a versioned view
of immutable keyframe evidence and independently optimized short submaps.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
import hashlib
import json
import math
import os
import time

import cv2
import numpy as np

from collab_geometry import (depth_points, geometry_descriptor, nearest, optimize_pose_graph,
                             pose_delta, transform, valid_transform, verify_rgbd, voxel_sample)
from collab_identity_inspector import inspect_link, render_witness
from collab_provisional import derive_previews, quality_summary
from collab_retrieval import VerificationSchedule, pair_key


# Display history, not graph constraints or an odometry product. Keep at most
# this many sources of path vertices per session (including the live endpoint).
TRAIL_POINTS = 384
TRAIL_STEP_M = .04
TRAIL_GAP_NS = 10_000_000_000


DEFAULTS = dict(shadow_mode=True, provisional_enabled=True, keyframe_seconds=8., keyframe_translation=.15,
                keyframe_rotation_deg=8., submap_translation=1., submap_rotation_deg=20.,
                submap_seconds=90., submap_keyframes=10, max_keyframes=256, max_submaps=64,
                max_store_bytes=2*1024**3, keyframe_points=2400, submap_points=5000,
                snapshot_points=16000, candidate_top_k=3, verifications_per_keyframe=2,
                min_visual_inliers=12, min_overlap=.25, max_registration_rmse=.065,
                corroboration_translation=.12, corroboration_rotation_deg=6.,
                transform_agreement_translation=.18, transform_agreement_rotation_deg=7.,
                entity_dino_similarity=.80, entity_clip_similarity=.50,
                entity_overlap=.25, entity_distance=.12, max_entities=256)


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [plain(item) for item in value]
    return value


def unit(vector):
    if vector is None:
        return None
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    return vector/norm if len(vector) and np.isfinite(vector).all() and norm > 1e-7 else None


def cosine(a, b):
    a, b = unit(a), unit(b)
    return float(a @ b) if a is not None and b is not None and a.shape == b.shape else None


def identifier(*parts):
    # Hash canonical tuples: embedded slash characters can never alias identities.
    return hashlib.sha256(json.dumps(parts, separators=(',', ':')).encode()).hexdigest()[:20]


def validate_config(config):
    unknown = set(config)-set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown collaborative backend options: {', '.join(sorted(unknown))}")
    values = dict(DEFAULTS, **config)
    for key in ('shadow_mode', 'provisional_enabled'):
        if type(values[key]) is not bool:
            raise ValueError(f'{key} must be a JSON boolean')
    integers = dict(submap_keyframes=(1, 64), max_keyframes=(1, 512), max_submaps=(1, 256),
                    max_store_bytes=(1024, 64*1024**3), keyframe_points=(12, 10000),
                    submap_points=(12, 20000), snapshot_points=(32, 100000), candidate_top_k=(1, 10),
                    verifications_per_keyframe=(0, 10), min_visual_inliers=(6, 200), max_entities=(1, 512))
    for key, (minimum, maximum) in integers.items():
        value = values[key]
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f'{key} must be an integer in [{minimum}, {maximum}]')
    ranges = dict(keyframe_seconds=(0., 3600.), keyframe_translation=(0., 10.), keyframe_rotation_deg=(0., 180.),
                  submap_translation=(.05, 20.), submap_rotation_deg=(.1, 180.), submap_seconds=(.1, 3600.),
                  min_overlap=(.05, 1.), max_registration_rmse=(.005, .25), corroboration_translation=(.01, 5.),
                  corroboration_rotation_deg=(.1, 90.), transform_agreement_translation=(.01, 2.),
                  transform_agreement_rotation_deg=(.1, 45.), entity_dino_similarity=(0., 1.),
                  entity_clip_similarity=(-1., 1.), entity_overlap=(.05, 1.), entity_distance=(.01, .5))
    for key, (minimum, maximum) in ranges.items():
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f'{key} must be finite and in [{minimum}, {maximum}]')
    return values


class Backend:
    """Single-owner backend. Callers must serialize ingest/command/snapshot.

    Capacity limits stop *new retained evidence*, rather than silently deleting
    anchors or pretending an evicted trajectory is still reconstructible. Local
    preview/perception may continue; the dashboard exposes capacity_reached.
    """
    def __init__(self, config=None, store_dir=None):
        self._overrides = dict(config or {})
        self.config = validate_config(config or {})
        self.store_dir = Path(store_dir).resolve() if store_dir else None
        self.revision = 0
        self.sessions = {}
        self.robots = {}
        self.keyframes = {}
        self.submaps = {}
        self.constraints = {}
        self.candidates = {}
        self.verification_schedule = VerificationSchedule()
        self.entity_links = {}
        self.entity_exclusions = set()
        self.entity_registry = {}
        self.entity_aliases = {}
        self.events = deque(maxlen=80)
        self.reference_session = None
        self.pending_anchors = []
        self.graph_status = {}
        self.store_bytes = 0
        self.capacity_reason = None
        self._entity_cache = []
        self._preview_cache = ([], {})
        self._preview_key = None
        if self.store_dir:
            self.store_dir.mkdir(parents=True, exist_ok=True)
            self._restore()

    def _event(self, kind, message, **details):
        self.events.append(dict(time=time.time(), kind=kind, message=message, **plain(details)))

    def _session(self, robot, session):
        return identifier('source', robot, session)

    def mark_trajectory_gap(self, robot_id, session_id):
        """Called under the backend lock when pose validation/TF drops.

        Keep the last accepted marker; do not create a session or a pose from a
        preview. A transition changes revision once, not on every rejected image.
        """
        source = self.sessions.get(self._session(robot_id, session_id))
        trail = source.get('trail') if source else None
        if not trail or not trail['latest'] or trail['tracking_lost']:
            return False
        trail['tracking_lost'] = True
        self.revision += 1
        return True

    def _record_trajectory_pose(self, source, obs, anchor, *, disconnected=False):
        """Keep accepted body poses relative to measured *keyframes*, not world.

        An internal keyframe can move relative to its own submap during a graph
        solve. Store the offset to that keyframe even for non-keyframe updates.
        """
        trail = source.setdefault('trail', dict(points=[], latest=None, tracking_lost=False))
        mount = obs.get('T_camera_base')
        frame = 'camera_optical' if mount is None else 'base'
        local_pose = obs['T_local_camera'] if mount is None else obs['T_local_camera'] @ mount
        relative = np.linalg.inv(anchor['T_local_camera']) @ local_pose
        last = trail['latest']
        split = (disconnected or trail['tracking_lost'] or last is None or frame != last['frame']
                 or obs['stamp_ns'] - last['stamp_ns'] > TRAIL_GAP_NS)
        points = trail['points']
        # Do not consume the path budget idling in place. Latest heading/position
        # still updates on *every* accepted observation, including a turn in place.
        store = split or not points
        if not store:
            previous = points[-1]
            other = self.keyframes[previous['anchor']]
            old_p = other['T_local_camera'][:3, :3] @ previous['p'] + other['T_local_camera'][:3, 3]
            store = (other['submap_id'] != anchor['submap_id'] or
                     np.linalg.norm(local_pose[:3, 3] - old_p) >= TRAIL_STEP_M)
        if store:
            points.append(dict(anchor=anchor['id'], seq=obs['seq'], p=relative[:3, 3].tolist(),
                               split=split, frame=frame))
            if len(points) >= TRAIL_POINTS:  # reserve one transient live endpoint
                del points[:len(points) - TRAIL_POINTS + 1]
        trail.update(latest=dict(anchor=anchor['id'], seq=obs['seq'], stamp_ns=obs['stamp_ns'],
                                 relative=relative.tolist(), frame=frame), tracking_lost=False)

    def _trajectory_snapshot(self, source):
        trail = source.get('trail')
        if not trail or not trail['latest']:
            return dict(latest=None, segments=[], tracking_lost=False)
        latest = trail['latest']
        owner = self.keyframes[latest['anchor']]
        measured = owner['T_submap_camera'] @ latest['relative']
        tip = dict(submap_id=owner['submap_id'], seq=latest['seq'], stamp_ns=latest['stamp_ns'],
                   frame=latest['frame'], T_submap_pose=measured)
        points = trail['points']
        if not points or points[-1]['seq'] != latest['seq']:
            points = [*points, dict(anchor=latest['anchor'], p=np.asarray(latest['relative'])[:3, 3],
                                    frame=latest['frame'], split=False)]
        segments = []
        for vertex in points:
            anchor = self.keyframes[vertex['anchor']]
            T, sid = anchor['T_submap_camera'], anchor['submap_id']
            p = np.round(T[:3, :3] @ vertex['p'] + T[:3, 3], 4).tolist()
            if not segments or vertex['split'] or sid != segments[-1]['submap_id']:
                segments.append(dict(submap_id=sid, frame=vertex['frame'],
                                     break_before=not segments or vertex['split'], points=[]))
            if not segments[-1]['points'] or p != segments[-1]['points'][-1]:
                segments[-1]['points'].append(p)
        return dict(latest=tip, segments=segments, tracking_lost=trail['tracking_lost'])

    def _cache_features(self, frame):
        matrix = {}
        reliable = sorted((surface for surface in frame['surfaces'] if surface['quality'] >= .3), key=lambda surface: -surface['quality'])[:32]
        for name in ('dino', 'clip'):
            vectors = [surface[name] for surface in reliable if surface[name] is not None]
            matrix[name] = np.vstack(vectors) if vectors and all(vector.shape == vectors[0].shape for vector in vectors) else None
        frame['_surface_features'] = matrix
        gray = cv2.resize(cv2.cvtColor(frame['rgb'], cv2.COLOR_BGR2GRAY), (64, 48)).astype(np.float32)
        frame['_view_thumbnail'] = (gray-float(gray.mean()))/max(10., float(gray.std()))

    def _distinct_views(self, a, b):
        distance, angle = pose_delta(a['T_local_camera'], b['T_local_camera'])
        moved = distance >= self.config['corroboration_translation'] or angle >= math.radians(self.config['corroboration_rotation_deg'])
        # External odometry alone is not evidence of an independent view: it can
        # drift while the camera image remains unchanged. Brightness-normalize.
        changed = float(np.mean(np.abs(a['_view_thumbnail']-b['_view_thumbnail']))) >= .12
        return moved and changed

    def identity_inspection(self, link_id):
        """Lazy explanation of a stored identity proposal, not a new decision."""
        return inspect_link(self, link_id, identifier)

    def identity_evidence_image(self, link_id, member_id, frame_id):
        """JPEG of measured camera-space support for this link's witness only."""
        return render_witness(self, link_id, member_id, frame_id, identifier)

    def _validated(self, observation):
        result = dict(observation)
        for key in ('robot_id', 'session_id', 'seq', 'stamp_ns', 'T_local_camera', 'K', 'rgb', 'depth'):
            if key not in result:
                raise ValueError(f"Observation missing {key}")
        result['robot_id'], result['session_id'] = str(result['robot_id']), str(result['session_id'])
        if not result['robot_id'] or not result['session_id']:
            raise ValueError('Robot and session IDs must not be empty')
        result['seq'], result['stamp_ns'] = int(result['seq']), int(result['stamp_ns'])
        result['T_local_camera'] = valid_transform(result['T_local_camera'])
        if result.get('T_camera_base') is not None:
            # Optional display metadata cannot invalidate otherwise sound mapping
            # evidence. A malformed mount falls back to the labelled optical pose.
            try:
                result['T_camera_base'] = valid_transform(result['T_camera_base'])
            except (TypeError, ValueError, OverflowError):
                result['T_camera_base'] = None
        result['K'] = np.asarray(result['K'], dtype=float)
        result['rgb'] = np.asarray(result['rgb'], dtype=np.uint8)
        result['depth'] = np.asarray(result['depth'], dtype=np.float32)
        if result['K'].shape != (3, 3) or not np.isfinite(result['K']).all() or min(result['K'][0, 0], result['K'][1, 1]) <= 0:
            raise ValueError('Invalid camera intrinsic matrix')
        if result['rgb'].ndim != 3 or result['rgb'].shape[2] != 3 or result['depth'].shape != result['rgb'].shape[:2]:
            raise ValueError('Registered RGB/depth dimensions differ')
        result['feature_space_id'] = str(result.get('feature_space_id', 'unknown'))
        result['calibration_id'] = str(result.get('calibration_id', 'unknown'))
        result['global_descriptor'] = unit(result.get('global_descriptor'))
        points, colors = result.get('points_camera'), result.get('colors')
        if points is None:
            points, colors = depth_points(result['depth'], result['K'], result['rgb'])
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if colors is None or np.asarray(colors).shape != points.shape:
            colors = np.full(points.shape, 170, dtype=np.uint8)
        result['points_camera'], result['colors'] = voxel_sample(points, np.asarray(colors, dtype=np.uint8), limit=self.config['keyframe_points'])
        result['geometry_descriptor'] = geometry_descriptor(result['points_camera'])
        surfaces = []
        for item in result.get('surfaces', [])[:256]:
            support, _ = voxel_sample(item.get('points_camera', []), size=.04, limit=180)
            if len(support) < 3:
                continue
            surfaces.append(dict(local_id=str(item['local_id']), entity_id=None if item.get('entity_id') is None else str(item['entity_id']),
                                 points_camera=support, dino=unit(item.get('dino')), clip=unit(item.get('clip')),
                                 quality=float(np.clip(item.get('quality', 0.), 0, 1))))
        result['surfaces'] = surfaces
        self._cache_features(result)
        return result

    def ingest(self, observation):
        obs = self._validated(observation)
        robot, session = obs['robot_id'], obs['session_id']
        source_id = self._session(robot, session)
        source = self.sessions.get(source_id)
        if source is None:
            source = self.sessions[source_id] = dict(id=source_id, robot_id=robot, session_id=session, frames=0,
                last_seq=-1, last_stamp_ns=-1, keyframe_ids=[], submap_ids=[], status='observing', rejected_poses=0)
            previous = self.robots.get(robot)
            self.robots[robot] = source_id
            self._event('session', f'{robot}: new source session; local IDs are not inherited', previous_session=previous, source=source_id)
            if self.reference_session is None:
                self.reference_session = source_id
        if obs['seq'] <= source['last_seq'] or obs['stamp_ns'] <= source['last_stamp_ns']:
            return self.snapshot()  # duplicate / reordered evidence must not vote twice
        source.update(last_seq=obs['seq'], last_stamp_ns=obs['stamp_ns'], frames=source['frames']+1,
                      last_received=time.time(), pose_quality=plain(obs.get('pose_quality', 'external_tf_not_validated')))
        if not obs.get('integration_allowed', True):
            source['status'] = 'pose_inconsistent'
            source['rejected_poses'] += 1
            self.mark_trajectory_gap(robot, session)
            self._event('pose_rejected', f'{robot}: observation excluded from mapping', pose_quality=source['pose_quality'])
            self.revision += 1
            return self.snapshot()
        source['status'] = 'observing'
        last = self.keyframes.get(source['keyframe_ids'][-1]) if source['keyframe_ids'] else None
        if last is not None:
            distance, angle = pose_delta(last['T_local_camera'], obs['T_local_camera'])
            elapsed = (obs['stamp_ns']-last['stamp_ns'])/1e9
            same_calibration = last['calibration_id'] == obs['calibration_id']
            if same_calibration and elapsed < self.config['keyframe_seconds'] and distance < self.config['keyframe_translation'] and angle < math.radians(self.config['keyframe_rotation_deg']):
                self._record_trajectory_pose(source, obs, last)
                self.revision += 1
                return self.snapshot()
        if len(self.keyframes) >= self.config['max_keyframes'] or self.store_bytes >= self.config['max_store_bytes']:
            self.capacity_reason = 'Retained evidence capacity reached; increase limits or start a new run. Existing evidence preserved.'
            source['status'] = 'capacity_reached'
            self.mark_trajectory_gap(robot, session)
            return self.snapshot()
        submap = self.submaps.get(source['submap_ids'][-1]) if source['submap_ids'] else None
        new_submap = submap is None
        if submap:
            distance, angle = pose_delta(submap['T_local_submap'], obs['T_local_camera'])
            age = (obs['stamp_ns']-submap['start_stamp_ns'])/1e9
            new_submap = (distance >= self.config['submap_translation'] or angle >= math.radians(self.config['submap_rotation_deg'])
                          or age >= self.config['submap_seconds'] or len(submap['keyframe_ids']) >= self.config['submap_keyframes']
                          or submap['calibration_id'] != obs['calibration_id'])
        if new_submap:
            if len(self.submaps) >= self.config['max_submaps']:
                self.capacity_reason = 'Submap capacity reached; existing graph preserved. Start a new run or increase max_submaps.'
                source['status'] = 'capacity_reached'
                self.mark_trajectory_gap(robot, session)
                return self.snapshot()
            previous = submap
            submap_id = identifier('submap', robot, session, len(source['submap_ids']))
            submap = dict(id=submap_id, source_id=source_id, robot_id=robot, session_id=session,
                          start_stamp_ns=obs['stamp_ns'], calibration_id=obs['calibration_id'], keyframe_ids=[],
                          T_local_submap=obs['T_local_camera'].copy(), T_world_submap=obs['T_local_camera'].copy(),
                          aligned=source_id == self.reference_session, points=np.empty((0, 3)), colors=np.empty((0, 3), dtype=np.uint8))
            if previous:
                submap['T_world_submap'] = previous['T_world_submap'] @ np.linalg.inv(previous['T_local_submap']) @ submap['T_local_submap']
                submap['aligned'] = previous['aligned']
                edge_id = identifier('odometry', previous['id'], submap_id)
                endpoint_quality = (self.keyframes[previous['keyframe_ids'][0]].get('pose_quality'), obs.get('pose_quality'))
                has_recovery = any(isinstance(q, dict) and q.get('mapping_pose_corrected') for q in endpoint_quality)
                self.constraints[edge_id] = dict(id=edge_id, source=submap_id, target=previous['id'],
                    T_target_source=np.linalg.inv(previous['T_local_submap']) @ submap['T_local_submap'],
                    kind='odometry', state='accepted', active=True, translation_sigma=.10, rotation_sigma=math.radians(8),
                    provenance='local_pose_with_rgbd_recovery' if has_recovery else 'external_odometry_not_validated',
                    corroborations=0)
            self.submaps[submap_id] = submap
            source['submap_ids'].append(submap_id)
        obs['id'] = identifier('keyframe', robot, session, obs['seq'])
        obs['source_id'], obs['submap_id'] = source_id, submap['id']
        obs['T_submap_camera'] = np.linalg.inv(submap['T_local_submap']) @ obs['T_local_camera']
        self.keyframes[obs['id']] = obs
        source['keyframe_ids'].append(obs['id'])
        submap['keyframe_ids'].append(obs['id'])
        self._record_trajectory_pose(source, obs, obs)
        self._rebuild_submap(submap)
        self._retrieve(obs)
        self._apply_pending_anchors()
        self._optimize()
        self._update_entities()
        self.revision += 1
        self._persist_frame(obs)
        self._persist_state()
        return self.snapshot()

    def _rebuild_submap(self, submap):
        points, colors = [], []
        for key in submap['keyframe_ids']:
            frame = self.keyframes[key]
            points.append(transform(frame['points_camera'], frame.get('T_submap_camera', np.linalg.inv(submap['T_local_submap']) @ frame['T_local_camera'])))
            colors.append(frame['colors'])
        if points:
            submap['points'], submap['colors'] = voxel_sample(np.vstack(points), np.vstack(colors), size=.055, limit=self.config['submap_points'])

    def _scores(self, first, second):
        scores = {}
        appearance = cosine(first.get('global_descriptor'), second.get('global_descriptor'))
        if appearance is not None:
            scores['appearance'] = max(0., appearance)
        a, b = first.get('geometry_descriptor'), second.get('geometry_descriptor')
        if a is not None and b is not None:
            scores['geometry'] = float(np.exp(-np.linalg.norm(a-b)/max(.3, np.linalg.norm(a)+np.linalg.norm(b))))
        for field in ('dino', 'clip'):
            a, b = first['_surface_features'][field], second['_surface_features'][field]
            if a is not None and b is not None and a.shape[1] == b.shape[1]:
                similarities = (a @ b.T).reshape(-1)
                count = min(3, len(similarities))
                scores[field] = float(np.partition(similarities, -count)[-count:].mean())
        return scores

    def _retrieve(self, observation):
        schedule = self.verification_schedule
        current = set(self.robots.values())
        if observation['source_id'] not in current:
            return
        pool = []
        for target in self.keyframes.values():
            if (target['robot_id'] == observation['robot_id'] or target['source_id'] not in current
                    or target['feature_space_id'] != observation['feature_space_id']
                    or pair_key(target['id'], observation['id']) in schedule.checked):
                continue
            scores = self._scores(observation, target)
            if scores:
                pool.append((target, scores))
        selected = {}
        for cue in ('appearance', 'geometry', 'dino', 'clip'):
            ranked = sorted((item for item in pool if cue in item[1]), key=lambda item: (-item[1][cue], item[0]['id']))
            for target, scores in ranked[:self.config['candidate_top_k']]:
                selected[target['id']] = (target, scores)
        priority = lambda item: (-schedule.live_priority(observation, *item), item[0]['id'])
        diversified = sorted(pool, key=priority)
        for target, cues in diversified[:self.config['candidate_top_k']]:
            selected[target['id']] = (target, cues)
        # Also nominate one plausible frame from under-checked submaps outside
        # the top per-cue lists. Re-ranking a collapsed shortlist is not enough.
        cells = set()
        for target, cues in sorted(diversified, key=lambda item: schedule.cell_checks[
                pair_key(observation['submap_id'], item[0]['submap_id'])]):
            if target['submap_id'] not in cells:
                selected[target['id']] = (target, cues)
                cells.add(target['submap_id'])
                if len(cells) >= self.config['candidate_top_k']:
                    break
        received = observation.get('received_monotonic')
        recent_id = None
        if selected and received is not None:
            recent = [(target, cues) for target, cues in pool
                      if self.robots.get(target['robot_id']) == target['source_id']
                      and target.get('received_monotonic') is not None
                      and abs(target['received_monotonic']-received) <= self.config['keyframe_seconds']*2]
            if recent:
                latest = min(recent, key=lambda pair: (abs(pair[0]['received_monotonic']-received), pair[0]['id']))
                recent_id = latest[0]['id']
                selected[recent_id] = latest
        ranked = sorted(selected.values(), key=priority)
        live = []
        for target, scores in ranked:
            candidate = self._proposal(observation, target, scores,
                    'recent_local_receipt_and_retrieval' if recent_id == target['id'] else 'retrieval')
            self.candidates[candidate['id']] = candidate
            live.append(candidate)
        budget = self.config['verifications_per_keyframe']
        if live and budget:
            schedule.turn += 1
            # First try a strong but not repeatedly exhausted target. When
            # available, the second slot usually goes to a contemporaneous view.
            # Every third round reserve one slot for an older *unchecked* pair
            # (every fourth with budget=1); no extra full verifier calls.
            older = None
            if schedule.turn % (4 if budget == 1 else 3) == 0 or (budget >= 2 and recent_id is None):
                older = self._next_backfill(current, exclude_frame=observation['id'])
            nearest = next((c for c in live[1:] if c['target'] == recent_id), None)
            jobs = ([older] if budget == 1 and older is not None else
                    [live[0]] + [c for c in (older, nearest) if c is not None] + live[1:])
            used = 0
            for c in jobs:
                if self._check_candidate(c, 'historical_backfill' if c is older else
                                         'recent_capture' if c['target'] == recent_id else 'diverse_retrieval'):
                    used += 1
                if used >= budget:
                    break
        for c in live:
            if 'verification' not in c and budget:
                schedule.add(c)
        # Diagnostics are bounded; accepted constraints retain their complete evidence IDs.
        while len(self.candidates) > 300:
            self.candidates.pop(next(iter(self.candidates)))

    def _proposal(self, source, target, scores, basis):
        # Keep a retained review card's original orientation/ID when the
        # exploration lane rediscovers it from the opposite direction.
        for left, right in ((source, target), (target, source)):
            previous = self.candidates.get(identifier('candidate', left['id'], right['id']))
            if previous is not None:
                return previous
        times = [f.get('received_monotonic') for f in (source, target)]
        return dict(id=identifier('candidate', source['id'], target['id']), source=source['id'], target=target['id'],
                    source_robot=source['robot_id'], target_robot=target['robot_id'],
                    source_submap=source['submap_id'], target_submap=target['submap_id'],
                    state='proposed', score=max(scores.values()), cues=scores, reason='retrieved_not_verified',
                    receipt_gap_s=None if None in times else abs(times[0]-times[1]), proposal_basis=basis)

    def _next_backfill(self, current, exclude_frame=None):
        schedule = self.verification_schedule
        deferred = schedule.historical(self.keyframes, current, exclude_frame=exclude_frame)
        if deferred is not None and schedule.backfilled % 4 != 3:
            return deferred
        # A bounded review list alone can never nominate a pair that all
        # semantic top lists skipped. Periodically search outside those lists.
        sources = [self.sessions[sid] for sid in sorted(current)]
        for a, b in schedule.explore(sources, self.keyframes, exclude_frame=exclude_frame):
            cues = self._scores(a, b)
            if cues:
                return self._proposal(a, b, cues, 'retained_trajectory_exploration')
        return deferred

    def _check_candidate(self, candidate, basis):
        schedule = self.verification_schedule
        candidate_id = candidate['id']
        observation, target = (self.keyframes[candidate[k]] for k in ('source', 'target'))
        if (pair_key(observation['id'], target['id']) in schedule.checked
                or observation['source_id'] != self.robots.get(observation['robot_id'])
                or target['source_id'] != self.robots.get(target['robot_id'])
                or observation['robot_id'] == target['robot_id']
                or observation['feature_space_id'] != target['feature_space_id']):
            return False
        verification = verify_rgbd(observation, target, min_inliers=self.config['min_visual_inliers'],
                                   min_overlap=self.config['min_overlap'], max_rmse=self.config['max_registration_rmse'])
        schedule.mark(observation, target)
        if basis == 'historical_backfill':
            schedule.backfilled += 1
        candidate.update(verification=plain(verification), verification_basis=basis, reason=verification['reason'])
        # Bring an older item back to the review window with its original ID.
        self.candidates.pop(candidate_id, None)
        self.candidates[candidate_id] = candidate
        if not verification['ok']:
            candidate['state'] = 'rejected'
            return True
        a, b = self.submaps[observation['submap_id']], self.submaps[target['submap_id']]
        measurement = np.linalg.inv(b['T_local_submap']) @ target['T_local_camera'] @ verification['T_target_source'] @ np.linalg.inv(observation['T_local_camera']) @ a['T_local_submap']
        local_relation = target['T_local_camera'] @ verification['T_target_source'] @ np.linalg.inv(observation['T_local_camera'])
        edge = dict(id=candidate_id, source=a['id'], target=b['id'], source_frame=observation['id'], target_frame=target['id'],
                    T_target_source=measurement, T_targetlocal_sourcelocal=local_relation,
                    T_targetcamera_sourcecamera=verification['T_target_source'].copy(),
                    kind='loop', state='verified', active=False, corroborations=1,
                    translation_sigma=verification['translation_sigma'], rotation_sigma=verification['rotation_sigma'],
                    verification_quality=quality_summary(verification),
                    provenance='rgbd_'+verification.get('seed_method', 'depth_ransac'),
                    matcher=verification.get('matcher'), source_session=observation['source_id'], target_session=target['source_id'])
        self.constraints[candidate_id] = edge
        candidate.update(state='verified', constraint_id=candidate_id)
        self._corroborate(edge)
        return True

    def backfill_one(self):
        """Spend at most one verifier call on current-session deferred evidence.

        The live owner may call this on its rate-limited idle path. Snapshots,
        ordinary replay reads and operator commands never run it implicitly.
        """
        if not self.config['verifications_per_keyframe']:
            return False
        c = self._next_backfill(set(self.robots.values()))
        if c is None or not self._check_candidate(c, 'historical_backfill'):
            return False
        if c['verification']['ok']:
            self._optimize()
            self._update_entities()
        while len(self.candidates) > 300:
            self.candidates.pop(next(iter(self.candidates)))
        self.revision += 1
        self._persist_state()
        return True

    def _corroborate(self, edge):
        first = self.keyframes[edge['source_frame']]
        second = self.keyframes[edge['target_frame']]
        witnesses = [edge]
        for previous in self.constraints.values():
            if previous is edge or previous.get('kind') != 'loop' or previous.get('state') == 'rejected':
                continue
            same = (previous.get('source_session'), previous.get('target_session')) == (edge['source_session'], edge['target_session'])
            reverse = (previous.get('target_session'), previous.get('source_session')) == (edge['source_session'], edge['target_session'])
            if not same and not reverse:
                continue
            relation = np.asarray(previous['T_targetlocal_sourcelocal'])
            if reverse:
                relation = np.linalg.inv(relation)
            distance, angle = pose_delta(relation, edge['T_targetlocal_sourcelocal'])
            if distance > self.config['transform_agreement_translation'] or angle > math.radians(self.config['transform_agreement_rotation_deg']):
                continue
            old_source, old_target = previous['source_frame'], previous['target_frame']
            if reverse:
                old_source, old_target = old_target, old_source
            # A changed, pose-supported view on either rover supplies a new
            # observation, independent of which rover triggered retrieval.
            if not (self._distinct_views(self.keyframes[old_source], first) or
                    self._distinct_views(self.keyframes[old_target], second)):
                continue
            witnesses.append(previous)
        if len(witnesses) >= 2:
            for witness in witnesses:
                witness.update(corroborations=max(witness.get('corroborations', 1), len(witnesses)))
                if witness.get('state') != 'accepted':
                    witness['state'] = 'corroborated'
                if not self.config['shadow_mode'] and not witness.get('operator_decision'):
                    if self._alignment_compatible(witness):
                        witness.update(active=True, state='accepted')
                    else:
                        witness.update(active=False, state='conflicting', reason='disagrees_with_accepted_alignment')
                if witness['id'] in self.candidates:
                    self.candidates[witness['id']]['state'] = witness['state']

    def _alignment_compatible(self, edge):
        for previous in self.constraints.values():
            if previous is edge or not previous.get('active') or previous.get('kind') != 'loop':
                continue
            same = (previous.get('source_session'), previous.get('target_session')) == (edge['source_session'], edge['target_session'])
            reverse = (previous.get('target_session'), previous.get('source_session')) == (edge['source_session'], edge['target_session'])
            if same or reverse:
                relation = np.asarray(previous['T_targetlocal_sourcelocal'])
                distance, angle = pose_delta(np.linalg.inv(relation) if reverse else relation, edge['T_targetlocal_sourcelocal'])
                if distance > self.config['transform_agreement_translation'] or angle > math.radians(self.config['transform_agreement_rotation_deg']):
                    return False
        a, b = self.submaps[edge['source']], self.submaps[edge['target']]
        if a['aligned'] and b['aligned']:
            expected = np.linalg.inv(b['T_world_submap']) @ a['T_world_submap']
            distance, angle = pose_delta(expected, edge['T_target_source'])
            if distance > self.config['transform_agreement_translation'] or angle > math.radians(self.config['transform_agreement_rotation_deg']):
                return False
        return True

    def _optimize(self):
        self._preview_key = None  # Graph changes, including reject/undo, can leave edge count unchanged.
        reference = self.sessions.get(self.reference_session, {})
        roots = reference.get('keyframe_ids', [])
        if not roots:
            return
        active = []
        # Native pose nodes are keyframes, including poses *inside* each packet.
        # Submap membership and source-owned surface identities never change.
        for source in self.sessions.values():
            for previous_id, current_id in zip(source['keyframe_ids'], source['keyframe_ids'][1:]):
                previous, current = self.keyframes[previous_id], self.keyframes[current_id]
                relative = np.linalg.inv(previous['T_local_camera']) @ current['T_local_camera']
                distance, angle = pose_delta(previous['T_local_camera'], current['T_local_camera'])
                active.append(dict(source=current_id, target=previous_id, T_target_source=relative,
                    kind='odometry', translation_sigma=.05+distance*.15,
                    rotation_sigma=math.radians(3)+angle*.2))
        for edge in self.constraints.values():
            if not edge.get('active') or edge['kind'] == 'odometry':
                continue
            if edge['kind'] == 'loop':
                a, b = edge['source_frame'], edge['target_frame']
                if a not in self.keyframes or b not in self.keyframes:
                    continue
                measurement = edge.get('T_targetcamera_sourcecamera')
                if measurement is None:  # Earlier evidence stores use submap-relative loops.
                    first, second = self.keyframes[a], self.keyframes[b]
                    measurement = (np.linalg.inv(second['T_local_camera']) @ self.submaps[edge['target']]['T_local_submap']
                        @ np.asarray(edge['T_target_source']) @ np.linalg.inv(self.submaps[edge['source']]['T_local_submap']) @ first['T_local_camera'])
            else:  # A supplied local-frame alignment anchors packet origin keyframes.
                a = self.submaps[edge['source']]['keyframe_ids'][0]
                b = self.submaps[edge['target']]['keyframe_ids'][0]
                measurement = edge['T_target_source']
            active.append(dict(edge, source=a, target=b, T_target_source=measurement))
        # Initialize newly connected components using edge relations before optimization.
        known = {roots[0]}
        poses = {roots[0]: self.keyframes[roots[0]]['T_local_camera'].copy()}
        while True:
            before = len(known)
            for edge in active:
                a, b, z = edge['source'], edge['target'], np.asarray(edge['T_target_source'])
                if b in known and a not in known:
                    poses[a] = poses[b] @ z; known.add(a)
                elif a in known and b not in known:
                    poses[b] = poses[a] @ np.linalg.inv(z); known.add(b)
            if before == len(known):
                break
        relevant = [edge for edge in active if edge['source'] in known and edge['target'] in known]
        optimized, self.graph_status = optimize_pose_graph(poses, relevant, fixed=[roots[0]], iterations=4)
        rebuilt = corrected = 0
        for submap in self.submaps.values():
            origin = submap['keyframe_ids'][0]
            submap['aligned'] = origin in optimized
            submap['T_world_submap'] = optimized.get(origin, submap['T_local_submap']).copy()
            changed = False
            for key in submap['keyframe_ids']:
                frame = self.keyframes[key]
                raw = np.linalg.inv(submap['T_local_submap']) @ frame['T_local_camera']
                relative = np.linalg.inv(submap['T_world_submap']) @ optimized[key] if key in optimized else raw
                changed |= not np.allclose(relative, frame.get('T_submap_camera', raw), atol=1e-6)
                corrected += not np.allclose(relative, raw, atol=1e-4)
                frame['T_submap_camera'] = relative
            if changed:
                self._rebuild_submap(submap)
                rebuilt += 1
        self.graph_status.update(corrected_internal_keyframes=int(corrected), rebuilt_submaps=rebuilt)

    def _entity_evidence(self):
        entities = {}
        # Latest observations dominate geometry without deleting source history.
        for frame in reversed(list(self.keyframes.values())):
            submap = self.submaps[frame['submap_id']]
            if not submap['aligned']:
                continue
            camera_world = submap['T_world_submap'] @ frame.get('T_submap_camera', np.linalg.inv(submap['T_local_submap']) @ frame['T_local_camera'])
            for surface in frame['surfaces']:
                if surface['entity_id'] is None or surface['quality'] < .3:
                    continue
                uid = identifier('entity', frame['robot_id'], frame['session_id'], surface['entity_id'])
                entry = entities.setdefault(uid, dict(id=uid, robot_id=frame['robot_id'], session_id=frame['session_id'],
                    local_id=surface['entity_id'], points=[], dino=[], clip=[], frames=[], feature_space_id=frame['feature_space_id']))
                if len(entry['frames']) >= 5 and frame['id'] not in entry['frames']:
                    continue
                entry['points'].append(transform(surface['points_camera'], camera_world))
                for name in ('dino', 'clip'):
                    if surface[name] is not None:
                        entry[name].append(surface[name])
                if frame['id'] not in entry['frames']:
                    entry['frames'].append(frame['id'])
        bounded = dict(list(entities.items())[:self.config['max_entities']])
        for entry in bounded.values():
            entry['points'], _ = voxel_sample(np.vstack(entry['points']), size=.06, limit=400)
            for name in ('dino', 'clip'):
                values = entry[name]
                entry[name] = unit(np.mean(values, axis=0)) if values and all(v.shape == values[0].shape for v in values) else None
        return bounded

    def _update_entities(self):
        entities = self._entity_evidence()
        values = list(entities.values())
        observed = set()
        for index, a in enumerate(values):
            for b in values[index+1:]:
                if a['robot_id'] == b['robot_id'] or a['feature_space_id'] != b['feature_space_id']:
                    continue
                pair = sorted((a['id'], b['id']))
                link_id = identifier('entity_link', *pair)
                if link_id in self.entity_exclusions:
                    continue
                if np.linalg.norm(np.median(a['points'], axis=0)-np.median(b['points'], axis=0)) > 1.5:
                    continue
                appearance, semantic = cosine(a['dino'], b['dino']), cosine(a['clip'], b['clip'])
                if appearance is None or appearance < self.config['entity_dino_similarity']:
                    continue
                if semantic is not None and semantic < self.config['entity_clip_similarity']:
                    continue
                forward, _ = nearest(a['points'], b['points'])
                reverse, _ = nearest(b['points'], a['points'])
                overlap = min(float((forward < self.config['entity_distance']).mean()), float((reverse < self.config['entity_distance']).mean()))
                if overlap < self.config['entity_overlap']:
                    continue
                observed.add(link_id)
                witnesses = sorted(set(a['frames']+b['frames']))
                link = self.entity_links.get(link_id, dict(id=link_id, members=pair, state='proposed', active=False,
                                                         provenance='aligned_geometry_and_appearance', witnesses=[]))
                link.update(dino=appearance, clip=semantic, overlap=overlap, confidence=min(appearance, overlap), witnesses=witnesses)
                # Independent viewpoints in both sources, not repeated static/cached frames.
                diverse = []
                for entity in (a, b):
                    frames = [self.keyframes[k] for k in entity['frames']]
                    diverse.append(any(self._distinct_views(x, y) for x in frames for y in frames))
                if all(diverse) and link['state'] not in ('rejected', 'accepted'):
                    link['state'] = 'corroborated'
                    if not self.config['shadow_mode']:
                        link.update(active=True, state='accepted')
                self.entity_links[link_id] = link
        # A missing entity is not negative evidence. Lack of *alignment* suspends links.
        for link in self.entity_links.values():
            link['alignment_valid'] = all(member in entities for member in link['members'])
            link['compatible_current'] = link['id'] in observed if link['alignment_valid'] else None
            if link['alignment_valid'] and not link['compatible_current']:
                link['reason'] = 'current_aligned_evidence_contradicts_link; historical_decision_retained'
        groups = {uid: {uid} for uid in entities}
        for link in self.entity_links.values():
            if not link.get('active') or not link.get('alignment_valid') or not link.get('compatible_current'):
                continue
            left, right = link['members']
            union = groups[left] | groups[right]
            # Do not collapse two co-visible objects from one source transitively.
            source_keys = [(entities[key]['robot_id'], entities[key]['session_id']) for key in union]
            if len(set(source_keys)) < len(source_keys):
                link['reason'] = 'same_source_identity_conflict'
                continue
            for key in union:
                groups[key] = union
        self._entity_cache = []
        visited = set()
        self.entity_aliases = {}
        for uid in entities:
            if uid not in self.entity_registry:
                self.entity_registry[uid] = dict(id=identifier('global_entity', uid), order=len(self.entity_registry))
        for uid, group in groups.items():
            if uid in visited:
                continue
            visited.update(group)
            points, _ = voxel_sample(np.vstack([entities[key]['points'] for key in sorted(group)]), size=.07, limit=500)
            canonical = min((self.entity_registry[key] for key in group), key=lambda record: record['order'])['id']
            for key in group:
                original = self.entity_registry[key]['id']
                if original != canonical:
                    self.entity_aliases[original] = canonical
            self._entity_cache.append(dict(id=canonical,
                members=[dict(id=key, robot_id=entities[key]['robot_id'], session_id=entities[key]['session_id'], local_id=entities[key]['local_id']) for key in sorted(group)],
                points=points, aligned=True, objectness='unverified',
                confidence='cross_source_linked_hypothesis' if len(group)>1 else 'source_hypothesis'))

    def _anchor(self, command):
        robot, reference = str(command['robot_id']), str(command.get('reference_robot_id') or '')
        if not reference and self.reference_session:
            reference = self.sessions[self.reference_session]['robot_id']
        if robot == reference:
            raise ValueError('Anchor source and reference rover must differ')
        relation = valid_transform(command['T_reference_local'])
        source_id, target_id = self.robots.get(robot), self.robots.get(reference)
        if not source_id or not target_id or not self.sessions[source_id]['submap_ids'] or not self.sessions[target_id]['submap_ids']:
            self.pending_anchors = [item for item in self.pending_anchors if item['robot_id'] != robot]
            self.pending_anchors.append(dict(command, reference_robot_id=reference, T_reference_local=relation.tolist()))
            return dict(ok=True, pending=True, reason='Waiting for observations from both rovers')
        a = self.submaps[self.sessions[source_id]['submap_ids'][-1]]
        b = self.submaps[self.sessions[target_id]['submap_ids'][-1]]
        if not b['aligned']:
            raise ValueError('Reference rover must already be aligned to the world')
        for edge in self.constraints.values():
            if edge['kind'] == 'anchor' and edge.get('robot_id') == robot:
                edge.update(active=False, state='replaced')
        edge_id = identifier('anchor', source_id, target_id, self.revision)
        self.constraints[edge_id] = dict(id=edge_id, source=a['id'], target=b['id'],
            T_target_source=np.linalg.inv(b['T_local_submap']) @ relation @ a['T_local_submap'],
            T_reference_local=relation, robot_id=robot, reference_robot_id=reference,
            kind='anchor', state='accepted', active=True, corroborations=0,
            translation_sigma=.02, rotation_sigma=math.radians(1), provenance=str(command.get('provenance', 'operator_supplied_transform')),
            reason=str(command.get('reason', 'Explicit trusted alignment; not automatic recognition')))
        self._event('anchor', f'Trusted alignment: {robot} into {reference}', constraint_id=edge_id)
        return dict(ok=True, constraint_id=edge_id, convention='p_reference_local = T_reference_local @ p_robot_local')

    def _apply_pending_anchors(self):
        pending, self.pending_anchors = self.pending_anchors, []
        for command in pending:
            self._anchor(command)

    def command(self, command):
        action = command.get('action', command.get('type', command.get('command')))
        result = dict(ok=True)
        if action == 'anchor':
            result = self._anchor(command)
        elif action == 'clear_anchor':
            robot = str(command['robot_id'])
            self.pending_anchors = [item for item in self.pending_anchors if item['robot_id'] != robot]
            for edge in self.constraints.values():
                if edge['kind'] == 'anchor' and edge.get('robot_id') == robot:
                    edge.update(active=False, state='rejected')
            self._event('anchor_removed', f'Removed supplied alignment for {robot}; local evidence preserved')
        elif action in ('accept_constraint', 'reject_constraint'):
            edge = self.constraints.get(str(command['id']))
            if edge is None or edge['kind'] == 'odometry':
                raise ValueError('Unknown or non-editable constraint')
            accept = action == 'accept_constraint'
            if accept and edge.get('state') not in ('verified', 'corroborated', 'accepted'):
                raise ValueError('Only geometrically verified constraints can be accepted')
            edge.update(active=accept, state='accepted' if accept else 'rejected', operator_decision=True)
            if edge['id'] in self.candidates:
                self.candidates[edge['id']]['state'] = edge['state']
            self._event('constraint', f'{action}: evidence remains available', constraint_id=edge['id'])
        elif action in ('accept_entity_link', 'unlink_entities'):
            link = self.entity_links.get(str(command['id']))
            if link is None:
                raise ValueError('Unknown entity association')
            accept = action == 'accept_entity_link'
            if accept and not link.get('alignment_valid'):
                raise ValueError('Entity association requires current verified/supplied alignment')
            link.update(active=accept, state='accepted' if accept else 'rejected', operator_decision=True)
            if accept:
                self.entity_exclusions.discard(link['id'])
            else:
                self.entity_exclusions.add(link['id'])
        elif action == 'set_shadow_mode':
            if type(command.get('enabled')) is not bool:
                raise ValueError('enabled must be a JSON boolean')
            self.config['shadow_mode'] = command['enabled']
            # Changing mode never retroactively accepts historical hypotheses.
            self._event('mode', f"Automatic acceptance {'disabled' if self.config['shadow_mode'] else 'enabled for future corroborated evidence'}")
        elif action not in ('save', 'snapshot'):
            raise ValueError(f'Unknown backend command: {action}')
        self._optimize()
        self._update_entities()
        self.revision += 1
        self._persist_state()
        result['revision'] = self.revision
        return result

    def snapshot(self):
        # Do not run registration on HTTP polls. Only presentation uses these
        # separate poses; solver/identities/persistence still see confirmed data.
        key = (len(self.keyframes), len(self.constraints), tuple(self.robots.values()),
               self.config['shadow_mode'], self.config['provisional_enabled'])
        if self._preview_key != key:
            self._preview_cache = derive_previews(self)
            self._preview_key = key
        audits, preview = self._preview_cache
        robots = []
        for robot_id, source_id in self.robots.items():
            source = self.sessions[source_id]
            maps = [self.submaps[key] for key in source['submap_ids']]
            latest = maps[-1] if maps else None
            latest_frame = self.keyframes[source['keyframe_ids'][-1]] if source['keyframe_ids'] else None
            world_camera = latest['T_world_submap'] @ latest_frame['T_submap_camera'] if latest and latest['aligned'] else None
            robots.append(dict(robot_id=robot_id, session_id=source['session_id'], source_id=source_id,
                frames=source['frames'], keyframes=len(source['keyframe_ids']), submaps=len(maps), status=source['status'],
                aligned=bool(latest and latest['aligned']),
                alignment_state=('confirmed' if latest and latest['aligned'] else
                                 'provisional' if source_id in preview else 'unaligned'),
                last_stamp_ns=source['last_stamp_ns'],
                last_received=source.get('last_received'), pose_quality=source.get('pose_quality'), rejected_poses=source['rejected_poses'],
                trajectory=self._trajectory_snapshot(source),
                T_world_camera=world_camera,
                T_world_local=(world_camera @ np.linalg.inv(latest_frame['T_local_camera'])) if world_camera is not None else None))
        maps = []
        per_map = max(32, self.config['snapshot_points']//max(1, len(self.submaps)))
        for submap in self.submaps.values():
            indices = np.linspace(0, len(submap['points'])-1, min(per_map, len(submap['points']))).astype(int)
            maps.append(dict(id=submap['id'], robot_id=submap['robot_id'], session_id=submap['session_id'],
                keyframe_count=len(submap['keyframe_ids']), aligned=submap['aligned'],
                alignment_state=('confirmed' if submap['aligned'] else
                                 'provisional' if submap['source_id'] in preview else 'unaligned'),
                T_preview_world_submap=(preview[submap['source_id']] @ submap['T_local_submap']
                                        if not submap['aligned'] and submap['source_id'] in preview else None),
                T_world_submap=submap['T_world_submap'], T_local_submap=submap['T_local_submap'],
                points=submap['points'][indices], colors=submap['colors'][indices]))
        # The review list is paged to 100 items. Count retained candidates for
        # the current sessions before truncation so the summary explains the run.
        current = set(self.robots.values())
        candidates = [c for c in self.candidates.values() if all(
            self.keyframes.get(c[end], {}).get('source_id') in current for end in ('source', 'target'))]
        checked = [c['verification'] for c in candidates if 'verification' in c]
        failed = [v for v in checked if not v['ok']]
        best = max(failed, key=lambda v: (v.get('visual_inliers', v.get('inliers', 0)),
                                          v.get('depth_correspondences', 0), v.get('visual_matches', 0)), default=None)
        active = sum(e['kind']=='loop' and e['active'] for e in self.constraints.values()
                     if e.get('source_session') in current and e.get('target_session') in current)
        evidence = dict(retrieved=len(candidates), checked=len(checked), verified=sum(v['ok'] for v in checked),
                        provisional=[r for r in audits if r['status'] == 'provisional'],
                        provisional_diagnostics=audits,
                        active=active, best_rejection=None if best is None else dict(
                            reason=best['reason'], inliers=best.get('visual_inliers', best.get('inliers', 0)),
                            **{k: best[k] for k in ('visual_matches', 'verification_stage', 'matcher') if k in best}))
        return plain(dict(schema='crossmaps_collaborative_v1', revision=self.revision, shadow_mode=self.config['shadow_mode'],
            reference_session=self.reference_session, robots=robots, submaps=maps,
            alignment_evidence=evidence,
            candidates=list(self.candidates.values())[-100:], constraints=list(self.constraints.values()),
            global_entities=self._entity_cache, entity_links=list(self.entity_links.values()), entity_aliases=self.entity_aliases,
            pending_anchors=self.pending_anchors, events=list(self.events),
            stats=dict(keyframes=len(self.keyframes), submaps=len(self.submaps), sessions=len(self.sessions),
                       store_bytes=self.store_bytes, capacity_reason=self.capacity_reason, graph=self.graph_status,
                       verification_scheduler=self.verification_schedule.report(),
                       accepted_loops=sum(edge['kind']=='loop' and edge['active'] for edge in self.constraints.values()),
                entity_links=sum(link.get('active', False) and link.get('alignment_valid', False) and link.get('compatible_current', False) for link in self.entity_links.values())),
            semantics=dict(absence_is_negative_evidence=False, automatic_alignment='shadow' if self.config['shadow_mode'] else 'corroborated_only',
                           provisional_display_enabled=self.config['provisional_enabled'] and not self.config['shadow_mode'],
                           provisional_used_for_graph_or_identity=False,
                           local_ids_are_source_owned=True, robot_tf_modified=False, submap_internal_pose_correction=True,
                           geometry_rebuilt_from_original_camera_support=True,
                           T_world_local_is_latest_pose_diagnostic_not_rigid_map_transform=True)))

    def _persist_frame(self, frame):
        if not self.store_dir:
            return
        metadata = {key: frame[key] for key in ('id', 'source_id', 'submap_id', 'robot_id', 'session_id',
                    'seq', 'stamp_ns', 'rgb_stamp_ns', 'received_monotonic', 'calibration_id', 'feature_space_id', 'pose_quality') if key in frame}
        arrays = {key: frame[key] for key in ('rgb', 'depth', 'K', 'T_local_camera', 'points_camera', 'colors')}
        if frame.get('T_camera_base') is not None:
            arrays['T_camera_base'] = frame['T_camera_base']
        for key in ('global_descriptor', 'geometry_descriptor'):
            arrays[key] = np.asarray(frame[key] if frame[key] is not None else [], dtype=np.float32)
        metadata['surfaces'] = []
        for index, surface in enumerate(frame['surfaces']):
            metadata['surfaces'].append({key: surface[key] for key in ('local_id', 'entity_id', 'quality')})
            for key in ('points_camera', 'dino', 'clip'):
                arrays[f's{index}_{key}'] = np.asarray(surface[key] if surface[key] is not None else [], dtype=np.float32)
        arrays['metadata'] = np.asarray(json.dumps(plain(metadata), allow_nan=False))
        destination = self.store_dir/f"frame-{frame['id']}.npz"
        temporary = destination.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, destination)
        self.store_bytes += destination.stat().st_size

    def _persist_state(self):
        if not self.store_dir:
            return
        state = dict(schema='crossmaps_collaborative_store_v1', revision=self.revision, config=self.config,
            sessions=self.sessions, robots=self.robots, reference_session=self.reference_session,
            submaps={key: {name: value for name, value in item.items() if name not in ('points', 'colors')} for key, item in self.submaps.items()},
            constraints=self.constraints, candidates=self.candidates, entity_links=self.entity_links,
            verification_schedule=dict(checked=sorted(self.verification_schedule.checked),
                                       pending=list(self.verification_schedule.pending.values()),
                                       turn=self.verification_schedule.turn, backfilled=self.verification_schedule.backfilled,
                                       cursor=self.verification_schedule.cursor),
            entity_registry=self.entity_registry,
            entity_exclusions=sorted(self.entity_exclusions), pending_anchors=self.pending_anchors,
            events=list(self.events), capacity_reason=self.capacity_reason)
        temporary = self.store_dir/'state.json.tmp'
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(plain(state), stream, allow_nan=False, separators=(',', ':'))
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, self.store_dir/'state.json')

    def _restore(self):
        path = self.store_dir/'state.json'
        if not path.exists():
            return
        state = json.loads(path.read_text(encoding='utf-8'))
        if state.get('schema') != 'crossmaps_collaborative_store_v1':
            raise ValueError('Unsupported collaborative evidence schema')
        self.config = validate_config(dict(state.get('config', {}), **self._overrides))
        for key in ('sessions', 'robots', 'reference_session', 'constraints', 'candidates', 'entity_links', 'pending_anchors', 'capacity_reason', 'revision'):
            setattr(self, key, state[key])
        self.entity_exclusions = set(state.get('entity_exclusions', []))
        self.entity_registry = state.get('entity_registry', {})
        self.events = deque(state.get('events', []), maxlen=80)
        self.submaps = state['submaps']
        for submap in self.submaps.values():
            for key in ('T_local_submap', 'T_world_submap'):
                submap[key] = valid_transform(submap[key])
        # Only expected hashed filenames are opened; no pickle/object arrays accepted.
        for source in self.sessions.values():
            for key in source['keyframe_ids']:
                if len(key) != 20 or any(char not in '0123456789abcdef' for char in key):
                    raise ValueError('Invalid persisted frame ID')
                frame_path = self.store_dir/f'frame-{key}.npz'
                with np.load(frame_path, allow_pickle=False) as arrays:
                    frame = json.loads(str(arrays['metadata'].item()))
                    for name in ('rgb', 'depth', 'K', 'T_local_camera', 'points_camera', 'colors'):
                        frame[name] = arrays[name].copy()
                    if 'T_camera_base' in arrays:
                        frame['T_camera_base'] = valid_transform(arrays['T_camera_base'])
                    for name in ('global_descriptor', 'geometry_descriptor'):
                        frame[name] = arrays[name].copy() if arrays[name].size else None
                    for index, surface in enumerate(frame['surfaces']):
                        for name in ('points_camera', 'dino', 'clip'):
                            data = arrays[f's{index}_{name}']
                            surface[name] = data.copy() if data.size else None
                self.keyframes[key] = frame
                # A reference session may have no accepted root yet. Even when
                # the graph can't solve, replay must preserve local trajectories.
                frame['T_submap_camera'] = (np.linalg.inv(self.submaps[frame['submap_id']]['T_local_submap'])
                                            @ frame['T_local_camera'])
                self._cache_features(frame)
                self.store_bytes += frame_path.stat().st_size
            if 'trail' not in source:
                # Old recordings retained only keyframes. Missing sequence
                # numbers can hide rejected poses, so don't invent connections.
                previous = None
                for key in source['keyframe_ids'][-TRAIL_POINTS:]:
                    frame = self.keyframes[key]
                    self._record_trajectory_pose(source, frame, frame, disconnected=(
                        previous is None or frame['seq'] != previous['seq'] + 1))
                    previous = frame
                if previous and (source.get('status') != 'observing' or
                                 source.get('last_seq') != previous['seq']):
                    source['trail']['tracking_lost'] = True
        # Legacy recordings didn't persist rejected pairs once their 300-card
        # review window rolled over. Import every check we *can* prove, and
        # always honor retained loop/operator decisions even off that window.
        saved = state.get('verification_schedule', {})
        checked = [*saved.get('checked', ()),
                   *((c['source'], c['target']) for c in self.candidates.values() if 'verification' in c),
                   *((e['source_frame'], e['target_frame']) for e in self.constraints.values() if e['kind'] == 'loop')]
        schedule = self.verification_schedule
        for left, right in checked:
            if left in self.keyframes and right in self.keyframes:
                schedule.mark(self.keyframes[left], self.keyframes[right])
        schedule.turn = saved.get('turn', 0)
        schedule.backfilled = saved.get('backfilled', 0)
        schedule.cursor = saved.get('cursor', 0)
        for c in saved.get('pending', self.candidates.values()):
            if c.get('state') == 'proposed' and 'verification' not in c:
                schedule.add(c)
        for submap in self.submaps.values():
            self._rebuild_submap(submap)
        self._optimize()
        self._update_entities()
        self._event('restore', 'Restored source evidence and explicit association decisions; no robot TF modified')

    @classmethod
    def replay(cls, store_dir, config=None):
        """Read a stored run into an independent in-memory backend for diagnostics."""
        restored = cls(config=config, store_dir=store_dir)
        restored.store_dir = None
        return restored
