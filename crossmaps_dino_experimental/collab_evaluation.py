"""Read-only simulation scoring; ground truth must never enter the estimator.

All comparisons use capture simulation stamps. This evaluates the retained
keyframe graph, not ROS localization or instantaneous asynchronous rover poses.
No transform, threshold, acceptance decision or robot command is modified here.
"""
from collections import OrderedDict
import bisect
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np


def rigid(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or abs(np.linalg.det(matrix[:3, :3])-1) > 1e-5):
        raise ValueError('Expected a finite rigid SE(3) transform')
    return matrix.copy()


def mount(entry):
    q = np.asarray(entry['quaternion_xyzw'], dtype=float)
    t = np.asarray(entry['translation'], dtype=float)
    if (q.shape != (4,) or t.shape != (3,) or not np.isfinite(q).all()
            or not np.isfinite(t).all() or abs(float(q @ q)-1) > 1e-6):
        raise ValueError('Invalid optical mount calibration')
    x, y, z, w = q
    result = np.eye(4)
    result[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    result[:3, 3] = t
    return rigid(result)


def error(estimate, truth):
    delta = np.linalg.inv(truth) @ estimate
    return float(np.linalg.norm(delta[:3, 3])), math.degrees(float(np.linalg.norm(cv2.Rodrigues(delta[:3, :3])[0])))


def rmse(values):
    return float(np.sqrt(np.mean(np.square(values)))) if values else None


def interval(values):
    return dict(min=min(values), max=max(values)) if values else None


def associate(samples, stamp, max_gap_ns=200_000_000):
    """Linear translation / shortest-arc SO(3) interpolation; no extrapolation."""
    stamps = [row[0] for row in samples]
    index = bisect.bisect_left(stamps, stamp)
    if index < len(samples) and samples[index][0] == stamp:
        return samples[index][1].copy()
    if index == 0 or index == len(samples):
        return None
    before, after = samples[index-1], samples[index]
    if after[0]-before[0] > max_gap_ns:
        return None
    fraction = (stamp-before[0])/(after[0]-before[0])
    result = np.eye(4)
    result[:3, 3] = (1-fraction)*before[1][:3, 3] + fraction*after[1][:3, 3]
    relative = before[1][:3, :3].T @ after[1][:3, :3]
    result[:3, :3] = before[1][:3, :3] @ cv2.Rodrigues(cv2.Rodrigues(relative)[0]*fraction)[0]
    return result


class SimulationEvaluator:
    """Bounded, external truth cache for the currently retained evidence graph."""
    def __init__(self, truth_path, scene_path, *, clock=time.time):
        self.truth_path, self.scene_path = Path(truth_path), Path(scene_path)
        self.clock = clock
        scene = json.loads(self.scene_path.read_text())
        if scene.get('world') != 'crossmaps_lab':
            raise ValueError('Evaluation requires the generated CrossMaps simulation scene')
        self.mounts = {robot: mount(scene['robots'][robot]['camera_optical'])
                       for robot in ('rover_a', 'rover_b')}
        self.reset()

    def reset(self):
        self.run_id = None
        self.cache = OrderedDict()
        self.latest_truth = {}
        self.blocked = None

    def _empty(self, status, reason, age=None):
        return dict(enabled=True, schema=1, status=status, reason=reason, run_id=self.run_id,
                    updated_unix=self.clock(),
                    coverage=dict(keyframes=0, associated=0, missing=0, truth_age_s=age),
                    local=[], shared=dict(confirmed=self._summary([]), provisional=self._summary([])),
                    optimization=dict(samples=0, raw_position_rmse_m=None, optimized_position_rmse_m=None,
                                      raw_rotation_rmse_deg=None, optimized_rotation_rmse_deg=None,
                                      label='Local odometry with current first-keyframe alignment vs optimized'),
                    loops=dict(evaluated=0, correct=0, precision=None, translation_rmse_m=None,
                               rotation_rmse_deg=None, accepted_evaluated=0, accepted_correct=0,
                               accepted_precision=None, translation_threshold_m=.1, rotation_threshold_deg=5.),
                    unsupported=['Full-surface IoU/F-score', 'Loop-closure recall', 'Single-rover SLAM Toolbox accuracy'],
                    notes=['Ground truth is evaluation-only; never supplied to perception, matching, TF or optimization.',
                           'Local error is first-pose anchored, not a fitted TUM ATE; RPE uses variable retained-keyframe intervals.',
                           'Shared pose errors compare asynchronous retained camera capture pairs, not instantaneous rover separation.',
                           'Optimization comparison uses one fixed reference gauge and the same keyframes; it is not a captured before/after solve.',
                           'Loop correctness uses explicit 0.10 m / 5 degree thresholds; no overlap oracle means no recall.',
                           'Simulator calibration and physics are approximate; simulation accuracy is not hardware validation.'])

    @staticmethod
    def _summary(rows):
        return dict(pairs=len(rows), translation_rmse_m=rmse([r[0] for r in rows]),
                    rotation_rmse_deg=rmse([r[1] for r in rows]),
                    capture_skew_s=interval([r[2] for r in rows]))

    def _truth(self):
        if self.truth_path.stat().st_size > 2*1024*1024:
            raise ValueError('Truth sidecar exceeds bounded size')
        value = json.loads(self.truth_path.read_text())
        if value.get('schema') != 1 or type(value.get('schema')) is not int:
            raise ValueError('Unsupported truth schema')
        if value.get('valid') is not True:
            raise ValueError('Ground-truth collector unavailable: ' + str(value.get('error', 'invalid stream')))
        run_id, updated = value.get('run_id'), value.get('updated_unix')
        if not isinstance(run_id, str) or not run_id or len(run_id) > 200:
            raise ValueError('Missing truth run identity')
        if type(updated) not in (int, float) or not math.isfinite(updated):
            raise ValueError('Invalid truth wall timestamp')
        age = self.clock()-updated
        if age < -1:
            raise ValueError('Truth wall timestamp is in the future')
        if self.run_id is not None and run_id != self.run_id:
            self.cache.clear()
            self.blocked = 'Simulator run changed; start a fresh map/evaluation session'
        if self.blocked:
            raise ValueError(self.blocked)
        self.run_id = run_id
        robots = value.get('robots')
        if not isinstance(robots, dict) or not set(robots).issubset(self.mounts):
            raise ValueError('Invalid truth robot identities')
        result = {}
        for robot, rows in robots.items():
            if not isinstance(rows, list) or len(rows) > 300:
                raise ValueError('Truth history must contain at most 300 samples per rover')
            samples = []
            for row in rows:
                stamp = row.get('stamp_ns')
                if type(stamp) is not int or stamp < 0 or (samples and stamp <= samples[-1][0]):
                    raise ValueError('Truth samples require strictly increasing simulation timestamps')
                samples.append((stamp, rigid(row['T_world_base'])))
            if samples:
                received = value.get('last_received_unix', {}).get(robot)
                if received is None:
                    raise ValueError('Truth samples have no sensor receipt timestamp')
                if received is not None:
                    if type(received) not in (int, float) or not math.isfinite(received) or received > self.clock()+1:
                        raise ValueError('Invalid truth sensor receipt timestamp')
                    age = max(age, self.clock()-received)
                if samples[-1][0] < self.latest_truth.get(robot, -1):
                    self.cache.clear()
                    self.blocked = 'Simulation clock moved backwards; start a fresh map/evaluation session'
                    raise ValueError(self.blocked)
                self.latest_truth[robot] = samples[-1][0]
            result[robot] = samples
        return result, max(0., age)

    def snapshot(self, backend):
        """Caller holds the backend lock; reads backend state but never mutates it."""
        try:
            truth, age = self._truth()
        except FileNotFoundError:
            return self._empty('waiting', 'Waiting for simulator ground-truth samples')
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            return self._empty('invalid', str(exc))
        if age > 5:
            return self._empty('stale', 'Ground-truth stream is stale; live accuracy unavailable', age)
        report = self._empty('waiting', 'Waiting for capture-time ground-truth association', age)
        current_sources = set(backend.robots.values())
        frames = [frame for frame in backend.keyframes.values() if frame['source_id'] in current_sources]
        frames.sort(key=lambda frame: (frame['stamp_ns'], frame['id']))
        current_ids = {frame['id'] for frame in frames}
        self.cache = OrderedDict((key, value) for key, value in self.cache.items() if key in current_ids)
        for frame in frames:
            identity = (frame['robot_id'], frame['session_id'], frame['stamp_ns'])
            old = self.cache.get(frame['id'])
            if old is not None and old[0] == identity:
                continue
            self.cache.pop(frame['id'], None)
            if frame['robot_id'] not in self.mounts:
                continue
            pose = associate(truth.get(frame['robot_id'], []), frame['stamp_ns'])
            if pose is not None:
                self.cache[frame['id']] = (identity, pose @ self.mounts[frame['robot_id']])
        while len(self.cache) > 512:
            self.cache.popitem(last=False)
        gt = {key: value[1] for key, value in self.cache.items()}
        report['coverage'].update(keyframes=len(frames), associated=len(gt), missing=len(frames)-len(gt))
        if not gt:
            return report
        report.update(status='ready', reason='Capture-time retained-keyframe evaluation; inspect sample coverage')
        by_source = {}
        for frame in frames:
            by_source.setdefault(frame['source_id'], []).append(frame)
        for source_frames in by_source.values():
            first = source_frames[0]
            measurements, rpe, spans = [], [], []
            if first['id'] in gt:
                gauge = gt[first['id']] @ np.linalg.inv(first['T_local_camera'])
                valid = [frame for frame in source_frames if frame['id'] in gt]
                measurements = [error(gauge @ frame['T_local_camera'], gt[frame['id']]) for frame in valid]
                for previous, frame in zip(valid, valid[1:]):
                    span = (frame['stamp_ns']-previous['stamp_ns'])/1e9
                    if span <= 0:
                        continue
                    estimated = np.linalg.inv(previous['T_local_camera']) @ frame['T_local_camera']
                    actual = np.linalg.inv(gt[previous['id']]) @ gt[frame['id']]
                    rpe.append(error(estimated, actual)); spans.append(span)
            report['local'].append(dict(robot_id=first['robot_id'], samples=len(measurements),
                                       position_rmse_m=rmse([v[0] for v in measurements]),
                                       rotation_rmse_deg=rmse([v[1] for v in measurements]),
                                       rpe_pairs=len(rpe), rpe_translation_rmse_m=rmse([v[0] for v in rpe]),
                                       rpe_rotation_rmse_deg=rmse([v[1] for v in rpe]), rpe_interval_s=interval(spans)))
        estimated, states = {}, {}
        preview = getattr(backend, '_preview_cache', ([], {}))[1]
        for frame in frames:
            submap = backend.submaps[frame['submap_id']]
            if submap['aligned']:
                estimated[frame['id']] = submap['T_world_submap'] @ frame['T_submap_camera']
                states[frame['id']] = 'confirmed'
            elif frame['source_id'] in preview:
                estimated[frame['id']] = preview[frame['source_id']] @ frame['T_local_camera']
                states[frame['id']] = 'provisional'
        groups = list(by_source.values())
        pair_errors = dict(confirmed=[], provisional=[])
        seen = set()
        for i, group in enumerate(groups):
            for other in groups[i+1:]:
                targets = [f for f in other if f['id'] in gt and f['id'] in estimated]
                if not targets:
                    continue
                for frame in group:
                    a = frame['id']
                    if a not in gt or a not in estimated:
                        continue
                    match = min(targets, key=lambda f: abs(f['stamp_ns']-frame['stamp_ns']))
                    b = match['id']
                    if (a, b) in seen:
                        continue
                    seen.add((a, b))
                    category = 'confirmed' if states[a] == states[b] == 'confirmed' else 'provisional'
                    e = error(np.linalg.inv(estimated[b]) @ estimated[a], np.linalg.inv(gt[b]) @ gt[a])
                    pair_errors[category].append((*e, abs(frame['stamp_ns']-match['stamp_ns'])/1e9))
        report['shared'] = {name: self._summary(values) for name, values in pair_errors.items()}
        reference = backend.sessions.get(backend.reference_session, {}).get('keyframe_ids', [])
        if reference and reference[0] in gt:
            root = backend.keyframes[reference[0]]
            gauge = gt[root['id']] @ np.linalg.inv(root['T_local_camera'])
            raw_errors, optimized_errors = [], []
            for source_frames in groups:
                first = source_frames[0]
                if first['id'] not in estimated or states[first['id']] != 'confirmed':
                    continue
                alignment = estimated[first['id']] @ np.linalg.inv(first['T_local_camera'])
                for frame in source_frames:
                    key = frame['id']
                    if key not in gt or key not in estimated or states[key] != 'confirmed':
                        continue
                    raw_errors.append(error(gauge @ alignment @ frame['T_local_camera'], gt[key]))
                    optimized_errors.append(error(gauge @ estimated[key], gt[key]))
            report['optimization'].update(samples=len(raw_errors),
                raw_position_rmse_m=rmse([v[0] for v in raw_errors]), raw_rotation_rmse_deg=rmse([v[1] for v in raw_errors]),
                optimized_position_rmse_m=rmse([v[0] for v in optimized_errors]),
                optimized_rotation_rmse_deg=rmse([v[1] for v in optimized_errors]))
        loop_errors, correct, accepted, accepted_correct = [], 0, 0, 0
        for edge in backend.constraints.values():
            if edge.get('kind') != 'loop' or edge.get('state') not in ('verified', 'corroborated', 'accepted'):
                continue
            a, b = edge.get('source_frame'), edge.get('target_frame')
            if a not in gt or b not in gt or edge.get('T_targetcamera_sourcecamera') is None:
                continue
            e = error(rigid(edge['T_targetcamera_sourcecamera']), np.linalg.inv(gt[b]) @ gt[a])
            loop_errors.append(e)
            good = e[0] <= .1 and e[1] <= 5.
            correct += int(good)
            if edge.get('active'):
                accepted += 1; accepted_correct += int(good)
        report['loops'].update(evaluated=len(loop_errors), correct=correct,
            precision=correct/len(loop_errors) if loop_errors else None,
            translation_rmse_m=rmse([v[0] for v in loop_errors]), rotation_rmse_deg=rmse([v[1] for v in loop_errors]),
            accepted_evaluated=accepted, accepted_correct=accepted_correct,
            accepted_precision=accepted_correct/accepted if accepted else None)
        return report
