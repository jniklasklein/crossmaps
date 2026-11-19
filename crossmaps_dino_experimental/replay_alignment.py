#!/usr/bin/env python3
"""Compare the production RGB-D verifier to its ORB baseline on immutable evidence.

  python replay_alignment.py .runtime/collaboration/RUN --limit 8
  python replay_alignment.py .runtime/collaboration/RUN --pair SOURCE_ID:TARGET_ID

This is read-only: no ROS, backend restore, shared identities, TF or live constraints.
JSON goes to stdout. A verified pair is a rigid-pose hypothesis, not a map merge.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sys

import cv2
import numpy as np

from collab_geometry import depth_points, pose_delta, verify_rgbd, voxel_sample


ID = re.compile(r'[0-9a-f]{20}')
REPORT_FIELDS = ('ok', 'reason', 'matcher', 'verification_stage', 'source_features',
                 'target_features', 'visual_matches', 'depth_correspondences',
                 'ransac_inliers', 'visual_inliers', 'image_cells', 'icp_inliers', 'icp_rmse',
                 'overlap_source', 'overlap_target', 'free_space_conflict',
                 'seed_method', 'pnp_reason', 'pnp_forward_inliers', 'pnp_reverse_inliers',
                 'pnp_min_hull_fraction', 'attempts')
REPORT_FIELDS += ('rmse', 'residual_method', 'depth_rmse', 'reprojection_inliers',
                  'reprojection_holdout_inliers', 'reprojection_image_cells')


def frame_path(store, key):
    if not ID.fullmatch(key):
        raise ValueError(f'Invalid frame id: {key!r}')
    return store / f'frame-{key}.npz'


def sample(items, n):
    """Evenly sample deterministic input order, without duplicate indices."""
    if len(items) <= n:
        return items
    return [items[i] for i in np.linspace(0, len(items)-1, n, dtype=int)]


def select_pairs(store, state, strategy, limit, explicit):
    if explicit:
        pairs = []
        for text in explicit:
            a, sep, b = text.partition(':')
            if not sep or not ID.fullmatch(a) or not ID.fullmatch(b) or a == b:
                raise ValueError('Use --pair SOURCE_ID:TARGET_ID with two distinct saved frame IDs')
            pairs.append((a, b))
        return list(dict.fromkeys(pairs))
    if strategy == 'checked':
        return sample(sorted({(c['source'], c['target']) for c in state['candidates'].values()
                              if isinstance(c.get('verification'), dict)
                              and c['source_robot'] != c['target_robot']}), limit)
    # Near-in-time is a sampling convenience, not an assumption of clock sync,
    # overlapping fields of view, or an independent pose measurement.
    robots = defaultdict(list)
    for session in state['sessions'].values():
        for key in session['keyframe_ids']:
            with np.load(frame_path(store, key), allow_pickle=False) as z:
                meta = json.loads(str(z['metadata']))
            robots[meta['robot_id']].append((int(meta['stamp_ns']), key))
    if len(robots) != 2:
        raise ValueError('Nearest-time sampling expects exactly two rovers; use --pair otherwise')
    a, b = [sorted(robots[r]) for r in sorted(robots)]
    return [(key, min(b, key=lambda x: (abs(x[0]-stamp), x[1]))[1])
            for stamp, key in sample(a, limit)]


def load_frame(store, key):
    with np.load(frame_path(store, key), allow_pickle=False) as z:
        f = json.loads(str(z['metadata']))
        if f['id'] != key:
            raise ValueError(f'Frame id differs from filename: {key}')
        f.update({k: z[k] for k in ('rgb', 'depth', 'K', 'T_local_camera', 'points_camera')})
    return f


def controls(source):
    """Counterfactuals only: do not claim real-world negatives or precision."""
    rgb, depth, k = source['rgb'], source['depth'], source['K']
    independent = np.random.default_rng(31).integers(0, 256, rgb.shape, dtype=np.uint8)
    # Same image pixels, incompatible scene metric scale; always regenerate
    # the 3D cloud so that ICP cannot accidentally consult the original depth.
    scaled = np.where((depth > .15) & (depth*1.7 < 8.), depth*1.7, 0.).astype(np.float32)
    points, _ = depth_points(scaled, k)
    points, _ = voxel_sample(points, size=.05, limit=2400)
    # New dict/ids: a feature cache must not confuse the control with a real frame.
    # Both targets independently contradict the source (appearance vs geometry).
    # Start from immutable source fields, never a previously extracted matcher cache.
    base = {k: source[k] for k in ('rgb', 'depth', 'K', 'points_camera')}
    yield 'unrelated_texture', dict(base, id='control-texture-'+source['id'], rgb=independent)
    yield 'incompatible_depth_scale', dict(base, id='control-scale-'+source['id'],
                                            depth=scaled, points_camera=points)


def canonical_transform(source, target, estimate):
    src = f"{source['robot_id']}/{source['session_id']}"
    dst = f"{target['robot_id']}/{target['session_id']}"
    mat = target['T_local_camera'] @ estimate @ np.linalg.inv(source['T_local_camera'])
    return ((src, dst), mat) if src < dst else ((dst, src), np.linalg.inv(mat))


def consistency(verified, translation_gate, rotation_gate):
    """Compare only identified session-to-session transforms in same direction."""
    rows = []
    for (src, dst), poses in sorted(verified.items()):
        deltas = [(trans, math.degrees(angle)) for i, p in enumerate(poses)
                  for q in poses[i+1:] for trans, angle in (pose_delta(p, q),)]
        rows.append(dict(source=src, target=dst, verified_pairs=len(poses),
                         transform_comparisons=len(deltas),
                         agreeing_comparisons=sum(t <= translation_gate and a <= rotation_gate
                                                   for t, a in deltas),
                         median_translation_delta_m=float(np.median([t for t, _ in deltas])) if deltas else None,
                         median_rotation_delta_deg=float(np.median([a for _, a in deltas])) if deltas else None,
                         max_translation_delta_m=max((t for t, _ in deltas), default=None),
                         max_rotation_delta_deg=max((a for _, a in deltas), default=None)))
    return rows


def run(store, state, pairs, control_pairs, modes=('orb', 'auto')):
    config = state['config']
    params = dict(min_inliers=config['min_visual_inliers'], min_overlap=config['min_overlap'],
                  max_rmse=config['max_registration_rmse'])
    counts = {m: Counter() for m in modes}
    verified = {m: defaultdict(list) for m in modes}
    false_verifications = Counter()
    historical = {(c['source'], c['target']): c['verification']
                  for c in state['candidates'].values() if isinstance(c.get('verification'), dict)}
    stored_outcomes = Counter()
    rows = []

    def check(a, b, kind):
        row = dict(source=a['id'], target=b['id'], kind=kind, modes={})
        if kind == 'recorded':
            row['stamp_difference_s'] = (int(a['stamp_ns']) - int(b['stamp_ns']))*1e-9
            saved = historical.get((a['id'], b['id']))
            if saved is not None:
                row['stored_verification'] = {f: saved[f] for f in REPORT_FIELDS if f in saved}
                stored_outcomes[saved.get('reason', 'unknown')] += 1
        for mode in modes:
            r = verify_rgbd(a, b, matcher_mode=mode, **params)
            detail = {f: r[f] for f in REPORT_FIELDS if f in r}
            row['modes'][mode] = detail
            if kind == 'recorded':
                counts[mode][r['reason']] += 1
                if r['ok']:
                    group, matrix = canonical_transform(a, b, r['T_target_source'])
                    verified[mode][group].append(matrix)
                    detail.update(local_transform_from=group[0], local_transform_to=group[1],
                                  local_transform=matrix.tolist())
            elif r['ok']:
                false_verifications[mode] += 1
        rows.append(row)

    for i, (s, t) in enumerate(pairs):
        a, b = load_frame(store, s), load_frame(store, t)
        if a['robot_id'] == b['robot_id']:
            raise ValueError(f'Not a cross-rover pair: {s}:{t}')
        check(a, b, 'recorded')
        if i < control_pairs:
            for label, other in controls(a):
                check(a, other, label)
    return dict(revision=state['revision'], opencv=cv2.__version__, verification_settings=params,
                recorded_pairs=len(pairs), control_pairs_tested=len(rows)-len(pairs),
                outcomes=counts, original_stored_outcomes=stored_outcomes,
                control_false_verifications={m: false_verifications[m] for m in modes},
                transform_agreement=dict(translation_m=config['transform_agreement_translation'],
                                         rotation_deg=config['transform_agreement_rotation_deg']),
                consistency={m: consistency(verified[m], config['transform_agreement_translation'],
                                            config['transform_agreement_rotation_deg']) for m in modes},
                note='Pair verification and repeatability are not ground truth or independent corroboration. No merges applied.',
                pairs=rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path, help='collaboration run directory (or its evidence subdirectory)')
    p.add_argument('--selection', choices=('checked', 'nearest'), default='checked')
    p.add_argument('--limit', type=int, default=6, help='maximum recorded pairs for automatic selection')
    p.add_argument('--pair', action='append', default=[], help='explicit SOURCE_ID:TARGET_ID (repeatable)')
    p.add_argument('--control-pairs', type=int, default=1, help='run two negative controls on each of the first N pairs')
    args = p.parse_args(argv)
    if args.limit < 1 or args.control_pairs < 0:
        p.error('limit must be positive and control-pairs nonnegative')
    cv2.setNumThreads(1)
    store = args.run / 'evidence' if (args.run / 'evidence').is_dir() else args.run
    try:
        state = json.loads((store/'state.json').read_bytes())  # single atomic snapshot of live index
        pairs = select_pairs(store, state, args.selection, args.limit, args.pair)
        if not pairs:
            raise ValueError('No pairs selected; try --selection nearest or explicit --pair')
        report = run(store, state, pairs, args.control_pairs)
    except (KeyError, ValueError, OSError) as e:
        p.exit(2, f'Replay failed: {e}\n')
    report.update(store=str(store), selection='explicit' if args.pair else args.selection)
    print(json.dumps(report, indent=2, allow_nan=False))
    # Not knowing whether a recorded pair truly overlaps, absence of a match
    # is not a test failure. Accepting a deliberately contradictory pair is.
    return 1 if any(report['control_false_verifications'].values()) else 0


if __name__ == '__main__':
    sys.exit(main())
