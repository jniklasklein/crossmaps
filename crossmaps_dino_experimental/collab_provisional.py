"""Unconfirmed cross-rover poses, strictly for the Combined display.

Unlike the pose graph, this estimator may use repeated *parked* captures.
It never modifies a source pose, a constraint, an entity or its evidence.
Output is derived from session-scoped measurements, not carried across runs.
"""
import itertools
import math

import numpy as np

from collab_geometry import pose_delta


REQUIRED = 3
WINDOW = 32
QUALITY_FIELDS = ('inliers', 'rmse', 'overlap_source', 'overlap_target', 'image_cells', 'target_image_cells')


def quality_summary(verification):
    """Retain a small audit even when the candidate review list rolls over."""
    out = {k: verification[k] for k in QUALITY_FIELDS if k in verification}
    # PnP seeds may not report the target grid before verification. The
    # independent bidirectional pixel audit reports the two measured grids.
    grids = verification.get('reprojection_image_cells')
    if ('image_cells' not in out or 'target_image_cells' not in out) and isinstance(grids, (list, tuple)) and len(grids) == 2:
        out['image_cells'], out['target_image_cells'] = grids
    return out


def _strong(v, config):
    try:
        return (all(math.isfinite(v[k]) for k in QUALITY_FIELDS)
                and v['inliers'] >= max(24, config['min_visual_inliers'])
                and 0 <= v['rmse'] <= min(.060, config['max_registration_rmse'])
                and min(v['overlap_source'], v['overlap_target']) >= max(.35, config['min_overlap'])
                and min(v['image_cells'], v['target_image_cells']) >= 4)
    except (KeyError, ValueError, TypeError):
        return False  # Missing old diagnostics never qualify as strong evidence.


def _near(a, b, d, theta):
    distance, angle = pose_delta(a, b)
    return distance <= d and angle <= theta


def _cluster(rows, max_d, max_r):
    """Find a pairwise-compatible group using distinct captures on BOTH sides.

    Try every possible starting pair: if any valid triplet exists it will be
    found. Greedily extend for more supporting captures; never use chained
    transform agreement. Bound at WINDOW, not the entire growing graph.
    """
    n = len(rows)
    neighbors = [0] * n
    errors = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        a, b = rows[i], rows[j]
        d, r = pose_delta(a['T'], b['T'])
        errors[i, j] = errors[j, i] = d / max_d + r / max_r
        if (d <= max_d and r <= max_r and a['source_stamp'] != b['source_stamp']
                and a['target_stamp'] != b['target_stamp']):
            neighbors[i] |= 1 << j
            neighbors[j] |= 1 << i
    best = []
    for i in range(n):
        if not best:
            best = [i]
        for j in range(i + 1, n):
            if not neighbors[i] & (1 << j):
                continue
            group, other = [i, j], neighbors[i] & neighbors[j]
            while other:
                k = (other & -other).bit_length() - 1
                group.append(k)
                other &= neighbors[k]
            if len(group) > len(best):
                best = group
    if not best:
        return [], None
    # Medoid is a measured rigid transform, not an average of Euler angles.
    center = min(best, key=lambda i: (sum(errors[i, j] for j in best), i))
    return [rows[i] for i in best], rows[center]


def _pair(backend, source, target):
    sid, tid = source['id'], target['id']
    report = dict(source_robot=source['robot_id'], target_robot=target['robot_id'],
                  source_session=sid, target_session=tid, required_count=REQUIRED,
                  status='waiting', support_count=0, reason='Waiting for three strong RGB-D pairs with separate captures on both rovers.')
    recent_src, recent_dst = set(source['keyframe_ids'][-WINDOW:]), set(target['keyframe_ids'][-WINDOW:])
    rows, rejected = [], []
    inspected = 0
    for edge in reversed(backend.constraints.values()):
        if edge.get('kind') != 'loop':
            continue
        sessions = (edge.get('source_session'), edge.get('target_session'))
        if sessions not in ((sid, tid), (tid, sid)):
            continue
        veto = edge.get('operator_decision') and edge.get('state') == 'rejected'
        if not veto:
            inspected += 1
            if inspected > WINDOW or edge.get('state') not in ('verified', 'corroborated'):
                continue
        sf, tf = edge['source_frame'], edge['target_frame']
        relation = np.asarray(edge['T_targetlocal_sourcelocal'])
        if sessions[0] == tid:
            sf, tf, relation = tf, sf, np.linalg.inv(relation)
        if veto:
            rejected.append(relation)
            continue
        if sf not in recent_src or tf not in recent_dst:
            continue
        v = edge.get('verification_quality')
        if v is None:  # Backwards-compatible, but never infer missing metrics.
            v = quality_summary(backend.candidates.get(edge['id'], {}).get('verification', {}))
        if not _strong(v, backend.config):
            continue
        a, b = backend.keyframes[sf], backend.keyframes[tf]
        # Older RGB-D stores have one capture stamp. Never replace an
        # explicitly repeated RGB stamp with a changing depth/pair stamp.
        a_stamp, b_stamp = (f.get('rgb_stamp_ns', f['stamp_ns']) for f in (a, b))
        if a_stamp is None or b_stamp is None:
            continue
        rows.append(dict(id=edge['id'], T=relation, source=a, target=b,
                         source_stamp=a_stamp, target_stamp=b_stamp))
    cfg = backend.config
    limit_d, limit_r = cfg['transform_agreement_translation'], math.radians(cfg['transform_agreement_rotation_deg'])
    tight_d, tight_r = min(.10, limit_d), min(math.radians(4), limit_r)
    group, center = _cluster(rows, tight_d, tight_r)
    report['support_count'] = len(group)
    if len(group) < REQUIRED:
        return report, None
    if any(_near(center['T'], veto, limit_d, limit_r) for veto in rejected):
        report.update(status='withheld', reason='Operator rejected a constraint for this relative pose.')
        return report, None
    outsiders = [row for row in rows if not _near(center['T'], row['T'], limit_d, limit_r)]
    rival, _ = _cluster(outsiders, tight_d, tight_r)
    if len(rival) >= 2 or any(backend._distinct_views(row[side], center[side])
                              for row in outsiders for side in ('source', 'target')):
        report.update(status='withheld', reason='Contradictory verified geometry; waiting for a distinct-view confirmation.')
        return report, None
    # World frame of this *specific measured target frame*, rather than a
    # "latest local -> world" transform on a potentially corrected trajectory.
    frame = center['target']
    target_submap = backend.submaps[frame['submap_id']]
    world_target = target_submap['T_world_submap'] @ frame['T_submap_camera']
    world_source_local = world_target @ np.linalg.inv(frame['T_local_camera']) @ center['T']
    report.update(status='provisional', edge_ids=[r['id'] for r in group], medoid_id=center['id'],
                  T_targetlocal_sourcelocal=center['T'],
                  reason='Repeated compatible RGB-D measurements; no independent viewpoint confirmation yet.')
    return report, world_source_local


def derive_previews(backend):
    """Preview current unanchored sessions only, directly against confirmed ones."""
    records, world_from_local = [], {}
    if backend.config['shadow_mode'] or not backend.config['provisional_enabled']:
        return records, world_from_local
    anchored, waiting = [], []
    for sid in backend.robots.values():
        session = backend.sessions[sid]
        if session['submap_ids']:
            (anchored if backend.submaps[session['submap_ids'][-1]]['aligned'] else waiting).append(session)
    for src in waiting:
        options = [_pair(backend, src, dst) for dst in anchored if src['robot_id'] != dst['robot_id']]
        # Deliberately do not chain previews through other unconfirmed poses.
        viable = [(report, pose) for report, pose in options if pose is not None]
        if len(viable) == 1:
            world_from_local[src['id']] = viable[0][1]
        elif len(viable) > 1:
            for report, _ in viable:
                report.update(status='withheld', reason='Multiple candidate anchors; waiting for confirmed geometry.')
        records.extend(report for report, _ in options)
    return records, world_from_local
