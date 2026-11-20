"""CPU geometry for collaborative mapping (NumPy/OpenCV, no ROS or learned models).

Transforms always map column-vector source coordinates into target coordinates.
The registration functions return evidence, never a decision to merge identities.
"""
from __future__ import annotations

import hashlib
import math
import cv2
import numpy as np


def transform(points, matrix):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    matrix = np.asarray(matrix, dtype=np.float64)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def valid_transform(value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("Pose must be a finite 4x4 matrix")
    if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("Pose must have homogeneous last row")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-3):
        raise ValueError("Pose rotation must be right-handed and orthonormal")
    return value.copy()


def pose_delta(a, b):
    relative = np.linalg.inv(a) @ b
    return float(np.linalg.norm(relative[:3, 3])), float(np.linalg.norm(cv2.Rodrigues(relative[:3, :3])[0]))


def rigid_fit(source, target):
    source, target = np.asarray(source), np.asarray(target)
    if len(source) < 3 or len(source) != len(target):
        raise ValueError("At least three paired points required")
    a, b = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source-a).T @ (target-b))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, b-rotation @ a
    return result


def voxel_sample(points, colors=None, size=.05, limit=3000):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = np.asarray(colors)[finite] if colors is not None else None
    if not len(points):
        return points, None if colors is None else colors
    _, indices = np.unique(np.floor(points/size).astype(np.int64), axis=0, return_index=True)
    indices = np.sort(indices)
    if len(indices) > limit:
        indices = indices[np.linspace(0, len(indices)-1, limit).astype(int)]
    return points[indices], None if colors is None else colors[indices]


def depth_points(depth, intrinsic, rgb=None, stride=5, max_depth=8.):
    depth, intrinsic = np.asarray(depth), np.asarray(intrinsic)
    ys, xs = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[ys, xs].reshape(-1)
    good = np.isfinite(z) & (z > .15) & (z < max_depth)
    x, y, z = xs.reshape(-1)[good], ys.reshape(-1)[good], z[good]
    points = np.column_stack(((x-intrinsic[0, 2])*z/intrinsic[0, 0], (y-intrinsic[1, 2])*z/intrinsic[1, 1], z))
    return points.astype(np.float32), None if rgb is None else np.asarray(rgb)[y, x, ::-1].copy()


def nearest(source, target):
    """Exact bounded nearest neighbors, independent of FLANN random state."""
    source, target = np.asarray(source), np.asarray(target)
    if not len(target):
        return np.full(len(source), np.inf), np.zeros(len(source), dtype=int)
    distances, indices = [], []
    for start in range(0, len(source), 128):
        delta = source[start:start+128, None, :] - target[None, :, :]
        square = np.einsum('ijk,ijk->ij', delta, delta)
        selected = square.argmin(axis=1)
        indices.extend(selected)
        distances.extend(np.sqrt(square[np.arange(len(selected)), selected]))
    return np.asarray(distances), np.asarray(indices, dtype=int)


def ransac_rigid(source, target, threshold=.075, iterations=300, min_inliers=12, seed=7):
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    result = dict(ok=False, reason="insufficient_correspondences", inliers=0)
    if source.shape != target.shape or len(source) < min_inliers:
        return result
    rng = np.random.default_rng(seed)
    best = np.zeros(len(source), dtype=bool)
    for _ in range(iterations):
        idx = rng.choice(len(source), 3, replace=False)
        if min(np.linalg.norm(np.cross(*(source[idx][1:]-source[idx][0]))),
               np.linalg.norm(np.cross(*(target[idx][1:]-target[idx][0])))) < 1e-4:
            continue
        fit = rigid_fit(source[idx], target[idx])
        keep = np.linalg.norm(transform(source, fit)-target, axis=1) < threshold
        if keep.sum() > best.sum():
            best = keep
    if best.sum() < min_inliers:
        return dict(result, reason="insufficient_geometric_inliers", inliers=int(best.sum()))
    fit = rigid_fit(source[best], target[best])
    error = np.linalg.norm(transform(source, fit)-target, axis=1)
    best = error < threshold
    if best.sum() < min_inliers:
        return dict(result, reason="insufficient_refined_inliers", inliers=int(best.sum()))
    covariance = np.linalg.eigvalsh(np.cov(source[best].T))
    if covariance[1] < .0025:  # Collinear/tiny support cannot constrain a place match.
        return dict(result, reason="degenerate_correspondence_support", inliers=int(best.sum()))
    return dict(ok=True, reason="ransac_verified", T_target_source=fit,
                inliers=int(best.sum()), inlier_ratio=float(best.mean()), inlier_mask=best,
                rmse=float(np.sqrt(np.mean(error[best]**2))), spread=float(np.sqrt(covariance[1])))


def icp(source, target, initial, distance=.15, iterations=12):
    # Exact Euclidean nearest neighbors in a single KD tree. The old 1,200
    # point displays were too sparse to measure a 6.5 cm registration residual.
    source, _ = voxel_sample(source, size=.035, limit=12000)
    target, _ = voxel_sample(target, size=.035, limit=12000)
    current = np.array(initial, dtype=float, copy=True)
    if min(len(source), len(target)) < 12:
        return dict(T_target_source=current, overlap_source=0., overlap_target=0., rmse=1e6, inliers=0)
    source_tree = cv2.flann_Index(source, dict(algorithm=4, leaf_max_size=16))
    target_tree = cv2.flann_Index(target, dict(algorithm=4, leaf_max_size=16))
    def query(tree, xyz):
        ids, d2 = tree.knnSearch(np.ascontiguousarray(xyz, dtype=np.float32), 1, params={'checks': -1})
        return np.sqrt(np.maximum(d2[:, 0], 0)), ids[:, 0]
    for _ in range(iterations):
        warped = transform(source, current)
        distances, indices = query(target_tree, warped)
        keep = distances < distance
        if keep.sum() < 12:
            break
        cutoff = np.percentile(distances[keep], 80)
        keep &= distances <= cutoff
        update = rigid_fit(warped[keep], target[indices[keep]])
        current = update @ current
        if pose_delta(np.eye(4), update)[0] < .0002 and pose_delta(np.eye(4), update)[1] < .0002:
            break
    warped = transform(source, current)
    forward, _ = query(target_tree, warped)
    reverse, _ = query(source_tree, transform(target, np.linalg.inv(current)))
    keep = forward < distance
    return dict(T_target_source=current, overlap_source=float(keep.mean()) if len(keep) else 0.,
                overlap_target=float((reverse < distance).mean()) if len(reverse) else 0.,
                rmse=max(float(np.sqrt(np.mean(d[d < distance]**2))) if (d < distance).any() else 1e6
                         for d in (forward, reverse)), inliers=int(keep.sum()),
                reverse_inliers=int((reverse < distance).sum()), icp_source_samples=len(source), icp_target_samples=len(target))


def _backproject_pixels(keypoints, depth, intrinsic):
    uv = np.asarray(keypoints, dtype=float).reshape(-1, 2)
    height, width = depth.shape
    points, good = np.zeros((len(uv), 3)), np.zeros(len(uv), dtype=bool)
    rounded = np.rint(uv)
    inside = (rounded >= 1).all(axis=1) & (rounded < [width-1, height-1]).all(axis=1)
    if not inside.any():
        return points, good
    x, y = rounded[inside].astype(int).T
    dx, dy = np.tile([-1, 0, 1], 3), np.repeat([-1, 0, 1], 3)
    patch = depth[y[:, None]+dy, x[:, None]+dx]
    valid = np.isfinite(patch) & (patch > .15) & (patch < 8)
    count = valid.sum(axis=1)
    sorted_z = np.sort(np.where(valid, patch, np.inf), axis=1)
    rows = np.arange(len(count))
    z = np.where(count >= 5, .5*(sorted_z[rows, count//2] + sorted_z[rows, np.maximum(0, (count-1)//2)]), 0.)
    # Exclude depth discontinuities, also during the dense free-space check.
    good[inside] = (count >= 5) & (np.where(valid, patch, -np.inf).max(axis=1) < sorted_z[:, 0]+.15)
    points[inside, :2] = (uv[inside]-[intrinsic[0, 2], intrinsic[1, 2]]) * z[:, None] / [intrinsic[0, 0], intrinsic[1, 1]]
    points[inside, 2] = z
    return points, good


def free_space_conflicts(points_camera, target_depth, target_intrinsic, tolerance=.18):
    """Count predicted surfaces inside space measured free by the other camera.

    A point *behind* the measured surface is occluded, not a contradiction.
    Invalid depth and depth-edge neighborhoods provide no negative evidence.
    """
    points = np.asarray(points_camera)
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > .15)]
    if not len(points):
        return 0., 0
    intrinsic = np.asarray(target_intrinsic)
    pixels = np.column_stack((points[:, 0]*intrinsic[0, 0]/points[:, 2]+intrinsic[0, 2],
                              points[:, 1]*intrinsic[1, 1]/points[:, 2]+intrinsic[1, 2]))
    projected, valid = _backproject_pixels(pixels, np.asarray(target_depth), intrinsic)
    if not valid.any():
        return 0., 0
    return float((points[valid, 2] < projected[valid, 2]-tolerance).mean()), int(valid.sum())


def _visible_depth_score(cloud, pose, target):
    """Test the *fixed* pose against what the other depth camera actually saw.

    No nearest-neighbor association to a different wall/floor pixel. Unknown,
    out-of-frustum and occluded samples cannot increase overlap. Contradicted
    free space is counted over all mutually measured rays, not just inliers.
    """
    points = transform(cloud, pose)
    front = points[points[:, 2] > .15]
    pix = front @ target['K'].T
    observed, valid = _backproject_pixels(pix[:, :2]/pix[:, 2:], target['depth'], target['K'])
    dz = front[valid, 2]-observed[valid, 2]
    agree = np.abs(dz) < .15
    return dict(overlap=float(agree.sum()/max(1, len(cloud))), measured=len(dz),
                rmse=float(np.sqrt(np.mean(dz[agree]**2))) if agree.any() else 1e6,
                inliers=int(agree.sum()), free=float((dz < -.18).mean()) if len(dz) else 0.)


def _orb_matches(source, target):
    detector = cv2.ORB_create(nfeatures=1600, fastThreshold=12)
    keys_a, desc_a = detector.detectAndCompute(source['rgb'], None)
    keys_b, desc_b = detector.detectAndCompute(target['rgb'], None)
    meta = dict(matcher='orb_1600_mutual_ratio_0.72', source_features=len(keys_a), target_features=len(keys_b))
    if desc_a is None or desc_b is None or min(len(desc_a), len(desc_b)) < 2:
        return np.empty((0, 2)), np.empty((0, 2)), meta
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    ab, ba = matcher.knnMatch(desc_a, desc_b, k=2), matcher.knnMatch(desc_b, desc_a, k=2)
    reverse = {pair[0].queryIdx: pair[0].trainIdx for pair in ba if len(pair) == 2 and pair[0].distance < .72*pair[1].distance}
    matches = [pair[0] for pair in ab if len(pair) == 2 and pair[0].distance < .72*pair[1].distance
               and reverse.get(pair[0].trainIdx) == pair[0].queryIdx]
    return (np.asarray([keys_a[m.queryIdx].pt for m in matches]).reshape(-1, 2),
            np.asarray([keys_b[m.trainIdx].pt for m in matches]).reshape(-1, 2), meta)


def _seed_from_matches(source, target, pixels_a, pixels_b, meta, min_inliers):
    rejected = dict(meta, ok=False, reason="insufficient_visual_features", inliers=0,
                    verification_stage='visual_features', visual_matches=len(pixels_a),
                    depth_correspondences=0, ransac_inliers=0, visual_inliers=0, icp_inliers=0)
    if min(meta['source_features'], meta['target_features']) < min_inliers:
        return rejected, None
    if len(pixels_a) < min_inliers:
        return dict(rejected, reason="insufficient_distinctive_mutual_matches",
                    matches=len(pixels_a), verification_stage='distinctive_matching'), None
    points_a, good_a = _backproject_pixels(pixels_a, source['depth'], source['K'])
    points_b, good_b = _backproject_pixels(pixels_b, target['depth'], target['K'])
    valid = good_a & good_b
    rejected.update(depth_correspondences=int(valid.sum()),
                    source_matches_with_depth=int(good_a.sum()), target_matches_with_depth=int(good_b.sum()),
                    verification_stage='depth_correspondences' if valid.sum() < min_inliers else 'geometric_ransac')
    fit = ransac_rigid(points_a[valid], points_b[valid], min_inliers=min_inliers)
    rejected.update(inliers=fit['inliers'], ransac_inliers=fit['inliers'], visual_inliers=fit['inliers'],
                    ransac_inlier_ratio=float(fit['inliers']/max(1, valid.sum())))
    if not fit['ok']:
        return dict(rejected, **fit), None
    inlier_pixels = pixels_a[valid][fit['inlier_mask']]
    h, w = source['rgb'].shape[:2]
    cells = np.floor(inlier_pixels / [w/4, h/3]).astype(int)
    rejected.update(image_cells=int(len(np.unique(cells, axis=0))), ransac_rmse=fit['rmse'],
                    ransac_spread_m=fit['spread'], verification_stage='correspondence_distribution',
                    T_ransac_target_source=fit['T_target_source'])
    ht, wt = target['rgb'].shape[:2]
    other_cells = np.floor(pixels_b[valid][fit['inlier_mask']] / [wt/4, ht/3]).astype(int)
    rejected['target_image_cells'] = len(np.unique(other_cells, axis=0))
    if min(rejected['image_cells'], rejected['target_image_cells']) < 4 or fit['inlier_ratio'] < .3:
        return dict(rejected, reason="localized_or_ambiguous_correspondences"), None
    fit['_seed_indices'] = np.flatnonzero(valid)[fit['inlier_mask']]
    return dict(rejected, seed_method='depth_ransac'), fit


def _verify_seed(source, target, rejected, fit, min_overlap, max_rmse, pixels, hold, min_inliers=12, refine_pixels=True):
    """A visual proposal is never sufficient: all seeds face the same dense checks."""
    # Always measure the original calibrated depth, not a keyframe's decimated
    # visualization points. Bound preprocessing independently of camera size.
    def cloud(frame):
        stride = max(2, math.ceil(math.sqrt(frame['depth'].size/36000)))
        xyz, _ = depth_points(frame['depth'], frame['K'], stride=stride)
        return xyz
    cloud_a, cloud_b = cloud(source), cloud(target)
    seed_indices = fit.pop('_seed_indices', None)
    from collab_pnp import audit_reprojection, prepare_reprojection, refine_reprojection
    seed = fit['T_target_source']
    data = prepare_reprojection(source, target, *pixels, seed, min_inliers, seed_indices=seed_indices, hold_indices=hold)
    checked_pose = None
    if refine_pixels:
        audit = refine_reprojection(data, seed)
        rejected.update({k: v for k, v in audit.items() if k.startswith('reprojection_')})
        rejected.update(verification_stage='pixel_reprojection')
        if not audit['ok']:
            return dict(rejected, reason=audit['reason'])
        checked_pose = audit['T_target_source']
        fit['T_target_source'] = checked_pose
    refined = icp(cloud_a, cloud_b, fit['T_target_source'])
    if 'reverse_inliers' in refined:
        rejected['icp_reverse_inliers'] = refined['reverse_inliers']
    shift, rotation = pose_delta(fit['T_target_source'], refined['T_target_source'])
    rejected.update(icp_inliers=refined['inliers'], icp_rmse=refined['rmse'],
                    icp_translation_delta_m=shift, icp_rotation_delta_deg=math.degrees(rotation),
                    overlap_source=refined['overlap_source'], overlap_target=refined['overlap_target'],
                    T_target_source=refined['T_target_source'], verification_stage='icp_agreement')
    if shift > .15 or rotation > math.radians(8):
        return dict(rejected, reason="icp_disagrees_with_visual_geometry")
    if checked_pose is not None:
        # ICP on a featureless plane can slide while improving its NN residual.
        # Keep its independent veto, but never publish a slid transform. Score
        # the pose that explained the held-out pixels against original depths.
        f = _visible_depth_score(cloud_a, checked_pose, target)
        r = _visible_depth_score(cloud_b, np.linalg.inv(checked_pose), source)
        overlap = f['overlap'], r['overlap']
        residual = max(f['rmse'], r['rmse'])
        free = max(f['free'], r['free'])
        rejected.update(T_icp_target_source=refined['T_target_source'], T_target_source=checked_pose,
                        residual_method='bidirectional_visible_depth', depth_rmse=residual,
                        overlap_source=overlap[0], overlap_target=overlap[1], free_space_conflict=free,
                        free_space_samples=[f['measured'], r['measured']])
        if min(f['measured'], r['measured']) < 40 or min(overlap) < min_overlap:
            return dict(rejected, reason='insufficient_depth_overlap', verification_stage='depth_overlap')
        if residual > max_rmse:
            return dict(rejected, reason='depth_alignment_residual_too_large', verification_stage='depth_residual')
        if free > .18:
            return dict(rejected, reason='mutually_observed_free_space_conflict', verification_stage='free_space_consistency')
        fit.pop('inlier_mask', None)
        return dict(fit | rejected, ok=True, reason='rgbd_verified', verification_stage='verified',
                    rmse=residual, translation_sigma=max(.03, 2*residual),
                    rotation_sigma=max(math.radians(2), max(.03, 2*residual)/max(.3, fit['spread'])))
    rejected['verification_stage'] = 'depth_overlap'
    if min(refined['overlap_source'], refined['overlap_target']) < min_overlap:
        return dict(rejected, reason="insufficient_depth_overlap")
    rejected['verification_stage'] = 'depth_residual'
    if refined['rmse'] > max_rmse:
        return dict(rejected, reason="depth_alignment_residual_too_large")
    forward_conflict, forward_count = free_space_conflicts(transform(cloud_a, refined['T_target_source']), target['depth'], target['K'])
    reverse_conflict, reverse_count = free_space_conflicts(transform(cloud_b, np.linalg.inv(refined['T_target_source'])), source['depth'], source['K'])
    refined['free_space_conflict'] = max(forward_conflict if forward_count >= 40 else 0., reverse_conflict if reverse_count >= 40 else 0.)
    rejected.update(free_space_conflict=refined['free_space_conflict'], free_space_samples=[forward_count, reverse_count],
                    verification_stage='free_space_consistency')
    if min(forward_count, reverse_count) < 40:
        return dict(rejected, reason="insufficient_mutually_observed_depth")
    if refined['free_space_conflict'] > .18:
        return dict(rejected, reason="mutually_observed_free_space_conflict")
    # The ORB fast path skips iterative pixel refinement, but its final ICP
    # transform must also explain the reserved, spatially distributed pixels.
    audit = audit_reprojection(data, refined['T_target_source'])
    rejected.update({k:v for k,v in audit.items() if k.startswith('reprojection_')})
    if not audit['ok']:
        return dict(rejected, reason=audit['reason'], verification_stage='pixel_reprojection')
    fit.pop('inlier_mask', None)
    fit.update(refined)
    fit.update(rejected)
    fit.update(ok=True, reason="rgbd_verified", verification_stage='verified')
    # Registration uncertainty is conservative; it is NOT input odometry covariance.
    fit['translation_sigma'] = max(.03, 2*fit['rmse'])
    fit['rotation_sigma'] = max(math.radians(2), fit['translation_sigma']/max(.3, fit['spread']))
    return fit


def _stage_rank(result):
    stages = ('visual_features', 'distinctive_matching', 'depth_correspondences', 'geometric_ransac',
              'correspondence_distribution', 'pixel_reprojection', 'icp_agreement', 'depth_overlap', 'depth_residual',
              'free_space_consistency', 'verified')
    stage = result.get('verification_stage')
    return (stages.index(stage) if stage in stages else -1, result.get('visual_inliers', 0),
            result.get('depth_correspondences', 0), result.get('visual_matches', 0))


def _verify_ordered(source, target, min_inliers, min_overlap, max_rmse, matcher_mode):
    """Try inexpensive ORB first. Expand visual proposals, not acceptance tolerances.

    If an established seed contradicts dense geometry, reject that pair; do
    not keep searching for a transform that happens to pass. No latest-pose,
    semantic-similarity, or uninitialized ICP fallback.
    """
    if matcher_mode not in ('auto', 'orb'):
        raise ValueError("matcher_mode must be 'auto' or 'orb'")
    from collab_visual_matches import visual_matches
    from collab_pnp import reprojection_partition
    attempts = []
    for proposals in ([_orb_matches] if matcher_mode == 'orb' else [_orb_matches, visual_matches]):
        a, b, meta = proposals(source, target)
        train, hold = reprojection_partition(source, target, a, b)
        evidence, seed = _seed_from_matches(source, target, a[train], b[train], meta, min_inliers)
        if seed is not None:
            seed['_seed_indices'] = train[seed['_seed_indices']]
        evidence.update(visual_matches=len(a), fit_matches=len(train), holdout_matches=len(hold))
        if seed is None and proposals is visual_matches and min(
                evidence.get('source_matches_with_depth', 0), evidence.get('target_matches_with_depth', 0)) >= min_inliers:
            from collab_pnp import bidirectional_pnp_seed
            pnp = bidirectional_pnp_seed(source, target, a[train], b[train], min_inliers=min_inliers)
            evidence['pnp_reason'] = pnp['reason']
            evidence.update({k: v for k, v in pnp.items() if k.startswith('pnp_')})
            if pnp['ok']:
                seed = dict(pnp)
                seed['_seed_indices'] = train
                evidence.update(seed_method='bidirectional_pnp', inliers=pnp['inliers'],
                                visual_inliers=pnp['visual_inliers'], image_cells=pnp['image_cells'])
        if seed is not None:
            result = _verify_seed(source, target, evidence, seed, min_overlap, max_rmse,
                                  (a, b), hold, min_inliers, refine_pixels=proposals is visual_matches)
            attempts.append(result)
            break
        attempts.append(evidence)
    result = max(attempts, key=_stage_rank)
    # Store small diagnostics per attempt, not descriptors, masks or a runner-up transform.
    fields = ('matcher', 'reason', 'verification_stage', 'visual_matches', 'depth_correspondences',
              'ransac_inliers', 'pnp_reason', 'visual_inliers', 'seed_method')
    result['attempts'] = [{k: r[k] for k in fields if k in r} for r in attempts]
    return result


def _capture_key(frame):
    return tuple(str(frame.get(k, '')) for k in ('robot_id','source_id','id','session_id','stamp_ns'))


def _measured_key(frame):
    # Only for low-level anonymous frames / tied metadata. Never use object
    # address, wall-clock time, or a favourable verification outcome as a key.
    digest = hashlib.blake2b(digest_size=20)
    for field in ('rgb','depth','K'):
        x = np.ascontiguousarray(frame[field])
        digest.update(repr((x.shape,x.dtype.str)).encode())
        digest.update(memoryview(x).cast('B'))
    return digest.digest()


def _reverse_evidence(result):
    result = dict(result)
    for k in ('T_target_source','T_ransac_target_source','T_icp_target_source'):
        if k in result:
            result[k] = np.linalg.inv(result[k])
    for a, b in (('source_features','target_features'),('source_matches_with_depth','target_matches_with_depth'),
                 ('image_cells','target_image_cells'),('overlap_source','overlap_target'),
                 ('icp_source_samples','icp_target_samples'),('icp_inliers','icp_reverse_inliers'),
                 ('pnp_source_depth','pnp_target_depth'),('pnp_forward_inliers','pnp_reverse_inliers'),
                 ('pnp_forward_fit_inliers','pnp_reverse_fit_inliers'),
                 ('reprojection_median_px_forward','reprojection_median_px_reverse'),
                 ('reprojection_forward_inliers','reprojection_reverse_inliers'),
                 ('reprojection_holdout_forward_inliers','reprojection_holdout_reverse_inliers')):
        if a in result and b in result:
            result[a],result[b] = result[b],result[a]
    if 'reverse_inliers' in result and 'icp_reverse_inliers' in result:
        result['reverse_inliers'] = result['icp_reverse_inliers']
    for k in ('free_space_samples','reprojection_image_cells','reprojection_holdout_cells'):
        if k in result: result[k] = result[k][::-1]
    return result


def verify_rgbd(source, target, min_inliers=12, min_overlap=.25, max_rmse=.065, *, matcher_mode='auto'):
    """One fit per unordered pair, always returned in caller source→target coordinates."""
    a, b = _capture_key(source), _capture_key(target)
    if a == b:
        a, b = _measured_key(source), _measured_key(target)
    reverse = b < a
    first, second = (target, source) if reverse else (source, target)
    result = _verify_ordered(first, second, min_inliers, min_overlap, max_rmse, matcher_mode)
    if result.get('ok'):
        # Graph currently consumes two isotropic scales, not a 6×6 covariance.
        # Include the rotation lever arm in the common translation uncertainty
        # envelope; neither publisher order should make a long baseline cheaper.
        result['translation_sigma'] = math.hypot(result['translation_sigma'],
            result['rotation_sigma'] * np.linalg.norm(result['T_target_source'][:3, 3]))
    return _reverse_evidence(result) if reverse else result


def geometry_descriptor(points):
    points = np.asarray(points)
    if len(points) < 12:
        return None
    centered = points-np.median(points, axis=0)
    distances = np.linalg.norm(centered, axis=1)
    radii = np.quantile(distances, [.1, .25, .5, .75, .9])
    eigen = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(centered.T)), 0))
    return np.r_[radii, eigen].astype(np.float32)


def _increment(vector):
    result = np.eye(4)
    result[:3, :3] = cv2.Rodrigues(np.asarray(vector[3:], dtype=float))[0]
    result[:3, 3] = vector[:3]
    return result


def _residual(source_pose, target_pose, measurement):
    # measurement = inverse(target world pose) @ source world pose.
    error = np.linalg.inv(measurement) @ np.linalg.inv(target_pose) @ source_pose
    return np.r_[error[:3, 3], cv2.Rodrigues(error[:3, :3])[0].reshape(3)]


def optimize_pose_graph(poses, constraints, fixed, iterations=5):
    """Sparse 6x6-block SE(3) Gauss-Newton with Cauchy loop weights.

    The caller supplies only one connected component and fixes its gauge. Local
    odometry and trusted anchors remain constraints, not overwritten robot TF.
    Block-Jacobi preconditioned CG avoids SciPy and dense O(keyframes**3) solves.
    """
    current = {key: valid_transform(value) for key, value in poses.items()}
    variables = [key for key in current if key not in set(fixed)]
    offsets = {key: index for index, key in enumerate(variables)}
    if not variables or not constraints:
        return current, dict(iterations=0, cost=0., max_residual=0.)
    cost = max_residual = 0.
    for iteration in range(iterations):
        blocks = {(index, index): np.eye(6)*1e-4 for index in range(len(variables))}
        gradient = np.zeros((len(variables), 6))
        cost = max_residual = 0.
        for edge in constraints:
            a, b = edge['source'], edge['target']
            if a not in current or b not in current:
                continue
            z = np.asarray(edge['T_target_source'])
            scales = np.array([edge.get('translation_sigma', .1)]*3 + [edge.get('rotation_sigma', .1)]*3)
            scales = np.maximum(scales, 1e-3)
            raw = _residual(current[a], current[b], z)
            residual = raw/scales
            max_residual = max(max_residual, float(np.linalg.norm(raw)))
            # A false closure cannot exert arbitrary force on the trajectory.
            robust = 1. if edge.get('kind') in ('odometry', 'anchor') else 1./math.sqrt(1.+float(residual @ residual)/9.)
            jacobians = {}
            for key in (a, b):
                if key not in offsets:
                    continue
                jacobian = np.zeros((6, 6))
                for axis in range(6):
                    delta = np.zeros(6); delta[axis] = 1e-5
                    perturbed = _increment(delta) @ current[key]
                    rp = _residual(perturbed if key == a else current[a], perturbed if key == b else current[b], z)
                    jacobian[:, axis] = (rp-raw)/1e-5/scales
                jacobians[offsets[key]] = jacobian*robust
            for left, jl in jacobians.items():
                gradient[left] += jl.T @ (residual*robust)
                for right, jr in jacobians.items():
                    key = (left, right)
                    blocks[key] = blocks.get(key, np.zeros((6, 6))) + jl.T @ jr
            cost += float((residual*robust) @ (residual*robust))
        if not np.isfinite(gradient).all():
            break
        try:
            inverse_diagonal = np.asarray([np.linalg.inv(blocks[index, index]) for index in range(len(variables))])
        except np.linalg.LinAlgError:
            break
        # Vectorized scatter over sparse block entries keeps the iteration cost
        # linear in graph edges rather than building a dense normal matrix.
        block_items = list(blocks.items())
        row_index = np.asarray([key[0] for key, _ in block_items])
        col_index = np.asarray([key[1] for key, _ in block_items])
        block_values = np.asarray([value for _, value in block_items])
        def multiply(vector):
            output = np.zeros_like(vector)
            np.add.at(output, row_index, np.einsum('nij,nj->ni', block_values, vector[col_index]))
            return output
        delta = np.zeros_like(gradient)
        residual_cg = -gradient.copy()
        preconditioned = np.einsum('nij,nj->ni', inverse_diagonal, residual_cg)
        direction = preconditioned.copy()
        residual_dot = float(np.sum(residual_cg*preconditioned))
        initial = max(np.linalg.norm(residual_cg), 1e-12)
        for _ in range(min(160, len(variables)*6+12)):
            product = multiply(direction)
            denominator = float(np.sum(direction*product))
            if denominator <= 1e-20 or residual_dot <= 1e-20:
                break
            alpha = residual_dot/denominator
            delta += alpha*direction
            residual_cg -= alpha*product
            if np.linalg.norm(residual_cg) < initial*1e-7:
                break
            preconditioned = np.einsum('nij,nj->ni', inverse_diagonal, residual_cg)
            next_dot = float(np.sum(residual_cg*preconditioned))
            direction = preconditioned + (next_dot/residual_dot)*direction
            residual_dot = next_dot
        for key, index in offsets.items():
            step = delta[index]
            length = max(np.linalg.norm(step[:3])/.5, np.linalg.norm(step[3:])/.3, 1.)
            current[key] = _increment(step/length) @ current[key]
        if np.linalg.norm(delta) < 1e-5:
            break
    return current, dict(iterations=iteration+1, cost=cost, max_residual=max_residual,
                         nodes=len(current), solver='block_sparse_se3_pcg')
