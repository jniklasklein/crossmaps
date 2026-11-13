"""CPU geometry for collaborative mapping (NumPy/OpenCV, no ROS or learned models).

Transforms always map column-vector source coordinates into target coordinates.
The registration functions return evidence, never a decision to merge identities.
"""
from __future__ import annotations

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
    source, _ = voxel_sample(source, size=.06, limit=1200)
    target, _ = voxel_sample(target, size=.06, limit=1200)
    current = np.array(initial, dtype=float, copy=True)
    for _ in range(iterations):
        warped = transform(source, current)
        distances, indices = nearest(warped, target)
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
    forward, _ = nearest(warped, target)
    reverse, _ = nearest(target, warped)
    keep = forward < distance
    return dict(T_target_source=current, overlap_source=float(keep.mean()) if len(keep) else 0.,
                overlap_target=float((reverse < distance).mean()) if len(reverse) else 0.,
                rmse=float(np.sqrt(np.mean(forward[keep]**2))) if keep.any() else 1e6,
                inliers=int(keep.sum()))


def _backproject_pixels(keypoints, depth, intrinsic):
    points, good = [], []
    height, width = depth.shape
    for u, v in keypoints:
        x, y = int(round(u)), int(round(v))
        if not (1 <= x < width-1 and 1 <= y < height-1):
            points.append([0, 0, 0]); good.append(False); continue
        patch = depth[y-1:y+2, x-1:x+2]
        valid = patch[np.isfinite(patch) & (patch > .15) & (patch < 8)]
        z = float(np.median(valid)) if len(valid) >= 5 else 0.
        # Discontinuity neighborhoods create highly biased correspondences.
        reliable = z > 0 and float(np.ptp(valid)) < .15 if len(valid) else False
        points.append([(u-intrinsic[0, 2])*z/intrinsic[0, 0], (v-intrinsic[1, 2])*z/intrinsic[1, 1], z])
        good.append(reliable)
    return np.asarray(points), np.asarray(good)


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


def verify_rgbd(source, target, min_inliers=12, min_overlap=.25, max_rmse=.065):
    """Mutual distinctive ORB matches -> depth RANSAC -> bounded ICP.

    Pure geometric ICP from an arbitrary initial pose is deliberately not a
    retrieval verifier: repeated boxes and planes produce convincing false fits.
    """
    detector = cv2.ORB_create(nfeatures=1600, fastThreshold=12)
    keys_a, desc_a = detector.detectAndCompute(source['rgb'], None)
    keys_b, desc_b = detector.detectAndCompute(target['rgb'], None)
    rejected = dict(ok=False, reason="insufficient_visual_features", inliers=0)
    if desc_a is None or desc_b is None or min(len(desc_a), len(desc_b)) < min_inliers:
        return rejected
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    ab, ba = matcher.knnMatch(desc_a, desc_b, k=2), matcher.knnMatch(desc_b, desc_a, k=2)
    reverse = {pair[0].queryIdx: pair[0].trainIdx for pair in ba if len(pair) == 2 and pair[0].distance < .72*pair[1].distance}
    matches = [pair[0] for pair in ab if len(pair) == 2 and pair[0].distance < .72*pair[1].distance
               and reverse.get(pair[0].trainIdx) == pair[0].queryIdx]
    if len(matches) < min_inliers:
        return dict(rejected, reason="insufficient_distinctive_mutual_matches", matches=len(matches))
    pixels_a = np.asarray([keys_a[m.queryIdx].pt for m in matches])
    pixels_b = np.asarray([keys_b[m.trainIdx].pt for m in matches])
    points_a, good_a = _backproject_pixels(pixels_a, source['depth'], source['K'])
    points_b, good_b = _backproject_pixels(pixels_b, target['depth'], target['K'])
    valid = good_a & good_b
    fit = ransac_rigid(points_a[valid], points_b[valid], min_inliers=min_inliers)
    if not fit['ok']:
        return fit
    inlier_pixels = pixels_a[valid][fit['inlier_mask']]
    h, w = source['rgb'].shape[:2]
    cells = np.floor(inlier_pixels / [w/4, h/3]).astype(int)
    if len(np.unique(cells, axis=0)) < 4 or fit['inlier_ratio'] < .3:
        return dict(rejected, reason="localized_or_ambiguous_correspondences", inliers=fit['inliers'])
    cloud_a = source.get('points_camera')
    cloud_b = target.get('points_camera')
    if cloud_a is None:
        cloud_a, _ = depth_points(source['depth'], source['K'])
    if cloud_b is None:
        cloud_b, _ = depth_points(target['depth'], target['K'])
    refined = icp(cloud_a, cloud_b, fit['T_target_source'])
    shift, rotation = pose_delta(fit['T_target_source'], refined['T_target_source'])
    if shift > .15 or rotation > math.radians(8):
        return dict(rejected, reason="icp_disagrees_with_visual_geometry")
    if min(refined['overlap_source'], refined['overlap_target']) < min_overlap or refined['rmse'] > max_rmse:
        return dict(rejected, reason="insufficient_depth_overlap", **refined)
    forward_conflict, forward_count = free_space_conflicts(transform(cloud_a, refined['T_target_source']), target['depth'], target['K'])
    reverse_conflict, reverse_count = free_space_conflicts(transform(cloud_b, np.linalg.inv(refined['T_target_source'])), source['depth'], source['K'])
    refined['free_space_conflict'] = max(forward_conflict if forward_count >= 40 else 0., reverse_conflict if reverse_count >= 40 else 0.)
    if refined['free_space_conflict'] > .18:
        return dict(rejected, reason="mutually_observed_free_space_conflict", **refined)
    fit.pop('inlier_mask', None)
    fit.update(refined)
    fit.update(ok=True, reason="rgbd_verified", visual_inliers=int(len(inlier_pixels)), image_cells=int(len(np.unique(cells, axis=0))))
    # Registration uncertainty is conservative; it is NOT input odometry covariance.
    fit['translation_sigma'] = max(.03, 2*fit['rmse'])
    fit['rotation_sigma'] = max(math.radians(2), fit['translation_sigma']/max(.3, fit['spread']))
    return fit


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
