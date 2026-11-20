"""Pose proposal from correspondences with reliable depth on *either* image.

Forward and inverse PnP are fitted separately using depth in different cameras.
Their agreement is a guard, not independent observations of a loop closure.
The caller must still check depth overlap, ICP movement and free-space conflict.
"""

import math

import cv2
import numpy as np


def _unique_matches(a, b, radius=3.0):
    # SIFT may describe the same feature several times at different scales or
    # orientations. One image location should not get several geometric votes.
    accepted, tables = [], ({}, {})
    for i, pair in enumerate(zip(a, b)):
        cells = [tuple(np.floor(p / radius).astype(int)) for p in pair]
        if any(
            any(
                np.dot(p - q, p - q) < radius**2
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for q in table.get((x + dx, y + dy), ())
            )
            for p, (x, y), table in zip(pair, cells, tables)
        ):
            continue
        for p, k, table in zip(pair, cells, tables):
            table.setdefault(k, []).append(p)
        accepted.append(i)
    return np.array(accepted, dtype=int)


def _project_errors(xyz, uv, K, T):
    warped = xyz @ T[:3, :3].T + T[:3, 3]
    front = warped[:, 2] > 0.1
    errors = np.full(len(uv), np.inf)
    hp = warped[front] @ K.T
    errors[front] = np.linalg.norm(hp[:, :2] / hp[:, 2:] - uv[front], axis=1)
    return errors


def _fit(xyz, uv, K, threshold):
    if len(xyz) < 6:
        return None
    try:
        ok, r, t, ix = cv2.solvePnPRansac(
            xyz,
            uv,
            K,
            None,
            iterationsCount=350,
            confidence=0.998,
            reprojectionError=threshold,
            flags=cv2.SOLVEPNP_AP3P,
        )
        if not ok or ix is None or len(ix) < 6:
            return None
        ix = ix.ravel()
        r, t = cv2.solvePnPRefineLM(xyz[ix], uv[ix], K, None, r, t)
        if not np.isfinite(r).all() or not np.isfinite(t).all():
            return None
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(r)[0]
        T[:3, 3] = t.ravel()
        return T
    except cv2.error:
        return None


def _support(uv, shape):
    h, w = shape
    cells = len(np.unique(np.floor(uv / [w / 4, h / 3]), axis=0))
    hull = cv2.convexHull(uv.astype(np.float32))
    # Hull and cell counts alone can be dominated by one remote keypoint.
    lo, hi = np.quantile(uv / [w, h], [0.1, 0.9], axis=0)
    return cells, float(cv2.contourArea(hull) / (w * h)), float(np.prod(hi - lo))


def bidirectional_pnp_seed(source, target, pixels_source, pixels_target, min_inliers=12):
    """Return source→target seed, not a verified alignment (rectified cameras)."""
    # Imported here since collab_geometry calls this optional fallback.
    from collab_geometry import _backproject_pixels, pose_delta

    diag = dict(
        verification_stage="bidirectional_pnp",
        inliers=0,
        visual_inliers=0,
        pnp_forward_inliers=0,
        pnp_reverse_inliers=0,
    )

    def reject(reason):
        return dict(diag, ok=False, reason=reason)

    frames = (source, target)
    K = [np.ascontiguousarray(f["K"], dtype=float) for f in frames]
    if not all(k.shape == (3, 3) and np.isfinite(k).all() and min(k[0, 0], k[1, 1]) > 0 for k in K):
        return reject("pnp_invalid_intrinsics")
    a, b = (np.asarray(v, dtype=float) for v in (pixels_source, pixels_target))
    limit = max(8, int(min_inliers))
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2:
        return reject("pnp_insufficient_distinct_matches")
    inside = np.ones(len(a), dtype=bool)
    for uv, f in zip((a, b), frames):
        h, w = f["depth"].shape
        inside &= np.isfinite(uv).all(axis=1) & (uv >= 0).all(axis=1) & (uv < [w, h]).all(axis=1)
    a, b = a[inside], b[inside]
    ix = _unique_matches(a, b)
    a, b = a[ix], b[ix]
    diag["pnp_distinct_matches"] = len(a)
    if len(a) < limit:
        return reject("pnp_insufficient_distinct_matches")
    xyz_a, ga = _backproject_pixels(a, source["depth"], K[0])
    xyz_b, gb = _backproject_pixels(b, target["depth"], K[1])
    diag.update(pnp_source_depth=int(ga.sum()), pnp_target_depth=int(gb.sum()))
    if min(ga.sum(), gb.sum()) < limit:
        return reject("pnp_insufficient_depth_in_either_direction")
    # Roughly 0.25 degrees, bounded in pixels so degraded calibration cannot
    # buy a pose by silently making reprojection checks arbitrarily loose.
    ta, tb = (float(np.clip(0.0045 * 0.5 * (k[0, 0] + k[1, 1]), 1.5, 4.0)) for k in K)
    p, q = _fit(xyz_a[ga], b[ga], K[1], tb), _fit(xyz_b[gb], a[gb], K[0], ta)
    if p is None or q is None:
        return reject("pnp_pose_unavailable_in_either_direction")
    diag.update(
        pnp_forward_fit_inliers=int((_project_errors(xyz_a[ga], b[ga], K[1], p) < tb).sum()),
        pnp_reverse_fit_inliers=int((_project_errors(xyz_b[gb], a[gb], K[0], q) < ta).sum()),
    )
    q_inv = np.linalg.inv(q)
    dist, angle = pose_delta(p, q_inv)
    diag.update(pnp_translation_disagreement_m=dist, pnp_rotation_disagreement_deg=math.degrees(angle))
    if dist > 0.10 or angle > math.radians(4):
        return reject("pnp_directions_disagree")
    T = np.eye(4)
    T[:3, :3] = p[:3, :3] @ cv2.Rodrigues(0.5 * cv2.Rodrigues(p[:3, :3].T @ q_inv[:3, :3])[0])[0]
    T[:3, 3] = 0.5 * (p[:3, 3] + q_inv[:3, 3])
    errors_a = _project_errors(xyz_a, b, K[1], T)
    errors_b = _project_errors(xyz_b, a, K[0], np.linalg.inv(T))
    # A match with two measured depths must agree both ways, including depth
    # along the ray (small pixel reprojection alone cannot check that).
    both = ga & gb
    metric = np.linalg.norm(xyz_a @ T[:3, :3].T + T[:3, 3] - xyz_b, axis=1)
    consistent = ~both | ((errors_a < tb) & (errors_b < ta) & (metric < 0.09))
    fa = ga & consistent & (errors_a < tb)
    fb = gb & consistent & (errors_b < ta)
    n, m = int(fa.sum()), int(fb.sum())
    # Report the conservative independent-direction count, not the sum (the
    # same pixel match can be present in both). "Independent" here means fits,
    # not independent physical observations across keyframes.
    amount = min(n, m)
    ratio = float(min(n / ga.sum(), m / gb.sum()))
    diag.update(
        pnp_forward_inliers=n,
        pnp_reverse_inliers=m,
        pnp_paired_depth=int(both.sum()),
        inliers=amount,
        visual_inliers=amount,
        inlier_ratio=ratio,
    )
    if amount < limit or ratio < 0.30:
        return reject("pnp_insufficient_consistent_reprojections")
    coverage = [
        _support(px[mask], frame["depth"].shape) for px, frame in zip((a, b), frames) for mask in (fa, fb)
    ]
    cells, area, inner = (min(row[j] for row in coverage) for j in range(3))
    spread = min(math.sqrt(max(0.0, np.linalg.eigvalsh(np.cov(x.T))[1])) for x in (xyz_a[fa], xyz_b[fb]))
    diag.update(
        image_cells=cells,
        pnp_min_hull_fraction=area,
        pnp_min_inner_area_fraction=inner,
        spread=spread,
        pnp_median_reprojection_px=max(float(np.median(errors_a[fa])), float(np.median(errors_b[fb]))),
    )
    if cells < 4 or area < 0.014 or inner < 0.008 or spread < 0.05:
        return reject("pnp_localized_or_degenerate_support")
    # A second coherent and materially different solution among the rejected
    # pixels is evidence of repeated structure/foreground motion. Don't pick
    # one based solely on its slightly larger support.
    for x, uv, k, tol, rest, nbest, pose in (
        (xyz_a, b, K[1], tb, ga & (errors_a >= tb), n, T),
        (xyz_b, a, K[0], ta, gb & (errors_b >= ta), m, np.linalg.inv(T)),
    ):
        needed = max(limit, math.ceil(0.75 * nbest))
        if rest.sum() < needed:
            continue
        alt = _fit(x[rest], uv[rest], k, tol)
        if alt is not None and np.count_nonzero(_project_errors(x[rest], uv[rest], k, alt) < tol) >= needed:
            ds, da = pose_delta(alt, pose)
            if ds > 0.10 or da > math.radians(4):
                return reject("pnp_competing_pose")
    return dict(diag, ok=True, reason="bidirectional_pnp_seed", T_target_source=T)


def _balanced_holdout(a, b, shapes, valid):
    """Reserve ~1/4 of usable correspondences before fitting any pose.

    Both image grids get proportional coverage. Neither source designation,
    detector enumeration nor reprojection residuals select the witnesses.
    """
    eligible = np.flatnonzero(valid)
    hold = np.zeros(len(a), bool)
    if not len(eligible):
        return hold
    uv = [p[eligible] / [w, h] for p, (h, w) in zip((a, b), shapes)]
    cells = []
    for p in uv:
        xy = np.clip(np.floor(p * [6, 4]).astype(int), [0, 0], [5, 3])
        cells.append(xy[:, 0] + 6*xy[:, 1])
    # Tie-break in a stable, direction-free order before drawing fixed random
    # priorities: rows swapping or cameras swapping must reserve the same pairs.
    x, y = uv
    flip = (x[:, 0] > y[:, 0]) | ((x[:, 0] == y[:, 0]) & (x[:, 1] > y[:, 1]))
    first, second = np.where(flip[:, None], y, x), np.where(flip[:, None], x, y)
    order = np.lexsort((second[:, 1], second[:, 0], first[:, 1], first[:, 0]))
    tie = np.empty(len(order))
    tie[order] = np.random.default_rng(173).random(len(order))
    totals = [np.bincount(c, minlength=24).clip(1) for c in cells]
    counts = [np.zeros(24, int), np.zeros(24, int)]
    selected = np.zeros(len(eligible), bool)
    for _ in range(round(0.24*len(eligible))):
        pa, pb = [(count[c] + .5) / total[c] for c, count, total in zip(cells, counts, totals)]
        cost = np.maximum(pa, pb) + .25 * (pa+pb) + 1e-8*tie
        cost[selected] = np.inf
        i = int(np.argmin(cost))
        selected[i] = True
        for c, count in zip(cells, counts):
            count[c[i]] += 1
    hold[eligible[selected]] = True
    return hold


def reprojection_partition(source, target, a, b):
    """Indices of distinct original matches; reserve the same pixels for both directions."""
    from collab_geometry import _backproject_pixels
    ix = _unique_matches(a, b)
    shapes = [v['depth'].shape for v in (source, target)]
    good = [_backproject_pixels(p[ix], f['depth'], f['K'])[1]
            for p, f in zip((a, b), (source, target))]
    hold = _balanced_holdout(a[ix], b[ix], shapes, good[0] | good[1])
    return ix[~hold], ix[hold]


def prepare_reprojection(source, target, pixels_a, pixels_b, seed_T, min_inliers=12, seed_indices=None,
                         hold_indices=None):
    """Refinement and audit share a precommitted, bidirectional partition.

    Held-out correspondences contribute to neither initialization nor pixel
    refinement in verify_rgbd. They are not independent camera viewpoints.
    """
    from collab_geometry import _backproject_pixels

    a, b = map(lambda x: np.asarray(x, dtype=float).reshape(-1, 2), (pixels_a, pixels_b))
    ix = _unique_matches(a, b)
    a, b = a[ix], b[ix]
    Ka, Kb = np.asarray(source["K"]), np.asarray(target["K"])
    A, ga = _backproject_pixels(a, source["depth"], Ka)
    B, gb = _backproject_pixels(b, target["depth"], Kb)
    T = np.asarray(seed_T)
    ea = _project_errors(A, b, Kb, T)
    eb = _project_errors(B, a, Ka, np.linalg.inv(T))
    fa, fb = 0.5 * (Ka[0, 0] + Ka[1, 1]), 0.5 * (Kb[0, 0] + Kb[1, 1])
    # Loose capture window is a proposal filter only. Do not veto the only
    # valid depth because the other camera happened to hit a depth edge.
    possible = (ga & (ea < fb * 0.055)) | (gb & (eb < fa * 0.055))
    both = ga & gb
    metric = np.linalg.norm(A @ T[:3, :3].T + T[:3, 3] - B, axis=1)
    possible &= ~both | (metric < 0.19)
    shapes = [f["depth"].shape for f in (source, target)]
    hold = (_balanced_holdout(a, b, shapes, ga | gb) if hold_indices is None
            else np.isin(ix, hold_indices))
    seed = (
        np.isin(ix, np.asarray(seed_indices, dtype=int))
        if seed_indices is not None
        else np.ones(len(ix), bool)
    )
    # Small fixed training grid, deliberately offset from the regular dense
    # verification grid. Use original calibrated depth, never display points.
    surfaces = []
    for frame, K in zip((source, target), (Ka, Kb)):
        depth = frame["depth"]
        step = max(8, math.ceil(math.sqrt(depth.size / 1000.0)))
        y, x = np.mgrid[step // 2 : depth.shape[0] : step, step // 2 : depth.shape[1] : step]
        points, valid = _backproject_pixels(np.column_stack((x.ravel(), y.ravel())), depth, K)
        surfaces.append(points[valid])
    return dict(
        a=a,
        b=b,
        A=A,
        B=B,
        ga=ga,
        gb=gb,
        Ka=Ka,
        Kb=Kb,
        fa=fa,
        fb=fb,
        shapes=shapes,
        surfaces=surfaces,
        depths=(source["depth"], target["depth"]),
        limit=max(8, int(min_inliers)),
        possible=possible,
        hold=hold,
        seed=seed,
    )


def audit_reprojection(data, T):
    """Check rectified pixel placement at a *fixed* candidate transform."""
    p = data
    A, B = p["A"], p["B"]
    ea = _project_errors(A, p["b"], p["Kb"], T)
    eb = _project_errors(B, p["a"], p["Ka"], np.linalg.inv(T))
    # ~0.63° in either camera: 5.9px for a 540px focal length,
    # 10px at 900px; bounds keep this tighter than depth-only RANSAC.
    ta, tb = np.clip(0.011 * np.array([p["fa"], p["fb"]]), 2.5, 10.0)
    ga, gb = p["ga"], p["gb"]
    metric = np.linalg.norm(A @ T[:3, :3].T + T[:3, 3] - B, axis=1)
    ok = (ga | gb) & (~ga | (ea < tb)) & (~gb | (eb < ta)) & (~(ga & gb) | (metric < 0.09))
    n = int(ok.sum())
    # Match proposals include unrelated descriptors. As in the training branch,
    # evaluate the support fraction within the *initial seed's* fixed and
    # deliberately loose pixel/depth capture window. Never reselect it using
    # the refined transform; do not fit either seed or refinement on ANY
    # reserved correspondence, including those outside the window.
    witnesses = p["hold"] & p["possible"]
    hold = ok & witnesses
    cells = (0, 0)
    hull = inner = 0.0
    if n >= 3:
        areas = [_support(x[ok], s) for x, s in zip((p["a"], p["b"]), p["shapes"])]
        cells = tuple(v[0] for v in areas)
        hull, inner = (min(v[j] for v in areas) for j in (1, 2))
    # Do not call a rigid fit accurate if all metric depth support is from a
    # single view. Count each distinct matched pixel just once.
    dirs = [int((ok & valid).sum()) for valid in (ga, gb)]
    supported = n >= p["limit"] and min(dirs) >= max(6, math.ceil(0.55 * p["limit"]))
    distributed = min(cells) >= 4 and hull >= 0.014 and inner >= 0.008
    # If enough pairs exist for refinement, require measured error also on
    # unseen pixels. These are *held out from refinement*, not new images.
    independent = int((hold & ~p["seed"]).sum())
    held_n = int(witnesses.sum())
    h_ok = held_n >= 3 and int(hold.sum()) >= max(3, math.ceil(0.22 * held_n))
    held_direction = [int((hold & valid).sum()) for valid in (ga, gb)]
    h_ok &= min(held_direction) >= 2
    hold_regions = [
        len(np.unique(np.floor(v[hold] / [s[1] / 4, s[0] / 3]), axis=0))
        for v, s in zip((p["a"], p["b"]), p["shapes"])
    ]
    h_ok &= min(hold_regions) >= 2
    return dict(
        ok=bool(supported and distributed and h_ok),
        reason=(
            "reprojection_consistent"
            if supported and distributed and h_ok
            else "insufficient_bidirectional_pixel_support"
            if not supported
            else "localized_reprojection_support"
            if not distributed
            else "insufficient_held_out_pixel_support"
        ),
        reprojection_inliers=n,
        reprojection_forward_inliers=dirs[0],
        reprojection_reverse_inliers=dirs[1],
        reprojection_image_cells=list(cells),
        reprojection_min_hull_fraction=hull,
        reprojection_inner_area_fraction=inner,
        reprojection_holdout_inliers=int(hold.sum()),
        reprojection_holdout_count=held_n,
        reprojection_reserved_count=int(p["hold"].sum()),
        reprojection_holdout_forward_inliers=held_direction[0],
        reprojection_holdout_reverse_inliers=held_direction[1],
        reprojection_holdout_cells=hold_regions,
        reprojection_holdout_nonseed_inliers=independent,
        reprojection_median_px_forward=float(np.median(ea[ga & ok])) if dirs[0] else None,
        reprojection_median_px_reverse=float(np.median(eb[gb & ok])) if dirs[1] else None,
    )


def _surface_terms(samples, depth, K, pose):
    """Project onto the other view and estimate a measured local tangent plane.

    A bounded angular/projective association is recomputed outside each linear
    solve. Normals require smooth neighbourhoods in all four directions.
    """
    from collab_geometry import _backproject_pixels

    warp = samples @ pose[:3, :3].T + pose[:3, 3]
    good = warp[:, 2] > 0.15
    vp = warp[good] @ K.T
    pix = vp[:, :2] / vp[:, 2:]
    if not len(pix):
        return tuple(np.empty((0, 3)) for _ in range(3))
    dif = max(2, round(4 * K[0, 0] / 900))
    pts = []
    use = np.ones(len(pix), bool)
    for dx, dy in ((0, 0), (-dif, 0), (dif, 0), (0, -dif), (0, dif)):
        v, ok = _backproject_pixels(pix + [dx, dy], depth, K)
        use &= ok
        pts.append(v)
    center, lt, rt, up, dn = pts
    z = np.stack([v[:, 2] for v in pts])
    use &= (np.ptp(z, axis=0) < 0.075) & (np.abs(warp[good, 2] - center[:, 2]) < 0.18)
    normal = np.cross(rt - lt, dn - up)
    norm = np.linalg.norm(normal, axis=1)
    use &= norm > 1e-7
    return samples[good][use], center[use], normal[use] / norm[use, None]


def refine_reprojection(data, seed_T, *, joint_depth=True):
    """IRLS on signed pixel residuals, bounded to the supplied metric seed."""
    from collab_geometry import pose_delta

    p = data
    seed = np.array(seed_T, dtype=float)
    train = p["possible"] & ~p["hold"]
    T = seed.copy()
    if train.sum() < 8:
        return dict(audit_reprojection(p, T), T_target_source=T)

    # Use each surviving observation in its own camera. We optimize in
    # normalized camera coordinates to give differing intrinsics equal weight.
    def select(t):
        f = train & p["ga"] & (_project_errors(p["A"], p["b"], p["Kb"], t) < 0.055 * p["fb"])
        b = train & p["gb"] & (_project_errors(p["B"], p["a"], p["Ka"], np.linalg.inv(t)) < 0.055 * p["fa"])
        return f, b

    f, b = select(T)
    if min(f.sum(), b.sum()) < 4:
        return dict(audit_reprojection(p, T), T_target_source=T)

    def increment(step):
        M = np.eye(4)
        M[:3, 3] = step[:3]
        M[:3, :3] = cv2.Rodrigues(step[3:])[0]
        return M

    def pixel_residual(t):
        ar = p["A"][f] @ t[:3, :3].T + t[:3, 3]
        br = (p["B"][b] - t[:3, 3]) @ t[:3, :3]
        q1 = ar @ p["Kb"].T
        q2 = br @ p["Ka"].T
        r1 = (q1[:, :2] / q1[:, 2:].clip(0.1) - p["b"][f]) / p["fb"]
        r2 = (q2[:, :2] / q2[:, 2:].clip(0.1) - p["a"][b]) / p["fa"]
        return np.vstack((r1, r2))

    # Bound influence of high-texture objects, not only pixel outliers.
    weights = []
    for pix, s in zip((p["a"][f], p["b"][b]), p["shapes"]):
        cell = (np.floor(pix[:, 0] * 8 / s[1]) + 8 * np.floor(pix[:, 1] * 6 / s[0])).astype(int)
        _, ind, cnt = np.unique(cell, return_inverse=True, return_counts=True)
        weights.extend(1 / np.maximum(1, cnt[ind] / 4))
    weights = np.asarray(weights)
    for _ in range(12):
        residuals = pixel_residual(T)
        norms = np.linalg.norm(residuals, axis=1)
        w = np.repeat(np.sqrt(weights / np.maximum(1.0, (norms / 0.004) ** 2)), 2)

        def e_at(t):
            return pixel_residual(t).reshape(-1)

        if joint_depth:
            # Train on noisy surface distances AND measured pixels; do not
            # pick between several resulting poses according to the holdout.
            # 4m/rad maps a 1 cm plane error to a 0.0025 angular residual.
            pa, ma, na = _surface_terms(p["surfaces"][0], p["depths"][1], p["Kb"], T)
            pb, mb, nb = _surface_terms(p["surfaces"][1], p["depths"][0], p["Ka"], np.linalg.inv(T))
            if min(len(pa), len(pb)) >= 20:

                def plane_residual(t):
                    ra = np.sum((pa @ t[:3, :3].T + t[:3, 3] - ma) * na, axis=1)
                    rb = np.sum(((pb - t[:3, 3]) @ t[:3, :3] - mb) * nb, axis=1)
                    return np.r_[ra, rb] / 4.0

                re = plane_residual(T)
                # Both cameras together contribute a bounded number of votes.
                # Avoid magnifying depth just because the input grid is dense.
                wd = np.full(len(re), max(6.0, weights.sum()) / len(re))
                wd = np.sqrt(wd / np.maximum(1.0, (np.abs(re) / 0.0125) ** 2))
                w = np.r_[w, wd]

                def e_at(t):
                    return np.r_[pixel_residual(t).reshape(-1), plane_residual(t)]

        e = e_at(T)
        J = np.column_stack([(e_at(increment(vec) @ T) - e) / 1e-5 for vec in np.eye(6) * 1e-5])
        H = J * w[:, None]
        step = np.linalg.lstsq(H, -w * e, rcond=1e-7)[0]
        if not np.isfinite(step).all() or np.linalg.norm(step) < 1e-6:
            break
        scale = min(
            1.0, 0.035 / max(1e-9, np.linalg.norm(step[:3])), 0.025 / max(1e-9, np.linalg.norm(step[3:]))
        )
        accepted = False
        for fraction in (scale, scale * 0.5, scale * 0.25):
            proposal = increment(step * fraction) @ T
            dist, ang = pose_delta(seed, proposal)
            if (
                dist < 0.15
                and ang < np.radians(8)
                and np.linalg.norm(e_at(proposal) * w) < np.linalg.norm(e * w)
            ):
                T = proposal
                accepted = True
                break
        if not accepted:
            break
    return dict(audit_reprojection(p, T), T_target_source=T)
