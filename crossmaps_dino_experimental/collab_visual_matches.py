"""Read-only alternate RGB proposals for cross-camera registration.

Pairs are not inliers. The caller must verify actual depth, geometric spread,
registration residuals and free-space consistency before accepting anything.
"""
from collections import OrderedDict
from threading import Lock
import weakref

import cv2
import numpy as np


_CACHE = OrderedDict()
_LOCK = Lock()
_CACHE_LIMIT = 12
_MAX_FEATURES = 2400
_RATIO = .84
_NAME = 'grid_rootsift_akaze_mutual'


def _features(image, scale, kind='sift'):
    # Cache features, not camera payloads. The weak reference guards against
    # recycling a Python address. Stored keyframe images are immutable.
    key = (id(image), scale, kind)
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None and hit[0]() is image:
            _CACHE.move_to_end(key)
            return hit[1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if scale != 1.:
        gray = cv2.resize(gray, (max(1, round(image.shape[1]*scale)),
                                 max(1, round(image.shape[0]*scale))), interpolation=cv2.INTER_AREA)
        gray = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(gray)
    detector = (cv2.AKAZE_create(threshold=.00035) if kind == 'akaze' else
                cv2.SIFT_create(nfeatures=_MAX_FEATURES + 400, contrastThreshold=.015))
    keypoints, desc = detector.detectAndCompute(gray, None)
    # Avoid dense texture on a single object consuming the descriptor budget.
    # Round-robin spatial cells; unused quotas go to occupied cells.
    rows, cols = 5, 7
    h, w = gray.shape
    buckets = [[] for _ in range(rows * cols)]
    for idx in sorted(range(len(keypoints)), key=lambda i: -keypoints[i].response):
        kp = keypoints[idx]
        x, y = kp.pt
        cell = min(rows-1, int(y * rows / h)) * cols + min(cols-1, int(x * cols / w))
        buckets[cell].append(idx)
    occupied = [iter(bucket) for bucket in buckets if bucket]
    ordered = []
    while occupied and len(ordered) < _MAX_FEATURES:
        next_round = []
        for entries in occupied:
            kp = next(entries, None)
            if kp is not None:
                ordered.append(kp)
                next_round.append(entries)
                if len(ordered) == _MAX_FEATURES:
                    break
        occupied = next_round
    pixels = np.asarray([keypoints[i].pt for i in ordered], dtype=np.float32).reshape(-1, 2)
    pixels = (pixels + .5) * [image.shape[1]/w, image.shape[0]/h] - .5
    if desc is not None:
        desc = np.ascontiguousarray(desc[ordered])
        if kind != 'akaze':
            desc /= np.maximum(desc.sum(axis=1, keepdims=True), 1.e-8)
            np.sqrt(desc, out=desc)
    result = (pixels, desc)
    with _LOCK:
        _CACHE[key] = (weakref.ref(image), result)
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_LIMIT:
            _CACHE.popitem(last=False)
    return result


def visual_matches(source, target):
    """Return original-camera Nx2 pixel arrays and JSON-safe diagnostics."""
    empty = np.empty((0, 2), dtype=np.float32)
    diag = dict(matcher=_NAME, source_features=0, target_features=0, visual_matches=0)
    if not hasattr(cv2, 'SIFT_create'):
        return empty, empty, dict(diag, matcher_unavailable=True)
    def distinctive(matcher, ratio, d1, d2):
        out = {}
        for pair in matcher.knnMatch(d1, d2, k=2):
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                out[m.queryIdx] = (m.trainIdx, m.distance/n.distance)
        return out
    proposals = []
    # AKAZE's binary nonlinear-scale-space features complement SIFT on
    # contrast boundaries. Their distances are compared only within family.
    for scale, kind in ((1., 'sift'), (.75, 'sift'), (1., 'akaze')):
        a, da = _features(source['rgb'], scale, kind)
        b, db = _features(target['rgb'], scale, kind)
        diag['source_features'] += len(a)
        diag['target_features'] += len(b)
        if min(len(a), len(b)) < 2:
            continue
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING if kind == 'akaze' else cv2.NORM_L2)
        ratio = .86 if kind == 'akaze' else _RATIO
        ab = distinctive(matcher, ratio, da, db)
        ba = distinctive(matcher, ratio, db, da)
        before = len(proposals)
        for i, (j, q) in ab.items():
            if j in ba and ba[j][0] == i:
                proposals.append((max(q, ba[j][1]), a[i], b[j]))
        diag[f'{kind}_{scale}_mutual'] = len(proposals)-before
    # Distinct orientations / scale-space detections of the same physical
    # corner must not masquerade as independent support for a transform.
    pa, pb, seen_a, seen_b = [], [], set(), set()
    for _, a, b in sorted(proposals, key=lambda p: p[0]):
        ka, kb = tuple(np.rint(a/4).astype(int)), tuple(np.rint(b/4).astype(int))
        if ka not in seen_a and kb not in seen_b:
            seen_a.add(ka); seen_b.add(kb); pa.append(a); pb.append(b)
    if not pa:
        return empty, empty, diag
    diag['visual_matches'] = len(pa)
    return np.asarray(pa, np.float32), np.asarray(pb, np.float32), diag
