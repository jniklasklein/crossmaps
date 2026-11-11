"""Display-only mask filling. Never feed interpolated labels back to geometry."""
import colorsys
import cv2
import numpy as np


def id_color(identifier):
    hue = (int(identifier) * 0.61803398875) % 1.0
    return tuple(round(255 * v) for v in colorsys.hsv_to_rgb(hue, 0.72, 1.0))


def dense_display_ids(seeds, masks, radius=4):
    seeds = np.asarray(seeds, dtype=np.int32)
    dense = np.zeros_like(seeds)
    covered = np.zeros_like(seeds, dtype=bool)
    # Smaller SAM proposals retain their boundaries inside larger proposals.
    for mask in sorted((m for m in masks if m is not None and m.shape == seeds.shape), key=np.count_nonzero):
        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask & (seeds > 0)):
            # A larger overlapping proposal must not paint over an explicitly
            # unseeded small mask and make it look geometrically attributed.
            covered |= mask
            continue
        y, x = np.nonzero(mask)
        box = np.s_[y.min():y.max()+1, x.min():x.max()+1]
        local_mask, local_seeds = mask[box], seeds[box]
        known = local_mask & (local_seeds > 0)
        _, labels = cv2.distanceTransformWithLabels((~known).astype(np.uint8), cv2.DIST_L2, 5,
                                                   labelType=cv2.DIST_LABEL_PIXEL)
        lookup = np.zeros(int(labels.max()) + 1, dtype=np.int32)
        lookup[labels[known]] = local_seeds[known]
        region = dense[box]
        fill = local_mask & ~covered[box]
        region[fill] = lookup[labels[fill]]
        covered |= mask
    # Masks can temporarily disappear. Keep *current-depth* reprojection visible,
    # but only within half a sampling stride; never reuse yesterday's pixels.
    if np.any(seeds > 0) and radius > 0:
        known = seeds > 0
        distance, labels = cv2.distanceTransformWithLabels((~known).astype(np.uint8), cv2.DIST_L2, 5,
                                                          labelType=cv2.DIST_LABEL_PIXEL)
        lookup = np.zeros(int(labels.max()) + 1, dtype=np.int32)
        lookup[labels[known]] = seeds[known]
        fill = ~covered & (dense == 0) & (distance <= radius)
        dense[fill] = lookup[labels[fill]]
    dense[seeds > 0] = seeds[seeds > 0]
    return dense


def render_overlay(ids, masks, rgb, color, alpha=0.78):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    base = np.zeros((*ids.shape, 3), dtype=np.uint8) if rgb is None else np.asarray(rgb, dtype=np.uint8)
    result = (base.astype(np.float32) * 0.5).astype(np.uint8)
    mask_union = np.zeros(ids.shape, dtype=bool)
    for mask in masks:
        if mask is not None and mask.shape == ids.shape:
            mask_union |= np.asarray(mask, dtype=bool)
    missing = mask_union & (ids == 0)
    result[missing] = (0.35 * base[missing] + 0.65 * np.array([175, 175, 175])).astype(np.uint8)
    # One palette lookup per pixel, instead of scanning the entire RGB image
    # separately for every segment (costly for many fine masks on CPU/MPS).
    unique, inverse = np.unique(ids, return_inverse=True)
    palette = np.asarray([color(int(sid)) if sid > 0 else (0, 0, 0) for sid in unique], dtype=np.float32)
    colored = palette[inverse].reshape((*ids.shape, 3))
    blended = ((1-alpha) * base.astype(np.float32) + alpha * colored).astype(np.uint8)
    np.copyto(result, blended, where=(ids > 0)[..., None])
    return result


def mask_consensus_ids(shape, masks, decisions, map_id=lambda sid: sid, dominant=False):
    """Exact current SAM silhouettes; no nearest-seed internal partitions.

    A mixed mask is unknown unless all supported segments map to one entity.
    Smaller proposals take precedence; unassigned masks do not borrow labels.
    """
    labels = np.zeros(shape, dtype=np.int32)
    ambiguous = np.zeros(shape, dtype=bool)
    covered = np.zeros(shape, dtype=bool)
    by_index = {d['mask']: d for d in decisions}
    for index in sorted(range(len(masks)), key=lambda i: np.count_nonzero(masks[i]) if masks[i] is not None else 0):
        mask = masks[index]
        if mask is None or mask.shape != shape:
            continue
        region = np.asarray(mask, dtype=bool) & ~covered
        covered |= np.asarray(mask, dtype=bool)
        item = by_index.get(index, {})
        if dominant:
            owner = item.get('display_owner')
            if owner is not None:
                labels[region] = int(owner)
                ambiguous[region] = bool(item.get('display_provisional'))
            continue
        winner = item.get('winner')
        owner = map_id(winner) if winner is not None else None
        if owner is None:
            mapped = {map_id(s) for s in item.get('segments', [])}
            if len(mapped) == 1 and None not in mapped and item.get('reason') == 'multiple_supported_owners':
                owner = next(iter(mapped))
        if owner is not None:
            labels[region] = int(owner)
        else:
            ambiguous[region] = len(item.get('segments', [])) > 1 or item.get('reason') in ('appearance_conflict', 'pending_owner_switch')
    return labels, ambiguous


def render_mask_consensus(shape, masks, decisions, rgb, map_id, color, alpha=.78, prefix='S', seeds=None, radius=4, dominant=False):
    labels, ambiguous = mask_consensus_ids(shape, masks, decisions, map_id, dominant)
    if not dominant and seeds is not None and np.shape(seeds) == shape:
        # Mixed masks still have useful measured support. Show it locally,
        # without a full-mask Voronoi partition or changing world ownership.
        mapped = np.zeros(shape, dtype=np.int32)
        for sid in np.unique(seeds):
            owner = map_id(int(sid)) if sid > 0 else None
            if owner is not None:
                mapped[seeds == sid] = owner
        local = dense_display_ids(mapped, [], radius=max(0, radius))
        fill = ambiguous & (local > 0)
        labels[fill] = local[fill]
        ambiguous[fill] = False
    view = render_overlay(labels, masks, rgb, color, alpha)
    # Sparse amber hatching communicates unresolved evidence without hiding
    # the RGB image under a large, apparently solid 'amber object'.
    yy, xx = np.indices(shape)
    hatch = ambiguous & (((xx+yy) % 12) < 3) & (not dominant)
    view[hatch] = (0.25*view[hatch] + 0.75*np.array([235, 178, 60])).astype(np.uint8)
    # Contours and IDs make color collisions and mixed observations inspectable.
    for mask in masks:
        if mask is None or mask.shape != shape:
            continue
        contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view, contours, -1, (235, 235, 235), 1)
    if dominant:
        # Provisional choices retain a solid interior; uncertainty is carried
        # by the outline and ID label, not competing colors within the mask.
        boundary = ambiguous & ~cv2.erode(ambiguous.astype(np.uint8), np.ones((3, 3), np.uint8),
                                          borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        view[boundary & (((xx+yy) % 12) < 6)] = (235, 178, 60)
    for sid in np.unique(labels):
        region = labels == sid
        if sid <= 0 or np.count_nonzero(region) < 300:
            continue
        distance = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 3)
        _, _, _, position = cv2.minMaxLoc(distance)
        text = f'{prefix}{sid}' + ('?' if dominant and np.any(ambiguous & region) else '')
        cv2.putText(view, text, position, cv2.FONT_HERSHEY_SIMPLEX, .45, (15,15,15), 3, cv2.LINE_AA)
        cv2.putText(view, text, position, cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1, cv2.LINE_AA)
    return view
