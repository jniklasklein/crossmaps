"""Batched equivalent of the scalar voxel query/confidence calculation."""
import numpy as np


def score_voxels(keys, sums, weights, views, text, weight_max, bins, view_strength,
                 view_power, coherence_floor, coherence_power, use_view, use_conf, use_sem):
    kept, parts = [], []
    keys = list(keys)
    # Limit temporary descriptor matrices; returned embeddings are required by
    # memory promotion, as before. No cross-cycle cache can hide new evidence.
    for offset in range(0, len(keys), 1024):
        local = [k for k in keys[offset:offset+1024] if k in sums and weights.get(k, 0.) > 1e-6]
        if not local:
            continue
        matrix = np.asarray([sums[k] for k in local], dtype=np.float32)
        w = np.asarray([weights[k] for k in local], dtype=np.float64)
        norm = np.linalg.norm(matrix, axis=1)
        valid = np.isfinite(norm) & np.isfinite(w) & (norm > 1e-6)
        if not np.any(valid):
            continue
        local = [k for k, good in zip(local, valid) if good]
        matrix, w, norm = matrix[valid], w[valid], norm[valid]
        unit = matrix/norm[:, None]
        coherence = np.linalg.norm(matrix/w.astype(np.float32)[:, None], axis=1)
        coh = np.clip((coherence.astype(np.float64)-coherence_floor)/max(1.-coherence_floor, 1e-6), 0., 1.)**coherence_power
        scores = np.zeros(len(local), dtype=np.float64) if text is None else np.maximum(0., np.einsum('ij,j->i', unit, text))
        if use_sem:
            scores = scores*coh
        conf = np.clip(w/max(weight_max, 1e-6), 0., 1.)
        if use_conf:
            conf *= coh
        counts = np.asarray([int(views.get(k, 0)).bit_count() if bins > 1 else int(views.get(k, 0) != 0)
                             for k in local], dtype=np.int32)
        if use_view and bins > 1:
            conf = np.clip(conf*(1.+view_strength*(counts/float(bins))**view_power), 0., 1.)
        kept.extend(local)
        parts.append((unit, coh.astype(np.float32), scores.astype(np.float32), conf.astype(np.float32), counts))
    if not parts:
        return None
    return (kept, *(np.concatenate([p[i] for p in parts], axis=0) for i in range(5)))
