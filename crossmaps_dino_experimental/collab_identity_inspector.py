"""Read-only, on-demand identity evidence from retained source keyframes.

A link holds the frame IDs and scores from its last compatible evaluation.
An archived or suspended decision must not silently acquire *current* images.
Membership scopes every lookup; our hashes are identities, not file paths.
"""
from itertools import combinations

import cv2
import numpy as np


SUPPORT_NOTE = "Highlighted dots are measured 3D surface samples projected into their original camera image, not a filled SAM mask."


def _surfaces(frame, member, make_id):
    robot, session = frame['robot_id'], frame['session_id']
    return [s for s in frame.get('surfaces', ())
            if s.get('entity_id') is not None and s.get('quality', 0.) >= .3
            and make_id('entity', robot, session, s['entity_id']) == member]


def _pixels(frame, surfaces):
    """Return distinct in-frame pixels from original camera-space samples only."""
    parts = [s['points_camera'] for s in surfaces if s.get('points_camera') is not None]
    if not parts:
        return np.empty((0, 2), dtype=np.int32)
    points = np.vstack(parts)
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0)]
    # For rectified camera intrinsics K the stored samples already have the
    # correct optical frame. Never apply a world pose or use any other view.
    hom = points @ np.asarray(frame['K']).T
    h, w = frame['rgb'].shape[:2]
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        xy = np.rint(hom[:, :2] / hom[:, 2:])
    xy = xy[(xy >= 0).all(axis=1) & (xy < [w, h]).all(axis=1)]
    return np.unique(xy.astype(np.int32), axis=0)


def _member_details(backend, link, member, make_id, known):
    result = dict(id=member, robot_id=None, local_id=None, session_id=None,
                  distinct_views=False, distinct_pair=None, views=[])
    if member in known:
        result.update({k: known[member].get(k) for k in ('robot_id', 'session_id', 'local_id')})
    found = []
    for fid in set(link.get('witnesses', ())):
        frame = backend.keyframes.get(fid)
        if frame is None:
            continue
        surfaces = _surfaces(frame, member, make_id)
        if surfaces:
            found.append((frame, surfaces))
    found.sort(key=lambda item: (item[0]['seq'], item[0]['id']), reverse=True)
    for frame, surfaces in found:
        pixels = _pixels(frame, surfaces)
        result.update(robot_id=frame['robot_id'], session_id=frame['session_id'], local_id=surfaces[0]['entity_id'])
        result['views'].append(dict(frame_id=frame['id'], seq=int(frame['seq']),
                                    stamp_ns=str(frame.get('rgb_stamp_ns') or frame['stamp_ns']),
                                    surface_ids=sorted({s['local_id'] for s in surfaces}),
                                    projected_points=len(pixels),
                                    distinct_from_reference=bool(backend._distinct_views(found[0][0], frame))))
    for (a, _), (b, _) in combinations(found, 2):
        if backend._distinct_views(a, b):
            result.update(distinct_views=True, distinct_pair=[a['id'], b['id']])
            break
    return result


def inspect_link(backend, link_id, make_id):
    link = backend.entity_links.get(link_id)
    if link is None:
        return None
    groups = [g['members'] for g in backend._entity_cache]
    known = {m['id']: m for group in groups for m in group}
    members = [_member_details(backend, link, uid, make_id, known) for uid in link['members']]
    merged = any(set(link['members']).issubset({m['id'] for m in group}) for group in groups)
    aligned = bool(link.get('alignment_valid'))
    current = link.get('compatible_current') if aligned else None
    blocked = []
    if link['state'] == 'rejected' or link_id in backend.entity_exclusions:
        blocked.append("This identity link was explicitly rejected or unlinked; automatic updates respect that decision.")
    if not aligned:
        blocked.append("Both members are not currently available in aligned submaps.")
    elif current is False and link_id not in backend.entity_exclusions:
        blocked.append("The last compatible evidence is shown. This pair no longer passes all current eligibility checks.")
    if link.get('active') and current and not merged:
        blocked.append("Grouping would put competing identities from one source in the same group.")
    if not merged and current and link['state'] == 'proposed':
        for m in members:
            if not m['views']:
                blocked.append(f"No retained witness view for {m['robot_id'] or m['id']}.")
            elif not m['distinct_views']:
                blocked.append(f"{m['robot_id']} has no pair of distinct viewpoints among its {len(m['views'])} contributing keyframes.")
    if not merged and backend.config['shadow_mode'] and not link.get('active'):
        blocked.append("Shadow mode disables automatic acceptance.")

    return dict(id=link_id, state=link['state'], active=bool(link.get('active')), merged=merged,
                operator_decision=bool(link.get('operator_decision')), alignment_valid=aligned,
                compatible_current=current, shadow_mode=bool(backend.config['shadow_mode']),
                evidence_scope='current' if current else 'last_compatible', members=members,
                dino=link.get('dino'), clip=link.get('clip'), overlap=link.get('overlap'),
                thresholds={k: backend.config[k] for k in ('corroboration_translation', 'corroboration_rotation_deg',
                            'entity_overlap', 'entity_distance', 'entity_dino_similarity', 'entity_clip_similarity')},
                blocking_reasons=blocked, support_note=SUPPORT_NOTE)


def render_witness(backend, link_id, member_id, frame_id, make_id):
    link = backend.entity_links.get(link_id)
    if link is None or member_id not in link['members'] or frame_id not in link.get('witnesses', ()):
        return None
    frame = backend.keyframes.get(frame_id)
    if frame is None:
        return None
    pixels = _pixels(frame, _surfaces(frame, member_id, make_id))
    if not len(pixels):
        return None
    # The historical field is called rgb; the camera/preview payload stored
    # here is OpenCV BGR. imencode expects precisely that order.
    original = frame['rgb']
    h, w = original.shape[:2]
    scale = min(1., 800/max(w, h))
    width, height = max(1, round(w*scale)), max(1, round(h*scale))
    image = cv2.resize(original, (width, height), interpolation=cv2.INTER_AREA)
    # Project pixel centres using the same resize geometry as the source.
    xy = np.rint((pixels + .5) * [width/w, height/h] - .5).astype(int)
    xy = np.clip(xy, [0, 0], [width-1, height-1])
    dots = np.zeros((height, width), dtype=np.uint8)
    dots[xy[:, 1], xy[:, 0]] = 255
    rim = cv2.dilate(dots, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    core = cv2.dilate(dots, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    image[rim > 0] = (20, 28, 29)
    image[core > 0] = (43, 235, 249)
    ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes() if ok else None
