import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from collab_backend import Backend


def observation(name, seq=1):
    # Camera-space points at distinct pixels; BGR intentionally asymmetric.
    support = np.array([[-.3, 0., 1.], [-.22, .1, 1.], [-.35, -.1, 1.]], dtype='f')
    w, h = 120, 80
    image = np.full((h, w, 3), (160, 40, 20), dtype='u1')
    k = np.array([[100., 0, 60.], [0, 100., 40.], [0., 0., 1.]])
    pose = np.eye(4)
    pose[0, 3] = 3.  # Projecting optimized world geometry here would miss the object entirely.
    def surface(entity, local, points):
        return dict(entity_id=entity, local_id=local, quality=.8, points_camera=points,
                    dino=np.array([1., 0.]), clip=np.array([1., 0.]))
    return dict(robot_id=name, session_id="session_"+name, seq=seq, stamp_ns=seq*10**10,
                rgb_stamp_ns=seq*10**10 + 31, T_local_camera=pose, K=k, rgb=image,
                depth=np.ones((h, w), dtype='f'), points_camera=support,
                surfaces=[surface('bin','panel',support), surface('other','leaf',-support * [1, 1, -1])])


def prepared(store=None):
    b = Backend(dict(shadow_mode=False, verifications_per_keyframe=0), store)
    for name in ['A', 'B']:
        for seq in (1, 2):
            b.ingest(observation(name, seq))
    b.command(dict(action='anchor', robot_id='B', reference_robot_id='A', T_reference_local=np.eye(4)))
    for k in b.entity_links:
        if {m['local_id'] for m in b.identity_inspection(k)['members']} == {'bin'}:
            return b, k
    raise AssertionError("Expected cross-rover bin proposal")


class IdentityInspectorTests(unittest.TestCase):
    def test_explains_actual_stationary_witnesses_and_uses_rgb_stamp(self):
        b, k = prepared()
        record = b.identity_inspection(k)
        json.dumps(record, allow_nan=False)
        self.assertEqual(record['evidence_scope'], 'current')
        self.assertFalse(record['merged'])
        self.assertEqual(record['dino'], 1.)
        self.assertEqual(record['overlap'], 1.)
        self.assertEqual(len(record['blocking_reasons']), 2)
        for member in record['members']:
            self.assertFalse(member['distinct_views'])
            self.assertIsNone(member['distinct_pair'])
            self.assertEqual([v['seq'] for v in member['views']], [2, 1])
            self.assertEqual(member['views'][0]['stamp_ns'], '20000000031')
            self.assertEqual(member['views'][0]['surface_ids'], ['panel'])
            self.assertEqual(member['views'][0]['projected_points'], 3)
        self.assertIsNone(b.identity_inspection('missing'))

    def test_jpeg_is_camera_local_sparse_and_link_scoped(self):
        b, k = prepared()
        a, other = b.identity_inspection(k)['members']
        frame = a['views'][0]['frame_id']
        before = b.keyframes[frame]['rgb'].copy()
        jpg = b.identity_evidence_image(k, a['id'], frame)
        image = cv2.imdecode(np.frombuffer(jpg, 'u1'), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape, before.shape)
        # Left bin gets a yellow marker, not its mirror or the gap between points.
        self.assertGreater(image[40,30,1], 190)
        for p in [(40,90), (39,42), (12,10)]:
            # JPEG chroma subsampling can perturb a nearby unpainted pixel.
            np.testing.assert_allclose(image[p], before[p], atol=14)
        np.testing.assert_array_equal(b.keyframes[frame]['rgb'], before)
        self.assertIsNone(b.identity_evidence_image(k, other['id'], frame))
        self.assertIsNone(b.identity_evidence_image(k, a['id'], 'unrelated'))
        self.assertIsNone(b.identity_evidence_image('unknown', a['id'], frame))

    def test_historical_and_accepted_decisions_not_current_scores(self):
        b, k = prepared()
        b.command(dict(action='accept_entity_link', id=k))
        linked = b.identity_inspection(k)
        self.assertTrue(linked['merged'])
        self.assertTrue(linked['operator_decision'])
        self.assertFalse(linked['blocking_reasons'])
        b.command(dict(action='clear_anchor', robot_id='B'))
        suspended = b.identity_inspection(k)
        self.assertFalse(suspended['merged'])
        self.assertTrue(suspended['active'])  # Decision retained, not applied.
        self.assertIsNone(suspended['compatible_current'])
        self.assertEqual(suspended['evidence_scope'], 'last_compatible')
        self.assertTrue(all(m['views'] for m in suspended['members']))
        self.assertTrue(suspended['blocking_reasons'])

    def test_offscreen_support_is_not_clamped_to_image_border(self):
        b, k = prepared()
        m = b.identity_inspection(k)['members'][0]
        fid = m['views'][0]['frame_id']
        surf = next(s for s in b.keyframes[fid]['surfaces'] if s['entity_id'] == m['local_id'])
        # One sample behind the camera, two projecting outside the original RGB.
        surf['points_camera'] = np.array([[0., 0., -1.], [2., 0., 1.], [0., -2., 1.]])
        same_member = next(i for i in b.identity_inspection(k)['members'] if i['id'] == m['id'])
        self.assertEqual(same_member['views'][0]['projected_points'], 0)
        self.assertIsNone(b.identity_evidence_image(k, m['id'], fid))

    def test_replay_uses_existing_format_without_modifying_recording(self):
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname)
            _, k = prepared(path)
            def hashes():
                return {p.name: hashlib.sha256(p.read_bytes()).digest() for p in path.iterdir()}
            before = hashes()
            replay = Backend.replay(path)
            meta = replay.identity_inspection(k)
            m = meta['members'][0]
            rev = replay.revision
            self.assertIsNotNone(replay.identity_evidence_image(k, m['id'], m['views'][0]['frame_id']))
            self.assertEqual(rev, replay.revision)
            self.assertEqual(before, hashes())
