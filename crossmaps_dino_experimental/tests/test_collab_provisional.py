"""Contract tests for the stationary preview, using the real ingestion/graph lifecycle.

Only the RGB-D verifier is scripted: every returned pose has an explicit
direction, and only the chosen independent camera observations can verify.
Existing geometry tests exercise the verifier itself.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from collab_backend import Backend
from collab_geometry import verify_rgbd
from test_collab_backend import observation, pose


class ProvisionalAlignmentTests(unittest.TestCase):
    # Deliberately non-identity and noncommuting with B's local camera origin:
    # detecting a transform in the wrong direction is part of the contract.
    a_from_b = pose(.43, -.16, .13)
    b_origin = pose(.26, .12, -.08)

    def setUp(self):
        self.measurements = {}
        fake = patch("collab_backend.verify_rgbd", side_effect=self.verify)
        self.mock_verifier = fake.start()
        self.addCleanup(fake.stop)

    def backend(self, store=None, **config):
        return Backend(dict(dict(verifications_per_keyframe=10, candidate_top_k=10,
                                 shadow_mode=False), **config), store_dir=store)

    def verify(self, src, dst, **kwargs):
        a, b = (src, dst) if src['robot_id'] == 'A' else (dst, src)
        measurement = self.measurements.get((a['seq'], b['seq']))
        if measurement is None:
            return dict(ok=False, reason='fixture_not_a_match', inliers=0)
        local_relation, overrides = measurement
        camera_relation = np.linalg.inv(a['T_local_camera']) @ local_relation @ b['T_local_camera']
        if src is a:
            camera_relation = np.linalg.inv(camera_relation)
        result = dict(ok=True, reason='rgbd_verified', verification_stage='verified',
                      seed_method='depth_ransac', matcher='orb',
                      T_target_source=camera_relation, inliers=45, visual_inliers=45,
                      ransac_inliers=45, image_cells=8, target_image_cells=8,
                      overlap_source=.75, overlap_target=.74, rmse=.035,
                      translation_sigma=.065, rotation_sigma=.04)
        result.update(overrides)
        return result

    def pair(self, backend, seq, relation=None, order='AB', quality=None,
             b_pose=None, b_texture=8, b_rgb_stamp=None):
        self.measurements[seq, seq] = (self.a_from_b if relation is None else relation, quality or {})
        for robot in order:
            image = (observation(robot, seq, local_pose=self.b_origin if b_pose is None else b_pose, texture=b_texture)
                     if robot == 'B' else observation(robot, seq))
            if robot == 'B' and b_rgb_stamp is not None:
                image['rgb_stamp_ns'] = b_rgb_stamp
            backend.ingest(image)
        return backend.snapshot()

    def parked(self, backend, **kwargs):
        for seq, order in [(1, 'AB'), (2, 'BA'), (3, 'AB')]:
            snapshot = self.pair(backend, seq, order=order, **kwargs)
        return snapshot

    @staticmethod
    def rover(snap, robot='B'):
        return next(r for r in snap['robots'] if r['robot_id'] == robot)

    def test_stationary_mixed_directions_preview_only_after_three_distinct_pairs(self):
        backend = self.backend()
        for seq, order in [(1, 'AB'), (2, 'BA')]:
            snap = self.pair(backend, seq, order=order)
            self.assertEqual(self.rover(snap)['alignment_state'], 'unaligned')
        raw = {key: f['points_camera'].copy() for key, f in backend.keyframes.items()}
        snap = self.pair(backend, 3)
        self.assertEqual(self.rover(snap)['alignment_state'], 'provisional')
        self.assertFalse(self.rover(snap)['aligned'])
        self.assertIsNone(self.rover(snap)['T_world_camera'])
        b_map = next(m for m in snap['submaps'] if m['robot_id'] == 'B')
        np.testing.assert_allclose(b_map['T_preview_world_submap'], self.a_from_b @ self.b_origin, atol=1e-7)
        np.testing.assert_allclose(b_map['T_world_submap'], self.b_origin)
        self.assertEqual(self.rover(snap, 'A')['alignment_state'], 'confirmed')
        self.assertEqual(snap['stats']['accepted_loops'], 0)
        self.assertFalse(any(e['active'] for e in snap['constraints'] if e['kind'] == 'loop'))
        self.assertEqual(snap['entity_links'], [])
        self.assertFalse(snap['semantics']['robot_tf_modified'])
        for key, original in raw.items():
            np.testing.assert_array_equal(backend.keyframes[key]['points_camera'], original)
        proposal, = snap['alignment_evidence']['provisional']
        self.assertEqual(proposal['support_count'], 3)
        self.assertEqual(len(set(proposal['edge_ids'])), 3)

    def test_reused_target_is_not_three_corroborating_frame_pairs(self):
        backend = self.backend()
        backend.ingest(observation('A', 1))
        for seq in range(1, 5):
            self.measurements[1, seq] = self.a_from_b, {}
            snap = backend.ingest(observation('B', seq, local_pose=self.b_origin))
        self.assertEqual(snap['alignment_evidence']['verified'], 4)
        self.assertEqual(self.rover(snap)['alignment_state'], 'unaligned')

    def test_real_rgbd_verifier_supplies_required_preview_quality_fields(self):
        # Keep only three real frame pairs, so the test can establish the
        # minimum evidence on the ordinary ORB path without testing all NxN.
        def selected(src, dst, **params):
            if src['seq'] == dst['seq']:
                return verify_rgbd(src, dst, matcher_mode='orb', **params)
            return dict(ok=False, inliers=0, reason='fixture_not_a_match')
        self.mock_verifier.side_effect = selected
        snap = self.parked(self.backend())
        self.assertEqual(snap['alignment_evidence']['verified'], 3)
        self.assertEqual(self.rover(snap)['alignment_state'], 'provisional')

    def test_copied_rgb_stamp_is_not_fresh_evidence_even_with_new_frame_ids(self):
        backend = self.backend()
        snap = self.parked(backend, b_rgb_stamp=10_000_000_000)
        self.assertEqual(snap['alignment_evidence']['verified'], 3)
        self.assertEqual(self.rover(snap)['alignment_state'], 'unaligned')

    def test_verified_but_low_quality_or_one_sided_support_does_not_preview(self):
        for override in [dict(rmse=.064), dict(visual_inliers=15, inliers=15, ransac_inliers=15),
                         dict(overlap_target=.26), dict(target_image_cells=2)]:
            with self.subTest(override=override):
                snap = self.parked(self.backend(), quality=override)
                self.assertEqual(snap['alignment_evidence']['verified'], 3)
                self.assertEqual(self.rover(snap)['alignment_state'], 'unaligned')

    def test_an_independent_conflicting_cluster_withdraws_preview(self):
        backend = self.backend()
        snap = self.parked(backend)
        self.assertEqual(self.rover(snap)['alignment_state'], 'provisional')
        opposite = pose(-1.2, .5, -.4)
        for seq in (4, 5):
            snap = self.pair(backend, seq, relation=opposite)
        self.assertEqual(self.rover(snap)['alignment_state'], 'unaligned')
        self.assertEqual(snap['stats']['accepted_loops'], 0)
        self.assertEqual(snap['entity_links'], [])

    def test_promising_but_inconsistent_transform_chain_not_counted_as_consensus(self):
        backend = self.backend()
        for seq, delta in [(1, 0.), (2, .075), (3, .15)]:
            snap = self.pair(backend, seq, relation=self.a_from_b @ pose(delta))
        self.assertEqual(snap['alignment_evidence']['verified'], 3)
        self.assertEqual(self.rover(snap)['alignment_state'], 'unaligned')

    def test_one_contradictory_strong_moved_view_withdraws_before_a_false_promotion(self):
        backend = self.backend()
        self.assertEqual(self.rover(self.parked(backend))['alignment_state'], 'provisional')
        snap = self.pair(backend, 4, relation=pose(-1.), b_pose=pose(.8), b_texture=66)
        self.assertEqual(self.rover(snap)['alignment_state'], 'unaligned')
        self.assertEqual(snap['stats']['accepted_loops'], 0)

    def test_visual_change_plus_motion_promotes_without_operator_accept(self):
        backend = self.backend()
        self.assertEqual(self.rover(self.parked(backend))['alignment_state'], 'provisional')
        # Changing the pose alone (e.g. stationary odometric drift) is not promotion.
        self.pair(backend, 4, b_pose=pose(.55, .12, -.08))
        self.assertEqual(self.rover(backend.snapshot())['alignment_state'], 'provisional')
        snap = self.pair(backend, 5, b_pose=pose(.57, .12, -.08), b_texture=196)
        self.assertEqual(self.rover(snap)['alignment_state'], 'confirmed')
        self.assertTrue(self.rover(snap)['aligned'])
        self.assertGreater(snap['stats']['accepted_loops'], 0)
        self.assertTrue(all(not e.get('operator_decision') for e in snap['constraints'] if e['active']))
        b_map = next(m for m in snap['submaps'] if m['robot_id'] == 'B')
        self.assertIsNone(b_map.get('T_preview_world_submap'))
        np.testing.assert_allclose(b_map['T_world_submap'], self.a_from_b @ self.b_origin, atol=1e-5)

    def test_shadow_mode_and_rejection_withdraw_preview(self):
        backend = self.backend(shadow_mode=True)
        self.assertEqual(self.rover(self.parked(backend))['alignment_state'], 'unaligned')
        backend.command(dict(action='set_shadow_mode', enabled=False))
        snap = self.pair(backend, 4)
        self.assertEqual(self.rover(snap)['alignment_state'], 'provisional')
        backend.command(dict(action='set_shadow_mode', enabled=True))
        self.assertEqual(self.rover(backend.snapshot())['alignment_state'], 'unaligned')
        backend.command(dict(action='set_shadow_mode', enabled=False))
        snap = self.pair(backend, 5)
        proposal, = snap['alignment_evidence']['provisional']
        backend.command(dict(action='reject_constraint', id=proposal['edge_ids'][0]))
        self.assertEqual(self.rover(backend.snapshot())['alignment_state'], 'unaligned')

    def test_source_reset_does_not_share_previous_sessions_preview(self):
        backend = self.backend()
        self.assertEqual(self.rover(self.parked(backend))['alignment_state'], 'provisional')
        backend.ingest(observation('B', 11, session='after-reset'))
        self.assertEqual(self.rover(backend.snapshot())['alignment_state'], 'unaligned')

    def test_preview_expires_if_new_keyframes_stop_supporting_it(self):
        backend = self.backend()
        self.assertEqual(self.rover(self.parked(backend))['alignment_state'], 'provisional')
        for seq in range(4, 37):
            snap = backend.ingest(observation('B', seq, local_pose=self.b_origin))
        self.assertEqual(self.rover(snap)['alignment_state'], 'unaligned')
        self.assertEqual(snap['stats']['accepted_loops'], 0)

    def test_preview_replay_reconstructed_without_changing_recorded_decision(self):
        with tempfile.TemporaryDirectory() as folder:
            backend = self.backend(store=folder)
            original = self.parked(backend)
            state_path = Path(folder)/'state.json'
            before = state_path.read_bytes()
            replay = Backend.replay(folder)
            snap = replay.snapshot()
            self.assertEqual(self.rover(snap)['alignment_state'], 'provisional')
            self.assertEqual(snap['alignment_evidence']['provisional'], original['alignment_evidence']['provisional'])
            self.assertEqual(snap['stats']['accepted_loops'], 0)
            self.assertEqual(state_path.read_bytes(), before)

    def test_pre_feature_recording_uses_retained_quality_but_never_invents_missing_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            self.parked(self.backend(store=folder))
            state_path = Path(folder)/'state.json'
            state = json.loads(state_path.read_bytes())
            state['config'].pop('provisional_enabled', None)
            for edge in state['constraints'].values():
                edge.pop('verification_quality', None)
            # Model an older PnP case with the independently checked two-image
            # reprojection grid, instead of a separate target_image_cells field.
            for candidate in state['candidates'].values():
                verdict = candidate.get('verification', {})
                if verdict.get('ok'):
                    verdict.pop('target_image_cells', None)
                    verdict['reprojection_image_cells'] = [8, 7]
            state_path.write_text(json.dumps(state))
            self.assertEqual(self.rover(Backend.replay(folder).snapshot())['alignment_state'], 'provisional')
            state['candidates'].clear()  # Candidate diagnostics may have rolled out of an older recording.
            state_path.write_text(json.dumps(state))
            self.assertEqual(self.rover(Backend.replay(folder).snapshot())['alignment_state'], 'unaligned')


if __name__ == '__main__':
    unittest.main()
