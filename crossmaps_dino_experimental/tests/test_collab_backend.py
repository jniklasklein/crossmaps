import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from collab_backend import Backend, identifier
from collab_geometry import (depth_points, free_space_conflicts, icp, optimize_pose_graph, pose_delta, ransac_rigid,
                             rigid_fit, transform, valid_transform, verify_rgbd)


def pose(x=0., y=0., yaw=0.):
    result = np.eye(4)
    result[:3, :3] = cv2.Rodrigues(np.array([0., 0., yaw]))[0]
    result[:3, 3] = [x, y, 0]
    return result


def observation(robot='A', seq=1, session='s1', local_pose=None, texture=8, entity='sofa', feature_space='dino-v1'):
    rng = np.random.default_rng(texture)
    rgb = cv2.GaussianBlur(rng.integers(0, 256, (240, 320, 3), dtype=np.uint8), (3, 3), .4)
    ys, xs = np.mgrid[:240, :320]
    depth = (2. + .15*np.sin(xs/50.) + .1*np.cos(ys/70.)).astype(np.float32)
    intrinsic = np.array([[250., 0, 160], [0, 250, 120], [0, 0, 1]])
    points, colors = depth_points(depth, intrinsic, rgb, stride=8)
    return dict(robot_id=robot, session_id=session, seq=seq, stamp_ns=int(seq*10e9), rgb_stamp_ns=int(seq*10e9),
                T_local_camera=np.eye(4) if local_pose is None else local_pose,
                K=intrinsic, rgb=rgb, depth=depth, points_camera=points, colors=colors,
                calibration_id='camera-v1', feature_space_id=feature_space, global_descriptor=np.array([1., 0., 0]),
                surfaces=[dict(local_id='surface1', entity_id=entity, points_camera=points[100:300],
                               dino=np.array([1., 0., 0]), clip=np.array([1., 0]), quality=.9)])


class CollaborativeGeometryTests(unittest.TestCase):
    def test_visible_depth_measures_the_supplied_pose_in_both_camera_directions(self):
        from collab_geometry import _visible_depth_score
        a = observation()
        cloud, _ = depth_points(a['depth'], a['K'])
        aligned = _visible_depth_score(cloud, np.eye(4), a)
        self.assertGreater(aligned['overlap'], .9)
        self.assertLess(aligned['rmse'], .01)
        shifted = np.eye(4); shifted[2, 3] = -.8
        conflict = _visible_depth_score(cloud, shifted, a)
        self.assertLess(conflict['overlap'], .1)
        self.assertGreater(conflict['free'], .9)

    def test_rigid_ransac_rejects_outliers_and_preserves_direction(self):
        rng = np.random.default_rng(42)
        source = rng.normal(size=(80, 3))
        expected = pose(.4, -.2, .3)
        target = transform(source, expected)
        target[-20:] = rng.normal(size=(20, 3))*4
        result = ransac_rigid(source, target)
        self.assertTrue(result['ok'])
        self.assertGreaterEqual(result['inliers'], 60)
        np.testing.assert_allclose(result['T_target_source'], expected, atol=1e-5)

    def test_line_and_repeated_points_do_not_validate(self):
        source = np.column_stack((np.arange(30)*.1, np.zeros(30), np.zeros(30)))
        self.assertFalse(ransac_rigid(source, source)['ok'])
        self.assertFalse(ransac_rigid(np.zeros((20, 3)), np.zeros((20, 3)))['ok'])

    def test_rgbd_verification_and_textureless_rejection(self):
        first, second = observation('A'), observation('B')
        result = verify_rgbd(first, second)
        self.assertTrue(result['ok'], result)
        np.testing.assert_allclose(result['T_target_source'], np.eye(4), atol=1e-4)
        self.assertGreater(result['image_cells'], 3)
        second['rgb'][:] = 100
        self.assertFalse(verify_rgbd(first, second)['ok'])

    def test_registration_stage_counts_distinguish_visual_evidence_from_dense_icp(self):
        first, second = observation('A'), observation('B')
        result = verify_rgbd(first, second)
        self.assertTrue(result['ok'])
        self.assertEqual(result['verification_stage'], 'verified')
        self.assertGreaterEqual(result['visual_matches'], result['depth_correspondences'])
        self.assertGreaterEqual(result['depth_correspondences'], result['ransac_inliers'])
        self.assertEqual(result['inliers'], result['ransac_inliers'])
        self.assertEqual(result['visual_inliers'], result['ransac_inliers'])
        self.assertGreater(result['icp_inliers'], 0)
        self.assertIn('icp_rmse', result)
        self.assertIn('ransac_rmse', result)

    def test_depth_residual_failure_is_not_mislabeled_as_low_overlap(self):
        refinement = dict(T_target_source=np.eye(4), overlap_source=.8, overlap_target=.9, rmse=.08, inliers=900)
        with patch('collab_geometry.icp', return_value=refinement):
            result = verify_rgbd(observation('A'), observation('B'))
        self.assertFalse(result['ok'])
        self.assertEqual(result['reason'], 'depth_alignment_residual_too_large')
        self.assertEqual(result['verification_stage'], 'depth_residual')
        self.assertEqual(result['icp_inliers'], 900)
        self.assertEqual(result['icp_rmse'], .08)
        self.assertGreaterEqual(result['ransac_inliers'], 12)
        self.assertEqual(result['inliers'], result['ransac_inliers'])

    def test_missing_depth_reports_which_correspondences_were_removed(self):
        first, second = observation('A'), observation('B')
        second['depth'][:] = 0
        result = verify_rgbd(first, second)
        self.assertFalse(result['ok'])
        self.assertEqual(result['verification_stage'], 'depth_correspondences')
        self.assertGreaterEqual(result['visual_matches'], 12)
        self.assertEqual(result['target_matches_with_depth'], 0)
        self.assertEqual(result['depth_correspondences'], 0)

    def test_unrelated_images_not_accepted_by_similar_geometry(self):
        result = verify_rgbd(observation(texture=3), observation(texture=99))
        self.assertFalse(result['ok'])

    def test_icp_is_refinement_not_arbitrary_translation_guess(self):
        source = np.random.default_rng(3).normal(size=(200, 3))
        expected = pose(.03, .01, .02)
        result = icp(source, transform(source, expected), np.eye(4))
        np.testing.assert_allclose(result['T_target_source'], expected, atol=.015)
        self.assertGreater(result['overlap_source'], .9)

    def test_pose_graph_changes_bad_initial_pose_with_fixed_gauge(self):
        poses = dict(a=np.eye(4), b=pose(1.4, .2), c=pose(2.5, .3))
        edges = [dict(source='b', target='a', T_target_source=pose(1), kind='odometry', translation_sigma=.2, rotation_sigma=.1),
                 dict(source='c', target='b', T_target_source=pose(1), kind='odometry', translation_sigma=.2, rotation_sigma=.1),
                 dict(source='c', target='a', T_target_source=pose(2), kind='loop', translation_sigma=.1, rotation_sigma=.1)]
        fitted, stats = optimize_pose_graph(poses, edges, fixed=['a'])
        np.testing.assert_allclose(fitted['a'], np.eye(4))
        np.testing.assert_allclose(fitted['c'], pose(2), atol=.01)
        self.assertGreater(stats['iterations'], 0)

    def test_sparse_solver_handles_keyframe_chain_without_dense_dependency(self):
        poses = {str(i): pose(i*1.01) for i in range(40)}
        edges = [dict(source=str(i), target=str(i-1), T_target_source=pose(1.01), kind='odometry',
                      translation_sigma=.1, rotation_sigma=.1) for i in range(1, 40)]
        edges.append(dict(source='39', target='0', T_target_source=pose(39), kind='anchor', translation_sigma=.01, rotation_sigma=.01))
        fitted, stats = optimize_pose_graph(poses, edges, fixed=['0'], iterations=3)
        self.assertLess(abs(fitted['39'][0, 3]-39), .02)
        self.assertEqual(stats['solver'], 'block_sparse_se3_pcg')

    def test_invalid_transform_not_silently_projected(self):
        bad = np.eye(4); bad[0, 0] = 2
        with self.assertRaises(ValueError):
            valid_transform(bad)

    def test_occlusion_is_not_free_space_contradiction(self):
        intrinsic = np.array([[50., 0, 25], [0, 50., 25], [0, 0, 1]])
        depth = np.full((50, 50), 2., dtype=np.float32)
        behind = np.array([[0, 0, 3.], [.2, 0, 3.]])
        front = np.array([[0, 0, 1.], [.2, 0, 1.]])
        self.assertEqual(free_space_conflicts(behind, depth, intrinsic)[0], 0.)
        self.assertEqual(free_space_conflicts(front, depth, intrinsic)[0], 1.)


class CollaborativeBackendTests(unittest.TestCase):
    def backend(self, **config):
        return Backend(dict(verifications_per_keyframe=0, **config))

    def test_domains_are_not_identity_and_restarts_get_new_ownership(self):
        backend = self.backend()
        backend.ingest(observation('A'))
        backend.ingest(observation('B'))
        backend.ingest(observation('A', session='s2'))
        self.assertEqual(len(backend.sessions), 3)
        self.assertEqual(len(backend.keyframes), 3)
        self.assertNotEqual(identifier('entity', 'A', 's1', 'sofa'), identifier('entity', 'A', 's2', 'sofa'))
        latest_a = next(robot for robot in backend.snapshot()['robots'] if robot['robot_id']=='A')
        self.assertFalse(latest_a['aligned'])

    def test_duplicate_and_reordered_observations_never_accumulate_votes(self):
        backend = self.backend()
        obs = observation()
        backend.ingest(obs); backend.ingest(obs)
        backend.ingest(observation(seq=0))
        self.assertEqual(len(backend.keyframes), 1)
        self.assertEqual(backend.snapshot()['robots'][0]['frames'], 1)

    def test_guard_rejection_preserves_existing_geometry(self):
        backend = self.backend()
        backend.ingest(observation())
        bad = observation(seq=2, local_pose=pose(20))
        bad['integration_allowed'] = False
        backend.ingest(bad)
        self.assertEqual(len(backend.keyframes), 1)
        self.assertEqual(backend.snapshot()['robots'][0]['rejected_poses'], 1)

    def test_reference_and_unaligned_maps_remain_separate(self):
        backend = self.backend()
        backend.ingest(observation('A'))
        backend.ingest(observation('B', local_pose=pose(3)))
        snapshot = backend.snapshot()
        self.assertTrue(snapshot['robots'][0]['aligned'])
        self.assertFalse(snapshot['robots'][1]['aligned'])
        self.assertEqual(snapshot['stats']['accepted_loops'], 0)
        json.dumps(snapshot, allow_nan=False)

    def test_anchor_convention_clear_and_source_evidence_unchanged(self):
        backend = self.backend()
        backend.ingest(observation('A'))
        backend.ingest(observation('B', local_pose=pose(3)))
        before = backend.keyframes[next(reversed(backend.keyframes))]['points_camera'].copy()
        backend.command(dict(action='anchor', robot_id='B', reference_robot_id='A', T_reference_local=pose(-3)))
        snapshot = backend.snapshot()
        self.assertTrue(snapshot['robots'][1]['aligned'])
        np.testing.assert_allclose(snapshot['submaps'][1]['T_world_submap'], np.eye(4), atol=1e-6)
        self.assertGreaterEqual(len(snapshot['entity_links']), 1)
        link_id = snapshot['entity_links'][0]['id']
        original_ids = {entity['id'] for entity in snapshot['global_entities']}
        backend.command(dict(action='accept_entity_link', id=link_id))
        self.assertTrue(any(len(entity['members']) == 2 for entity in backend.snapshot()['global_entities']))
        merged_id = backend.snapshot()['global_entities'][0]['id']
        self.assertIn(merged_id, original_ids)
        self.assertEqual(len(backend.snapshot()['entity_aliases']), 1)
        backend.command(dict(action='unlink_entities', id=link_id))
        self.assertFalse(any(len(entity['members']) == 2 for entity in backend.snapshot()['global_entities']))
        self.assertEqual({entity['id'] for entity in backend.snapshot()['global_entities']}, original_ids)
        backend.command(dict(action='clear_anchor', robot_id='B'))
        self.assertFalse(backend.snapshot()['robots'][1]['aligned'])
        np.testing.assert_array_equal(backend.keyframes[next(reversed(backend.keyframes))]['points_camera'], before)

    def test_pending_anchor_applies_only_when_both_sources_exist(self):
        backend = self.backend()
        result = backend.command(dict(action='anchor', robot_id='B', reference_robot_id='A', T_reference_local=np.eye(4)))
        self.assertTrue(result['pending'])
        backend.ingest(observation('A')); backend.ingest(observation('B'))
        self.assertTrue(backend.snapshot()['robots'][1]['aligned'])

    def test_pose_revision_suspends_contradicted_entity_link_without_deleting_history(self):
        backend = self.backend()
        backend.ingest(observation('A')); backend.ingest(observation('B'))
        backend.command(dict(action='anchor', robot_id='B', reference_robot_id='A', T_reference_local=np.eye(4)))
        link_id = backend.snapshot()['entity_links'][0]['id']
        backend.command(dict(action='accept_entity_link', id=link_id))
        self.assertEqual(len(backend.snapshot()['global_entities']), 1)
        backend.command(dict(action='anchor', robot_id='B', reference_robot_id='A', T_reference_local=pose(3)))
        self.assertEqual(len(backend.snapshot()['global_entities']), 2)
        link = backend.snapshot()['entity_links'][0]
        self.assertTrue(link['active'])  # explicit decision retained, current grouping suspended
        self.assertFalse(link['compatible_current'])

    def test_external_yaw_drift_is_not_independent_visual_evidence(self):
        backend = self.backend()
        first = backend._validated(observation('A'))
        drifted = backend._validated(observation('A', seq=2, local_pose=pose(yaw=.4)))
        self.assertFalse(backend._distinct_views(first, drifted))

    def test_incompatible_automatic_cluster_cannot_replace_accepted_alignment(self):
        backend = self.backend()
        backend.ingest(observation('A')); backend.ingest(observation('B'))
        backend.command(dict(action='anchor', robot_id='B', reference_robot_id='A', T_reference_local=np.eye(4)))
        maps = list(backend.submaps)
        edge = dict(source=maps[1], target=maps[0], source_session='b', target_session='a', T_target_source=pose(3), T_targetlocal_sourcelocal=pose(3))
        self.assertFalse(backend._alignment_compatible(edge))

    def test_feature_incompatible_models_do_not_propose_matches(self):
        backend = self.backend()
        backend.ingest(observation('A', feature_space='one'))
        backend.ingest(observation('B', feature_space='two'))
        self.assertEqual(backend.snapshot()['candidates'], [])

    def test_real_rgbd_candidate_is_shadow_not_fused(self):
        backend = Backend(dict(verifications_per_keyframe=1))
        backend.ingest(observation('A')); backend.ingest(observation('B'))
        snapshot = backend.snapshot()
        verified = [candidate for candidate in snapshot['candidates'] if candidate['state']=='verified']
        self.assertEqual(len(verified), 1)
        self.assertFalse(snapshot['robots'][1]['aligned'])
        backend.command(dict(action='accept_constraint', id=verified[0]['constraint_id']))
        self.assertTrue(backend.snapshot()['robots'][1]['aligned'])
        backend.command(dict(action='reject_constraint', id=verified[0]['constraint_id']))
        self.assertFalse(backend.snapshot()['robots'][1]['aligned'])

    def test_alignment_summary_includes_retained_but_unchecked_candidates(self):
        backend = Backend(dict(verifications_per_keyframe=1, candidate_top_k=3))
        backend.ingest(observation('A', seq=1))
        backend.ingest(observation('A', seq=2))
        backend.ingest(observation('B'))
        summary = backend.snapshot()['alignment_evidence']
        self.assertEqual(summary, dict(retrieved=2, checked=1, verified=1, active=0, best_rejection=None))

    def test_recent_target_gets_a_bounded_proposal_not_a_free_verification(self):
        backend = Backend(dict(candidate_top_k=1, verifications_per_keyframe=2))
        with patch.object(backend, '_scores', side_effect=lambda source, target: {'clip': 1/target['seq']}), \
                patch('collab_backend.verify_rgbd', return_value=dict(ok=False, reason='test_rejected', inliers=0)) as check:
            for seq in (1, 2, 3):
                backend.ingest(dict(observation('A', seq=seq), received_monotonic=float(10*seq)))
            backend.ingest(dict(observation('B'), received_monotonic=31.))
        self.assertEqual([call.args[1]['seq'] for call in check.call_args_list], [1, 3])
        self.assertTrue(all(not edge['active'] for edge in backend.constraints.values() if edge['kind'] == 'loop'))

    def test_partial_masks_can_propose_same_entity_without_segment_merging(self):
        backend = self.backend()
        first, second = observation('A'), observation('B')
        surface = second['surfaces'][0]
        support = surface['points_camera']
        second['surfaces'] = [dict(surface, local_id='part1', points_camera=support[:100]),
                              dict(surface, local_id='part2', points_camera=support[100:])]
        backend.ingest(first); backend.ingest(second)
        backend.command(dict(action='anchor', robot_id='B', reference_robot_id='A', T_reference_local=np.eye(4)))
        self.assertEqual(len(backend.snapshot()['entity_links']), 1)
        latest = next(reversed(backend.keyframes.values()))
        self.assertEqual(len(latest['surfaces']), 2)

    def test_corroboration_requires_distinct_viewpoints_not_new_timestamps(self):
        backend = Backend(dict(verifications_per_keyframe=1))
        backend.ingest(observation('A')); backend.ingest(observation('B'))
        backend.ingest(observation('B', seq=2))
        self.assertTrue(all(edge['state']=='verified' for edge in backend.constraints.values() if edge['kind']=='loop'))

    def test_moving_only_target_rover_corroborates_in_either_match_order(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                backend = self.backend()
                backend.ingest(observation('A'))
                backend.ingest(observation('B'))
                backend.ingest(observation('B', 2, local_pose=pose(.25), texture=3))
                a, b, moved_b = backend.keyframes
                def edge(name, source, target):
                    return dict(id=name, kind='loop', state='verified', corroborations=1,
                                source_frame=source, target_frame=target,
                                source_session=backend.keyframes[source]['source_id'],
                                target_session=backend.keyframes[target]['source_id'],
                                T_targetlocal_sourcelocal=np.eye(4))
                prior = edge('prior', b, a) if reverse else edge('prior', a, b)
                current = edge('current', a, moved_b)
                backend.constraints.update(prior=prior, current=current)
                backend._corroborate(current)
                self.assertEqual(current['state'], 'corroborated')

    def test_capacity_is_explicit_preserves_existing_evidence(self):
        backend = self.backend(max_keyframes=1)
        backend.ingest(observation())
        backend.ingest(observation(seq=2))
        self.assertEqual(len(backend.keyframes), 1)
        self.assertIn('capacity', backend.snapshot()['stats']['capacity_reason'])

    def test_keyframe_loop_corrects_internal_submap_geometry_and_rejection_rebuilds(self):
        backend = Backend(dict(verifications_per_keyframe=1))
        backend.ingest(observation('A'))
        backend.ingest(observation('B'))
        backend.command(dict(action='anchor', robot_id='B', reference_robot_id='A', T_reference_local=np.eye(4)))
        backend.ingest(observation('A', seq=2, local_pose=pose(.3)))
        latest = backend.keyframes[next(reversed(backend.keyframes))]
        submap_id = latest['submap_id']
        original_cloud = backend.submaps[submap_id]['points'].copy()
        raw_camera_points = latest['points_camera'].copy()
        candidate = next(candidate for candidate in backend.candidates.values() if candidate['source']==latest['id'] and candidate['state']=='verified')
        backend.command(dict(action='accept_constraint', id=candidate['constraint_id']))
        self.assertEqual(latest['submap_id'], submap_id)
        self.assertLess(abs(latest['T_submap_camera'][0, 3]), .1)
        self.assertGreater(backend.graph_status['corrected_internal_keyframes'], 0)
        self.assertTrue(backend.snapshot()['semantics']['submap_internal_pose_correction'])
        np.testing.assert_array_equal(latest['points_camera'], raw_camera_points)
        backend.command(dict(action='reject_constraint', id=candidate['constraint_id']))
        np.testing.assert_allclose(latest['T_submap_camera'], pose(.3), atol=1e-5)
        np.testing.assert_allclose(backend.submaps[submap_id]['points'], original_cloud, atol=1e-5)

    def test_invalid_bounds_and_string_booleans_fail_before_storage_creation(self):
        for config in (dict(max_keyframes=0), dict(max_keyframes=100000), dict(submap_seconds=float('nan')),
                       dict(shadow_mode='false'), dict(candidate_top_k=-2), dict(min_overlap=2), dict(unknown_option=4)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                Backend(config)
        with self.assertRaises(ValueError):
            self.backend().command(dict(action='set_shadow_mode', enabled='false'))

    def test_safe_persistence_restore_and_replay_reversible_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = Backend(dict(verifications_per_keyframe=0), directory)
            backend.ingest(observation('A')); backend.ingest(observation('B'))
            backend.command(dict(action='anchor', robot_id='B', reference_robot_id='A', T_reference_local=np.eye(4)))
            restored = Backend(store_dir=directory)
            self.assertEqual(len(restored.keyframes), 2)
            self.assertTrue(restored.snapshot()['robots'][1]['aligned'])
            recorded_bytes = {path.name: path.read_bytes() for path in Path(directory).iterdir()}
            replay = Backend.replay(directory)
            replay.command(dict(action='clear_anchor', robot_id='B'))
            replay.command(dict(action='save'))
            self.assertFalse(replay.snapshot()['robots'][1]['aligned'])
            self.assertEqual({path.name: path.read_bytes() for path in Path(directory).iterdir()}, recorded_bytes)
            unchanged = Backend(store_dir=directory)
            self.assertTrue(unchanged.snapshot()['robots'][1]['aligned'])
            self.assertFalse(any(path.suffix=='.pkl' for path in Path(directory).iterdir()))


if __name__ == '__main__':
    unittest.main()
