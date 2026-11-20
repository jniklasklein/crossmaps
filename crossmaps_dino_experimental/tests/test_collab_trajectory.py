"""The displayed path is accepted local evidence, not a global pose history."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from collab_backend import Backend


def pose(x=0., yaw=0.):
    mat = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    mat[:2, :2] = [[c, -s], [s, c]]
    mat[0, 3] = x
    return mat


MOUNT = np.array([[0., -1., 0., -.05], [0., 0., -1., .10], [1., 0., 0., -.065], [0., 0., 0., 1.]])


def obs(seq, x=0., yaw=0., session='s', robot='A', sec=None, mount=None, allowed=True):
    return dict(robot_id=robot, session_id=session, seq=seq, stamp_ns=int((sec or seq)*1e9),
                T_local_camera=pose(x, yaw), T_camera_base=mount, integration_allowed=allowed,
                rgb=np.zeros((20, 32, 3), np.uint8), depth=np.ones((20, 32), np.float32),
                K=np.array([[24., 0, 16.], [0, 24., 10.], [0, 0, 1.]]))


def trail(state, robot='A'):
    return next(r['trajectory'] for r in state['robots'] if r['robot_id'] == robot)


class RoverTrajectoryTests(unittest.TestCase):
    def test_nonkeyframe_motion_and_spin_updates_pose_at_body_origin_with_correct_heading(self):
        b = Backend(dict(keyframe_seconds=120., keyframe_translation=1., keyframe_rotation_deg=180.))
        start = b.ingest(obs(1, 3., mount=MOUNT))
        sid = start['submaps'][0]['id']
        frame = b.ingest(obs(2, 3.1, .5, mount=MOUNT))
        end = b.ingest(obs(3, 3.1, .8, mount=MOUNT))
        t = trail(end)
        self.assertEqual(end['stats']['keyframes'], 1)
        self.assertEqual((t['latest']['submap_id'], t['latest']['seq'], t['latest']['frame']), (sid, 3, 'base'))
        in_submap = np.linalg.inv(pose(3.)) @ pose(3.1, .8) @ MOUNT
        np.testing.assert_allclose(t['latest']['T_submap_pose'], in_submap)
        np.testing.assert_allclose(t['segments'][-1]['points'][-1], in_submap[:3, 3], atol=.0001)
        self.assertFalse(np.allclose(trail(frame)['latest']['T_submap_pose'], t['latest']['T_submap_pose']))
        self.assertEqual(len(t['segments']), 1)

    def test_rejection_missing_tf_clock_gap_and_mount_change_each_break_path(self):
        b = Backend()
        prior = trail(b.ingest(obs(1, .1)))
        rejected = trail(b.ingest(obs(2, 9., allowed=False)))
        self.assertEqual(prior['latest'], rejected['latest'])
        self.assertTrue(rejected['tracking_lost'])
        self.assertFalse(b.mark_trajectory_gap('A', 's'))
        # Reordered packets cannot undo tracking loss.
        self.assertTrue(trail(b.ingest(obs(1)))['tracking_lost'])
        t = trail(b.ingest(obs(3, .2)))
        self.assertFalse(t['tracking_lost'])
        self.assertEqual(t['latest']['frame'], 'camera_optical')
        self.assertEqual(len(t['segments']), 2)
        self.assertFalse(b.mark_trajectory_gap('A', 'another-session'))
        self.assertTrue(b.mark_trajectory_gap('A', 's'))
        b.ingest(obs(4, .25))
        b.ingest(obs(5, .3, sec=25))
        t = trail(b.ingest(obs(6, .3, sec=26, mount=MOUNT)))
        self.assertEqual(len(t['segments']), 5)
        self.assertTrue(all(s['break_before'] for s in t['segments']))
        # New session starts a new marker, never inherits previous geometry.
        t = trail(b.ingest(obs(1, 40., session='next')))
        self.assertEqual(len(t['segments']), 1)
        self.assertTrue(t['segments'][0]['break_before'])

    def test_invalid_optional_base_frame_does_not_block_map_evidence(self):
        b = Backend()
        t = trail(b.ingest(obs(1, mount=np.zeros((4, 4)))))
        self.assertEqual(t['latest']['frame'], 'camera_optical')
        self.assertEqual(len(b.keyframes), 1)

    def test_submap_transition_is_continuous_but_uses_own_alignment(self):
        b = Backend(dict(submap_translation=.2))
        b.ingest(obs(1, 5.))
        state = b.ingest(obs(2, 5.3))
        first, second = state['submaps']
        t = trail(state)
        self.assertEqual([s['submap_id'] for s in t['segments']], [first['id'], second['id']])
        self.assertEqual([s['break_before'] for s in t['segments']], [True, False])
        self.assertAlmostEqual(t['segments'][1]['points'][0][0], 0.)
        self.assertAlmostEqual(second['T_local_submap'][0][3], 5.3)

    def test_after_optimization_every_vertex_follows_its_own_keyframe(self):
        b = Backend()
        b.ingest(obs(1, 0.))
        b.ingest(obs(2, .2))
        state = b.ingest(obs(3, .26))
        self.assertEqual((state['stats']['keyframes'], state['stats']['submaps']), (2, 1))
        # Same submap origin; internal keyframe 2 has a new optimized offset.
        second = list(b.keyframes.values())[1]
        second['T_submap_camera'] = pose(.6)
        t = trail(b.snapshot())
        np.testing.assert_allclose(t['segments'][0]['points'], [[0, 0, 0], [.6, 0, 0], [.66, 0, 0]])
        np.testing.assert_allclose(t['latest']['T_submap_pose'], pose(.66))

    def test_bound_history_without_accumulating_duplicate_standing_points(self):
        b = Backend(dict(keyframe_translation=10., keyframe_seconds=3600.))
        with patch('collab_backend.TRAIL_POINTS', 6):
            for seq in range(1, 12):
                state = b.ingest(obs(seq, seq*.1))
            for seq in range(12, 20):
                state = b.ingest(obs(seq, 1.101))
        t = trail(state)
        pts = t['segments'][0]['points']
        self.assertLessEqual(len(pts), 6)
        self.assertGreaterEqual(len(pts), 4)
        self.assertGreater(pts[0][0], 0)
        self.assertEqual(t['latest']['seq'], 19)
        self.assertTrue(t['segments'][0]['break_before'])

    def test_replay_preserves_latest_nonkeyframe_and_legacy_never_bridges_unknown_gaps(self):
        with tempfile.TemporaryDirectory() as store:
            b = Backend(store_dir=store)
            b.ingest(obs(1, mount=MOUNT))
            b.ingest(obs(3, .4, mount=MOUNT))
            b.ingest(obs(4, .42, mount=MOUNT))
            b.ingest(obs(5, .8, allowed=False))
            b.command({'action': 'save'})
            old = trail(b.snapshot())
            self.assertEqual(trail(Backend.replay(store).snapshot()), old)
            state_file = Path(store, 'state.json')
            data = json.loads(state_file.read_text())
            for src in data['sessions'].values():
                del src['trail']
            state_file.write_text(json.dumps(data))
            legacy = trail(Backend.replay(store).snapshot())
            self.assertEqual((legacy['latest']['seq'], legacy['latest']['frame']), (3, 'base'))
            self.assertTrue(legacy['tracking_lost'])
            self.assertEqual([s['break_before'] for s in legacy['segments']], [True, True])


if __name__ == '__main__':
    unittest.main()
