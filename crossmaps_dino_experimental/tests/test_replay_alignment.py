import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from replay_alignment import (canonical_transform, consistency, controls,
                              frame_path, load_frame, run, select_pairs)


def pose(x):
    m = np.eye(4)
    m[0, 3] = x
    return m


class AlignmentReplayTests(unittest.TestCase):
    def test_recovers_canonical_local_transform_across_poses_and_reversed_order(self):
        a = dict(robot_id='A', session_id='one', T_local_camera=pose(.3))
        b = dict(robot_id='B', session_id='one', T_local_camera=pose(.1))
        group, ab = canonical_transform(a, b, pose(.7))
        group2, ba = canonical_transform(b, a, pose(-.7))
        self.assertEqual(group, group2)
        np.testing.assert_allclose(ab, pose(.5))
        np.testing.assert_allclose(ba, ab)
        summary = consistency({group: [ab, ba, pose(1.1)]}, .18, 7.)[0]
        self.assertEqual(summary['transform_comparisons'], 3)
        self.assertEqual(summary['agreeing_comparisons'], 1)
        self.assertAlmostEqual(summary['max_translation_delta_m'], .6)

    def test_recorded_and_controls_keep_distinct_evidence_without_reusing_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            keys = ['a'*20, 'b'*20]
            k = np.array([[10., 0, 5], [0., 10., 5], [0., 0., 1.]])
            for i, key in enumerate(keys):
                meta = dict(id=key, robot_id=key[0], session_id='session',
                            stamp_ns=1000*i)
                np.savez(frame_path(p, key), metadata=json.dumps(meta),
                         rgb=np.ones((16, 16, 3), dtype=np.uint8),
                         depth=np.ones((16, 16), dtype=np.float32), K=k,
                         T_local_camera=pose(0), points_camera=np.array([[0.,0.,1.]]))
            state = dict(revision=123, sessions={k: dict(keyframe_ids=[k]) for k in keys},
                         config=dict(min_visual_inliers=12, min_overlap=.25, max_registration_rmse=.065,
                                     transform_agreement_translation=.18, transform_agreement_rotation_deg=7),
                         candidates={'c': dict(source=keys[0], target=keys[1], source_robot='a',
                                               target_robot='b', verification={'ok': False}),
                                     'unchecked': dict(source=keys[1], target=keys[0])})
            pairs = [tuple(keys)]
            self.assertEqual(select_pairs(p, state, 'checked', 3, []), pairs)
            self.assertEqual(select_pairs(p, state, 'nearest', 3, []), pairs)
            self.assertEqual(select_pairs(p, state, 'checked', 3, [':'.join(keys)]), pairs)
            with self.assertRaises(ValueError):
                select_pairs(p, state, 'checked', 3, ['../bad:{}'.format(keys[1])])
            frame = load_frame(p, keys[0])
            frame['_feature_cache'] = object()
            cases = dict(controls(frame))
            self.assertNotIn('_feature_cache', cases['unrelated_texture'])
            np.testing.assert_allclose(cases['incompatible_depth_scale']['depth'], 1.7)
            np.testing.assert_allclose(cases['incompatible_depth_scale']['points_camera'][:, 2], 1.7)
            np.testing.assert_allclose(frame['depth'], 1.)
            def fake(a, b, matcher_mode, **settings):
                self.assertEqual(settings['min_inliers'], 12)
                ok = matcher_mode == 'auto' and not b['id'].startswith('control-')
                return dict(ok=ok, reason='verified' if ok else 'rejected', T_target_source=pose(0))
            with patch('replay_alignment.verify_rgbd', side_effect=fake):
                report = run(p, state, pairs, 1)
            self.assertEqual(report['control_pairs_tested'], 2)
            self.assertEqual(report['control_false_verifications'], {'orb': 0, 'auto': 0})
            self.assertEqual(report['consistency']['auto'][0]['verified_pairs'], 1)
            self.assertEqual(report['consistency']['orb'], [])
            self.assertEqual(report['outcomes']['auto']['verified'], 1)
            json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
