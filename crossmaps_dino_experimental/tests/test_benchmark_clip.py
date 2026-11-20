import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmark_clip import agreement, recorded_crops


class ClipBenchmarkTests(unittest.TestCase):
    def test_crops_are_repeatable_rgb_from_the_recorded_projection(self):
        with tempfile.TemporaryDirectory() as work:
            run = Path(work)
            (run / "evidence").mkdir()
            bgr = np.full((45, 50, 3), [10, 20, 30], np.uint8)
            # x=[3,39], y=[2,36] at z=1 with identity intrinsics.
            points = np.array([[3, 2, 1], [39, 36, 1], [5, 3, 1],
                               [500, 500, 1], [300, 300, -1]])
            for i in range(8):
                meta = dict(id=str(i), robot_id="rover", session_id="s", surfaces=[{"local_id": "4"}])
                np.savez(run / "evidence" / f"frame-{i}.npz", rgb=bgr, K=np.eye(3),
                         metadata=json.dumps(meta), s0_points_camera=points)
            crops, keys, digest = recorded_crops(run)
            self.assertEqual(len(crops), 8)
            self.assertEqual(crops[0].size, (37, 35))
            self.assertEqual(crops[0].getpixel((0, 0)), (30, 20, 10))
            self.assertEqual((keys, digest), recorded_crops(run)[1:])
            with self.assertRaisesRegex(ValueError, "at least 8"):
                recorded_crops(run, frames=1)

    def test_agreement_requires_another_frame_in_the_same_session_and_a_distractor(self):
        keys = [("r", "s", "f1", "a"), ("r", "s", "f1", "b"),
                ("r", "s", "f2", "a"), ("r", "s", "f2", "b"),
                ("r", "new_session", "f3", "a"), ("r", "s", "f3", "unseen")]
        features = np.eye(3)[[0, 1, 0, 1, 0, 2]]
        self.assertEqual(agreement(keys, features),
                         {"top1_matches": 4, "eligible_queries": 4, "fraction": 1.0})
        self.assertIsNone(agreement(keys[:2], features[:2])["fraction"])
        self.assertIsNone(agreement([keys[0], keys[2]], features[[0, 2]])["fraction"])


if __name__ == "__main__":
    unittest.main()
