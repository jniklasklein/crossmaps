import math
import unittest

try:
    from .validate import continuity, pose_summary, synthetic_replay
except ImportError:
    from validate import continuity, pose_summary, synthetic_replay


class ValidationTests(unittest.TestCase):
    def test_wraparound_and_reverse(self):
        radians = math.radians
        rows = [(0., 1, 0., 0., radians(175)), (.1, 2, 0., 0., radians(-175)),
                (.2, 3, math.cos(radians(175)) * -.5, math.sin(radians(175)) * -.5, radians(-165))]
        s = pose_summary(rows)
        self.assertAlmostEqual(s['yaw_change_deg'], 20)
        self.assertAlmostEqual(s['forward_displacement_m'], -.5)
        self.assertAlmostEqual(s['lateral_displacement_m'], 0)

    def test_live_subscriber_freshness(self):
        rows = [(i*.1, i) for i in range(51)]
        self.assertTrue(continuity(rows, 5)['fresh'])
        self.assertFalse(continuity(rows[:20], 5)['fresh'])  # dies before interval end
        self.assertFalse(continuity(rows[20:], 5)['fresh'])  # appears too late
        self.assertFalse(continuity([(t,0) for t,_ in rows],5)['fresh'])  # frozen stamps
        self.assertFalse(continuity(rows[:10]+rows[31:],5)['fresh'])  # mid-trial stall

    def test_black_box_replay(self):
        result = synthetic_replay()
        self.assertTrue(result['pass_test'], result)


if __name__ == '__main__':
    unittest.main()
