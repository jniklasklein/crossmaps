"""Pure motion/rate checks: these tests never publish robot commands."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import motion_check
from smoke_check import rate_report, rate_sample


class MotionCheckTests(unittest.TestCase):
    def test_motion_requires_explicit_execute_before_ros_import_or_probe(self):
        with patch.object(motion_check, 'ros_python') as ros, redirect_stdout(StringIO()):
            self.assertEqual(motion_check.main(['--robot', 'rover_a']), 0)
        ros.assert_not_called()

    def test_invalid_robot_domains_or_unbounded_timeout_are_rejected(self):
        for args in (['--robot', 'physical'], ['--domain', '0'], ['--timeout', 'nan'], ['--timeout', 'inf'], ['--timeout', '181']):
            with self.subTest(args=args), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                motion_check.main(args)

    def test_motion_requires_odometry_tf_image_change_and_new_captures(self):
        before = dict(stamp=1, odom_xy=[0., 0.], capture_xy=[0., 0.], odom_yaw=0., capture_yaw=0.)
        forward = dict(before, stamp=2, odom_xy=[.075, 0.], capture_xy=[.074, 0.])
        turn = dict(forward, stamp=3, odom_yaw=.225, capture_yaw=.22, image_change=10.)
        self.assertTrue(motion_check.evaluate_motion(before, forward, turn)['ok'])
        self.assertFalse(motion_check.evaluate_motion(before, forward, dict(turn, image_change=0.))['ok'])
        self.assertFalse(motion_check.evaluate_motion(before, forward, dict(turn, stamp=2))['ok'])
        self.assertFalse(motion_check.evaluate_motion(before, forward, dict(turn, capture_yaw=0.))['ok'])
        self.assertAlmostEqual(motion_check.angle_delta(math.radians(179), math.radians(-179)), math.radians(2))
        self.assertAlmostEqual(motion_check.yaw(SimpleNamespace(x=0., y=0., z=math.sin(.1), w=math.cos(.1))), .2)

    def test_rates_count_source_progress_not_repeated_callbacks(self):
        rate = {}
        rate_sample(rate, 1_000_000_000, 100.)
        rate_sample(rate, 1_000_000_000, 101.)
        rate_sample(rate, 900_000_000, 102.)
        rate_sample(rate, 1_500_000_000, 104.)
        report = rate_report(rate)
        self.assertEqual(report['advancing_frames'], 2)
        self.assertAlmostEqual(report['hz_wall'], .25)
        self.assertAlmostEqual(report['hz_simulation'], 2.)
        self.assertIsNone(rate_report({})['hz_wall'])


if __name__ == '__main__':
    unittest.main()
