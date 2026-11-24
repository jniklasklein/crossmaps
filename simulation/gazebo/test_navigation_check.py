"""Pure checker tests: no ROS node or navigation command is created."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import math
import unittest
from unittest.mock import patch

import navigation_check as check


class NavigationCheckTests(unittest.TestCase):
    def test_fixed_array_uuid_equality_is_scalar_and_exact(self):
        import numpy as np
        left = np.arange(16, dtype=np.uint8)
        self.assertTrue(check.same_goal_id(left, left.copy()))
        self.assertTrue(check.same_goal_id(left, list(left)))
        right = left.copy()
        right[15] = 42
        self.assertFalse(check.same_goal_id(left, right))

    def test_execution_is_explicit_before_ros_probe(self):
        with patch.object(check, "ros_python") as ros, patch.object(check, "validate_simulation_config") as validate:
            with redirect_stdout(StringIO()):
                self.assertEqual(check.main(["--robot", "rover_a"]), 0)
                self.assertEqual(check.main(["--robot", "rover_b", "--cancel"]), 0)
            ros.assert_not_called()
            validate.assert_not_called()

    def test_invalid_scope_and_bounds(self):
        for options in (["--domain", "0"], ["--robot", "physical"], ["--timeout", "nan"],
                        ["--timeout", "181"], ["--timeout", "14"], ["--forward", "inf"],
                        ["--forward", "0"], ["--lateral", "0.6"], ["--forward", "1", "--lateral", ".5"]):
            with self.subTest(options=options), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                check.main(options)

    def test_goal_is_relative_to_heading(self):
        x, y, heading = check.relative_goal([1., 2., math.pi / 2.], .5, .25)
        self.assertAlmostEqual(x, .75)
        self.assertAlmostEqual(y, 2.5)
        self.assertAlmostEqual(heading, math.pi / 2.)
        _, angle = check.pose_errors([0., 0., math.pi - .1], [0., 0., -math.pi + .1])
        self.assertAlmostEqual(angle, .2)

    def report(self, **changes):
        report = dict(error=None, goal_accepted=True, planned_path_count=1, max_path_poses=5,
                      scan_advances_during_test=3, scan_fresh=True, last_command_zero=True,
                      command_fresh=True, nonzero_command_seen=True, cancel_test=False,
                      action_status=4, cancel_acknowledged=False, start_pose=[0., 0., 0.],
                      goal_pose=[.5, .25, 0.], final_pose=[.49, .24, .02])
        report.update(changes)
        return report

    def test_success_needs_actual_motion_path_stop_and_fresh_scan(self):
        self.assertTrue(check.evaluate_navigation(self.report())["ok"])
        for missing in (dict(planned_path_count=0), dict(max_path_poses=0), dict(scan_fresh=False),
                        dict(scan_advances_during_test=0), dict(last_command_zero=False),
                        dict(command_fresh=False), dict(nonzero_command_seen=False),
                        dict(final_pose=[0., 0., 0.]), dict(action_status=6), dict(error="timeout")):
            with self.subTest(missing=missing):
                self.assertFalse(check.evaluate_navigation(self.report(**missing))["ok"])

    def test_cancel_requires_owned_cancel_ack_and_terminal_state(self):
        report = self.report(cancel_test=True, cancel_acknowledged=True, action_status=5,
                             final_pose=[.05, .01, 0.])
        self.assertTrue(check.evaluate_navigation(report)["ok"])
        for change in (dict(cancel_acknowledged=False), dict(action_status=4), dict(planned_path_count=0)):
            self.assertFalse(check.evaluate_navigation(dict(report, **change))["ok"])

    def test_control_graph_excludes_only_own_observer(self):
        check.validate_control_graph(["checker", "crossmaps_gazebo_rover_a"],
                                     ["velocity_smoother", "behavior_server"],
                                     ["crossmaps_gazebo_rover_a"], "rover_a", "checker")
        for receivers, publishers, sources in (
            (["crossmaps_gazebo_rover_b"], ["velocity_smoother"], ["crossmaps_gazebo_rover_a"]),
            (["crossmaps_gazebo_rover_a", "physical_driver"], ["velocity_smoother"], ["crossmaps_gazebo_rover_a"]),
            (["crossmaps_gazebo_rover_a"], ["crossmaps_sim_keyboard"], ["crossmaps_gazebo_rover_a"]),
            (["crossmaps_gazebo_rover_a"], ["velocity_smoother"], ["unexpected_scan"]),
        ):
            with self.assertRaises(RuntimeError):
                check.validate_control_graph(receivers, publishers, sources, "rover_a", "checker")


if __name__ == "__main__":
    unittest.main()
