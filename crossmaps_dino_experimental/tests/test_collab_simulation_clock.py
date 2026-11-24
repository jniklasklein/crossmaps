"""ROS-clock policy without ROS dependencies; real /clock coverage is opt-in."""
import unittest

from collab_config import validate_config
from collab_transport import SimulationClockState, pair_timing


class SimulationClockTests(unittest.TestCase):
    def setUp(self):
        self.wall = 0.
        self.clock = SimulationClockState(clock=lambda: self.wall)

    def test_zero_waits_then_advancement_runs_and_pause_is_reversible(self):
        self.assertEqual(self.clock.observe(0), 'waiting_for_clock')
        self.wall = 5.
        self.assertEqual(self.clock.observe(0), 'waiting_for_clock')
        self.assertEqual(self.clock.observe(1_000_000), 'running')
        self.wall = 9.
        self.assertEqual(self.clock.observe(1_000_000), 'paused_or_stalled')
        self.assertEqual(self.clock.observe(2_000_000), 'running')

    def test_any_backward_jump_latches_even_after_clock_overtakes_old_time(self):
        self.clock.observe(10_000_000)
        self.assertEqual(self.clock.observe(9_999_999), 'reset_restart_required')
        self.assertEqual(self.clock.observe(0), 'reset_restart_required')
        self.assertEqual(self.clock.observe(100_000_000), 'reset_restart_required')

    def test_jump_callback_can_invalidate_between_polls(self):
        self.clock.observe(1_000_000)
        self.clock.invalidate()
        self.assertEqual(self.clock.observe(2_000_000), 'reset_restart_required')

    def test_simulation_age_does_not_accumulate_wall_wait_but_tf_deadline_does(self):
        timing = pair_timing(10_000_000_000, 10_100_000_000, 50., 54., 2., 6.,
                             current_clock_ns=10_200_000_000)
        self.assertEqual(timing['timing_status'], 'within_latency_budget')
        self.assertAlmostEqual(timing['selected_frame_age_s'], .2)
        self.assertAlmostEqual(timing['pending_wait_s'], 4.)
        expired = pair_timing(10_000_000_000, 10_100_000_000, 50., 57., 2., 6.,
                             current_clock_ns=10_100_000_000)
        self.assertEqual(expired['timing_status'], 'timestamped_tf_wait_expired')

    def test_explicit_opt_in_and_no_hardware_relay(self):
        self.assertFalse(validate_config({})['input']['use_sim_time'])
        self.assertTrue(validate_config({'input': {'use_sim_time': True}})['input']['use_sim_time'])
        for value in ('true', 1, None):
            with self.assertRaisesRegex(ValueError, 'use_sim_time must be boolean'):
                validate_config({'input': {'use_sim_time': value}})
        with self.assertRaisesRegex(ValueError, 'camera_relay.mode=off'):
            validate_config({'input': {'use_sim_time': True}, 'robots': [
                {'id': 'sim_a', 'domain': 80, 'camera_relay': {'mode': 'auto', 'ssh_host': 'rover'}}]})


if __name__ == '__main__':
    unittest.main()
