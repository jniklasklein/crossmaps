"""Configuration and bounded scheduling regressions; no ROS or model imports."""
import copy
import math
import unittest

from collab_config import DEFAULT_CONFIG, validate_config
from collab_runtime import CycleTiming, FairLatestScheduler


class ConfigurationTests(unittest.TestCase):
    def test_defaults_isolate_domains_and_do_not_mutate_input(self):
        supplied = {"robots": [{"id": "alpha", "domain": 4}, {"id": "beta", "domain": 7}]}
        before = copy.deepcopy(supplied)
        result = validate_config(supplied)
        self.assertEqual(supplied, before)
        self.assertEqual(result["central_domain"], 42)
        self.assertEqual(result["robots"][0]["depth_registration"], "auto")
        result["robots"][0]["parameters"]["stride"] = 99
        self.assertEqual(DEFAULT_CONFIG["robots"][0], {"id": "rover_a", "domain": 0})

    def test_invalid_root_and_section_types(self):
        for supplied in (None, [], "config", 2):
            with self.subTest(supplied=supplied), self.assertRaises(ValueError):
                validate_config(supplied)
        for section in ("dashboard", "input", "perception", "pose_guard", "backend"):
            for value in (None, [], "enabled", 12):
                with self.subTest(section=section, value=value), self.assertRaises(ValueError):
                    validate_config({section: value})
        with self.assertRaises(ValueError):
            validate_config({"perception": {"parameters": []}})

    def test_unknown_root_section_and_robot_fields_fail_loudly(self):
        invalid = [{"central_domian": 42}, {"dashboard": {"bind": "0.0.0.0"}},
                   {"input": {"max_hzz": 5}}, {"pose_guard": {"enable": False}},
                   {"perception": {"models": {}}},
                   {"robots": [{"id": "alpha", "domain": 4, "rgb_topc": "/image"}]}]
        for supplied in invalid:
            with self.subTest(supplied=supplied), self.assertRaisesRegex(ValueError, "unknown|known"):
                validate_config(supplied)

    def test_domain_collisions_boolean_fractional_and_out_of_range_rejected(self):
        invalid = [{"central_domain": 0}, {"central_domain": True}, {"central_domain": 42.5},
                   {"central_domain": -1}, {"central_domain": 102},
                   {"robots": [{"id": "alpha", "domain": 3}, {"id": "beta", "domain": 3}]},
                   {"robots": [{"id": "alpha", "domain": 3}, {"id": "alpha", "domain": 4}]},
                   {"robots": [{"id": "alpha", "domain": 42}]}]
        invalid += [{"robots": [{"id": "alpha", "domain": val}]} for val in (True, 3.2, "3", -1, 102)]
        for supplied in invalid:
            with self.subTest(supplied=supplied), self.assertRaises(ValueError):
                validate_config(supplied)

    def test_robot_shapes_names_and_sensor_configuration(self):
        invalid = [{"robots": []}, {"robots": "two"}, {"robots": [None]},
                   {"robots": [{"id": "Alpha", "domain": 4}]},
                   {"robots": [{"id": "../alpha", "domain": 4}]},
                   {"robots": [{"id": "alpha", "domain": 4, "rgb_compressed": "true"}]},
                   {"robots": [{"id": "alpha", "domain": 4, "depth_scale": 0}]},
                   {"robots": [{"id": "alpha", "domain": 4, "depth_registration": "resize"}]},
                   {"robots": [{"id": "alpha", "domain": 4, "parameters": []}]},
                   {"robots": [{"id": "alpha", "domain": 4, "rgb_topic": ""}]}]
        for supplied in invalid:
            with self.subTest(supplied=supplied), self.assertRaises(ValueError):
                validate_config(supplied)

    def test_input_numeric_types_limits_and_order(self):
        invalid = [("qos_depth", 3.5), ("max_pixels", 2048.5), ("max_hz", True),
                   ("max_hz", "2"), ("max_hz", 0), ("max_age_s", math.inf),
                   ("max_age_s", math.nan), ("tf_timeout_s", -.1), ("sync_slop_s", 2.)]
        for name, value in invalid:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                validate_config({"input": {name: value}})
        with self.assertRaisesRegex(ValueError, "min_depth_m"):
            validate_config({"input": {"min_depth_m": 2., "max_depth_m": 1.}})

    def test_pose_guard_bounds_and_boolean_enabled(self):
        for value in (0, 1, "false", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_config({"pose_guard": {"enabled": value}})
        for name, lower, upper in (("max_translation_error_m", .01, 10.),
                                   ("max_rotation_error_deg", .1, 90.),
                                   ("max_speed_m_s", .01, 20.),
                                   ("max_yaw_rate_deg_s", 1., 720.)):
            for value in (lower / 2., upper + 1., math.nan, math.inf, True):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    validate_config({"pose_guard": {name: value}})
            self.assertEqual(validate_config({"pose_guard": {name: lower}})["pose_guard"][name], lower)
            self.assertEqual(validate_config({"pose_guard": {name: upper}})["pose_guard"][name], upper)

    def test_network_exposure_and_pose_lock_are_not_silent_options(self):
        for host in ("0.0.0.0", "192.168.1.1", "example.org"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                validate_config({"dashboard": {"host": host}})
        for port in (True, 8766.5, 80, 70000):
            with self.subTest(port=port), self.assertRaises(ValueError):
                validate_config({"dashboard": {"port": port}})
        for supplied in ({"perception": {"parameters": {"stationary_pose_lock": True}}},
                         {"robots": [{"id": "alpha", "domain": 4,
                                      "parameters": {"stationary_pose_lock": True}}]}):
            with self.assertRaisesRegex(ValueError, "stationary pose lock"):
                validate_config(supplied)


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.scheduler = FairLatestScheduler(["a", "b"], max_age_s=2., clock=lambda: self.now)

    def packet(self, robot="a", seq=1, session="first", received=None):
        return dict(robot_id=robot, seq=seq, session_id=session,
                    received_monotonic=self.now if received is None else received)

    def test_fair_round_robin_despite_one_busy_source(self):
        for robot in ("a", "b"):
            self.scheduler.offer(self.packet(robot))
        self.assertEqual(self.scheduler.take()["robot_id"], "a")
        for seq in range(2, 15):
            self.scheduler.offer(self.packet("a", seq))
        self.assertEqual(self.scheduler.take()["robot_id"], "b")
        self.assertEqual(self.scheduler.take()["seq"], 14)
        self.assertIsNone(self.scheduler.take())
        self.assertEqual(self.scheduler.snapshot()["a"]["taken"], 2)

    def test_one_latest_slot_replaces_without_backlog(self):
        for seq in range(1, 101):
            self.assertTrue(self.scheduler.offer(self.packet(seq=seq)))
        state = self.scheduler.snapshot()["a"]
        self.assertEqual(state["offered"], 100)
        self.assertEqual(state["replaced"], 99)
        self.assertTrue(state["pending"])
        self.assertEqual(len(self.scheduler.pending), 1)
        self.assertEqual(self.scheduler.take()["seq"], 100)

    def test_duplicate_and_out_of_order_sequences_are_not_new_evidence(self):
        self.scheduler.offer(self.packet(seq=7))
        for seq in (7, 6, 0):
            self.assertFalse(self.scheduler.offer(self.packet(seq=seq)))
        self.assertEqual(self.scheduler.take()["seq"], 7)
        self.assertFalse(self.scheduler.offer(self.packet(seq=7)))
        self.assertEqual(self.scheduler.snapshot()["a"]["duplicate"], 4)

    def test_restart_accepts_new_epoch_and_rejects_retired_epoch_forever(self):
        self.scheduler.offer(self.packet(seq=99, session="old"))
        self.assertTrue(self.scheduler.offer(self.packet(seq=0, session="new")))
        self.assertFalse(self.scheduler.offer(self.packet(seq=1000, session="old")))
        self.assertEqual(self.scheduler.take()["session_id"], "new")
        self.assertTrue(self.scheduler.offer(self.packet(seq=1, session="third")))
        self.assertFalse(self.scheduler.offer(self.packet(seq=9999, session="new")))
        self.assertEqual(self.scheduler.take()["session_id"], "third")

    def test_stale_and_nonfinite_or_future_input_is_discarded_without_starving_peer(self):
        for received in (97., 101., math.nan, math.inf):
            scheduler = FairLatestScheduler(["a", "b"], max_age_s=2., clock=lambda: self.now)
            scheduler.offer(self.packet("a", received=received))
            scheduler.offer(self.packet("b"))
            self.assertEqual(scheduler.take()["robot_id"], "b")
            self.assertEqual(scheduler.snapshot()["a"]["expired"], 1)
            self.assertIsNone(scheduler.take())

    def test_staleness_is_checked_when_taken_not_when_offered(self):
        self.scheduler.offer(self.packet())
        self.now += 3.
        self.assertIsNone(self.scheduler.take())
        self.assertFalse(self.scheduler.snapshot()["a"]["pending"])

    def test_unknown_source_and_invalid_identity_rejected(self):
        for packet in (self.packet("unknown"), self.packet(session=""), self.packet(seq=-1)):
            with self.subTest(packet=packet), self.assertRaises(ValueError):
                self.scheduler.offer(packet)

    def test_snapshot_is_independent_of_internal_counters(self):
        snapshot = self.scheduler.snapshot()
        snapshot["a"]["offered"] = 999
        self.assertEqual(self.scheduler.snapshot()["a"]["offered"], 0)


class TimingTests(unittest.TestCase):
    def test_bounded_live_cycle_average_and_percentile(self):
        timing = CycleTiming(window=3)
        self.assertIsNone(timing.snapshot()["average_ms"])
        for value in (100., 1., 2., 3.):
            timing.add(value, value / 2., value + .5)
        self.assertEqual(timing.snapshot(), dict(count=3, cycle_ms=3000., average_ms=2000.,
                         p95_ms=2900., inference_ms=1500., input_to_map_ms=3500.))


if __name__ == "__main__":
    unittest.main()
