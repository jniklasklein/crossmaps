"""No ROS graph, goals, or simulator motion is needed for these contract tests."""
import json
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

import navigation


class NavigationConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nav = yaml.safe_load((navigation.HERE / "nav2_params.yaml").read_text())
        cls.slam = yaml.safe_load((navigation.HERE / "slam_params.yaml").read_text())["slam_toolbox"]["ros__parameters"]
        cls.rviz = yaml.safe_load((navigation.HERE / "navigation.rviz").read_text())

    def test_all_nodes_use_sim_time_and_only_laser_slam_owns_map(self):
        for name in navigation.LIFECYCLE_NODES:
            self.assertIs(self.nav[name]["ros__parameters"]["use_sim_time"], True)
        self.assertIs(self.slam["use_sim_time"], True)
        self.assertEqual(self.slam["mode"], "mapping")
        self.assertEqual(self.slam["scan_topic"], "/scan")
        self.assertEqual(self.slam["odom_frame"], "odom")
        self.assertEqual(self.slam["base_frame"], "base_footprint")
        self.assertNotIn("amcl", self.nav)
        self.assertNotIn("map_server", self.nav)
        self.assertNotIn("map_file_name", self.slam)

    def test_nonholonomic_envelope_and_watchdog_compatible_rate(self):
        controller = self.nav["controller_server"]["ros__parameters"]
        planner = controller["FollowPath"]
        velocity = self.nav["velocity_smoother"]["ros__parameters"]
        self.assertEqual(planner["max_vel_y"], 0.0)
        self.assertEqual(planner["vy_samples"], 1)
        self.assertLessEqual(planner["max_vel_x"], 0.35)
        self.assertLessEqual(planner["max_vel_theta"], 0.7)
        self.assertGreaterEqual(controller["controller_frequency"], 10.0)
        self.assertGreaterEqual(velocity["smoothing_frequency"], 10.0)
        self.assertLess(velocity["velocity_timeout"], 0.5)
        self.assertIn("ObstacleFootprint", planner["critics"])
        self.assertIs(self.nav["planner_server"]["ros__parameters"]["GridBased"]["allow_unknown"], False)

    def test_consistent_physical_footprint_and_scan(self):
        footprints = []
        for name in ("local_costmap", "global_costmap"):
            parameters = self.nav[name][name]["ros__parameters"]
            footprints.append(json.loads(parameters["footprint"]))
            self.assertEqual(parameters["robot_base_frame"], "base_footprint")
            self.assertEqual(parameters["obstacle_layer"]["scan"]["topic"], "/scan")
            self.assertEqual(parameters["obstacle_layer"]["scan"]["expected_update_rate"], 0.5)
            self.assertEqual(parameters["obstacle_layer"]["scan"]["observation_persistence"], 0.0)
            self.assertIs(parameters["use_sim_time"], True)
        self.assertEqual(*footprints)
        self.assertGreaterEqual(max(x for x, _ in footprints[0]) - min(x for x, _ in footprints[0]), 0.251 + 0.04)
        self.assertGreaterEqual(max(y for _, y in footprints[0]) - min(y for _, y in footprints[0]), 0.2304 + 0.039)

    def test_native_goal_tool_and_readable_real_sensor_displays(self):
        manager = self.rviz["Visualization Manager"]
        self.assertEqual(manager["Global Options"]["Fixed Frame"], "map")
        self.assertIn("nav2_rviz_plugins/GoalTool", [tool["Class"] for tool in manager["Tools"]])
        topics = {display.get("Topic", {}).get("Value") for display in manager["Displays"]}
        self.assertTrue({"/scan", "/map", "/odom", "/plan", "/local_plan"} <= topics)
        self.assertNotIn("/ground_truth", topics)


class SupervisorTests(unittest.TestCase):
    def test_lifecycle_pause_resume_reset_and_startup_preserve_stack(self):
        policy = navigation.LifecyclePolicy()
        active = {name: 3 for name in navigation.LIFECYCLE_NODES}
        paused = {name: 2 for name in navigation.LIFECYCLE_NODES}
        reset = {name: 1 for name in navigation.LIFECYCLE_NODES}
        mixed = dict(active, controller_server=2)
        self.assertEqual(policy.evaluate(active, 1.0), ("ready", True))
        self.assertEqual(policy.evaluate(mixed, 2.0), ("lifecycle_transition", False))
        self.assertEqual(policy.evaluate(paused, 3.0), ("paused", False))
        self.assertEqual(policy.evaluate(paused, 100.0), ("paused", False))
        self.assertEqual(policy.evaluate(mixed, 101.0), ("lifecycle_transition", False))
        self.assertEqual(policy.evaluate(active, 102.0), ("ready", True))
        self.assertEqual(policy.evaluate(dict(paused, controller_server=11), 103.0), ("lifecycle_transition", False))
        self.assertEqual(policy.evaluate(reset, 104.0), ("reset", False))
        self.assertEqual(policy.evaluate(reset, 200.0), ("reset", False))
        self.assertEqual(policy.evaluate(dict(reset, controller_server=10), 201.0), ("lifecycle_transition", False))
        self.assertEqual(policy.evaluate(active, 202.0), ("ready", True))

    def test_lifecycle_mixed_state_cannot_hang_forever_and_terminal_failures_stop(self):
        active = {name: 3 for name in navigation.LIFECYCLE_NODES}
        mixed = dict(active, controller_server=2)
        policy = navigation.LifecyclePolicy()
        self.assertEqual(policy.evaluate(mixed, 1.0), ("lifecycle_transition", False))
        self.assertEqual(policy.evaluate(mixed, 5.9), ("lifecycle_transition", False))
        with self.assertRaisesRegex(RuntimeError, "transition exceeded"):
            policy.evaluate(mixed, 6.1)
        for bad_state in (0, 4, 12, 15, 99):
            with self.assertRaisesRegex(RuntimeError, "unavailable or failed"):
                navigation.LifecyclePolicy().evaluate(dict(active, controller_server=bad_state), 1.0)
        with self.assertRaisesRegex(RuntimeError, "unavailable or failed"):
            navigation.LifecyclePolicy().evaluate({}, 1.0)

    def test_domain_allowlist_and_environment_scrubbing(self):
        self.assertEqual(navigation.domain_for("rover_a"), 80)
        self.assertEqual(navigation.domain_for("rover_b"), 81)
        for invalid in ("192.168.88.244", "0", "rover_c", None):
            with self.assertRaises(ValueError):
                navigation.domain_for(invalid)
        with patch.dict(os.environ, ROS_DOMAIN_ID="0", CYCLONEDDS_URI="remote", ROS_DISCOVERY_SERVER="remote",
                        ROS_STATIC_PEERS="192.168.88.244", AMENT_PREFIX_PATH="untrusted"):
            environment = navigation.isolated_environment("/env/bin/python3", Path("/logs"), "rover_b")
        self.assertEqual(environment["ROS_DOMAIN_ID"], "81")
        self.assertEqual(environment["ROS_LOCALHOST_ONLY"], "1")
        self.assertEqual(environment["ROS_AUTOMATIC_DISCOVERY_RANGE"], "LOCALHOST")
        self.assertEqual(environment["AMENT_PREFIX_PATH"], "/env")
        for key in ("CYCLONEDDS_URI", "ROS_DISCOVERY_SERVER", "ROS_STATIC_PEERS"):
            self.assertNotIn(key, environment)

    def test_commands_do_not_send_goals_and_viewer_does_not_duplicate_stack(self):
        plan = navigation.commands("/env/bin/python3", "rover_a", no_rviz=True)
        self.assertEqual([name for name, _ in plan], ["slam", "nav2"])
        joined = " ".join(arg for _, command in plan for arg in command)
        self.assertNotIn("send_goal", joined)
        self.assertNotIn("/cmd_vel", joined)
        self.assertIn("use_sim_time:=true", joined)
        self.assertIn("use_respawn:=False", joined)
        view = navigation.commands("/env/bin/python3", "rover_b", rviz_only=True)
        self.assertEqual(len(view), 1)
        self.assertEqual(view[0][0], "rviz")
        self.assertIn("__node:=rviz_rover_b", view[0][1])

    def test_freshness_requires_advancing_all_inputs_and_reset_is_latched(self):
        health = navigation.InputHealth()
        self.assertFalse(health.fresh(1))
        for key in ("clock", "odom", "scan"):
            health.update(key, 100, 1.0)
            health.update(key, 200, 1.1)
        self.assertTrue(health.fresh(2.0))
        for key in ("clock", "odom", "scan"):
            health.update(key, 200, 9.0)
        self.assertFalse(health.fresh(9.0))
        health.update("clock", 1, 10)
        for key in ("clock", "odom", "scan"):
            health.update(key, 400, 10.1)
        self.assertFalse(health.fresh(10.2))
        self.assertTrue(health.clock_reset)

    def test_ready_file_is_pid_scoped_and_replaced_on_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ready.json"
            navigation.publish_health(path, "rover_a", True, "ready")
            health = json.loads(path.read_text())
            self.assertIs(health["ready"], True)
            self.assertEqual(health["pid"], os.getpid())
            self.assertEqual(health["domain"], 80)
            navigation.publish_health(path, "rover_a", False, "stopped")
            self.assertIs(json.loads(path.read_text())["ready"], False)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_rviz_windows_identify_selected_rover_without_changing_topics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = navigation.write_rviz_config(directory, "rover_b")
            self.assertEqual(path.name, "rover_b-navigation.rviz")
            config = yaml.safe_load(path.read_text())
            self.assertIn("ROVER_B", config["Panels"][0]["Name"])
            self.assertIn("domain 81", config["Panels"][0]["Name"])
            command = navigation.commands("/env/bin/python3", "rover_b", rviz_only=True, rviz_config=path)[0][1]
            self.assertEqual(command[command.index("-d") + 1], str(path))
            bundled = navigation.commands("/env/bin/python3", "rover_b", rviz_only=True,
                                          rviz_executable="/runtime/Rover B.app/Contents/MacOS/RViz")[0][1]
            self.assertEqual(bundled[0], "/runtime/Rover B.app/Contents/MacOS/RViz")
            self.assertIn("__node:=rviz_rover_b", bundled)

    def test_status_write_failure_cannot_skip_controller_cleanup(self):
        events = []
        probe, stream, lock = Mock(), Mock(), Mock()
        probe.close.side_effect = lambda **kwargs: events.append("zero_and_close_probe")
        stream.close.side_effect = lambda: events.append("close_log")
        lock.close.side_effect = lambda: events.append("unlock")
        with patch.object(navigation, "publish_health", side_effect=OSError("disk full")), \
                patch.object(navigation, "stop_owned", side_effect=lambda *a, **k: events.append("stop_controllers")), \
                patch.object(navigation.signal, "signal", side_effect=lambda *a: events.append("restore_handler")):
            with self.assertRaisesRegex(OSError, "disk full"):
                navigation.cleanup_navigation([], [stream], probe, lock, {2: "handler"},
                                              Path("unwritable"), "rover_a", "stopped", True)
        self.assertEqual(events, ["stop_controllers", "zero_and_close_probe", "close_log", "restore_handler", "unlock"])
        probe.close.assert_called_once_with(stop_robot=True)

    def test_teardown_failure_still_releases_every_other_owned_resource(self):
        probe, stream, lock = Mock(), Mock(), Mock()
        probe.close.side_effect = RuntimeError("probe already closed")
        with patch.object(navigation, "publish_health"), \
                patch.object(navigation, "stop_owned", side_effect=OSError("group race")) as stop, \
                patch.object(navigation.signal, "signal") as handler:
            with self.assertRaisesRegex(RuntimeError, "probe already closed"):
                navigation.cleanup_navigation([], [stream], probe, lock, {2: "handler"},
                                              None, "rover_b", "failed", True)
        stop.assert_called_once()
        stream.close.assert_called_once()
        handler.assert_called_once_with(2, "handler")
        lock.close.assert_called_once()

    def test_cached_active_lifecycle_replies_expire_and_hung_requests_retry(self):
        messages, services = ModuleType("lifecycle_msgs"), ModuleType("lifecycle_msgs.srv")
        services.GetState = SimpleNamespace(Request=lambda: object())
        messages.srv = services
        probe = navigation.NavigationProbe.__new__(navigation.NavigationProbe)
        probe.clients = {name: Mock() for name in navigation.LIFECYCLE_NODES}
        for client in probe.clients.values():
            client.service_is_ready.return_value = False
        probe.states = {name: 3 for name in navigation.LIFECYCLE_NODES}
        probe.state_replies = {name: 1.0 for name in navigation.LIFECYCLE_NODES}
        pending = Mock()
        pending.done.return_value = False
        probe.pending = {"controller_server": (pending, 1.0)}
        probe.last_query = 10.0
        with patch.dict("sys.modules", {"lifecycle_msgs": messages, "lifecycle_msgs.srv": services}), \
                patch.object(navigation.time, "monotonic", return_value=10.0):
            self.assertFalse(probe.lifecycle_ready())
            self.assertTrue(all(state == 0 for state in probe.states.values()))
            pending.cancel.assert_called_once()
            probe.clients["controller_server"].remove_pending_request.assert_called_once_with(pending)
            self.assertEqual(probe.pending, {})
            # A later responsive ACTIVE reply may restore readiness; stale cache
            # alone never can. Startup remains retryable without permanent hangs.
            for name in navigation.LIFECYCLE_NODES:
                future = Mock()
                future.done.return_value = True
                future.result.return_value = SimpleNamespace(current_state=SimpleNamespace(id=3))
                probe.pending[name] = (future, 9.5)
            self.assertTrue(probe.lifecycle_ready())
            self.assertTrue(all(stamp == 10.0 for stamp in probe.state_replies.values()))

    def test_cached_tf_must_cover_current_simulation_time(self):
        rclpy, ros_time = ModuleType("rclpy"), ModuleType("rclpy.time")
        ros_time.Time = lambda: object()
        rclpy.time = ros_time
        probe = navigation.NavigationProbe.__new__(navigation.NavigationProbe)
        probe.map_received = True
        probe.buffer = Mock()
        probe.buffer.lookup_transform.return_value = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=10, nanosec=0)))
        probe.health = navigation.InputHealth()
        with patch.dict("sys.modules", {"rclpy": rclpy, "rclpy.time": ros_time}):
            probe.health.stamps["clock"] = 10_100_000_000
            self.assertTrue(probe.geometry_ready())
            probe.health.stamps["clock"] = 12_000_000_000
            self.assertFalse(probe.geometry_ready())
            probe.health.stamps["clock"] = 9_000_000_000
            self.assertFalse(probe.geometry_ready())
            probe.buffer.lookup_transform.side_effect = RuntimeError("no map transform")
            self.assertFalse(probe.geometry_ready())


if __name__ == "__main__":
    unittest.main()
