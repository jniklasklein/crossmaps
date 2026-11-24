"""Opt-in, localhost-only ROS adapter integration; never uses rover domains.

Run with CROSSMAPS_ROS_SMOKE=1 ROS_LOCALHOST_ONLY=1. Uses diagnostic domains
86 and 87, real ROS messages and adapter child processes, but no models,
physical cameras or robot control topics. All owned processes are cleaned up.
"""
import multiprocessing
import os
import queue
import time
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from collab_config import validate_config
from collab_transport import run_adapter, stamp_ns


@unittest.skipUnless(os.environ.get("CROSSMAPS_ROS_SMOKE") == "1",
                     "opt-in ROS localhost integration test")
class AdapterRosSmokeTests(unittest.TestCase):
    def setUp(self):
        self.assertEqual(os.environ.get("ROS_LOCALHOST_ONLY"), "1",
                         "ROS smoke tests must explicitly restrict DDS to localhost")
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, CompressedImage
        from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
        self.rclpy = rclpy
        self.channels = []
        self.config = validate_config({"central_domain": 88, "robots": [
            {"id": "smoke_a", "domain": 86, "base_frame": "base_link"}, {"id": "smoke_b", "domain": 87}],
            "input": {"max_hz": 10., "max_age_s": 2., "tf_timeout_s": .3,
                      "use_sim_time": self._testMethodName.startswith('test_simulation_')}})
        self.mp = multiprocessing.get_context("spawn")
        self.addCleanup(self._cleanup)
        for robot in self.config["robots"]:
            source = SimpleNamespace(robot=robot, process=None, node=None, context=None, executor=None,
                                     output=self.mp.Queue(maxsize=2), health=self.mp.Queue(maxsize=2),
                                     stop=self.mp.Event())
            self.channels.append(source)
            source.process = self.mp.Process(target=run_adapter,
                args=(robot, self.config["input"], source.output, source.health, source.stop))
            source.process.start()
            source.context = Context()
            rclpy.init(args=[], context=source.context, domain_id=robot["domain"])
            source.node = Node("crossmaps_synthetic_" + robot["id"], context=source.context)
            source.executor = SingleThreadedExecutor(context=source.context)
            source.executor.add_node(source.node)
            source.rgb = source.node.create_publisher(CompressedImage, robot["rgb_topic"], qos_profile_sensor_data)
            source.depth = source.node.create_publisher(CompressedImage, robot["depth_topic"], qos_profile_sensor_data)
            source.info = source.node.create_publisher(CameraInfo, robot["camera_info_topic"], qos_profile_sensor_data)
            source.depth_info = source.node.create_publisher(CameraInfo, robot["depth_camera_info_topic"], qos_profile_sensor_data)
            source.tf = TransformBroadcaster(source.node)
            source.tf_static = StaticTransformBroadcaster(source.node)
        self._wait(lambda: all(c.rgb.get_subscription_count() and c.depth.get_subscription_count()
                              and c.info.get_subscription_count() and c.tf.pub_tf.get_subscription_count()
                              and c.tf_static.pub_tf.get_subscription_count() for c in self.channels),
                   timeout=12., message="isolated adapter subscriber discovery")
        for source in self.channels:
            info = self._info(source.node.get_clock().now().to_msg())
            source.info.publish(info)
            source.depth_info.publish(info)
            static = self._tf(info.header.stamp, "base_link", "camera", (.1, 0., .2))
            source.tf_static.sendTransform(static)
        self._spin_for(.25)

    def _cleanup(self):
        for source in getattr(self, "channels", []):
            source.stop.set()
        for source in getattr(self, "channels", []):
            if source.process is not None and source.process.pid is not None:
                source.process.join(timeout=5.)
                if source.process.is_alive():
                    source.process.terminate()
                    source.process.join(timeout=3.)
                if source.process.is_alive():
                    source.process.kill()
                    source.process.join(timeout=2.)
            if source.executor is not None:
                source.executor.shutdown()
            if source.node is not None:
                source.node.destroy_node()
            if source.context is not None:
                source.context.try_shutdown()
            for channel in (source.output, source.health):
                channel.cancel_join_thread()
                channel.close()

    def _spin_for(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            for source in self.channels:
                source.executor.spin_once(timeout_sec=.005)

    def _health(self):
        result = {}
        for source in self.channels:
            while True:
                try:
                    health = source.health.get_nowait()
                    # Avoid printing JPEG bytes on assertion failures.
                    result[source.robot["id"]] = {k: v for k, v in health.items() if k not in ("preview_jpeg", "depth_preview_jpeg")}
                except queue.Empty:
                    break
        return result

    def _wait(self, predicate, timeout=6., message="condition"):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            for source in self.channels:
                if source.process is not None and source.process.exitcode is not None:
                    self.fail(f"Adapter {source.robot['id']} exited: {source.process.exitcode}")
            self._spin_for(.025)
        self.fail(f"Timed out waiting for {message}; health={self._health()}")

    @staticmethod
    def _stamp(value):
        from builtin_interfaces.msg import Time
        return Time(sec=value // 1_000_000_000, nanosec=value % 1_000_000_000)

    @staticmethod
    def _info(stamp):
        from sensor_msgs.msg import CameraInfo
        msg = CameraInfo()
        msg.header.frame_id = "camera"
        msg.header.stamp = stamp
        msg.width, msg.height = 48, 32
        msg.k = [45., 0., 24., 0., 45., 16., 0., 0., 1.]
        msg.p = [45., 0., 24., 0., 0., 45., 16., 0., 0., 0., 1., 0.]
        return msg

    @staticmethod
    def _tf(stamp, parent, child, translation):
        from geometry_msgs.msg import TransformStamped
        msg = TransformStamped()
        msg.header.stamp, msg.header.frame_id, msg.child_frame_id = stamp, parent, child
        msg.transform.rotation.w = 1.
        (msg.transform.translation.x, msg.transform.translation.y,
         msg.transform.translation.z) = map(float, translation)
        return msg

    def _pair(self, source, color, depth_m, pose, delayed_tf=False, omit_tf=False, now_ns=None):
        from sensor_msgs.msg import CompressedImage
        now_ns = source.node.get_clock().now().nanoseconds if now_ns is None else now_ns
        image_stamp = self._stamp(now_ns - 100_000_000)
        latest_stamp = self._stamp(now_ns)
        rgb = CompressedImage()
        rgb.header.stamp, rgb.header.frame_id = image_stamp, "camera"
        rgb.format = "jpeg"
        ok, data = cv2.imencode(".jpg", np.full((32, 48, 3), color, np.uint8),
                                [cv2.IMWRITE_JPEG_QUALITY, 100])
        self.assertTrue(ok)
        rgb.data = data.tobytes()
        depth = CompressedImage()
        depth.header.stamp, depth.header.frame_id = image_stamp, "camera"
        depth.format = "16UC1; compressedDepth png"
        ok, data = cv2.imencode(".png", np.full((32, 48), round(depth_m * 1000), np.uint16))
        self.assertTrue(ok)
        depth.data = bytes(12) + data.tobytes()
        at_capture = self._tf(image_stamp, "odom", "base_link", pose)
        # A latest-TF fallback would give a wildly incorrect pose and fail.
        later = self._tf(latest_stamp, "odom", "base_link", np.asarray(pose) + [20., 0., 0.])
        if not delayed_tf and not omit_tf:
            source.tf.sendTransform([at_capture, later])
            self._spin_for(.05)
        source.rgb.publish(rgb)
        source.depth.publish(depth)
        if delayed_tf:
            # One image pair only, so success requires retained-pair TF retry.
            self._spin_for(.25)
            with self.assertRaises(queue.Empty):
                source.output.get_nowait()
            source.tf.sendTransform([at_capture, later])
        return stamp_ns(image_stamp)

    def _packet(self, source):
        received = []
        def available():
            try:
                received.append(source.output.get_nowait())
                return True
            except queue.Empty:
                return False
        self._wait(available, message="observation from " + source.robot["id"])
        return received[0]

    def test_two_domains_keep_colors_geometry_timestamps_and_sessions_separate(self):
        first, second = self.channels
        a_stamp = self._pair(first, (10, 30, 230), 1.4, (1., 2., .3))
        a = self._packet(first)
        self._spin_for(.15)
        with self.assertRaises(queue.Empty):
            second.output.get_nowait()
        b_stamp = self._pair(second, (220, 50, 10), 2.2, (9., -4., .7))
        b = self._packet(second)
        for packet, source, color, depth_m, pose, stamp in (
                (a, first, (10, 30, 230), 1.4, (1.1, 2., .5), a_stamp),
                (b, second, (220, 50, 10), 2.2, (9.1, -4., .9), b_stamp)):
            self.assertEqual(packet["robot_id"], source.robot["id"])
            self.assertEqual(packet["stamp_ns"], stamp)
            self.assertEqual(packet["rgb_stamp_ns"], stamp)
            self.assertEqual(packet["seq"], 1)
            self.assertEqual(packet["depth_registration"], "same_optical_frame")
            self.assertEqual(packet["rgb"].dtype, np.uint8)
            self.assertEqual(packet["depth"].dtype, np.float32)
            np.testing.assert_allclose(packet["rgb"][0, 0], color, atol=3)
            np.testing.assert_allclose(packet["depth"], depth_m, atol=1e-5)
            np.testing.assert_allclose(packet["T_local_camera"][:3, 3], pose, atol=1e-6)
            if source is first:
                np.testing.assert_allclose(packet["T_camera_base"][:3, 3], [-.1, 0., -.2], atol=1e-6)
                np.testing.assert_allclose((packet["T_local_camera"] @ packet["T_camera_base"])[:3, 3],
                                           [1., 2., .3], atol=1e-6)
            else:
                # An unknown base frame must not suppress valid RGB-D or claim a body pose.
                self.assertNotIn("T_camera_base", packet)
            np.testing.assert_allclose(packet["K"], [[45., 0., 24.], [0., 45., 16.], [0., 0., 1.]])
        self.assertNotEqual(a["session_id"], b["session_id"])
        self._pair(first, (10, 30, 230), 1.4, (1.05, 2., .3))
        next_a = self._packet(first)
        self.assertEqual(next_a["session_id"], a["session_id"])
        self.assertEqual(next_a["seq"], 2)
        with self.assertRaises(queue.Empty):
            second.output.get_nowait()

    def test_image_pair_waits_for_delayed_timestamped_tf(self):
        source = self.channels[0]
        stamp = self._pair(source, (45, 100, 200), 1.8, (3., 1., .4), delayed_tf=True)
        packet = self._packet(source)
        self.assertEqual(packet["stamp_ns"], stamp)
        np.testing.assert_allclose(packet["T_local_camera"][:3, 3], [3.1, 1., .6], atol=1e-6)

    def test_sustained_tf_gap_offers_model_preview_then_same_stamp_mapping_completion(self):
        source = self.channels[0]
        stamp = self._pair(source, (45, 100, 200), 1.8, (3., 1., .4), omit_tf=True)
        preview = self._packet(source)
        self.assertTrue(preview['preview_only'])
        self.assertFalse(preview['integration_allowed'])
        self.assertIsNone(preview['T_local_camera'])
        self.assertEqual(preview['stamp_ns'], stamp)
        self.assertEqual(preview['depth_registration'], 'same_optical_frame')
        self.assertGreaterEqual(preview['transport_timing']['pending_wait_s'], .35)
        self._spin_for(2.1)  # TF is later than the complete camera freshness budget.
        capture = self._tf(self._stamp(stamp), 'odom', 'base_link', (3., 1., .4))
        later = self._tf(source.node.get_clock().now().to_msg(), 'odom', 'base_link', (30., 1., .4))
        source.tf.sendTransform([capture, later])
        mapped = self._packet(source)
        self.assertFalse(mapped['preview_only'])
        self.assertTrue(mapped['integration_allowed'])
        self.assertEqual(mapped['stamp_ns'], preview['stamp_ns'])
        self.assertGreater(mapped['pose_ready_monotonic']-mapped['received_monotonic'], 2.)
        self.assertGreater(mapped['seq'], preview['seq'])
        np.testing.assert_allclose(mapped['T_local_camera'][:3, 3], [3.1, 1., .6], atol=1e-6)

    def test_raw_rgb_and_depth_diagnostics_update_without_pose_tf(self):
        source = self.channels[0]
        first_stamp = self._pair(source, (45, 100, 200), 1., (0., 0., 0.), omit_tf=True)
        reports = []
        def new_health(stamp):
            def available():
                try:
                    report = source.health.get_nowait()
                except queue.Empty:
                    return False
                if report.get('depth_preview_stamp_ns') == stamp and report.get('preview_stamp_ns') == stamp:
                    reports.append(report)
                    return True
                return False
            return available
        self._wait(new_health(first_stamp), message='acquisition RGB/depth previews without TF')
        second_stamp = self._pair(source, (180, 50, 90), 3., (0., 0., 0.), omit_tf=True)
        self._wait(new_health(second_stamp), message='fresh acquisition previews despite held TF pair')
        self.assertNotEqual(reports[0]['depth_preview_jpeg'], reports[1]['depth_preview_jpeg'])
        self.assertNotEqual(reports[0]['preview_jpeg'], reports[1]['preview_jpeg'])
        self.assertGreater(reports[1]['depth_preview_monotonic'], reports[0]['depth_preview_monotonic'])
        self.assertEqual(reports[1]['depth_preview_registration'], 'depth_camera_native_grid')

    def test_stalled_rgb_rebind_preserves_session_and_real_capture_timestamps(self):
        from sensor_msgs.msg import CompressedImage
        source = self.channels[0]
        first_stamp = self._pair(source, (45, 100, 200), 1.3, (0., 0., 0.))
        first = self._packet(source)
        depth = CompressedImage()
        depth.header.frame_id = 'camera'
        depth.format = '16UC1; compressedDepth png'
        _, payload = cv2.imencode('.png', np.full((32, 48), 1300, np.uint16))
        depth.data = bytes(12) + payload.tobytes()
        def depth_only_until_rebound():
            depth.header.stamp = source.node.get_clock().now().to_msg()
            source.depth.publish(depth)
            h = self._health().get(source.robot['id'], {})
            if not h.get('rgb_reader_awaiting_frame'):
                return False
            self.assertEqual(h['status'], 'rgb_stale')
            self.assertEqual(h['rgb_reader_attempts'], 1)
            self.assertEqual(h['rgb_reader_recoveries'], 0)
            self.assertEqual(h['preview_stamp_ns'], first_stamp)
            return True
        self._wait(depth_only_until_rebound, timeout=17., message='stalled RGB reader rebind')
        self._wait(lambda: source.rgb.get_subscription_count() == 1)
        next_stamp = self._pair(source, (80, 160, 15), 1.6, (.4, .2, .1))
        second = self._packet(source)
        self.assertEqual(second['rgb_stamp_ns'], next_stamp)
        self.assertEqual(second['session_id'], first['session_id'])
        self.assertGreater(second['seq'], first['seq'])
        self._wait(lambda: self._health().get(source.robot['id'], {}).get('rgb_reader_recoveries') == 1)

    def test_simulation_clock_pause_resume_and_reset(self):
        from rosgraph_msgs.msg import Clock
        from rclpy.qos import qos_profile_sensor_data
        source = self.channels[0]
        clock = source.node.create_publisher(Clock, '/clock', qos_profile_sensor_data)
        self._wait(lambda: clock.get_subscription_count() > 0, message='simulation clock subscriber')
        self._pair(source, (25, 50, 200), 1.2, (0., 0., 0.), now_ns=1_000_000_000)
        self._spin_for(.2)
        with self.assertRaises(queue.Empty):
            source.output.get_nowait()
        self._wait(lambda: self._health().get(source.robot['id'], {}).get('simulation_clock_status') == 'waiting_for_clock')

        def set_clock(ns):
            clock.publish(Clock(clock=self._stamp(ns)))
            self._spin_for(.15)

        set_clock(10_000_000_000)
        stamp = self._pair(source, (25, 50, 200), 1.2, (0., 0., 0.), now_ns=10_000_000_000)
        packet = self._packet(source)
        self.assertEqual(packet['stamp_ns'], stamp)
        self.assertEqual(packet['transport_timing']['clock_mode'], 'simulation')
        self.assertAlmostEqual(packet['transport_timing']['selected_frame_age_at_receipt_s'], .1)
        self.assertFalse(packet['preview_only'])
        self._wait(lambda: self._health().get(source.robot['id'], {}).get('simulation_clock_status') == 'paused_or_stalled',
                   timeout=5., message='wall-time health while /clock is paused')
        set_clock(11_000_000_000)
        self._pair(source, (30, 70, 210), 1.2, (.1, 0., 0.), now_ns=11_000_000_000)
        resumed = self._packet(source)
        self.assertEqual(resumed['session_id'], packet['session_id'])
        self.assertGreater(resumed['seq'], packet['seq'])
        set_clock(0)
        set_clock(20_000_000_000)  # Must remain latched even after overtaking old clock.
        self._pair(source, (30, 70, 210), 1.2, (.1, 0., 0.), now_ns=20_000_000_000)
        self._wait(lambda: self._health().get(source.robot['id'], {}).get('simulation_clock_status') == 'reset_restart_required')
        with self.assertRaises(queue.Empty):
            source.output.get_nowait()


if __name__ == "__main__":
    unittest.main()
