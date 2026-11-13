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
            {"id": "smoke_a", "domain": 86}, {"id": "smoke_b", "domain": 87}],
            "input": {"max_hz": 10., "max_age_s": 10., "tf_timeout_s": .3}})
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
                    result[source.robot["id"]] = {k: v for k, v in health.items() if k != "preview_jpeg"}
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

    def _pair(self, source, color, depth_m, pose, delayed_tf=False):
        from sensor_msgs.msg import CompressedImage
        now_ns = source.node.get_clock().now().nanoseconds
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
        if not delayed_tf:
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


if __name__ == "__main__":
    unittest.main()
