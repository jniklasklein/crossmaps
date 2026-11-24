"""Real lidar scene/codec contracts, independent of ROS installation/runtime."""
import ast
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from bridge_codec import LIDAR_FRAME, laser_scan, lidar_extrinsic, stamp_ns
from generate_world import build, VENDOR


def scan_message():
    return dict(header={"stamp": {"sec": "5", "nsec": 123}}, frame=LIDAR_FRAME,
                count=720, vertical_count=1, angle_min=-math.pi,
                angle_max=math.pi-2*math.pi/720, angle_step=2*math.pi/720,
                range_min=.12, range_max=12., ranges=[2.] * 720,
                intensities=[0.] * 720)


class LidarTests(unittest.TestCase):
    def test_odom_publisher_is_reliable_for_nav2_not_camera_qos(self):
        tree = ast.parse((Path(__file__).parent / "ros_bridge.py").read_text())
        publishers = {call.args[1].value: call for call in ast.walk(tree)
                      if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                      and call.func.attr == "create_publisher" and len(call.args) >= 3
                      and isinstance(call.args[1], ast.Constant)}
        odom_qos = publishers["/odom"].args[2]
        self.assertIsInstance(odom_qos, ast.Call)
        self.assertEqual(odom_qos.func.id, "QoSProfile")
        settings = {keyword.arg: keyword.value for keyword in odom_qos.keywords}
        self.assertEqual(settings["depth"].value, 10)
        self.assertEqual(settings["reliability"].attr, "RELIABLE")
        for topic in ("/scan", "/oak/rgb/image_rect/compressed", "/oak/stereo/image_raw/compressedDepth"):
            self.assertEqual(publishers[topic].args[2].id, "qos_profile_sensor_data")

    def test_cad_mount_and_real_planar_sensor(self):
        vendor = ET.parse(VENDOR / "urdf/ugv_rover.urdf").getroot()
        mount = vendor.find("joint[@name='base_lidar_link_joint']/origin")
        base = vendor.find("joint[@name='base_joint']/origin")
        base_xyz = list(map(float, base.get("xyz").split()))
        mount_xyz = list(map(float, mount.get("xyz").split()))
        yaw = float(mount.get("rpy").split()[2])
        with tempfile.TemporaryDirectory() as directory:
            path = build(Path(directory))
            world = ET.parse(path).getroot().find("world")
            scene = json.loads((Path(directory) / "scene.json").read_text())
            for robot in ("rover_a", "rover_b"):
                sensor = world.find(f"model[@name='{robot}']/link[@name='base_lidar_link']/sensor")
                self.assertEqual(sensor.get("type"), "gpu_lidar")
                self.assertEqual(sensor.findtext("topic"), f"/{robot}/scan")
                self.assertEqual(sensor.findtext("gz_frame_id"), LIDAR_FRAME)
                self.assertEqual(sensor.findtext("update_rate"), "10.0")
                sensor_pose = list(map(float, sensor.findtext("pose").split()))
                self.assertEqual(sensor_pose[:2], [0., 0.])
                self.assertAlmostEqual(sensor_pose[2], .028)
                self.assertAlmostEqual(sensor_pose[5], -yaw)
                horizontal = sensor.find("lidar/scan/horizontal")
                samples = int(horizontal.findtext("samples"))
                self.assertEqual(samples, 720)
                span = float(horizontal.findtext("max_angle")) - float(horizontal.findtext("min_angle"))
                self.assertAlmostEqual(span * samples / (samples - 1), 2 * math.pi)
                self.assertEqual(sensor.findtext("lidar/scan/vertical/samples"), "1")
                self.assertEqual(sensor.findtext("lidar/range/min"), "0.12")
                self.assertEqual(sensor.findtext("lidar/range/max"), "12.0")
                self.assertEqual(sensor.find("lidar/noise").get("type"), "gaussian")
                self.assertEqual(sensor.findtext("lidar/noise/stddev"), "0.01")
                extrinsic = lidar_extrinsic(scene, robot)
                expected = [base_xyz[i] + mount_xyz[i] + (sensor_pose[2] if i == 2 else 0.) for i in range(3)]
                self.assertEqual(list(extrinsic["translation"]), expected)
                self.assertEqual(extrinsic["quaternion_xyzw"], (0., 0., 0., 1.))
                self.assertAlmostEqual(extrinsic["translation"][2], .148)
                self.assertIn("not hardware calibrated", extrinsic["calibration_status"])

    def test_scan_preserves_capture_time_geometry_and_instantaneous_rays(self):
        message = scan_message()
        result = laser_scan(message)
        self.assertEqual(stamp_ns(message), 5_000_000_123)
        self.assertEqual(len(result["ranges"]), 720)
        self.assertEqual(result["ranges"], [2.] * 720)
        self.assertEqual(result["angle_increment"], message["angle_step"])
        self.assertAlmostEqual(result["angle_min"] + 719 * result["angle_increment"], result["angle_max"])
        self.assertEqual(result["scan_time"], .1)
        self.assertEqual(result["time_increment"], 0.)

    def test_invalid_returns_never_become_fake_free_space(self):
        message = scan_message()
        message["ranges"][:7] = ["Infinity", "-Infinity", "NaN", 0., 12.1, .12, 12.]
        result = laser_scan(message)["ranges"]
        self.assertEqual(result[:2], [math.inf, -math.inf])
        self.assertTrue(math.isnan(result[2]))
        self.assertEqual(result[3:7], [-math.inf, math.inf, .12, 12.])
        message["intensities"] = []
        self.assertEqual(laser_scan(message)["intensities"], [])

    def test_malformed_scan_fails_closed(self):
        for change in ({"count": 5000}, {"count": 0}, {"count": 719},
                       {"vertical_count": 2}, {"ranges": []}, {"angle_step": -.1},
                       {"angle_step": float("nan")}, {"angle_max": 1.},
                       {"range_min": 12.}, {"range_max": math.inf},
                       {"frame": "odom"}, {"intensities": [1.]},
                       {"intensities": [math.nan] * 720}):
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                laser_scan(dict(scan_message(), **change))
        for rate in (0., math.nan, math.inf):
            with self.assertRaises(ValueError):
                laser_scan(scan_message(), rate)

    def test_laser_mount_requires_generated_valid_metadata(self):
        laser = dict(translation=[.04, 0., .148], quaternion_xyzw=[0., 0., 0., 1.],
                     frame_id=LIDAR_FRAME, update_rate_hz=10., calibration_status="test")
        scene = {"robots": {"rover_a": {"lidar": laser}}}
        self.assertEqual(lidar_extrinsic(scene, "rover_a")["update_rate_hz"], 10.)
        with self.assertRaises(ValueError):
            lidar_extrinsic(scene, "rover_b")
        for change in ({"translation": [1., 0., math.nan]}, {"translation": [0., 0.]},
                       {"quaternion_xyzw": [0., 0., 0., 0.]}, {"frame_id": "base_link"},
                       {"update_rate_hz": 0.}, {"calibration_status": ""}):
            invalid = copy.deepcopy(scene)
            invalid["robots"]["rover_a"]["lidar"].update(change)
            with self.assertRaises(ValueError):
                lidar_extrinsic(invalid, "rover_a")


if __name__ == "__main__":
    unittest.main()
