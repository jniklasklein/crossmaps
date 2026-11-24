"""Opt-in native lifecycle regression tests; no Gazebo server is started.

CROSSMAPS_GZ_NATIVE_TEST=1 python -m unittest discover -s simulation/gazebo -p test_bridge_native.py
Needs localhost socket permission because Gazebo Transport initializes sockets.
"""
import os
import json
from pathlib import Path
import select
import shutil
import subprocess
import unittest
import uuid


@unittest.skipUnless(os.environ.get("CROSSMAPS_GZ_NATIVE_TEST") == "1", "opt-in native Gazebo Transport test")
class NativeBridgeTests(unittest.TestCase):
    def setUp(self):
        self.helper = Path(__file__).parent / ".build" / "gz_capture"
        self.assertTrue(self.helper.is_file(), "Build gz_capture first")
        self.env = dict(os.environ, GZ_IP="127.0.0.1", GZ_PARTITION="crossmaps_sim_native_test_"+uuid.uuid4().hex)

    def test_devnull_eof_exits_without_busy_loop(self):
        result = subprocess.run([str(self.helper),"rover_a"], stdin=subprocess.DEVNULL,
                                capture_output=True,env=self.env,timeout=5.)
        self.assertEqual(result.returncode,0,result.stderr.decode())
        self.assertEqual(result.stdout,b"")

    def test_command_pipe_eof_exits(self):
        result = subprocess.run([str(self.helper),"rover_b"],input=b"0.1 0.2\n",
                                capture_output=True,env=self.env,timeout=5.)
        self.assertEqual(result.returncode,0,result.stderr.decode())
        self.assertEqual(result.stdout,b"")

    def test_unsafe_partition_rejected(self):
        result = subprocess.run([str(self.helper),"rover_a"],input=b"",
                                capture_output=True,env=dict(self.env,GZ_PARTITION="default"),timeout=5.)
        self.assertEqual(result.returncode,2)
        self.assertIn(b"GZ_PARTITION",result.stderr)

    def test_lidar_transport_retains_stamp_and_nonfinite_range_semantics(self):
        # A small protobuf transport fixture, not simulated environmental data.
        # Real GPU ray rendering is separately checked by the runtime smoke test.
        gz = shutil.which("gz")
        self.assertIsNotNone(gz)
        helper = subprocess.Popen([str(self.helper), "rover_a"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env)
        try:
            message = ('header { stamp { sec: 12 nsec: 345 } } frame: "laser_frame" '
                       'count: 3 vertical_count: 1 angle_min: -1 angle_max: 1 angle_step: 1 '
                       'range_min: 0.12 range_max: 12 ranges: 1.5 ranges: inf ranges: nan')
            packet = None
            for _ in range(3):
                subprocess.run([gz, "topic", "-t", "/rover_a/scan", "-m", "gz.msgs.LaserScan", "-p", message],
                               env=self.env, capture_output=True, check=True, timeout=5.)
                if select.select([helper.stdout], [], [], 1.)[0]:
                    packet = json.loads(helper.stdout.readline())
                    break
            self.assertIsNotNone(packet, "No scan received from isolated transport fixture")
            self.assertEqual(packet["kind"], "scan")
            value = packet["message"]
            self.assertEqual(int(value["header"]["stamp"]["sec"]), 12)
            self.assertEqual(value["header"]["stamp"]["nsec"], 345)
            self.assertEqual(value["ranges"], [1.5, "Infinity", "NaN"])
        finally:
            helper.stdin.close()
            try:
                helper.wait(timeout=3.)
            except subprocess.TimeoutExpired:
                helper.kill()
                helper.wait(timeout=3.)
            helper.stdout.close()
            helper.stderr.close()

    def test_evaluation_is_read_only_and_preserves_named_pose(self):
        gz = shutil.which('gz')
        self.assertIsNotNone(gz)
        helper = subprocess.Popen([str(self.helper), 'evaluation'], stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env)
        try:
            message = ('header { stamp { sec: 7 nsec: 123 } } '
                       'pose { name: "rover_b" position { x: 1.2 y: -3 } orientation { w: 1 } }')
            packet = None
            for _ in range(3):
                subprocess.run([gz, 'topic', '-t', '/model/rover_b/pose', '-m', 'gz.msgs.Pose_V', '-p', message],
                               env=self.env, capture_output=True, check=True, timeout=5.)
                if select.select([helper.stdout], [], [], 1.)[0]:
                    packet = json.loads(helper.stdout.readline()); break
            self.assertIsNotNone(packet)
            self.assertEqual(packet['kind'], 'rover_b')
            self.assertEqual(packet['message']['pose'][0]['name'], 'rover_b')
            self.assertEqual(int(packet['message']['header']['stamp']['sec']), 7)
            topics = subprocess.run([gz, 'topic', '-l'], env=self.env, capture_output=True,
                                    check=True, timeout=5.).stdout
            self.assertNotIn(b'cmd_vel', topics)
        finally:
            helper.terminate()
            try:
                helper.wait(timeout=3.)
            except subprocess.TimeoutExpired:
                helper.kill(); helper.wait(timeout=3.)
            helper.stdout.close(); helper.stderr.close()


if __name__ == "__main__":
    unittest.main()
