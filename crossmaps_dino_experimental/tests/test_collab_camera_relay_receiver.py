"""Wire/ownership tests: local pipes only, never SSH or rover connections."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collab_camera_relay as relay


def frame(stamp=1234567891, rgb=b'jpeg', depth=b'png'):
    meta = dict(frame_id='optical', stamp_ns=stamp)
    info = dict(meta, width=8, height=8, k=[1.]*9, r=[1.]*9, p=[1.]*12, d=[], binning_x=0, binning_y=0,
                distortion_model='plumb_bob',
                roi=dict(width=0, height=0, x_offset=0, y_offset=0, do_rectify=False))
    obj = dict(rgb=dict(meta, format='rgb8; jpeg compressed'), depth=dict(meta, format='16UC1; compressedDepth png'),
               rgb_info=info, depth_info=info)
    header = json.dumps(obj).encode()
    return relay.HEADER.pack(relay.MAGIC, len(header), len(rgb), len(depth))+header+rgb+depth, obj


def settings():
    return ({'id':'rover_a', 'domain':0, 'rgb_topic':'/foo/compressed', 'depth_topic':'/bar/compressedDepth',
             'camera_info_topic':'/foo/info', 'depth_camera_info_topic':'/bar/info',
             'camera_relay':{'ssh_host':'jetson@192.0.2.3','container':'ros','mode':'auto','max_hz':1.5}},
            {'max_age_s':2.,'sync_slop_s':.08})


class RelayTests(unittest.TestCase):
    def test_only_exact_finite_rigid_source_pose_can_accompany_a_capture(self):
        _, meta = frame()
        stamp = meta['depth']['stamp_ns']
        matrix = [[0.,-1.,0.,.7],[1.,0.,0.,-.2],[0.,0.,1.,.3],[0.,0.,0.,1.]]
        p = dict(target_frame='odom', source_frame='optical', stamp_ns=stamp, matrix=matrix)
        check = lambda v: relay.checked_pose(v, 'odom', 'optical', stamp)
        self.assertEqual(check(p), p)
        for change in (dict(target_frame='map'), dict(source_frame='depth'), dict(stamp_ns=stamp+1),
                       dict(stamp_ns=True), dict(stamp_ns=float(stamp)), dict(matrix=matrix[:3]),
                       dict(matrix=matrix[:3]+[[0,0,1,1]]),
                       dict(matrix=[[-1,0,0,0],[0,1,0,0],[0,0,1,0],matrix[3]]),
                       dict(matrix=[[1,1,0,0],*matrix[1:]]),dict(matrix=[[1,0,0,float('nan')],*matrix[1:]]),
                       dict(matrix=[[0,-1,False,0],*matrix[1:]])):
            with self.subTest(change=change):
                self.assertIsNone(check(dict(p, **change)))
        self.assertIsNone(check(None))

    def test_missing_compact_codec_reports_action_not_remote_traceback(self):
        for module in ('PIL', 'numpy'):
            error = relay._failure(f"private remote path: No module named '{module}'".encode())
            self.assertIn('image_stride 1', error)
            self.assertNotIn('private remote path', error)

    def test_wire_fragmented_multiple_packets_exact(self):
        one, m = frame(); two, n = frame(2345678912)
        reader = relay.FrameReader()
        records = []
        for i in range(0, len(one+two), 17):
            records.extend(reader.feed((one+two)[i:i+17]))
        self.assertEqual(records, [(m,b'jpeg',b'png'),(n,b'jpeg',b'png')])

    def test_reject_before_large_allocation_and_bad_metadata(self):
        for header in [relay.HEADER.pack(relay.MAGIC, 1, relay.MAX_IMAGE + 1, 1),
                       relay.HEADER.pack(b'wrong', 1, 1, 1), relay.HEADER.pack(relay.MAGIC, 0, 1, 1)]:
            with self.assertRaises(ValueError):
                list(relay.FrameReader().feed(header))
        _, obj = frame()
        obj['rgb_info']['r'] = [float('nan')]*9
        with self.assertRaises(ValueError): relay._metadata(json.dumps(obj))

    def test_secure_auth_args_no_secret_in_command_no_implicit_sshpass(self):
        robot, cfg = settings()
        argv, env = relay.ssh_command(robot, cfg, {relay.PASSWORD_ENV:'dummy-example', 'SSHPASS':'old-example'},
                                      password_auth=True)
        self.assertNotIn('dummy-example',' '.join(argv))
        self.assertEqual(argv[0],'ssh')
        self.assertIn('BatchMode=no',argv)
        self.assertIn('StrictHostKeyChecking=yes',argv)
        self.assertNotIn(relay.PASSWORD_ENV,env)
        self.assertNotIn('SSHPASS',env)
        self.assertEqual(shlex.split(argv[-1])[:6], ['docker','exec','-i','ros','bash','-c'])
        self.assertIn('--image-stride 1',argv[-1])
        robot['camera_relay']['image_stride'] = 4
        argv, env = relay.ssh_command(robot,cfg,{'SSHPASS':'ignore-me'})
        self.assertIn('--image-stride 4',argv[-1])
        self.assertEqual(argv[0],'ssh'); self.assertNotIn('SSHPASS',env); self.assertIn('BatchMode=yes',argv)
        robot['camera_relay']['ssh_host'] = '-SomethingBad'
        with self.assertRaises(ValueError): relay.ssh_command(robot,cfg,{})

    def test_subprocess_lease_ends_idle_remote_on_eof(self):
        code = b"import time; print('ready', flush=True); time.sleep(60)"
        with subprocess.Popen([sys.executable,'-u','-c',relay._BOOT],stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE) as proc:
            proc.stdin.write(str(len(code)).encode()+b'\n'+code)
            proc.stdin.flush()
            self.assertEqual(proc.stdout.readline(), b'ready\n')
            proc.stdin.close()
            self.assertEqual(proc.wait(timeout=3),0)

    def test_latest_only_dedup_generational_change_and_close(self):
        robot, cfg = settings()
        stamp = time.time_ns()
        one, _ = frame(stamp)
        two, _ = frame(stamp=stamp+1,rgb=b'new')
        # Run a local fixture via our bootloader to exercise the real pipe
        # reader, latest-slot replacement and subprocess/lease ownership.
        code = b"import os,time;os.write(1,"+repr(one+two+two).encode()+b");time.sleep(60)"
        argv = [sys.executable,'-u','-c',relay._BOOT]
        client = relay.CameraRelay(robot,cfg)
        with patch.object(relay,'ssh_command',return_value=(argv,dict(os.environ))), patch.object(
                relay.Path,'read_bytes',return_value=code), patch.object(relay,'_messages',side_effect=lambda *x:x):
            client.start(); client.start()
            try:
                end = time.monotonic()+4
                while client.status()['replaced_pairs'] < 1 and time.monotonic()<end: time.sleep(.01)
                packet = client.poll()
                self.assertIsNotNone(packet, client.status())
                self.assertEqual(packet[1], b'new')
                self.assertEqual(packet[-1], 1)
                self.assertEqual(client.status()['replaced_pairs'], 1)
                self.assertIsNone(client.poll())
                client._pending = (*packet[:3], time.monotonic()-4, *packet[4:])
                self.assertIsNone(client.poll(), 'must discard stale pending pairs')
            finally:
                client.close()
            self.assertFalse(client._thread.is_alive())
            client.close(); client.start()
            self.assertEqual(client.status()['attempts'], 1)

    def test_ros_headers_and_calibration_round_trip_when_available(self):
        try: import sensor_msgs.msg
        except ImportError: self.skipTest('ROS messages unavailable')
        _, obj = frame()
        pair = relay._messages(obj,b'jpeg',b'png',1.,123,2)
        self.assertEqual((pair.rgb.header.stamp.sec,pair.rgb.header.stamp.nanosec),(1,234567891))
        self.assertEqual(pair.depth.header.frame_id,'optical')
        self.assertEqual(pair.rgb_info.distortion_model,'plumb_bob')
        self.assertEqual(bytes(pair.depth.data),b'png')

    def test_crashed_source_reconnects_with_new_generation_and_original_reset_stamp(self):
        robot, cfg = settings()
        stamp=time.time_ns()
        first, _ = frame(stamp)
        restarted, _ = frame(stamp-1_000_000_000, rgb=b'after-reset')
        cfg['max_age_s'] = 5.
        def script(data, hold):
            return b"import os,time;os.write(1,"+repr(data).encode()+b");time.sleep("+str(hold).encode()+b")"
        with patch.object(relay, 'ssh_command', return_value=([sys.executable,'-u','-c',relay._BOOT],dict(os.environ))), \
             patch.object(relay.Path,'read_bytes',side_effect=[script(first,.1),script(restarted,60)]), \
             patch.object(relay,'_messages',side_effect=lambda *args:args):
            client = relay.CameraRelay(robot,cfg)
            client.start()
            try:
                end = time.monotonic()+5
                packet = None
                while time.monotonic()<end:
                    p = client.poll()
                    if p and p[-1]==2: packet=p; break
                    time.sleep(.01)
                self.assertIsNotNone(packet, client.status())
                self.assertEqual(packet[0]['rgb']['stamp_ns'], stamp-1_000_000_000)
                self.assertEqual(packet[1], b'after-reset')
                self.assertEqual(client.status()['attempts'], 2)
            finally:
                client.close(wait=False)
                self.assertIsNone(client.poll())
                client.close()

    def test_receiver_acks_even_rejected_stale_frames_and_accepts_newest(self):
        robot, cfg = settings()
        old, _ = frame(time.time_ns()-32_000_000_000)
        good, _ = frame(time.time_ns(),rgb=b'current')
        code = (b"import os,time;assert CMCP_CREDIT.is_set();CMCP_CREDIT.clear();os.write(1,"+repr(old).encode()
                +b");assert CMCP_CREDIT.wait(2.);CMCP_CREDIT.clear();os.write(1,"+repr(good).encode()+b");time.sleep(60)")
        argv = [sys.executable,'-u','-c',relay._BOOT]
        client = relay.CameraRelay(robot,cfg)
        with patch.object(relay,'ssh_command',return_value=(argv,dict(os.environ))), patch.object(
                relay.Path,'read_bytes',return_value=code), patch.object(relay,'_messages',side_effect=lambda *x:x):
            client.start()
            try:
                end = time.monotonic()+3; packet=None
                while packet is None and time.monotonic()<end:
                    packet = client.poll()
                    if packet is None: time.sleep(.01)
                self.assertIsNotNone(packet,client.status())
                self.assertEqual(packet[1],b'current')
                self.assertEqual(client.status()['stale_pairs'],1)
                self.assertEqual(client.status()['replaced_pairs'],0)
            finally: client.close()


if __name__ == '__main__':
    unittest.main()
