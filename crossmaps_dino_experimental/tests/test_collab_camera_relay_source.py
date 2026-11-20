"""ROS-free wire, freshness, calibration and backpressure tests."""
import contextlib
import io
import json
import os
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from collab_camera_relay_source import (LENGTHS, MAGIC, MAX_IMAGE, PAIR_HISTORY_COUNT, CapturePose, LatestWriter,
                                         PairedCapture, arguments, take_with_credit)


class Clock:
    def __init__(self):
        self.now = 123.

    def __call__(self):
        return self.now


def header(ns, frame):
    return NS(stamp=NS(sec=ns//10**9, nanosec=ns%10**9), frame_id=frame)


def image(ns, frame, payload, fmt):
    return NS(header=header(ns, frame), format=fmt, data=payload)


def calib(ns, frame):
    return NS(header=header(ns, frame), height=33, width=44,
              d=[.02,-.03], k=[3.,0.,2.,0.,4.,2.,0.,0.,1.], r=[0.]*9,
              p=[3.,0.,2.,0.,0.,4.,2.,0.,0.,0.,1.,0.], distortion_model='plumb_bob',
              binning_x=0, binning_y=1, roi=NS(x_offset=1, y_offset=2, width=10, height=20, do_rectify=False))


def decode(parts):
    packet = b''.join(parts)
    assert packet[:5] == MAGIC
    hl, al, bl = LENGTHS.unpack_from(packet, 5)
    start = 5+LENGTHS.size
    assert len(packet) == start+hl+al+bl
    return json.loads(packet[start:start+hl]), packet[start+hl:start+hl+al], packet[-bl:]


class Sampling(unittest.TestCase):
    def setUp(self):
        self.base = 2_500_000_000
        self.clock = Clock()
        self.s = PairedCapture(.08, 1.5, self.clock)
        self.s.calibration('rgb', calib(0, 'optical_rgb'))
        self.s.calibration('depth', calib(0, 'optical_depth'))

    def pair(self, stamp, delta=2_000_000, rgb=b'rgb', depth=b'png'):
        self.s.image('rgb', image(stamp, 'optical_rgb', rgb, 'bgr8; jpeg compressed bgr8'))
        self.s.image('depth', image(stamp+delta, 'optical_depth', depth, '16UC1; compressedDepth png'))

    def test_exact_payload_calibration_wire_and_timestamps(self):
        self.pair(self.base, rgb=b'\x00\xffCMCP1', depth=b'depth\x00')
        m,a,b = decode(self.s.take())
        self.assertEqual((a,b),(b'\x00\xffCMCP1',b'depth\x00'))
        self.assertEqual(m['rgb']['stamp_ns'], self.base)
        self.assertEqual(m['depth']['stamp_ns'], self.base+2_000_000)
        self.assertEqual(m['depth']['format'],'16UC1; compressedDepth png')
        self.assertEqual(m['rgb_info']['k'][0], 3.)
        self.assertEqual(m['rgb_info']['d'], [.02,-.03])
        self.assertEqual(m['depth_info']['frame_id'], 'optical_depth')
        self.assertEqual(m['depth_info']['roi']['x_offset'],1)

    def test_slop_nonrepetition_rate_and_local_age(self):
        self.pair(self.base, delta=90_000_000)
        self.assertIsNone(self.s.take())
        self.pair(self.base)
        self.assertIsNotNone(self.s.take())
        self.pair(self.base+500_000_000)
        self.assertIsNone(self.s.take())  # publish cap
        self.clock.now += .8
        self.assertIsNotNone(self.s.take())
        self.clock.now += .8
        self.pair(self.base+500_000_000)  # duplicate compressed RGB+depth
        self.assertIsNone(self.s.take())
        self.pair(self.base+2_000_000_000)
        self.clock.now += 2
        self.assertIsNone(self.s.take())  # expired, although not yet sent

    def test_recent_precisely_matched_images_win_over_adjacent_frame(self):
        self.pair(self.base)
        self.s.image('rgb', image(self.base+66_000_000,'optical_rgb',b'new', 'jpg'))
        self.assertEqual(decode(self.s.take())[0]['rgb']['stamp_ns'], self.base)

    def test_pose_at_depth_stamp_prefers_recent_covered_pair_but_never_blocks_camera(self):
        available = self.base+2_000_000
        self.s.pose_lookup = lambda frame, ns: dict(source_frame=frame, stamp_ns=ns) if ns==available else None
        self.pair(self.base)
        self.pair(self.base+100_000_000)
        meta,_,_ = decode(self.s.take())
        self.assertEqual(meta['pose'], dict(source_frame='optical_rgb', stamp_ns=meta['depth']['stamp_ns']))
        self.clock.now += 1.
        self.pair(self.base+1_000_000_000)
        fresh,_,_ = decode(self.s.take())
        self.assertNotIn('pose', fresh)
        self.assertEqual(fresh['depth']['stamp_ns'], self.base+1_002_000_000)

    def test_startup_adjacent_depth_gets_a_short_chance_to_be_replaced(self):
        self.pair(self.base,delta=-66_000_000)
        self.assertIsNone(self.s.take())
        self.clock.now+=.04
        self.s.image('depth',image(self.base+200_000,'optical_depth',b'true','png'))
        self.assertEqual(decode(self.s.take())[2],b'true')

    def test_base_mount_is_queried_for_selected_capture_and_never_required(self):
        calls = []
        def lookup(frame, ns):
            calls.append((frame, ns))
            return dict(source_frame='base', target_frame=frame, stamp_ns=ns)
        self.s.mount_lookup = lookup
        self.pair(self.base)
        self.pair(self.base + 100_000_000)
        meta, _, _ = decode(self.s.take())
        ns = meta['depth']['stamp_ns']
        self.assertEqual(calls, [('optical_rgb', ns)])
        self.assertEqual(meta['mount'], dict(source_frame='base', target_frame='optical_rgb', stamp_ns=ns))
        self.clock.now += 1.
        self.s.mount_lookup = lambda frame, ns: None
        self.pair(self.base + 1_000_000_000)
        self.assertNotIn('mount', decode(self.s.take())[0])

    def test_calibration_mismatch_or_oversize_cannot_be_forwarded(self):
        self.s.info['depth'].clear()
        self.s.calibration('depth', calib(0,'wrong_depth_frame'))
        self.pair(self.base)
        self.assertIsNone(self.s.take())
        self.s.calibration('depth', calib(0,'optical_depth'))
        self.s.images['rgb'].clear()
        self.s.image('rgb',image(self.base,'optical_rgb',bytearray(MAX_IMAGE+1),'jpg'))
        self.assertIsNone(self.s.take())
        self.pair(self.base)
        broken = calib(0,'optical_rgb'); broken.k[0] = float('nan')
        self.s.calibration('rgb', broken)
        self.assertIsNone(self.s.take())

    def test_queue_is_bounded_and_raw_topic_options_fail(self):
        for i in range(50): self.pair(self.base+i*10**6)
        self.assertEqual(len(self.s.images['rgb']),PAIR_HISTORY_COUNT)
        self.assertEqual(len(self.s.images['depth']),PAIR_HISTORY_COUNT)
        for option in ('--rgb-topic','--depth-topic'):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                arguments(['--domain','0',option,'/oak/camera/image_raw'])

    def test_late_compressed_rgb_uses_original_depth_and_matching_calibration(self):
        self.s.info['depth'].clear()
        self.s.info['rgb'].clear()
        # Compressed RGB can arrive half a second later than depth/info.
        # More than four depth frames and six info messages have passed.
        for i in range(13):
            stamp = self.base + i * 40_000_000
            self.s.image('depth', image(stamp, 'optical_depth', bytes([i]), 'png'))
            for k in ('rgb','depth'):
                self.s.calibration(k, calib(stamp, 'optical_'+k))
            self.clock.now += .04
        self.s.image('rgb', image(self.base, 'optical_rgb', b'late', 'jpg'))
        meta, rgb, depth = decode(self.s.take())
        self.assertEqual((rgb,depth), (b'late',b'\x00'))
        self.assertTrue(all(meta[k]['stamp_ns'] == self.base for k in meta))
        self.assertIsNone(self.s.take())

    def test_retained_payloads_also_bounded_by_bytes_and_local_age(self):
        with patch('collab_camera_relay_source.PAIR_HISTORY_BYTES', 9):
            for i in range(10):
                self.s.image('depth', image(self.base+i, 'optical_depth', b'1234', 'png'))
            self.assertEqual(len(self.s.images['depth']), 2)
            self.clock.now += 1.6
            self.s.image('depth', image(self.base+10, 'optical_depth', b'new', 'png'))
            self.assertEqual(len(self.s.images['depth']), 1)


class CapturePoseTests(unittest.TestCase):
    def test_real_tf2_composes_static_mount_only_with_tight_dynamic_coverage(self):
        try:
            import tf2_py
            from rclpy.duration import Duration
            from rclpy.time import Time
            from geometry_msgs.msg import TransformStamped
        except ImportError:
            self.skipTest('requires real ROS tf2 bindings')
        pose = CapturePose('odom', tf2_py.BufferCore(Duration(seconds=8)),
                           lambda ns: Time(nanoseconds=ns), tf2_py.TransformException)
        def tf(parent, child, ns, x, static=False):
            m = TransformStamped()
            m.header.stamp, m.header.frame_id, m.child_frame_id = Time(nanoseconds=ns).to_msg(), parent, child
            m.transform.translation.x, m.transform.rotation.w = float(x), 1.
            pose.ingest(NS(transforms=[m]), static)
        tf('base', 'cam', 0, .3, True)
        mount = pose.lookup('base', 1_100_000_000, target='cam')
        self.assertEqual((mount['target_frame'], mount['source_frame'], mount['stamp_ns']),
                         ('cam', 'base', 1_100_000_000))
        self.assertAlmostEqual(mount['matrix'][0][3], -.3)
        tf('odom', 'base', 1_000_000_000, 1.)
        tf('odom', 'base', 1_200_000_000, 2.)
        p = pose.lookup('cam', 1_100_000_000)
        self.assertEqual((p['target_frame'],p['source_frame'],p['stamp_ns']),('odom','cam',1_100_000_000))
        self.assertAlmostEqual(p['matrix'][0][3], 1.8)
        tf('odom', 'base', 3_000_000_000, 30.)
        for ns in (0, 100_000_000, 2_000_000_000, 3_100_000_000):
            self.assertIsNone(pose.lookup('cam', ns))
        self.assertIsNone(pose.lookup('disconnected', 1_100_000_000))


class PipeTests(unittest.TestCase):
    def setUp(self):
        self.r,self.w=os.pipe(); self.clock=Clock()
        self.addCleanup(os.close,self.r);self.addCleanup(os.close,self.w)
        self.writer=LatestWriter(self.w,self.clock)

    def test_unstarted_frame_replaced_and_short_real_write(self):
        self.writer.offer((b'old',))
        self.writer.offer((b'new',b'data'))
        self.writer.pump()
        self.assertEqual(os.read(self.r,100), b'newdata')

    def test_partial_frame_is_never_interleaved_under_backpressure(self):
        sent=bytearray()
        def slow(fd, chunks):
            prefix=b''.join(chunks)[:2]; sent.extend(prefix); return len(prefix)
        with patch('collab_camera_relay_source.os.writev', side_effect=slow):
            self.writer.offer((b'aaa', b'111'))
            self.writer.pump()
            self.writer.offer((b'old', b'222'))
            self.writer.offer((b'new', b'333'))
            for i in range(7): self.writer.pump()
        self.assertEqual(sent,b'aaa111new333')

    def test_stuck_partial_expires_and_empty_reader_terminates(self):
        self.writer.offer((b'abcdef',))
        with patch('collab_camera_relay_source.os.writev',return_value=1):
            self.writer.pump()
            self.clock.now += 9
            with self.assertRaises(TimeoutError): self.writer.pump()
        other_r,other_w=os.pipe()
        try:
            writer=LatestWriter(other_w)
            os.close(other_r)
            with self.assertRaises(BrokenPipeError): writer.pump()
        finally:
            os.close(other_w)


class CompactAndCredit(unittest.TestCase):
    def fixture(self, stride, bad=None):
        import cv2
        import numpy as np
        self.cv,self.np=cv2,np
        w,h=71,53
        y,x=np.mgrid[:h,:w]
        bgr=np.dstack([60+x,50+y,80+(x+y)//2]).astype('uint8')
        depth=(32000+125*x-82*y).astype('uint16')
        depth[:,w//2:]+=13000
        depth[::9,::11]=0
        self.cv.setNumThreads(1)
        rgb_bytes=cv2.imencode('.jpg',bgr,[cv2.IMWRITE_JPEG_QUALITY,95])[1].tobytes()
        depth_bytes=b'\x00'*12+cv2.imencode('.png',depth)[1].tobytes()
        self.rgb=cv2.imdecode(np.frombuffer(rgb_bytes,np.uint8),cv2.IMREAD_COLOR)
        self.dep=depth
        s=PairedCapture(.08,1.5,image_stride=stride)
        for key in ('rgb','depth'):
            c=calib(2000,key);c.width,c.height=w,h
            c.roi=NS(x_offset=0,y_offset=0,width=w,height=h,do_rectify=False)
            c.k=[108.,.3,35.,0.,111.,24.,0.,0.,1.]
            c.p=[108.,.3,35.,-13.,0.,111.,24.,.4,0.,0.,1.,0.]
            if bad and key=='depth':
                if bad=='roi': c.roi.x_offset=3
                if bad=='binning': c.binning_x=2
                if bad=='dims': c.width+=1
            s.calibration(key,c)
            rt='bgr8; jpeg compressed bgr8' if key=='rgb' else ('32FC1; compressedDepth png' if bad=='float' else '16UC1; compressedDepth png')
            s.image(key,image(2001 if key=='rgb' else 2021,key, rgb_bytes if key=='rgb' else depth_bytes,rt))
        return s

    def test_compact_preserves_3d_stamps_rgb_grid_and_exact_depth_samples(self):
        import numpy as np
        for stride in (2,3,4):
            s=self.fixture(stride)
            meta,c,d=decode(s.take())
            np.testing.assert_array_equal(d[:12],b'\x00'*12)
            actual=self.cv.imdecode(np.frombuffer(d[12:],np.uint8),self.cv.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(actual,self.dep[::stride,::stride])
            self.assertEqual(actual.dtype,np.uint16)
            actual_rgb=self.cv.imdecode(np.frombuffer(c,np.uint8),self.cv.IMREAD_COLOR)
            diff=np.abs(actual_rgb.astype(float)-self.rgb[::stride,::stride].astype(float))
            self.assertLess(diff.mean(),3.0)
            self.assertEqual(meta['depth']['stamp_ns'],2021)
            self.assertEqual(meta['rgb']['stamp_ns'],2001)
            for k in ('rgb','depth'):
                info=meta[k+'_info']
                self.assertEqual(info['frame_id'],k)
                self.assertEqual((info['height'],info['width']),actual.shape)
                self.assertEqual(info['roi']['width'],actual.shape[1])
                for fld,cols in [('k',3),('p',4)]:
                    old=np.array(s.info[k][0][fld]).reshape(3,cols)
                    new=np.array(info[fld]).reshape(3,cols)
                    point=np.array([.1,-.5,3.,1.])[:cols]
                    before,after=old@point,new@point
                    np.testing.assert_allclose(after[:2]/after[2],before[:2]/before[2]/stride)

    def test_compact_rejects_unsupported_depth_and_calibration(self):
        for failure in ('roi','binning','dims','float'):
            s=self.fixture(3,bad=failure)
            self.assertIsNone(s.take(),failure)
            self.assertNotEqual(s.reason,'streaming')

    def test_one_in_flight_no_old_frames_materialized_during_stall(self):
        s=PairedCapture(.08,10,clock=lambda: t[0])
        for key in ('rgb','depth'): s.calibration(key,calib(0,key))
        credit=threading.Event(); credit.set()
        writer=NS(current=None)
        def pair(ns):
            for key in ('rgb','depth'): s.image(key,image(ns,key,bytes([ns]),'jpg'))
        t=[1.]
        pair(1);self.assertIsNotNone(take_with_credit(s,writer,credit));self.assertFalse(credit.is_set())
        for n in range(2,20):
            t[0]+=.3;pair(n)
            self.assertIsNone(take_with_credit(s,writer,credit))
        self.assertEqual(s.last['rgb'],1)
        credit.set()
        writer.current=(b'partial',)
        self.assertIsNone(take_with_credit(s,writer,credit))
        writer.current=None
        meta,_,_=decode(take_with_credit(s,writer,credit))
        self.assertEqual(meta['rgb']['stamp_ns'],19)
        self.assertFalse(credit.is_set())


if __name__ == '__main__':
    unittest.main()
