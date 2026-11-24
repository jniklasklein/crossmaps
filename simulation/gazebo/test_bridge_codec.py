import base64
from pathlib import Path
import struct
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bridge_codec import (CAMERA_FRAME, SIM_DOMAINS, LatestRgbdPair, camera_calibration,
                          camera_extrinsic, encode_depth, encode_rgb, image_array, safe_velocity, sensor_noise,
                          stamp_ns, stamp_parts)


def image_message(image, fmt):
    return dict(width=image.shape[1], height=image.shape[0], step=image.strides[0],
                data=base64.b64encode(image.tobytes()).decode(), pixel_format_type=fmt,
                header={"stamp": {"sec": "12", "nsec": 345}})


class BridgeCodecTests(unittest.TestCase):
    def test_pair_join_exact_stamp_latest_complete_only(self):
        join = LatestRgbdPair(capacity=4)
        def msg(t):
            return {"header":{"stamp":{"nsec":t}}}
        for t in (10,20,30):
            join.add("rgb",msg(t))
        join.add("depth",msg(10))
        join.add("depth",msg(20))
        a,b = join.take_latest()
        self.assertEqual(stamp_ns(a),20)
        self.assertEqual(stamp_ns(b),20)
        self.assertIsNone(join.take_latest())
        join.add("depth",msg(30))
        self.assertEqual(stamp_ns(join.take_latest()[0]),30)
        join.add("rgb",msg(20)); join.add("depth",msg(20))
        self.assertIsNone(join.take_latest())

    def test_pair_join_bounds_unmatched_backlog(self):
        join = LatestRgbdPair(capacity=2)
        for t in range(10):
            join.add("rgb",{"header":{"stamp":{"nsec":t}}})
        self.assertEqual(len(join.pending["rgb"]),2)
        self.assertEqual(join.dropped,8)
        self.assertIsNone(join.take_latest())

    def test_stamp_preserved_without_wall_time_substitution(self):
        self.assertEqual(stamp_ns({"header": {"stamp": {"sec": "12", "nsec": 345}}}), 12_000_000_345)
        self.assertEqual(stamp_parts({}), (0,0))
        for value in ({"sec": -1}, {"nsec": -1}, {"nsec": 1_000_000_000}):
            with self.assertRaises(ValueError):
                stamp_parts(value)

    def test_rgb_not_reversed_and_jpeg_high_quality(self):
        rgb = np.zeros((48,64,3), np.uint8)
        rgb[:,:,0] = 220
        message = image_message(rgb, "RGB_INT8")
        converted = image_array(message)
        self.assertEqual(converted[0,0].tolist(), [0,0,220])
        fmt, payload = encode_rgb(message)
        self.assertIn("jpeg", fmt)
        decoded = cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_COLOR)
        self.assertLess(np.abs(decoded.astype(float)-converted).max(), 3)

    def test_rgb_padding_and_alpha(self):
        bgra = np.array([[[10,20,30,255], [40,50,60,255]]],np.uint8)
        self.assertEqual(image_array(image_message(bgra,"BGRA_INT8")).tolist(), [[[10,20,30],[40,50,60]]])
        message = dict(width=1,height=2,step=4,pixel_format_type="RGB_INT8",
                       data=base64.b64encode(bytes([1,2,3,99,4,5,6,99])).decode())
        self.assertEqual(image_array(message).tolist(), [[[3,2,1]],[[6,5,4]]])

    def test_depth_lossless_mm_and_invalid_is_zero(self):
        depth = np.array([[1.,2.345,np.inf,np.nan,-1.,0.,70.]],dtype="<f4")
        fmt, payload = encode_depth(image_message(depth,"R_FLOAT32"))
        self.assertEqual(fmt,"16UC1; compressedDepth png")
        self.assertEqual(struct.unpack("<iff",payload[:12]),(0,0.,0.))
        result = cv2.imdecode(np.frombuffer(payload[12:],np.uint8),cv2.IMREAD_UNCHANGED)
        self.assertEqual(result.dtype, np.uint16)
        self.assertEqual(result.tolist(), [[1000,2345,0,0,0,0,0]])

    def test_bad_inputs_fail_closed(self):
        message = image_message(np.zeros((3,4,3),np.uint8),"RGB_INT8")
        for change in ({"width":0}, {"step":1}, {"data":"badbase64!"}, {"pixel_format_type":"L_INT16"}):
            with self.assertRaises(ValueError):
                image_array(dict(message,**change))
        with self.assertRaises(ValueError):
            encode_depth(message)

    def test_calibration_uses_published_intrinsics(self):
        k = [380.,0.,320.,0.,380.,180.,0.,0.,1.]
        result = camera_calibration(dict(width=640,height=360,intrinsics={"k":k}))
        self.assertEqual(result["k"], k)
        self.assertEqual(result["p"], [380.,0.,320.,0.,0.,380.,180.,0.,0.,0.,1.,0.])
        with self.assertRaises(ValueError):
            camera_calibration(dict(width=640,height=360,intrinsics={"k":k},distortion={"k":[.1]}))

    def test_optical_axes(self):
        x,y,z,w = (-.5,.5,-.5,.5)
        rot = np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],
                        [2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
                        [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]])
        np.testing.assert_allclose(rot[:,2],[1,0,0])
        np.testing.assert_allclose(rot[:,0],[0,-1,0])
        np.testing.assert_allclose(rot[:,1],[0,0,-1])

    def test_scene_extrinsic_required_and_not_hardcoded(self):
        optical = dict(translation=[.06,.01,.11],quaternion_xyzw=[-.5,.5,-.5,.5],
                       frame_id=CAMERA_FRAME,calibration_status="fixture")
        scene = {"robots":{"rover_a":{"camera_optical":optical}}}
        self.assertEqual(camera_extrinsic(scene,"rover_a")["translation"],(.06,.01,.11))
        with self.assertRaises(ValueError):
            camera_extrinsic(scene,"rover_b")
        for change in ({"translation":[float("nan"),0,0]}, {"quaternion_xyzw":[0,0,0,0]},
                       {"frame_id":"wrong"}, {"translation":[60,10,110]}):
            invalid = {"robots":{"rover_a":{"camera_optical":dict(optical,**change)}}}
            with self.assertRaises(ValueError):
                camera_extrinsic(invalid,"rover_a")

    def test_noisy_rgb_deterministic_by_capture_and_rover(self):
        rgb = np.full((80,80,3),128,np.uint8)
        options = dict(seed=9,robot="rover_a",capture_ns=123)
        noisy = sensor_noise(rgb,"rgb","noisy",**options)
        np.testing.assert_array_equal(noisy,sensor_noise(rgb,"rgb","noisy",**options))
        self.assertFalse(np.array_equal(noisy,sensor_noise(rgb,"rgb","noisy",**dict(options,robot="rover_b"))))
        self.assertFalse(np.array_equal(noisy,sensor_noise(rgb,"rgb","noisy",**dict(options,capture_ns=124))))
        np.testing.assert_array_equal(rgb,sensor_noise(rgb,"rgb","clean"))
        self.assertGreater(np.std(noisy.astype(float)-rgb),.8)
        self.assertLess(np.std(noisy.astype(float)-rgb),1.6)

    def test_depth_noise_range_dropout_quantization_and_invalid(self):
        near = np.ones((128,128),np.float32)
        far = near*5.
        a,b = [sensor_noise(data,"depth","noisy",capture_ns=99) for data in (near,far)]
        self.assertGreater(np.std(b[b>0]-5.),np.std(a[a>0]-1.)*5)
        self.assertGreater(np.mean(b==0),np.mean(a==0)*2)
        np.testing.assert_allclose(b/.002,np.rint(b/.002),atol=.0003)
        invalid = np.array([[np.nan,np.inf,0.,-1.]],np.float32)
        np.testing.assert_array_equal(sensor_noise(invalid,"depth","noisy"),np.zeros_like(invalid))
        clean = sensor_noise(far,"depth","clean")
        np.testing.assert_array_equal(clean,far)

    def test_sim_domains_and_motion_limits(self):
        self.assertEqual(SIM_DOMAINS,{"rover_a":80,"rover_b":81})
        self.assertEqual(safe_velocity(99,-99),(.35,-.7))
        self.assertEqual(safe_velocity(float("nan"),.2),(0.,0.))


if __name__ == "__main__":
    unittest.main()
