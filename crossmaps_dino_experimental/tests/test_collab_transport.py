"""Dependency-light image, calibration and deferred-TF queue regressions."""
import copy
import math
import queue
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from collab_transport import (PendingPair, SensorSequencer, cached_rgbd, calibration_id, camera_matrix, decode_depth,
                              decode_rgb, depth_preview_jpeg, latest_put, pair_timing, prepare_registered_rgbd,
                              preview_put, register_depth, require_compressed_transport, stamp_ns, tf_matrix)


class ImageDecodeTests(unittest.TestCase):
    def test_compressed_rgb_is_bgr(self):
        source = np.full((8, 12, 3), (12, 70, 220), np.uint8)
        ok, data = cv2.imencode(".jpg", source, [cv2.IMWRITE_JPEG_QUALITY, 100])
        self.assertTrue(ok)
        np.testing.assert_allclose(decode_rgb(data.tobytes()), source, atol=3)
        with self.assertRaisesRegex(ValueError, "invalid compressed"):
            decode_rgb(b"not an image")

    def test_raw_rgb_respects_padding_and_converts_rgb_once(self):
        pixels = np.array([[[1, 2, 3], [4, 5, 6]], [[20, 30, 40], [50, 60, 70]]], np.uint8)
        rows = np.full((2, 9), 255, np.uint8)
        rows[:, :6] = pixels.reshape(2, 6)
        bgr = decode_rgb(rows.tobytes(), False, "bgr8", 2, 2, 9)
        rgb = decode_rgb(rows.tobytes(), False, "rgb8", 2, 2, 9)
        np.testing.assert_array_equal(bgr, pixels)
        np.testing.assert_array_equal(rgb, pixels[:, :, ::-1])
        self.assertTrue(bgr.flags.writeable)

    def test_invalid_raw_rgb_encoding_stride_and_payload(self):
        for encoding, width, height, step, payload in (
                ("mono8", 2, 2, 2, bytes(4)), ("rgb8", 2, 2, 5, bytes(10)),
                ("rgb8", 2, 2, 6, bytes(11)), ("bgr8", 0, 2, 0, bytes(0))):
            with self.subTest(encoding=encoding, step=step), self.assertRaises(ValueError):
                decode_rgb(payload, False, encoding, width, height, step)

    def test_uint16_png_with_or_without_compressed_depth_header(self):
        source = np.array([[0, 1000, 2000], [250, 1500, 65535]], np.uint16)
        ok, png = cv2.imencode(".png", source)
        self.assertTrue(ok)
        for prefix in (b"", bytes(12)):
            result = decode_depth(prefix + png.tobytes(), True, "16UC1; compressedDepth png")
            self.assertEqual(result.dtype, np.float32)
            np.testing.assert_allclose(result, source.astype(np.float32) * .001)

    def test_unsupported_quantized_float_rvl_and_uint8_png_are_rejected(self):
        _, uint16 = cv2.imencode(".png", np.ones((3, 4), np.uint16))
        _, uint8 = cv2.imencode(".png", np.ones((3, 4), np.uint8))
        for data, encoding in ((uint16.tobytes(), "32FC1; compressedDepth png"),
                               (b"not png RVL", "16UC1; compressedDepth rvl"),
                               (uint8.tobytes(), "16UC1; compressedDepth png")):
            with self.subTest(encoding=encoding), self.assertRaises(ValueError):
                decode_depth(data, True, encoding)

    def test_raw_uint16_depth_handles_endianness_and_padding(self):
        values = np.array([[1000, 2000], [3000, 0]], np.uint16)
        for bigendian in (False, True):
            array = values.astype(">u2" if bigendian else "<u2")
            rows = np.full((2, 6), 127, np.uint8)
            rows[:, :4] = np.frombuffer(array.tobytes(), np.uint8).reshape(2, 4)
            decoded = decode_depth(rows.tobytes(), False, "16UC1", 2, 2, 6, bigendian)
            np.testing.assert_allclose(decoded, [[1., 2.], [3., 0.]])

    def test_raw_float_depth_preserves_metres_and_removes_invalid_values(self):
        values = np.array([[1.25, 2.5, math.nan], [-1., math.inf, 0.]], np.float32)
        for bigendian in (False, True):
            array = values.astype(">f4" if bigendian else "<f4")
            rows = np.zeros((2, 16), np.uint8)
            rows[:, :12] = np.frombuffer(array.tobytes(), np.uint8).reshape(2, 12)
            decoded = decode_depth(rows.tobytes(), False, "32FC1", 3, 2, 16, bigendian, scale=.001)
            np.testing.assert_allclose(decoded, [[1.25, 2.5, 0.], [0., 0., 0.]])

    def test_invalid_raw_depth_format_stride_and_payload(self):
        for encoding, step, payload in (("64FC1", 16, bytes(32)), ("16UC1", 3, bytes(6)),
                                        ("16UC1", 4, bytes(7))):
            with self.subTest(encoding=encoding, step=step), self.assertRaises(ValueError):
                decode_depth(payload, False, encoding, 2, 2, step)


class CalibrationGeometryTests(unittest.TestCase):
    def info(self):
        return SimpleNamespace(width=4, height=3,
            p=[100., 0., 2., 0., 0., 110., 1., 0., 0., 0., 1., 0.],
            k=[80., 0., 2., 0., 90., 1., 0., 0., 1.])

    def test_rectified_projection_intrinsics_preferred_over_raw_k(self):
        info = self.info()
        np.testing.assert_array_equal(camera_matrix(info, (3, 4)), [[100., 0., 2.], [0., 110., 1.], [0., 0., 1.]])
        info.p = [0.] * 12
        np.testing.assert_array_equal(camera_matrix(info, (3, 4)), np.asarray(info.k).reshape(3, 3))

    def test_calibration_shape_mismatch_does_not_resize_depth(self):
        with self.assertRaisesRegex(ValueError, "differs"):
            camera_matrix(self.info(), (6, 8))
        info = self.info()
        info.p = [0.] * 12
        info.k[0] = -5.
        with self.assertRaisesRegex(ValueError, "invalid camera"):
            camera_matrix(info, (3, 4))

    def test_registration_uses_projection_and_leaves_unobserved_pixels_empty(self):
        depth = np.ones((2, 2), np.float32)
        k_rgb = np.diag([2., 2., 1.])
        result = register_depth(depth, np.eye(3), k_rgb, np.eye(4), (4, 4))
        expected = np.zeros((4, 4), np.float32)
        expected[0, 0] = expected[0, 2] = expected[2, 0] = expected[2, 2] = 1.
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(depth, np.ones((2, 2), np.float32))

    def test_registration_nearest_surface_wins_independent_of_source_order(self):
        k_rgb = np.diag([.1, 1., 1.])
        for values in ([[2., 1.]], [[1., 2.]]):
            result = register_depth(np.array(values, np.float32), np.eye(3), k_rgb, np.eye(4), (1, 2))
            np.testing.assert_array_equal(result, [[1., 0.]])

    def test_registration_extrinsics_transform_depth_and_reject_behind_camera(self):
        transform = np.eye(4)
        transform[2, 3] = 1.
        result = register_depth(np.array([[2.]], np.float32), np.eye(3), np.eye(3), transform, (1, 1))
        self.assertEqual(float(result[0, 0]), 3.)
        transform[2, 3] = -5.
        result = register_depth(np.array([[2.]], np.float32), np.eye(3), np.eye(3), transform, (1, 1))
        self.assertEqual(float(result[0, 0]), 0.)

    def test_calibration_identifier_includes_registration_and_extrinsics(self):
        k, extrinsic = np.eye(3), np.eye(4)
        original = calibration_id(k, (3, 4), "rgb", "depth", "projected", k, extrinsic)
        self.assertEqual(original, calibration_id(k.copy(), (3, 4), "rgb", "depth", "projected", k.copy(), extrinsic.copy()))
        changes = []
        moved = extrinsic.copy()
        moved[0, 3] = .03
        changes.append(calibration_id(k, (3, 4), "rgb", "depth", "projected", k, moved))
        changes.append(calibration_id(k, (3, 4), "rgb", "depth", "registered", k, extrinsic))
        changes.append(calibration_id(k, (3, 4), "new_rgb", "depth", "projected", k, extrinsic))
        changes.append(calibration_id(k, (6, 8), "rgb", "depth", "projected", k, extrinsic))
        different_depth_k = k.copy()
        different_depth_k[0, 0] = 1.1
        changes.append(calibration_id(k, (3, 4), "rgb", "depth", "projected", different_depth_k, extrinsic))
        self.assertTrue(all(value != original for value in changes))

    def test_tf_normalizes_quaternion_and_validates_nonfinite_values(self):
        transform = SimpleNamespace(transform=SimpleNamespace(
            translation=SimpleNamespace(x=1., y=2., z=3.),
            rotation=SimpleNamespace(x=0., y=0., z=0., w=2.)))
        expected = np.eye(4)
        expected[:3, 3] = [1., 2., 3.]
        np.testing.assert_array_equal(tf_matrix(transform), expected)
        transform.transform.rotation.w = 0.
        with self.assertRaisesRegex(ValueError, "quaternion"):
            tf_matrix(transform)
        transform.transform.rotation.w = 1.
        transform.transform.translation.x = math.nan
        with self.assertRaisesRegex(ValueError, "translation"):
            tf_matrix(transform)
        self.assertEqual(stamp_ns(SimpleNamespace(sec=12, nanosec=345)), 12_000_000_345)


class PendingPairTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.
        self.pending = PendingPair(2., clock=lambda: self.now)

    def test_keeps_active_tf_waiting_frame_while_newer_frames_arrive(self):
        self.pending.offer("rgb_1", "depth_1")
        first = self.pending.take()
        self.now = .5
        self.pending.offer("rgb_2", "depth_2")
        self.now = 1.
        self.pending.offer("rgb_3", "depth_3")
        self.assertEqual(self.pending.take(), first)
        self.pending.done()
        self.assertEqual(self.pending.take(), ("rgb_3", "depth_3", 1.))
        self.pending.done()
        self.assertIsNone(self.pending.take())

    def test_ttl_replaces_expired_active_with_latest_not_fifo(self):
        self.pending.offer("rgb_1", "depth_1")
        self.pending.take()
        self.now = 1.8
        self.pending.offer("rgb_2", "depth_2")
        self.now = 2.1
        self.assertEqual(self.pending.take(), ("rgb_2", "depth_2", 1.8))

    def test_stale_active_and_stale_latest_both_discarded(self):
        self.pending.offer("first", "first_depth")
        self.pending.take()
        self.now = .1
        self.pending.offer("last", "last_depth")
        self.now = 3.
        self.assertIsNone(self.pending.take())
        self.assertIsNone(self.pending.active)
        self.assertIsNone(self.pending.latest)

    def test_offering_before_take_keeps_only_newest_pair(self):
        self.pending.offer("first", "first_depth")
        self.pending.offer("second", "second_depth")
        self.assertEqual(self.pending.take()[:2], ("second", "second_depth"))

    def test_bounded_queue_replaces_old_value_without_blocking(self):
        channel = queue.Queue(maxsize=1)
        self.assertTrue(latest_put(channel, "first"))
        self.assertTrue(latest_put(channel, "latest"))
        self.assertEqual(channel.qsize(), 1)
        self.assertEqual(channel.get_nowait(), "latest")


class SensorSequencerTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.
        self.sequence = SensorSequencer(clock=lambda: self.now)
        self.sequence.commit(100_000_000_000)

    def test_small_reordered_pair_and_duplicates_never_reset_session(self):
        for stamp, reason in ((99_900_000_000, 'out_of_order_depth_timestamp'),
                              (100_000_000_000, 'duplicate_depth_timestamp')):
            result = self.sequence.inspect(stamp)
            self.assertFalse(result['accept'])
            self.assertFalse(result['reset'])
            self.assertEqual(result['reason'], reason)
        self.assertTrue(self.sequence.inspect(101_000_000_000)['accept'])
        self.assertEqual(self.sequence.last_stamp, 100_000_000_000)


class DiagnosticTransportTests(unittest.TestCase):
    def test_waiting_for_tf_does_not_turn_arrival_latency_into_clock_error(self):
        first = pair_timing(100_000_000_000, 100_100_000_000, 50., 50.2, 2.)
        expired = pair_timing(100_000_000_000, 100_100_000_000, 50., 51.95, 2.)
        self.assertEqual(first['timing_status'], 'within_latency_budget')
        self.assertEqual(expired['timing_status'], 'timestamped_tf_wait_expired')
        self.assertAlmostEqual(expired['selected_frame_age_at_receipt_s'], .1)
        self.assertAlmostEqual(expired['pending_wait_s'], 1.95)
        self.assertAlmostEqual(expired['selected_frame_age_s'], 2.05)

    def test_stale_arrival_and_future_sensor_stamp_are_distinct(self):
        stale = pair_timing(90_000_000_000, 100_000_000_000, 50., 50., 2.)
        future = pair_timing(105_000_000_000, 100_000_000_000, 50., 50., 2.)
        self.assertEqual(stale['timing_status'], 'frame_stale_on_arrival')
        self.assertEqual(future['timing_status'], 'sensor_timestamp_ahead_of_receiver')

    def test_delayed_exact_tf_has_separate_bounded_deadline_for_fresh_inputs_only(self):
        def status(arrival, wait):
            return pair_timing(100_000_000_000, int((100+arrival)*1e9), 50., 50.+wait, 2., 6.)
        recovered = status(1.35, 4.34)
        self.assertEqual(recovered['timing_status'], 'within_latency_budget')
        self.assertAlmostEqual(recovered['selected_frame_age_s'], 5.69)
        self.assertEqual(status(2.1, 0)['timing_status'], 'frame_stale_on_arrival')
        self.assertEqual(status(-2.1, 0)['timing_status'], 'sensor_timestamp_ahead_of_receiver')
        self.assertEqual(status(.4, 6.01)['timing_status'], 'timestamped_tf_wait_expired')

    def test_preview_never_evicts_mapping_but_mapping_replaces_preview(self):
        channel = queue.Queue(maxsize=1)
        self.assertTrue(latest_put(channel, dict(preview_only=False, seq=1)))
        self.assertFalse(preview_put(channel, dict(preview_only=True, seq=2)))
        self.assertEqual(channel.get_nowait()['seq'], 1)
        self.assertTrue(preview_put(channel, dict(preview_only=True, seq=2)))
        self.assertTrue(latest_put(channel, dict(preview_only=False, seq=3)))
        self.assertEqual(channel.get_nowait()['seq'], 3)

    def test_real_adapters_require_compressed_inputs(self):
        require_compressed_transport(dict(rgb_compressed=True, depth_compressed=True))
        for field in ('rgb_compressed', 'depth_compressed'):
            with self.assertRaisesRegex(ValueError, 'compressed'):
                require_compressed_transport({field: False})

    def test_repeated_tf_retries_reuse_one_calibrated_payload(self):
        metadata = {}
        payload = dict(rgb=np.zeros((2, 2, 3), np.uint8), depth=np.ones((2, 2), np.float32))
        prepare = Mock(return_value=payload)
        for _ in range(20):
            self.assertIs(cached_rgbd(metadata, prepare), payload)
        prepare.assert_called_once()

    def test_depth_acquisition_preview_needs_neither_rgb_nor_pose(self):
        first = depth_preview_jpeg(np.full((16, 24), 1., np.float32))
        second = depth_preview_jpeg(np.full((16, 24), 4., np.float32))
        self.assertNotEqual(first, second)
        decoded = cv2.imdecode(np.frombuffer(first, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, (16, 24, 3))

    def test_calibrated_model_preview_rejects_missing_registration_tf_before_decode(self):
        rgb = SimpleNamespace(header=SimpleNamespace(frame_id='rgb'), data=b'not decoded')
        depth = SimpleNamespace(header=SimpleNamespace(frame_id='depth'), data=b'not decoded', format='16UC1')
        info = SimpleNamespace(header=SimpleNamespace(frame_id='rgb'))
        depth_info = SimpleNamespace(header=SimpleNamespace(frame_id='depth'))
        robot = dict(depth_registration='auto', rgb_compressed=True, depth_compressed=True)
        lookup = Mock(side_effect=RuntimeError('no calibration transform'))
        with patch('collab_transport.decode_rgb') as decode:
            with self.assertRaisesRegex(RuntimeError, 'calibration transform'):
                prepare_registered_rgbd(rgb, depth, robot, {}, dict(rgb=info, depth=depth_info), lookup)
            decode.assert_not_called()

    def test_same_frame_calibrated_preview_contains_no_invented_pose(self):
        _, rgb_data = cv2.imencode('.jpg', np.full((12, 16, 3), 100, np.uint8))
        _, depth_data = cv2.imencode('.png', np.full((12, 16), 1500, np.uint16))
        header = SimpleNamespace(frame_id='camera')
        rgb = SimpleNamespace(header=header, data=rgb_data.tobytes())
        depth = SimpleNamespace(header=header, data=depth_data.tobytes(), format='16UC1')
        info = SimpleNamespace(header=header, width=16, height=12,
            k=[20., 0., 8., 0., 20., 6., 0., 0., 1.], p=[0.]*12)
        robot = dict(depth_registration='auto', local_frame='odom', depth_scale=.001)
        settings = dict(max_pixels=1000, min_depth_m=.2, max_depth_m=8.)
        lookup = Mock(side_effect=AssertionError('same optical frame needs no extrinsic lookup'))
        payload = prepare_registered_rgbd(rgb, depth, robot, settings, dict(rgb=info), lookup)
        self.assertNotIn('T_local_camera', payload)
        self.assertEqual(payload['depth_registration'], 'same_optical_frame')
        np.testing.assert_allclose(payload['depth'], 1.5)


class SensorEpochConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.
        self.sequence = SensorSequencer(clock=lambda: self.now)
        self.sequence.commit(100_000_000_000)

    def test_one_large_old_pair_rechecked_for_tf_is_not_epoch_evidence(self):
        for _ in range(8):
            self.now += .25
            result = self.sequence.inspect(20_000_000_000)
            self.assertFalse(result['accept'])
            self.assertFalse(result['reset'])
            self.assertEqual(result['confirmations'], 1)

    def test_corroborated_advancing_rollback_opens_new_epoch_only_on_commit(self):
        for index, stamp in enumerate((20_000_000_000, 20_100_000_000, 20_200_000_000)):
            self.now += .25
            result = self.sequence.inspect(stamp)
            self.assertEqual(result['reset'], index == 2)
        self.assertEqual(result['reason'], 'corroborated_sensor_clock_rollback')
        self.assertEqual(self.sequence.last_stamp, 100_000_000_000)
        self.sequence.commit(20_200_000_000)
        self.assertTrue(self.sequence.inspect(20_300_000_000)['accept'])
        self.assertFalse(self.sequence.inspect(20_300_000_000)['reset'])

    def test_healthy_timestamp_commit_cancels_suspected_rollback(self):
        self.sequence.inspect(20_000_000_000)
        self.now = .25
        self.sequence.inspect(20_100_000_000)
        self.sequence.commit(101_000_000_000)
        self.now = .5
        result = self.sequence.inspect(20_200_000_000)
        self.assertFalse(result['reset'])
        self.assertEqual(result['confirmations'], 1)

    def test_clock_rollback_requires_duration_and_bounded_confirmation_window(self):
        self.sequence.inspect(20_000_000_000)
        self.now = .01
        self.sequence.inspect(20_100_000_000)
        self.now = .02
        self.assertFalse(self.sequence.inspect(20_200_000_000)['reset'])
        self.now = 6.
        result = self.sequence.inspect(20_300_000_000)
        self.assertFalse(result['reset'])
        self.assertEqual(result['confirmations'], 1)

    def test_missing_timestamp_does_not_change_frontier(self):
        result = self.sequence.inspect(0)
        self.assertFalse(result['accept'])
        self.assertEqual(self.sequence.last_stamp, 100_000_000_000)


if __name__ == "__main__":
    unittest.main()
