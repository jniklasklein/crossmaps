"""Fleet tests run real v2 fusion/identity without ROS, weights or a robot."""
import copy
import math
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from collab_perception import (PerceptionFleet, PoseConsistencyGuard, SHARED_ATTRIBUTES,
                              depth_points, pooled_dino_descriptor, validate_observation)
from test_node_identity_integration import make_node


def frame(robot="A", seq=1, session="test_session"):
    # The real fusion fixture uses positive local XY and a 5 cm voxel grid.
    rgb = np.full((16, 24, 3), (10, 70, 230), np.uint8)
    return dict(robot_id=robot, session_id=session, seq=seq, stamp_ns=seq * 1_000_000_000,
                rgb_stamp_ns=seq * 1_000_000_000, calibration_id="calib_1", rgb=rgb,
                depth=np.full(rgb.shape[:2], .525, np.float32),
                K=np.array([[10.5, 0, -2.5], [0, 10.5, -2.5], [0, 0, 1.]]),
                T_local_camera=np.eye(4))


def config():
    return dict(robots=[dict(id="A", domain=0), dict(id="B", domain=1)],
                perception=dict(parameters={"stride": 1, "entity_birth_observations": 2}))


class Factory:
    def __init__(self):
        self.created = []

    def __call__(self, robot, parameters, shared, context, sink):
        node = make_node()
        node.params.update(parameters)
        node.params["dinov2_context_contrast_weight"] = 0.
        node.params["segment_mask_interior_px"] = 0
        node.external_inputs = True
        node.device = "test_cpu"
        node.stride = 1
        node.inference_lock = threading.Lock()
        node.model = object()
        node.preprocess = object()
        node.tokenizer = object()
        node.esam_model = object()
        node.dinov2_model = object()
        node.dinov2_patch_size = 14
        if shared:
            for name in SHARED_ATTRIBUTES:
                setattr(node, name, shared[name])
        node.destroyed = False
        node.destroy_node = lambda: setattr(node, "destroyed", True)
        node.model_inputs = []
        def masks(rgb):
            node.model_inputs.append(rgb.copy())
            return [np.ones(rgb.shape[:2], bool)], np.ones(1, np.float32)
        node.segment_esam_masks = masks
        def dense(rgb):
            features = np.zeros((2, 2, 2), np.float32)
            features[..., 0] = 1.
            return dict(features=features, meta=dict(input_size=2, resized_w=2, resized_h=2,
                                                     offset_x=0, offset_y=0))
        node.compute_dinov2_dense_features = dense
        node.compute_dinov2_mask_embeddings = lambda masks, dense, return_quality: (
            np.tile([1., 0.], (len(masks), 1)).astype(np.float32), np.ones(len(masks), np.float32))
        node.compute_mask_embeddings_batched = lambda rgb, masks, dense_clip: np.tile(
            [1., 0.], (len(masks), 1)).astype(np.float32)
        self.created.append(node)
        return node


class FleetTests(unittest.TestCase):
    def setUp(self):
        self.factory = Factory()
        self.fleet = PerceptionFleet(config(), node_factory=self.factory)

    def tearDown(self):
        self.fleet.close()

    def test_lazily_shares_models_but_not_surface_or_entity_state(self):
        self.assertEqual(self.fleet.nodes, ())
        for seq in range(1, 5):
            a = self.fleet.process(frame("A", seq))
            b = self.fleet.process(frame("B", seq))
        left, right = self.fleet.nodes
        self.assertIs(left.model, right.model)
        self.assertIs(left.esam_model, right.esam_model)
        self.assertIs(left.dinov2_model, right.dinov2_model)
        self.assertIs(left.inference_lock, right.inference_lock)
        self.assertIsNot(left.entity_memory, right.entity_memory)
        self.assertIsNot(left.seg_w, right.seg_w)
        self.assertNotEqual(left.world_model_id, right.world_model_id)
        self.assertEqual(left.seg_obs_count[101], 4)
        self.assertEqual(right.seg_obs_count[101], 4)
        self.assertEqual(a["surfaces"][0]["entity_id"], "1")
        self.assertEqual(b["surfaces"][0]["entity_id"], "1")
        self.assertEqual(a["feature_space_id"], b["feature_space_id"])
        self.assertEqual(self.fleet.health()["A"]["status"], "processed")

    def test_bgr_contract_and_rgb_model_point_overlay_colors(self):
        original = frame()
        result = self.fleet.process(original)
        node = self.fleet.nodes[0]
        np.testing.assert_array_equal(node.model_inputs[0][0, 0], [230, 70, 10])
        np.testing.assert_array_equal(node.last_assignment_rgb[0, 0], [230, 70, 10])
        np.testing.assert_array_equal(result["colors"][0], [230, 70, 10])
        np.testing.assert_array_equal(result["rgb"], original["rgb"])
        encoded, stamp = self.fleet.get_image("A", "rgb")
        decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_allclose(decoded[0, 0], original["rgb"][0, 0], atol=3)
        self.assertEqual(stamp, original["stamp_ns"])

    def test_duplicate_cannot_update_overlay_or_accumulate_evidence(self):
        self.fleet.process(frame())
        snapshot = self.fleet.get_image("A", "rgb")
        node = self.fleet.nodes[0]
        before = copy.deepcopy(node.seg_obs_count)
        duplicate = frame()
        duplicate["rgb"][:] = 0
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.fleet.process(duplicate)
        self.assertEqual(snapshot, self.fleet.get_image("A", "rgb"))
        self.assertEqual(before, node.seg_obs_count)
        old_stamp = frame(seq=2)
        old_stamp["stamp_ns"] = 1_000_000_000
        with self.assertRaisesRegex(ValueError, "out-of-order"):
            self.fleet.process(old_stamp)

    def test_session_restart_reuses_models_but_starts_fresh_identity(self):
        first = self.fleet.process(frame())
        node = self.fleet.nodes[0]
        second = self.fleet.process(frame(session="restarted"))
        replacement = self.fleet.nodes[0]
        self.assertTrue(node.destroyed)
        self.assertIs(node.model, replacement.model)
        self.assertNotEqual(first["local_world_id"], second["local_world_id"])
        self.assertEqual(replacement.seg_obs_count[101], 1)

    def test_no_masks_export_no_historical_or_display_support(self):
        self.fleet.process(frame())
        node = self.fleet.nodes[0]
        node.segment_esam_masks = lambda rgb: ([], np.zeros(0, np.float32))
        result = self.fleet.process(frame(seq=2))
        self.assertEqual(result["surfaces"], [])
        self.assertEqual(node.seg_obs_count[101], 1)
        self.assertEqual(result["segment_count"], 1)

    def test_geometry_is_current_camera_frame_not_accumulated_local_map(self):
        observation = frame()
        observation["T_local_camera"][0, 3] = 1.
        result = self.fleet.process(observation)
        expected, _, _ = depth_points(observation["depth"], observation["K"], 1)
        np.testing.assert_allclose(result["points_camera"], expected)
        surface = result["surfaces"][0]
        self.assertEqual(surface["observation_support"], "accepted_current_depth")
        self.assertLessEqual(float(surface["points_camera"][:, 0].max()), float(expected[:, 0].max()))

    def test_guard_rejection_does_not_touch_local_memory(self):
        self.fleet.process(frame())
        node = self.fleet.nodes[0]
        before = copy.deepcopy(node.seg_obs_count)
        with patch.object(self.fleet._guards["A"], "check", return_value={
                "integration_allowed": False, "status": "visual_odometry_contradiction"}):
            result = self.fleet.process(frame(seq=2))
        self.assertFalse(result["integration_allowed"])
        self.assertEqual(before, node.seg_obs_count)
        self.assertEqual(result["surfaces"], [])
        self.assertEqual(self.fleet.get_image("A", "rgb")[1], 1_000_000_000)

    def test_bad_registration_and_reflected_transform_rejected_before_loading(self):
        bad = frame()
        bad["depth"] = bad["depth"][:, :-1]
        with self.assertRaisesRegex(ValueError, "registered"):
            self.fleet.process(bad)
        bad = frame()
        bad["T_local_camera"][0, 0] = -1
        with self.assertRaisesRegex(ValueError, "proper rigid"):
            self.fleet.process(bad)
        self.assertEqual(self.fleet.nodes, ())

    def test_incompatible_model_parameters_rejected(self):
        settings = config()
        settings["robots"][1]["parameters"] = {"clip_model_name": "ViT-B-32"}
        with self.assertRaisesRegex(ValueError, "same shared model"):
            PerceptionFleet(settings, node_factory=self.factory)

    def test_missing_required_models_fail_explicitly_before_fusion(self):
        self.fleet.process(frame())
        node = self.fleet.nodes[0]
        before = copy.deepcopy(node.seg_obs_count)
        node.dinov2_model = None
        with self.assertRaisesRegex(RuntimeError, "DINO model failed to load"):
            self.fleet.process(frame(seq=2))
        self.assertEqual(before, node.seg_obs_count)
        self.assertEqual(self.fleet.health()["A"]["status"], "error")

    def test_feature_preprocessing_changes_have_different_spaces(self):
        settings = config()
        settings["robots"][1]["parameters"] = {"dinov2_input_size": 700}
        fleet = PerceptionFleet(settings, node_factory=self.factory)
        try:
            a = fleet.process(frame("A"))
            b = fleet.process(frame("B"))
            self.assertNotEqual(a["feature_space_id"], b["feature_space_id"])
        finally:
            fleet.close()


class DescriptorAndPoseTests(unittest.TestCase):
    @staticmethod
    def textured_scene():
        rng = np.random.default_rng(31)
        texture = rng.integers(0, 256, (480, 640), dtype=np.uint8)
        rgb = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
        depth = np.full(texture.shape, 3., np.float32)
        K = np.array([[300., 0, 320.], [0, 300., 240.], [0, 0, 1.]])
        return rgb, depth, K

    def test_global_descriptor_excludes_letterbox_padding(self):
        features = np.array([[[0., 1.], [0., 1.]], [[1., 0.], [1., 0.]]], np.float32)
        dense = dict(features=features, meta=dict(input_size=2, resized_w=2, resized_h=1,
                                                offset_x=0, offset_y=1))
        np.testing.assert_allclose(pooled_dino_descriptor(dense), [1., 0.])
        self.assertEqual(pooled_dino_descriptor(None).size, 0)

    def test_pose_guard_does_not_lock_legitimate_motion(self):
        reference, current = np.eye(4), np.eye(4)
        current[0, 3] = 1.
        visual = np.linalg.inv(current)
        result = PoseConsistencyGuard.compare(reference, current, visual)
        self.assertTrue(result["integration_allowed"])
        self.assertFalse(result["is_localization_fix"])

    def test_stationary_visual_evidence_detects_accumulated_yaw_error(self):
        current = np.eye(4)
        angle = math.radians(12.)
        current[:2, :2] = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        result = PoseConsistencyGuard.compare(np.eye(4), current, np.eye(4))
        self.assertFalse(result["integration_allowed"])
        self.assertAlmostEqual(result["rotation_disagreement_deg"], 12.)

    def test_real_orb_depth_pnp_catches_slow_cumulative_yaw_drift(self):
        rgb, depth, K = self.textured_scene()
        guard = PoseConsistencyGuard(rotation_deg=5.)
        states = []
        for seq, angle_deg in enumerate((0, 2, 4, 6, 8), 1):
            angle = math.radians(angle_deg)
            T = np.eye(4)
            T[:2, :2] = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
            states.append(guard.check(rgb, depth, K, T, seq * 1_000_000_000))
        self.assertTrue(states[1]["integration_allowed"])
        self.assertEqual(states[1]["status"], "visual_pose_consistent")
        self.assertFalse(states[3]["integration_allowed"])
        self.assertFalse(states[4]["integration_allowed"])
        np.testing.assert_array_equal(guard.anchor["T"], np.eye(4))
        blank = guard.check(np.zeros_like(rgb), depth, K, T, 6_000_000_000)
        self.assertFalse(blank["integration_allowed"])
        self.assertEqual(blank["status"], "pose_revalidation_required")
        recovered = guard.check(rgb, depth, K, np.eye(4), 7_000_000_000)
        self.assertTrue(recovered["integration_allowed"])

    def test_real_camera_translation_updates_reference_without_pose_lock(self):
        rgb, depth, K = self.textured_scene()
        guard = PoseConsistencyGuard()
        guard.check(rgb, depth, K, np.eye(4), 1_000_000_000)
        # A plane at 3 m viewed after a +.5 m lateral camera movement shifts
        # -50 px at fx=300. Its depth remains 3 m.
        moved = cv2.warpAffine(rgb, np.array([[1., 0., -50.], [0., 1., 0.]]), (640, 480))
        T = np.eye(4)
        T[0, 3] = .5
        result = guard.check(moved, depth, K, T, 2_000_000_000)
        self.assertEqual(result["status"], "visual_pose_consistent")
        self.assertTrue(result["integration_allowed"])
        np.testing.assert_array_equal(guard.anchor["T"], T)

    def test_disabled_guard_and_pose_jump_have_explicit_status(self):
        rgb, depth, K = self.textured_scene()
        disabled = PoseConsistencyGuard(enabled=False)
        result = disabled.check(rgb, depth, K, np.eye(4), 1)
        self.assertEqual(result["status"], "external_tf_not_validated")
        self.assertFalse(result["guard_enabled"])
        guard = PoseConsistencyGuard(max_speed_m_s=2.)
        guard.check(rgb, depth, K, np.eye(4), 1_000_000_000)
        T = np.eye(4)
        T[0, 3] = 10.
        result = guard.check(rgb, depth, K, T, 2_000_000_000)
        self.assertEqual(result["status"], "external_pose_discontinuity")
        self.assertFalse(result["integration_allowed"])


if __name__ == "__main__":
    unittest.main()
