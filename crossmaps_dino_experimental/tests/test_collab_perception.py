"""Fleet tests run real v2 fusion/identity without ROS, weights or a robot."""
import copy
import json
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
            features = np.zeros((2, 2, 4), np.float32)
            features[..., 0] = 1.
            return dict(features=features, meta=dict(input_size=2, resized_w=2, resized_h=2,
                                                     offset_x=0, offset_y=0))
        node.compute_dinov2_dense_features = dense
        node.compute_dinov2_mask_embeddings = lambda masks, dense, return_quality: (
            np.tile([1., 0., 0., 0.], (len(masks), 1)).astype(np.float32), np.ones(len(masks), np.float32))
        node.compute_mask_embeddings_batched = lambda rgb, masks, dense_clip: np.tile(
            [1., 0., 0., 0.], (len(masks), 1)).astype(np.float32)
        # Execute the production camera-space renderers, stubbing only ROS IO.
        del node.publish_esam_mask_overlay
        del node.publish_feature_diagnostics
        node.feature_color_projections = {}
        node.debug_rgb_image_has_subscribers = lambda *args: True
        for attribute, view in {"esam_mask_overlay": "sam", "dinov2_feature_view": "dino",
                "dinov2_similarity_view": "dino_similarity", "dinov2_instance_view": "dino_instances",
                "clip_feature_view": "clip", "clip_similarity_view": "clip_similarity"}.items():
            setattr(node, "pub_" + attribute, view)
            setattr(node, "pub_" + attribute + "_compressed", None)
        def capture(view, compressed_pub, rgb, header, **kwargs):
            stamp = header.stamp.sec * 1_000_000_000 + header.stamp.nanosec
            sink(robot, view, cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2BGR), stamp)
        node.publish_debug_rgb_image = capture
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
        self.assertEqual(self.fleet.get_image("A", "rgb")[1], 2_000_000_000)
        self.assertEqual(self.fleet.get_image("A", "depth")[1], 2_000_000_000)
        self.assertEqual(self.fleet.health()["A"]["session_id"], "test_session")
        self.assertFalse(result["map_integrated"])
        self.assertTrue(result["visual_processed"])

    def test_validated_pose_is_shared_by_local_fusion_and_backend_with_raw_provenance(self):
        original = frame()
        self.fleet.process(original)
        node = self.fleet.nodes[0]
        incoming = frame(seq=2)
        incoming["T_local_camera"][0, 3] = 1.
        corrected = np.eye(4)
        quality = dict(integration_allowed=True, status="visual_rgbd_recovered",
                       mapping_pose_corrected=True, corrected_T_local_camera=corrected.tolist(),
                       external_T_local_camera=incoming["T_local_camera"].tolist())
        with patch.object(self.fleet._guards["A"], "check", return_value=quality), \
                patch.object(node, "apply_semantic_result", wraps=node.apply_semantic_result) as apply:
            result = self.fleet.process(incoming)
        np.testing.assert_allclose(result["T_local_camera"], corrected)
        np.testing.assert_allclose(apply.call_args.args[0]["pts_int"], result["points_camera"])
        self.assertEqual(incoming["T_local_camera"][0, 3], 1.)  # Input/raw source is immutable.
        self.assertEqual(result["pose_quality"]["external_T_local_camera"][0][3], 1.)
        json.dumps(result["pose_quality"], allow_nan=False)

    def test_pose_recovery_config_reaches_guard_and_health(self):
        settings = config()
        settings["pose_guard"] = dict(rgbd_recovery=False)
        fleet = PerceptionFleet(settings, node_factory=self.factory)
        try:
            fleet.process(frame())
            self.assertFalse(fleet._guards["A"].rgbd_recovery)
            self.assertFalse(fleet.health()["A"]["pose_quality"]["rgbd_recovery_enabled"])
        finally:
            fleet.close()

    def test_rejected_pose_keeps_all_2d_diagnostics_live_without_recoloring_map_ids(self):
        initial = frame()
        initial["received_monotonic"] = 100.
        self.fleet.process(initial)
        node = self.fleet.nodes[0]
        identity = copy.deepcopy(node.entity_memory.to_dict())
        counts = copy.deepcopy(node.seg_obs_count)
        for view in ("segments", "entities"):
            self.fleet._put_image("A", view, initial["rgb"], initial["stamp_ns"])
        old_ids = {view: self.fleet.get_image("A", view) for view in ("segments", "entities")}
        incoming = frame(seq=2)
        incoming["received_monotonic"] = 101.
        with patch.object(self.fleet._guards["A"], "check", return_value={
                "integration_allowed": False, "status": "visual_odometry_contradiction"}):
            result = self.fleet.process(incoming)
        for view in ("rgb", "depth", "sam", "dino", "dino_similarity", "dino_instances", "clip", "clip_similarity"):
            self.assertEqual(self.fleet.get_image("A", view, session_id="test_session")[1], 2_000_000_000, view)
            self.assertEqual(self.fleet.image_metadata("A", "test_session")[view], 101., view)
        for view in old_ids:
            self.assertEqual(self.fleet.get_image("A", view), old_ids[view])
            self.assertEqual(self.fleet.image_metadata("A", "test_session")[view], 100.)
        self.assertEqual(identity, node.entity_memory.to_dict())
        self.assertEqual(counts, node.seg_obs_count)
        self.assertEqual(node.last_applied_observation_seq, 1)
        self.assertEqual(len(node.model_inputs), 2)
        self.assertEqual(result["last_integrated_stamp_ns"], 1_000_000_000)
        self.assertEqual(result["last_visual_stamp_ns"], 2_000_000_000)
        self.assertFalse(result.get("preview_only", False))
        self.assertEqual(result["timings_ms"]["fusion_ms"], 0.)

    def test_missing_tf_preview_runs_models_without_seeding_pose_or_world_state(self):
        incoming = frame()
        incoming.update(preview_only=True, T_local_camera=None, received_monotonic=90.)
        result = self.fleet.process(incoming)
        node = self.fleet.nodes[0]
        self.assertTrue(result["preview_only"])
        self.assertFalse(result["integration_allowed"])
        self.assertFalse(result["map_integrated"])
        self.assertFalse(result["pose_quality"]["guard_evaluated"])
        self.assertIsNone(self.fleet._guards["A"].anchor)
        self.assertIsNone(self.fleet._guards["A"].previous_pose)
        self.assertEqual(node.seg_w, {})
        self.assertEqual(node.entity_memory.records, {})
        self.assertEqual(node.last_applied_observation_seq, -1)
        self.assertEqual(result["surfaces"], [])
        self.assertIsNone(self.fleet.get_image("A", "segments"))
        self.assertIsNone(self.fleet.get_image("A", "entities"))
        for view in ("sam", "dino", "dino_similarity", "dino_instances", "clip", "clip_similarity", "depth"):
            self.assertEqual(self.fleet.get_image("A", view)[1], incoming["stamp_ns"])
        self.assertEqual(self.fleet.health()["A"]["session_id"], "test_session")

    def test_same_stamp_preview_promotes_once_and_reuses_inference(self):
        incoming = frame()
        incoming.update(preview_only=True, T_local_camera=None, received_monotonic=90.)
        self.fleet.process(incoming)
        promoted = frame(seq=2)
        promoted.update(stamp_ns=incoming["stamp_ns"], rgb_stamp_ns=incoming["rgb_stamp_ns"], received_monotonic=95.)
        result = self.fleet.process(promoted)
        node = self.fleet.nodes[0]
        self.assertTrue(result["inference_reused"])
        self.assertTrue(result["map_integrated"])
        self.assertEqual(node.seg_obs_count[101], 1)
        self.assertEqual(node.last_applied_observation_seq, 2)
        self.assertEqual(len(node.model_inputs), 1)
        # Arrival of TF must not make the same old camera image look newer.
        self.assertEqual(self.fleet.image_metadata("A", "test_session")["sam"], 90.)
        for preview in (False, True):
            repeated = dict(promoted, seq=3, preview_only=preview)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                self.fleet.process(repeated)
        self.assertEqual(node.seg_obs_count[101], 1)

    def test_promotion_does_not_reuse_different_image_or_calibration(self):
        incoming = frame()
        incoming.update(preview_only=True, T_local_camera=None)
        self.fleet.process(incoming)
        promoted = frame(seq=2)
        promoted.update(stamp_ns=incoming["stamp_ns"], rgb_stamp_ns=incoming["rgb_stamp_ns"])
        promoted["rgb"][:] = [200, 50, 40]
        result = self.fleet.process(promoted)
        self.assertFalse(result["inference_reused"])
        self.assertEqual(len(self.fleet.nodes[0].model_inputs), 2)

    def test_preview_does_not_advance_existing_guard_and_valid_frame_recovers(self):
        self.fleet.process(frame())
        guard = self.fleet._guards["A"]
        old_pose = (guard.previous_pose[0], guard.previous_pose[1].copy())
        incoming = frame(seq=2)
        incoming.update(preview_only=True)
        incoming.pop("T_local_camera")
        with patch.object(guard, "check", side_effect=AssertionError("Preview must not evaluate the pose guard")):
            self.fleet.process(incoming)
        self.assertEqual(guard.previous_pose[0], old_pose[0])
        np.testing.assert_array_equal(guard.previous_pose[1], old_pose[1])
        result = self.fleet.process(frame(seq=3))
        self.assertTrue(result["map_integrated"])
        self.assertEqual(self.fleet.nodes[0].seg_obs_count[101], 2)

    def test_image_session_provenance_does_not_depend_on_current_health(self):
        self.fleet.process(frame())
        self.fleet._health["A"] = {"status": "error"}
        self.assertIsNotNone(self.fleet.get_image("A", "sam", session_id="test_session"))
        self.assertIsNone(self.fleet.get_image("A", "sam", session_id="different_session"))
        self.assertTrue(self.fleet.image_metadata("A", "test_session"))
        self.assertEqual(self.fleet.image_metadata("A", "different_session"), {})
        incoming = frame(session="restarted")
        incoming.update(preview_only=True, T_local_camera=None)
        self.fleet.process(incoming)
        self.assertIsNone(self.fleet.get_image("A", "sam", session_id="test_session"))
        self.assertIsNotNone(self.fleet.get_image("A", "sam", session_id="restarted"))
        self.assertIsNone(self.fleet.get_image("A", "entities", session_id="restarted"))

    def test_missing_pose_requires_explicit_preview_mode(self):
        incoming = frame()
        incoming.pop("T_local_camera")
        with self.assertRaisesRegex(ValueError, "Invalid T_local_camera"):
            self.fleet.process(incoming)
        incoming["preview_only"] = "true"
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            self.fleet.process(incoming)

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

    @classmethod
    def nonplanar_scene(cls):
        rgb, depth, K = cls.textured_scene()
        depth[:, :160], depth[:, 160:320] = 2., 3.
        depth[:, 320:480], depth[:, 480:] = 4., 2.5
        return rgb, depth, K

    @staticmethod
    def yaw(angle_deg):
        angle = math.radians(angle_deg)
        T = np.eye(4)
        T[:2, :2] = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        return T

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

    def test_distributed_nonplanar_rgbd_recovers_only_after_three_fresh_observations(self):
        rgb, depth, K = self.nonplanar_scene()
        guard = PoseConsistencyGuard()
        anchor = self.yaw(7.)
        anchor[:3, 3] = [.7, .2, 1.]
        guard.check(rgb, depth, K, anchor, 1_000_000_000)
        raw = anchor @ self.yaw(18.)
        first = guard.check(rgb, depth, K, raw, 2_000_000_000)
        self.assertFalse(first["integration_allowed"])
        self.assertEqual(first["status"], "rgbd_recovery_pending")
        self.assertEqual(first["recovery_confirmations"], 1)
        duplicate = guard.check(rgb, depth, K, raw, 2_000_000_000)
        self.assertFalse(duplicate["integration_allowed"])
        self.assertEqual(duplicate["recovery_confirmations"], 1)
        self.assertFalse(guard.check(rgb, depth, K, raw, 3_000_000_000)["integration_allowed"])
        recovered = guard.check(rgb, depth, K, raw, 4_000_000_000)
        self.assertTrue(recovered["integration_allowed"])
        self.assertEqual(recovered["status"], "visual_rgbd_recovered")
        np.testing.assert_allclose(recovered["corrected_T_local_camera"], anchor, atol=1e-5)
        np.testing.assert_allclose(recovered["external_T_local_camera"], raw)
        np.testing.assert_allclose(guard.previous_pose[1], anchor, atol=1e-5)
        self.assertEqual(recovered["anchor_stamp_ns"], 1_000_000_000)
        self.assertGreaterEqual(recovered["rgbd_support_cells"], 6)
        self.assertGreater(recovered["rgbd_dense_agreement"], .99)
        self.assertFalse(recovered["is_localization_fix"])
        json.dumps(recovered, allow_nan=False)

    def test_recovery_never_falls_back_to_unvalidated_raw_or_reanchors_when_lost(self):
        rgb, depth, K = self.nonplanar_scene()
        guard = PoseConsistencyGuard()
        guard.check(rgb, depth, K, np.eye(4), 1_000_000_000)
        for seq in (2, 3, 4):
            result = guard.check(rgb, depth, K, self.yaw(30.), seq * 1_000_000_000)
        self.assertTrue(result["integration_allowed"])
        anchor = guard.anchor
        weak = guard.check(np.zeros_like(rgb), depth, K, np.eye(4), 5_000_000_000)
        self.assertFalse(weak["integration_allowed"])
        self.assertIs(guard.anchor, anchor)
        # Even a raw pose inside the normal 5-degree gate must not silently
        # jump away from the already corrected camera/local coordinates.
        for seq in (6, 7, 8):
            near = guard.check(rgb, depth, K, self.yaw(4.), seq * 1_000_000_000)
        self.assertTrue(near["mapping_pose_corrected"])
        self.assertIs(guard.anchor, anchor)
        resumed = guard.check(rgb, depth, K, np.eye(4), 9_000_000_000)
        self.assertTrue(resumed["integration_allowed"])
        self.assertTrue(resumed["resumed_external_pose"])
        self.assertFalse(resumed["rgbd_recovery_active"])
        self.assertNotIn("external_T_local_camera", resumed)
        self.assertNotIn("corrected_T_local_camera", resumed)

    def test_planar_or_disabled_recovery_does_not_override_pose_rejection(self):
        for use_nonplanar, enabled in ((False, True), (True, False)):
            rgb, depth, K = self.nonplanar_scene() if use_nonplanar else self.textured_scene()
            guard = PoseConsistencyGuard(rgbd_recovery=enabled)
            guard.check(rgb, depth, K, np.eye(4), 1_000_000_000)
            for seq in range(2, 6):
                result = guard.check(rgb, depth, K, self.yaw(20.), seq * 1_000_000_000)
                self.assertFalse(result["integration_allowed"])
            if enabled:
                self.assertEqual(result["recovery_reason"], "support_not_distributed_at_multiple_depths")

    def test_recovery_tracks_validated_camera_motion_instead_of_freezing_pose(self):
        rgb, depth, K = self.nonplanar_scene()
        guard = PoseConsistencyGuard()
        guard.check(rgb, depth, K, np.eye(4), 1_000_000_000)
        for seq in (2, 3, 4):
            guard.check(rgb, depth, K, self.yaw(20.), seq * 1_000_000_000)
        camera = self.yaw(3.)
        H = K @ camera[:3, :3].T @ np.linalg.inv(K)
        moved_rgb = cv2.warpPerspective(rgb, H, (640, 480))
        moved_depth = cv2.warpPerspective(depth, H, (640, 480), flags=cv2.INTER_NEAREST)
        result = guard.check(moved_rgb, moved_depth, K, self.yaw(20.), 5_000_000_000)
        self.assertTrue(result["integration_allowed"])
        np.testing.assert_allclose(result["corrected_T_local_camera"], camera, atol=.02)
        self.assertIsNotNone(guard.anchor)
        np.testing.assert_array_equal(guard.anchor["T"], np.eye(4))

    def test_raw_resume_outside_recovery_window_still_requires_strong_support_and_continuity(self):
        guard = PoseConsistencyGuard()
        guard.visual_tracking_active = True
        guard.recovery_hits = 3
        guard.recovery_candidate = (1_000_000_000, np.eye(4))
        camera = self.yaw(18.)
        reference = dict(T=np.eye(4), stamp_ns=1_000_000_000)
        # Registration proof is independently tested above; isolate the source
        # transition here. Raw TF now agrees exactly, with a plausible rate.
        with patch.object(guard, "_recovery_evidence", return_value={"recovery_validated": True}):
            result = guard._recover(reference, {}, [], [], np.linalg.inv(camera), None, None, camera,
                                    2_000_000_000, dict(integration_allowed=True))
        self.assertTrue(result["integration_allowed"])
        self.assertTrue(result["resumed_external_pose"])
        self.assertFalse(guard.visual_tracking_active)
        self.assertNotIn("corrected_T_local_camera", result)

    def test_recovered_pose_cannot_be_chained_beyond_trusted_anchor_window(self):
        rgb, depth, K = self.nonplanar_scene()
        guard = PoseConsistencyGuard()
        guard.check(rgb, depth, K, np.eye(4), 1_000_000_000)
        for seq in (2, 3, 4):
            guard.check(rgb, depth, K, self.yaw(20.), seq * 1_000_000_000)
        # Project actual camera rotation about the optical axis; depth follows
        # the same homography. Recovery must stop, not create a new VO anchor.
        camera = self.yaw(18.)
        H = K @ camera[:3, :3].T @ np.linalg.inv(K)
        moved_rgb = cv2.warpPerspective(rgb, H, (640, 480))
        moved_depth = cv2.warpPerspective(depth, H, (640, 480), flags=cv2.INTER_NEAREST)
        result = guard.check(moved_rgb, moved_depth, K, self.yaw(40.), 5_000_000_000)
        self.assertFalse(result["integration_allowed"])
        np.testing.assert_array_equal(guard.anchor["T"], np.eye(4))
        self.assertTrue(guard.visual_tracking_active)

    def test_moving_foreground_fit_cannot_override_conflicting_dense_background(self):
        rgb, depth, K = self.nonplanar_scene()
        guard = PoseConsistencyGuard()
        reference = guard._features(rgb, depth, K, np.eye(4))
        current = guard._features(rgb, depth, K, np.eye(4))
        # Give PnP an apparently perfect, broadly distributed moving foreground
        # fit, but leave independent background RGB-D unchanged. Feature count
        # and depth fit alone must not authorize a camera movement.
        u, v = np.meshgrid(np.linspace(60., 575., 15), np.linspace(50., 425., 10))
        z = np.where(u < 320., 2., 4.)
        xyz = np.column_stack(((u.ravel()-320.)*z.ravel()/300.,
                               (v.ravel()-240.)*z.ravel()/300., z.ravel()))
        visual = np.eye(4)
        visual[2, 3] = .3
        moved = xyz + visual[:3, 3]
        reference["points"] = xyz
        current["points"] = moved
        current["pixels"] = moved[:, :2]/moved[:, 2, None]*[300., 300.]+[320., 240.]
        matches = [SimpleNamespace(queryIdx=i, trainIdx=i) for i in range(len(xyz))]
        evidence = guard._recovery_evidence(reference, current, matches, np.arange(len(xyz)), visual, depth, K)
        self.assertFalse(evidence["recovery_validated"])
        self.assertEqual(evidence["recovery_reason"], "background_or_dense_geometry_conflict")


if __name__ == "__main__":
    unittest.main()
