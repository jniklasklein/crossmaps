"""Shared inference, isolated v2 surface/entity memory for collaborative mapping.

ROS and the heavy model stack are imported lazily. The standalone v2 node is
unchanged unless its external-input constructor is explicitly requested.
Arrays crossing this interface use OpenCV BGR images, RGB point colors and
metres in the calibrated optical-camera frame. No robot TF is published here.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np


MODEL_KEYS = (
    "clip_model_name", "clip_pretrained", "esam_variant", "esam_checkpoint",
    "use_esam_masks", "use_dinov2_features", "dinov2_model_name", "dinov2_repo",
    "dinov2_local_repo", "dinov2_checkpoint", "dinov2_allow_download",
)
FEATURE_KEYS = MODEL_KEYS + (
    "dinov2_input_size", "dinov2_mask_min_weight", "dinov2_context_ring_tokens",
    "dinov2_context_contrast_weight", "clip_mask_embedding_variants", "clip_context_margin_frac",
    "clip_context_bg_alpha", "clip_mask_min_crop_px",
)
SHARED_ATTRIBUTES = (
    "model", "preprocess", "tokenizer", "device", "esam_model",
    "dinov2_model", "dinov2_patch_size", "inference_lock",
)
VIEW_TOPICS = {
    "rgb": "debug/last_observation_rgb", "sam": "esam/mask_overlay",
    "segments": "debug/segment_id_projection", "entities": "debug/entity_id_projection",
    "dino": "dino/feature_view", "dino_similarity": "dino/similarity_view",
    "dino_instances": "dino/instance_view", "clip": "clip/feature_view",
    "clip_similarity": "clip/similarity_view", "depth": "debug/last_observation_depth",
}
VIEW_ALIASES = {"dino_sim": "dino_similarity", "dino_inst": "dino_instances", "clip_sim": "clip_similarity"}


def normalize_descriptor(value):
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    if not value.size or not np.isfinite(value).all():
        return np.zeros(0, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-8 else np.zeros(0, dtype=np.float32)


def pooled_dino_descriptor(dense):
    """Pool real-image tokens only; letterbox padding is not place evidence."""
    if not dense:
        return np.zeros(0, dtype=np.float32)
    features = np.asarray(dense.get("features", []), dtype=np.float32)
    if features.ndim != 3 or not features.size:
        return np.zeros(0, dtype=np.float32)
    meta = dense.get("meta", {})
    side = int(meta.get("input_size", features.shape[0]))
    run_w, run_h = int(meta.get("resized_w", side)), int(meta.get("resized_h", side))
    off_x, off_y = int(meta.get("offset_x", 0)), int(meta.get("offset_y", 0))
    canvas = np.zeros((side, side), np.float32)
    canvas[off_y:off_y + run_h, off_x:off_x + run_w] = 1.
    weights = cv2.resize(canvas, (features.shape[1], features.shape[0]), interpolation=cv2.INTER_AREA)
    return normalize_descriptor(np.sum(features * weights[..., None], axis=(0, 1)))


def validate_observation(observation):
    """Reject bad input before changing any persistent state or camera output."""
    for key in ("robot_id", "session_id", "seq", "stamp_ns", "calibration_id"):
        if key not in observation:
            raise ValueError(f"Observation missing {key}")
    if not observation["session_id"] or not observation["calibration_id"]:
        raise ValueError("Session and calibration identifiers must be nonempty")
    if int(observation["seq"]) < 0 or int(observation["stamp_ns"]) <= 0:
        raise ValueError("Observation sequence/stamp must be valid")
    rgb = np.asarray(observation["rgb"])
    depth = np.asarray(observation["depth"], dtype=np.float32)
    K = np.asarray(observation["K"], dtype=np.float64).reshape(3, 3)
    preview_only = observation.get("preview_only", False)
    if not isinstance(preview_only, bool):
        raise ValueError("preview_only must be an explicit boolean")
    transform = observation.get("T_local_camera")
    T = None if transform is None and preview_only else np.asarray(transform, dtype=np.float64)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be HxWx3 BGR uint8")
    if depth.shape != rgb.shape[:2]:
        raise ValueError("Depth must already be registered to RGB pixel coordinates")
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid camera intrinsics")
    if T is None:
        return rgb, depth, K, T
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("Invalid T_local_camera")
    if (not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=2e-3)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1., atol=2e-3)):
        raise ValueError("T_local_camera must be a proper rigid transform")
    return rgb, depth, K, T


def depth_points(depth, K, stride=8, min_depth=.2, max_depth=8.):
    vv, uu = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[vv, uu]
    good = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
    uu, vv, z = uu[good], vv[good], z[good]
    points = np.column_stack(((uu - K[0, 2]) * z / K[0, 0],
                              (vv - K[1, 2]) * z / K[1, 1], z)).astype(np.float32)
    return points, uu.astype(np.int32), vv.astype(np.int32)


def _rotation_angle(T):
    return math.degrees(math.acos(float(np.clip((np.trace(T[:3, :3]) - 1.) / 2., -1., 1.))))


class PoseConsistencyGuard:
    """Conservative contradiction detector, NOT odometry or a stationary lock.

    A retained RGB-D anchor makes accumulated yaw drift detectable while the
    scene remains visible. Recovery is bounded registration to that anchor,
    never an unchecked visual-odometry chain or a stationary pose substitute.
    """
    def __init__(self, rotation_deg=5., translation_m=.15, *, enabled=True,
                 max_speed_m_s=2., max_yaw_rate_deg_s=150., rgbd_recovery=True):
        self.rotation_deg = float(rotation_deg)
        self.translation_m = float(translation_m)
        self.enabled = bool(enabled)
        self.max_speed_m_s = float(max_speed_m_s)
        self.max_yaw_rate_deg_s = float(max_yaw_rate_deg_s)
        self.rgbd_recovery = bool(rgbd_recovery)
        self.anchor = None
        self.previous_pose = None
        self.previous_external_pose = None
        self.revalidation_required = False
        self.visual_tracking_active = False
        self.recovery_hits = 0
        self.recovery_candidate = None
        self.orb = cv2.ORB_create(nfeatures=900)

    def _features(self, rgb, depth, K, T):
        scale = min(1., 640. / max(rgb.shape[:2]))
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        if scale != 1.:
            gray = cv2.resize(gray, None, fx=scale, fy=scale)
        keys, descriptors = self.orb.detectAndCompute(gray, None)
        pixels = np.asarray([k.pt for k in keys], dtype=np.float32).reshape(-1, 2) / scale
        pix = np.rint(pixels).astype(int)
        if len(pix):
            pix[:, 0] = np.clip(pix[:, 0], 0, depth.shape[1] - 1)
            pix[:, 1] = np.clip(pix[:, 1], 0, depth.shape[0] - 1)
        z = depth[pix[:, 1], pix[:, 0]]
        points = np.column_stack(((pixels[:, 0] - K[0, 2]) * z / K[0, 0],
                                  (pixels[:, 1] - K[1, 2]) * z / K[1, 1], z))
        dense, u, v = depth_points(depth, K, max(8, int(math.ceil(max(depth.shape)/64))))
        smooth = cv2.GaussianBlur(gray, (5, 5), 0)
        values = cv2.remap(smooth, (u*scale).astype(np.float32).reshape(-1, 1),
                           (v*scale).astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR).reshape(-1) if len(u) else np.zeros(0)
        return dict(pixels=pixels, descriptors=descriptors, points=points,
                    T=T.copy(), K=K.copy(), shape=depth.shape, dense_points=dense,
                    dense_gray=values, gray=smooth, scale=scale)

    @staticmethod
    def _recovery_evidence(reference, current, matches, ids, visual, depth, K):
        """Independent 3D and dense hold-out checks; PnP count is not enough."""
        evidence = {"recovery_validated": False}
        if (len(ids) < 60 or len(ids)/max(1, len(matches)) < .7
                or len({matches[i].trainIdx for i in ids}) < 60):
            return {**evidence, "recovery_reason": "insufficient_correspondences"}
        a = np.asarray([reference["points"][matches[i].queryIdx] for i in ids])
        b = np.asarray([current["points"][matches[i].trainIdx] for i in ids])
        uv = np.asarray([current["pixels"][matches[i].trainIdx] for i in ids])
        transformed = a @ visual[:3, :3].T + visual[:3, 3]
        errors = np.linalg.norm(transformed-b, axis=1)
        valid = np.isfinite(b).all(axis=1) & (b[:, 2] > .2) & (b[:, 2] < 8.)
        if np.count_nonzero(valid) < 60:
            return {**evidence, "recovery_reason": "insufficient_independent_depth"}
        tolerance = .06 + .02*float(np.median(b[valid, 2]))
        good = valid & (errors < tolerance)
        evidence.update(rgbd_depth_support_fraction=float(np.mean(good)),
                        rgbd_median_error_m=float(np.median(errors[valid])))
        if np.count_nonzero(good) < 60 or np.mean(good) < .75 or np.median(errors[good]) > .045:
            return {**evidence, "recovery_reason": "3d_correspondence_conflict"}
        xyz, pixels = a[good], uv[good]
        extent = np.ptp(pixels, axis=0)/[depth.shape[1], depth.shape[0]]
        bins = np.floor(pixels/[depth.shape[1], depth.shape[0]]*[4, 3]).astype(int)
        cells = len(set(map(tuple, bins)))
        singular = np.linalg.svd(xyz-xyz.mean(axis=0), compute_uv=False)
        ratio = float(singular[-1]/max(1e-8, singular[0]))
        depth_range = float(np.quantile(xyz[:, 2], .9)-np.quantile(xyz[:, 2], .1))
        evidence.update(rgbd_support_cells=cells, rgbd_support_span_fraction=extent.tolist(), rgbd_depth_range_m=depth_range,
                        rgbd_nonplanarity_ratio=ratio)
        if extent[0] < .65 or extent[1] < .4 or cells < 6 or depth_range < .25 or ratio < .02:
            return {**evidence, "recovery_reason": "support_not_distributed_at_multiple_depths"}
        # Validate on a regular depth grid not used by the feature fit. This
        # rejects a moving textured foreground whose background disagrees.
        dense = reference["dense_points"]
        projected = dense @ visual[:3, :3].T + visual[:3, 3]
        positive = projected[:, 2] > .2
        uv_dense = projected[:, :2]/np.maximum(projected[:, 2, None], .01)*[K[0, 0], K[1, 1]]+[K[0, 2], K[1, 2]]
        rounded = np.rint(uv_dense).astype(int)
        inside = positive & (rounded[:, 0] >= 0) & (rounded[:, 0] < depth.shape[1]) & (rounded[:, 1] >= 0) & (rounded[:, 1] < depth.shape[0])
        selected = np.flatnonzero(inside)
        measured = depth[rounded[selected, 1], rounded[selected, 0]]
        good_depth = np.isfinite(measured) & (measured > .2) & (measured < 8.)
        selected, measured = selected[good_depth], measured[good_depth]
        if len(selected) < 200 or len(selected)/max(1, len(dense)) < .6:
            return {**evidence, "recovery_reason": "insufficient_dense_overlap"}
        depth_agrees = np.abs(measured-projected[selected, 2]) < .08+.015*measured
        sampled = cv2.remap(current["gray"], (uv_dense[selected, 0]*current["scale"]).astype(np.float32).reshape(-1, 1),
                            (uv_dense[selected, 1]*current["scale"]).astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR).reshape(-1)
        difference = sampled.astype(float)-reference["dense_gray"][selected].astype(float)
        difference -= np.median(difference)  # Global exposure offset is not motion.
        photo_agrees = np.abs(difference) < 25.
        joint = float(np.mean(depth_agrees & photo_agrees))
        evidence.update(rgbd_dense_samples=len(selected), rgbd_dense_agreement=joint,
                        rgbd_photometric_median=float(np.median(np.abs(difference))))
        if joint < .70 or np.mean(photo_agrees) < .85:
            return {**evidence, "recovery_reason": "background_or_dense_geometry_conflict"}
        return {**evidence, "recovery_validated": True, "recovery_reason": "distributed_rgbd_and_background_support"}

    def _recover(self, reference, current, matches, ids, visual, depth, K, T, stamp_ns, result):
        evidence = self._recovery_evidence(reference, current, matches, ids, visual, depth, K)
        result.update(evidence)
        candidate = reference["T"] @ np.linalg.inv(visual)
        correction = np.linalg.inv(T) @ candidate
        resume_external = (self.visual_tracking_active and result["recovery_validated"]
                           and not result.get("external_pose_discontinuity", False)
                           and _rotation_angle(correction) <= .5
                           and np.linalg.norm(correction[:3, 3]) <= .02)
        # Bounded local registration, not chained VO: preserve the trusted
        # anchor and stop when it is no longer sufficiently observable. A raw
        # pose independently revalidated against this anchor may resume, with
        # the same continuity check below, without starting a corrected chain.
        if not resume_external and (np.linalg.norm(visual[:3, 3]) > .4 or _rotation_angle(visual) > 15.):
            result.update(recovery_validated=False, recovery_reason="outside_trusted_anchor_window")
        if not result["recovery_validated"] or stamp_ns is None:
            self.recovery_hits, self.recovery_candidate = 0, None
            result.update(integration_allowed=False, recovery_confirmations=0)
            return result
        stamp_ns = int(stamp_ns)
        old = self.recovery_candidate
        if old is not None and stamp_ns <= old[0]:
            result.update(integration_allowed=False, recovery_reason="duplicate_recovery_observation",
                          recovery_confirmations=self.recovery_hits)
            return result
        continuous = True
        if old is not None:
            dt = (stamp_ns-old[0])*1e-9
            motion = np.linalg.inv(old[1]) @ candidate
            continuous = (dt <= 30. and np.linalg.norm(motion[:3, 3]) <= .03+self.max_speed_m_s*dt
                          and _rotation_angle(motion) <= 1.+self.max_yaw_rate_deg_s*dt)
        self.recovery_hits = min(3, self.recovery_hits+1) if old is not None and continuous else 1
        self.recovery_candidate = (stamp_ns, candidate)
        result["recovery_confirmations"] = self.recovery_hits
        if self.recovery_hits < 3:
            result.update(integration_allowed=False, status="rgbd_recovery_pending")
            return result
        # Do not switch between a corrected pose and a raw pose merely because
        # raw odometry falls back inside the much looser contradiction gate.
        # Return only after the same independent evidence supports near-equality
        # and the raw source itself has no abrupt discontinuity this frame.
        if resume_external:
            self.visual_tracking_active = False
            self.recovery_hits, self.recovery_candidate = 0, None
            result.update(integration_allowed=True, status="visual_pose_consistent",
                          pose_source="external_tf_rgbd_revalidated", recovery_confirmations=0,
                          resumed_external_pose=True)
            return result
        self.visual_tracking_active = True
        result.update(integration_allowed=True, status="visual_rgbd_recovered",
                      pose_source="validated_rgbd_trusted_anchor", mapping_pose_corrected=True,
                      external_T_local_camera=T.tolist(), corrected_T_local_camera=candidate.tolist(),
                      anchor_stamp_ns=reference.get("stamp_ns"),
                      correction_rotation_deg=_rotation_angle(correction),
                      correction_translation_m=float(np.linalg.norm(correction[:3, 3])))
        return result

    @staticmethod
    def compare(reference_T, current_T, visual_current_reference, rotation_deg=8., translation_m=.25):
        expected = np.linalg.inv(current_T) @ reference_T
        difference = np.linalg.inv(visual_current_reference) @ expected
        rotation, translation = _rotation_angle(difference), float(np.linalg.norm(difference[:3, 3]))
        consistent = rotation <= rotation_deg and translation <= translation_m
        return {"integration_allowed": consistent,
                "status": "visual_pose_consistent" if consistent else "visual_odometry_contradiction",
                "rotation_disagreement_deg": rotation, "translation_disagreement_m": translation,
                "is_localization_fix": False}

    def check(self, rgb, depth, K, T, stamp_ns=None):
        unvalidated = {"integration_allowed": True, "status": "external_tf_not_validated",
                       "is_localization_fix": False, "guard_enabled": self.enabled,
                       "rgbd_recovery_enabled": self.rgbd_recovery}
        if not self.enabled:
            return unvalidated
        blocked = self.revalidation_required or self.visual_tracking_active
        if blocked:
            unvalidated.update(integration_allowed=False, status="pose_revalidation_required")
        discontinuity = {}
        if stamp_ns is not None and self.previous_external_pose is not None:
            previous_stamp, previous_T = self.previous_external_pose
            dt = (int(stamp_ns) - previous_stamp) * 1e-9
            if dt > 0.:
                delta = np.linalg.inv(previous_T) @ T
                speed = float(np.linalg.norm(delta[:3, 3])) / dt
                rotation_rate = _rotation_angle(delta) / dt
                if speed > self.max_speed_m_s or rotation_rate > self.max_yaw_rate_deg_s:
                    self.revalidation_required = True
                    blocked = True
                    discontinuity = {"external_pose_discontinuity": True, "speed_m_s": speed,
                                     "rotation_rate_deg_s": rotation_rate}
                    unvalidated.update(integration_allowed=False, status="external_pose_discontinuity", **discontinuity)
        if stamp_ns is not None and (self.previous_external_pose is None or int(stamp_ns) > self.previous_external_pose[0]):
            self.previous_external_pose = int(stamp_ns), T.copy()
        current = self._features(rgb, depth, K, T)
        current["stamp_ns"] = None if stamp_ns is None else int(stamp_ns)
        def accepted(value):
            value["rgbd_recovery_active"] = self.visual_tracking_active
            value.setdefault("recovery_confirmations", self.recovery_hits)
            if stamp_ns is not None and value["integration_allowed"]:
                pose = np.asarray(value["corrected_T_local_camera"]) if value.get("mapping_pose_corrected") else T
                self.previous_pose = int(stamp_ns), pose.copy()
            return value
        def unknown():
            # A weak intervening frame cannot contribute to three consecutive
            # independent confirmations or release a previously latched guard.
            self.recovery_hits, self.recovery_candidate = 0, None
            return accepted(unvalidated)
        if self.anchor is None:
            self.anchor = current
            return accepted(unvalidated)
        reference = self.anchor
        if reference["descriptors"] is None:
            if not blocked:
                self.anchor = current
            return unknown()
        if current["descriptors"] is None:
            return unknown()
        if reference["shape"] != current["shape"] or not np.allclose(reference["K"], K):
            if not blocked:
                self.anchor = current
            return unknown()
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(reference["descriptors"], current["descriptors"], k=2)
        matches = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < .72 * pair[1].distance]
        matches = [m for m in matches if np.isfinite(reference["points"][m.queryIdx]).all()
                   and .2 < reference["points"][m.queryIdx, 2] < 8.]
        if len(matches) < 24:
            # A changed viewpoint can require a new anchor. Do not reset merely
            # because a valid stationary contradiction has been observed.
            if not blocked:
                self.anchor = current
            return unknown()
        xyz = np.asarray([reference["points"][m.queryIdx] for m in matches], np.float64)
        uv = np.asarray([current["pixels"][m.trainIdx] for m in matches], np.float64)
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(xyz, uv, K, None, iterationsCount=150,
                    reprojectionError=2.5, confidence=.995, flags=cv2.SOLVEPNP_EPNP)
        except cv2.error:
            return unknown()
        if not ok or inliers is None or len(inliers) < 24 or len(inliers) / len(matches) < .55:
            return unknown()
        ids = inliers.reshape(-1)
        spread = np.ptp(uv[ids], axis=0) / np.array([depth.shape[1], depth.shape[0]])
        if np.any(spread < .25):
            return unknown()
        visual = np.eye(4)
        visual[:3, :3] = cv2.Rodrigues(rvec)[0]
        visual[:3, 3] = tvec.reshape(3)
        transformed = xyz[ids] @ visual[:3, :3].T + visual[:3, 3]
        rounded = np.rint(uv[ids]).astype(int)
        measured = depth[rounded[:, 1], rounded[:, 0]]
        good = np.isfinite(measured) & (measured > .2)
        if np.count_nonzero(good) < 16 or np.median(np.abs(transformed[good, 2] - measured[good])) > .08:
            return unknown()
        result = self.compare(reference["T"], T, visual, self.rotation_deg, self.translation_m)
        result["visual_inliers"] = len(ids)
        result["guard_enabled"] = True
        result["rgbd_recovery_enabled"] = self.rgbd_recovery
        if discontinuity:
            result.update(discontinuity, integration_allowed=False, status="external_pose_discontinuity")
        if not result["integration_allowed"] or self.visual_tracking_active:
            if self.rgbd_recovery:
                result = self._recover(reference, current, matches, ids, visual, depth, K, T, stamp_ns, result)
            else:
                result.update(integration_allowed=False)
        else:
            self.recovery_hits, self.recovery_candidate = 0, None
        self.revalidation_required = not result["integration_allowed"]
        if (result["integration_allowed"] and not self.visual_tracking_active
                and (np.linalg.norm(visual[:3, 3]) > .4 or _rotation_angle(visual) > 15.)):
            self.anchor = current
        return accepted(result)


def _make_node(robot_id, parameters, shared_models, context, image_sink):
    """Import ROS only when a real rover observation actually needs inference."""
    from rclpy.parameter import Parameter
    from crossmaps_v2_node import CrossMapsNode

    class FleetNode(CrossMapsNode):
        def create_publisher(self, msg_type, topic, qos_profile, **kwargs):
            if topic.startswith("/crossmaps/"):
                topic = f"/crossmaps/robots/{robot_id}/" + topic[len("/crossmaps/"):]
            return super().create_publisher(msg_type, topic, qos_profile, **kwargs)

        def debug_rgb_image_has_subscribers(self, raw_pub, compressed_pub):
            return True  # HTTP camera consumers do not appear in ROS discovery.

        def publish_debug_rgb_image(self, raw_pub, compressed_pub, rgb_img, header_msg, compressed_format="jpeg"):
            suffix = raw_pub.topic_name.split(f"/crossmaps/robots/{robot_id}/", 1)[-1]
            view = next((v for v, topic in VIEW_TOPICS.items() if topic == suffix), None)
            if view:
                stamp = header_msg.stamp
                image_sink(robot_id, view, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR),
                           int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec))
            return super().publish_debug_rgb_image(raw_pub, compressed_pub, rgb_img, header_msg, compressed_format)

        def lookup_T_publish_internal(self, stamp_msg=None):
            return np.eye(4, dtype=np.float32)

    return FleetNode(node_name=f"crossmaps_collaborative_{robot_id}", context=context,
                     parameter_overrides=[Parameter(name, value=value) for name, value in parameters.items()],
                     external_inputs=True, shared_models=shared_models)


class PerceptionFleet:
    """Serialized perception work; image/health snapshots are HTTP-thread safe.

    Optional ``node_factory`` provides a lightweight seam for deterministic
    tests, not a different production perception implementation.
    """
    def __init__(self, config, context=None, node_factory=None):
        self.config, self.context = config, context
        self.node_factory = node_factory or _make_node
        self.robots = {str(item["id"]): item for item in config.get("robots", [])}
        if not self.robots or len(self.robots) != len(config.get("robots", [])):
            raise ValueError("At least one uniquely named rover is required")
        if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", robot) for robot in self.robots):
            raise ValueError("Rover IDs must be valid ROS name components")
        self._parameters = {}
        self._pose_config = dict(config.get("pose_guard", {}))
        base = dict(config.get("perception", {}).get("parameters", {}))
        model_signature = None
        for robot, settings in self.robots.items():
            params = {**base, **settings.get("parameters", {})}
            if str(params.get("segment_frame_assignment_mode", "fine_image")).lower() in ("", "off", "false", "legacy", "per_mask"):
                raise ValueError("Collaborative observation export requires fine-image segment assignment")
            signature = {key: params.get(key) for key in MODEL_KEYS}
            if model_signature is not None and signature != model_signature:
                raise ValueError("All rovers must use the same shared model configuration")
            model_signature = signature
            params.update(use_async_semantic_worker=False, autoload_map_state=False, autosave_map_state=False,
                          stationary_pose_lock=False, use_ray_consistency=False, use_clip_dense_features=False,
                          internal_frame=f"{robot}/local", publish_frame=f"{robot}/local", world_model_source_id=robot)
            self._parameters[robot] = params
        self._nodes, self._sessions, self._last, self._guards = {}, {}, {}, {}
        self._last_integrated, self._preview_cache, self._active_frames = {}, {}, {}
        self._models = None
        self._images, self._health = {}, {}
        self._snapshot_lock = threading.Lock()
        self._process_lock = threading.Lock()

    @property
    def nodes(self):
        return tuple(self._nodes.values())

    def _put_image(self, robot, view, bgr, stamp):
        ok, data = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if ok:
            with self._snapshot_lock:
                frame = self._active_frames.get(robot, {})
                self._images[(robot, view)] = (data.tobytes(), int(stamp),
                    frame.get("session_id", self._sessions.get(robot)),
                    frame.get("received_monotonic", time.monotonic()))

    def get_image(self, robot, view, session_id=None):
        with self._snapshot_lock:
            image = self._images.get((str(robot), VIEW_ALIASES.get(view, view)))
            return image[:2] if image is not None and (session_id is None or image[2] == session_id) else None

    def image_with_receipt(self, robot, view, session_id):
        """Bytes and input receipt time from the same stored observation."""
        with self._snapshot_lock:
            image = self._images.get((str(robot), VIEW_ALIASES.get(view, view)))
            return (image[0], image[3]) if image is not None and image[2] == session_id else None

    def image_metadata(self, robot, session_id=None):
        """Acquisition monotonic times, independent of transient worker health."""
        with self._snapshot_lock:
            return {view: image[3] for (source, view), image in self._images.items()
                    if source == str(robot) and (session_id is None or image[2] == session_id)}

    def health(self):
        with self._snapshot_lock:
            return {robot: dict(data) for robot, data in self._health.items()}

    def _get_node(self, robot, session):
        if self._sessions.get(robot) != session:
            old = self._nodes.pop(robot, None)
            if old is not None:
                old.destroy_node()
            self._last.pop(robot, None)
            self._last_integrated.pop(robot, None)
            self._preview_cache.pop(robot, None)
            self._active_frames.pop(robot, None)
            guard = self._pose_config
            self._guards[robot] = PoseConsistencyGuard(
                enabled=guard.get("enabled", True),
                rotation_deg=guard.get("max_rotation_error_deg", 5.),
                translation_m=guard.get("max_translation_error_m", .15),
                max_speed_m_s=guard.get("max_speed_m_s", 2.),
                max_yaw_rate_deg_s=guard.get("max_yaw_rate_deg_s", 150.),
                rgbd_recovery=guard.get("rgbd_recovery", True))
            with self._snapshot_lock:
                self._images = {key: val for key, val in self._images.items() if key[0] != robot}
            node = self.node_factory(robot, self._parameters[robot], self._models, self.context, self._put_image)
            self._nodes[robot] = node
            self._sessions[robot] = session
            if self._models is None:
                self._models = {name: getattr(node, name) for name in SHARED_ATTRIBUTES}
        return self._nodes[robot]

    @staticmethod
    def _header(stamp_ns, frame_id):
        try:
            from std_msgs.msg import Header
            msg = Header()
        except ImportError:
            msg = SimpleNamespace(stamp=SimpleNamespace())
        msg.stamp.sec, msg.stamp.nanosec = divmod(int(stamp_ns), 1_000_000_000)
        msg.frame_id = frame_id
        return msg

    def process(self, observation):
        with self._process_lock:
            try:
                return self._process(observation)
            except Exception as error:
                robot = str(observation.get("robot_id", "unknown"))
                with self._snapshot_lock:
                    previous = self._health.get(robot, {})
                    self._health[robot] = {**previous, "status": "error", "error": str(error)}
                raise

    def _process(self, observation):
        bgr, depth, K, T = validate_observation(observation)
        robot, session = str(observation["robot_id"]), str(observation["session_id"])
        if robot not in self.robots:
            raise ValueError(f"Unknown rover {robot}")
        seq, stamp = int(observation["seq"]), int(observation["stamp_ns"])
        preview_only = observation.get("preview_only", False)
        previous = self._last.get(robot) if self._sessions.get(robot) == session else None
        promoting = bool(previous and stamp == previous[1] and previous[2] and not preview_only)
        if previous and (seq <= previous[0] or stamp < previous[1] or (stamp == previous[1] and not promoting)):
            raise ValueError("Duplicate or out-of-order observation; not independent evidence")
        started = time.perf_counter()
        def stage(name):
            with self._snapshot_lock:
                self._health[robot] = {"seq": seq, "stamp_ns": stamp, "session_id": session,
                                      "status": "processing", "stage": name}
        stage("loading_models" if robot not in self._nodes else "pose_validation")
        node = self._get_node(robot, session)
        received = float(observation.get("received_monotonic", time.monotonic()))
        if not math.isfinite(received):
            received = time.monotonic()
        rgb_stamp = int(observation.get("rgb_stamp_ns", stamp))
        signature = (session, stamp, rgb_stamp, observation["calibration_id"])
        cached = self._preview_cache.get(robot) if promoting else None
        reused = bool(cached is not None and cached["signature"] == signature
                      and np.array_equal(cached["rgb"], bgr)
                      and np.array_equal(cached["depth"], depth, equal_nan=True)
                      and np.array_equal(cached["K"], K))
        if reused:
            received = cached["received_monotonic"]
        with self._snapshot_lock:
            self._active_frames[robot] = dict(session_id=session, received_monotonic=received)
        # Camera-space previews never depend on pose validity or map fusion.
        self._put_image(robot, "rgb", bgr, rgb_stamp)
        valid = np.isfinite(depth) & (depth > 0.)
        depth_u8 = np.clip(255. * (1. - np.where(valid, depth, 8.) / 8.), 0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        depth_color[~valid] = 0
        self._put_image(robot, "depth", depth_color, stamp)
        value = lambda key: node.get_parameter(key).value
        if node.model is None:
            raise RuntimeError("Shared CLIP model is unavailable; semantic integration is disabled")
        if bool(value("use_esam_masks")) and node.esam_model is None:
            raise RuntimeError("eSAM model failed to load; verify esam_checkpoint and the EfficientSAM installation")
        if bool(value("use_dinov2_features")) and node.dinov2_model is None:
            raise RuntimeError("DINO model failed to load; verify its local repository/checkpoint or download configuration")
        stage("pose_validation")
        if preview_only:
            provided = observation.get("pose_quality")
            pose = dict(provided) if isinstance(provided, dict) else {"status": provided or "timestamped_tf_unavailable"}
            pose.update(integration_allowed=False, is_localization_fix=False, guard_evaluated=False)
        else:
            pose = self._guards[robot].check(bgr, depth, K, T, stamp)
            pose["guard_evaluated"] = True
        if pose["integration_allowed"] and pose.get("mapping_pose_corrected"):
            # This is a validated camera-to-local transform, never a change to
            # robot TF. Preserve the raw source matrix in pose_quality; backend
            # keyframes and local surface fusion must use the exact same pose.
            T = np.asarray(pose["corrected_T_local_camera"], dtype=np.float64)
        result = {**observation, "T_local_camera": T,
                  "integration_allowed": pose["integration_allowed"], "pose_quality": pose}
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        points, uu, vv = depth_points(depth, K, max(1, int(value("stride"))),
                                     float(value("min_depth_m")), float(value("max_depth_m")))
        if not len(points) and pose["integration_allowed"]:
            raise ValueError("No valid depth support; no semantic map update performed")
        node.fx, node.fy, node.cx, node.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        header = self._header(stamp, f"{robot}/camera_optical")
        rgb_header = self._header(rgb_stamp, f"{robot}/camera_optical")
        phases = dict(sam_ms=0., dino_ms=0., pool_clip_ms=0., fusion_ms=0.)
        if reused:
            masks, scores, dense, dino, quality, clip = cached["features"]
        else:
            stage("eSAM")
            mark = time.perf_counter()
            masks, scores = node.segment_esam_masks(rgb)
            phases["sam_ms"] = (time.perf_counter() - mark) * 1000
            stage("DINO")
            mark = time.perf_counter()
            dense = node.compute_dinov2_dense_features(rgb)
            phases["dino_ms"] = (time.perf_counter() - mark) * 1000
            stage("CLIP_and_pooling")
            mark = time.perf_counter()
            if masks:
                dino, quality = node.compute_dinov2_mask_embeddings(masks, dense, return_quality=True)
                clip = node.compute_mask_embeddings_batched(rgb, masks, dense_clip=None)
            else:
                dino, quality, clip = np.zeros((0, 0), np.float32), np.zeros(0, np.float32), np.zeros((0, 0), np.float32)
            phases["pool_clip_ms"] = (time.perf_counter() - mark) * 1000
        surfaces = []
        if pose["integration_allowed"]:
            point_local = (points @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
            semantic = dict(seq=seq, obs_time_s=node.now_sec(), rgb_cv=rgb, rgb_header=rgb_header, depth_header=header,
                            pts_int=point_local, cam_int=T[:3, 3].astype(np.float32), uu=uu, vv=vv,
                            z_cam=points[:, 2], image_shape=depth.shape, masks=masks, mask_scores=scores,
                            mask_emb=clip, dino_mask_emb=dino, dino_mask_quality=quality,
                            dense_clip=None, dense_dino=dense)
            mark = time.perf_counter()
            stage("local_fusion_and_identity")
            if not node.apply_semantic_result(semantic, "collaborative"):
                raise ValueError("Local perception rejected this observation")
            phases["fusion_ms"] = (time.perf_counter() - mark) * 1000
            self._last_integrated[robot] = stamp
            surfaces = self._surface_evidence(node, points, dino, quality, masks, uu, vv)
        else:
            # These methods render camera-space diagnostics only. In particular
            # never call apply_semantic_result with an invented/frozen pose.
            stage("camera_diagnostics_fusion_paused")
            node.publish_esam_mask_overlay(rgb, masks, rgb_header)
            node.publish_feature_diagnostics(rgb, masks, clip, dino, quality, dense, rgb_header)
        self._last[robot] = (seq, stamp, preview_only)
        if preview_only:
            self._preview_cache[robot] = dict(signature=signature, rgb=bgr.copy(), depth=depth.copy(), K=K.copy(),
                received_monotonic=received, features=(masks, scores, dense, dino, quality, clip))
        else:
            self._preview_cache.pop(robot, None)
        model_settings = {}
        for key in FEATURE_KEYS:
            try:
                model_settings[key] = value(key)
            except Exception:
                pass
        feature_id = "crossmaps_dino_clip_v1_" + hashlib.sha256(json.dumps(model_settings, sort_keys=True).encode()).hexdigest()[:20]
        result.update(points_camera=points, colors=rgb[vv, uu].copy(), surfaces=surfaces,
                      global_descriptor=pooled_dino_descriptor(dense), feature_space_id=feature_id,
                      local_world_id=node.world_model_id, mask_count=len(masks),
                      segment_count=len(node.seg_w), entity_count=len(node.entity_memory.records),
                      timings_ms=phases, visual_processed=True, map_integrated=pose["integration_allowed"],
                      inference_reused=reused, last_visual_stamp_ns=rgb_stamp,
                      last_integrated_stamp_ns=self._last_integrated.get(robot))
        result["processing_ms"] = (time.perf_counter() - started) * 1000
        with self._snapshot_lock:
            self._health[robot] = {"seq": seq, "stamp_ns": stamp, "session_id": session,
                                  "status": "preview_only" if preview_only else "processed" if pose["integration_allowed"] else "pose_rejected",
                                  "processing_ms": result["processing_ms"], "timings_ms": phases,
                                  "segments": result["segment_count"], "entities": result["entity_count"],
                                  "visual_processed": True, "map_integrated": pose["integration_allowed"],
                                  "last_visual_stamp_ns": rgb_stamp,
                                  "last_integrated_stamp_ns": self._last_integrated.get(robot),
                                  "pose_quality": pose, "device": str(node.device),
                                  "sam_available": node.esam_model is not None, "dino_available": node.dinov2_model is not None}
        return result

    @staticmethod
    def _surface_evidence(node, points, dino, quality, masks, uu, vv):
        owners = np.asarray(getattr(node, "last_observation_point_segments", []), dtype=np.int32)
        if len(owners) != len(points):
            return []
        surfaces = []
        for sid in np.unique(owners[owners > 0]):
            selected = np.flatnonzero(owners == sid)
            if len(selected) < 3 or int(sid) not in node.seg_w:
                continue
            # Descriptor memory is local and bounded, but support is strictly
            # this observed surface; no historical geometry enters a keyframe.
            embedding = node.segment_dino_embedding(int(sid))
            if embedding is not None:
                embedding = np.asarray(embedding)[:int(getattr(node, "dino_base_dim", 0)) or None]
            clip = node.segment_embedding_and_coherence(int(sid))[0]
            eid = node._effective_entity_id(node.entity_memory.segment_entity.get(int(sid)))
            mask_quality = [float(quality[i]) for i, mask in enumerate(masks)
                            if i < len(quality) and np.count_nonzero(mask[vv[selected], uu[selected]]) >= 3]
            take = selected[np.linspace(0, len(selected) - 1, min(len(selected), 512), dtype=int)]
            surfaces.append(dict(local_id=str(sid), entity_id=None if eid is None else str(eid),
                                 points_camera=points[take].copy(),
                                 dino=normalize_descriptor([] if embedding is None else embedding),
                                 clip=normalize_descriptor([] if clip is None else clip),
                                 quality=float(np.mean(mask_quality)) if mask_quality else 0.,
                                 support_points=len(selected), observation_support="accepted_current_depth"))
        return surfaces

    def close(self):
        with self._process_lock:
            for node in self._nodes.values():
                node.destroy_node()
            self._nodes.clear()
            self._models = None
