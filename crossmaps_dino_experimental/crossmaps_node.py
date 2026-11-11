#!/usr/bin/env python3
"""CrossMaps v2: RGB-D perception, persistent surfaces and entity hypotheses."""
import hashlib
import json
import os
import math
import platform
import pickle
import queue
import struct
import tempfile
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from identity_graph import IdentityConfig, ObservationEdges, PersistentEntities
from object_assembly import AssemblyConfig
from segment_overlay import dense_display_ids, id_color, render_overlay, render_mask_consensus
from surface_tracking import SurfaceConfig, SurfaceConsensus, BoundaryOwnershipVotes
from dominant_overlay import DominantOverlay
from world_model import new_world_id, validate_world_id, node_uid, interior_descriptor, surface_role, dominant_surface_core, EvidenceRecorder
from pipeline_runtime import LatestFrame, PipelineHealth
from pipeline_instance import PipelineInstance, check_existing_publishers
from voxel_scoring import score_voxels

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

import tf2_ros

from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Header, String
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from cv_bridge import CvBridge

import torch

_TORCHVISION_LIBRARY_SHIM = None


def _patch_torchvision_import_for_jetson():
    global _TORCHVISION_LIBRARY_SHIM
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system != "linux" or machine not in ("aarch64", "arm64"):
        return
    try:
        from torch.library import Library

        _TORCHVISION_LIBRARY_SHIM = Library("torchvision", "DEF")
        _TORCHVISION_LIBRARY_SHIM.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
    except Exception:
        pass


_patch_torchvision_import_for_jetson()

import open_clip
from PIL import Image as PILImage

import message_filters


# -----------------------------
# utils
# -----------------------------
def quat_to_rot_matrix(x, y, z, w):
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def transform_to_matrix(tf):
    t = tf.transform.translation
    q = tf.transform.rotation
    R = quat_to_rot_matrix(q.x, q.y, q.z, q.w)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[0, 3] = t.x
    T[1, 3] = t.y
    T[2, 3] = t.z
    return T


def pack_rgb(r, g, b):
    rgb_uint32 = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack("f", struct.pack("I", rgb_uint32))[0]


def blue_to_red(val01: float):
    v = float(np.clip(val01, 0.0, 1.0))
    r = int(255 * v)
    b = int(255 * (1.0 - v))
    g = int(40 * (1.0 - abs(v - 0.5) * 2.0))
    return r, g, b


def make_cloud_xyz(points_xyz: np.ndarray, frame_id: str, stamp_msg):
    """Creates a PointCloud2 message with XYZ points and white color."""
    msg = PointCloud2()
    msg.header = Header(stamp=stamp_msg, frame_id=frame_id)
    msg.height = 1
    msg.width = int(points_xyz.shape[0])
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = points_xyz.astype(np.float32).tobytes()
    return msg


def make_cloud_xyzrgb(points_xyz: np.ndarray, rgb_floats: np.ndarray, frame_id: str, stamp_msg):
    """Creates a PointCloud2 message with XYZ points and RGB colors packed as floats."""
    msg = PointCloud2()
    msg.header = Header(stamp=stamp_msg, frame_id=frame_id)
    msg.height = 1
    msg.width = int(points_xyz.shape[0])
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True

    data = np.zeros((points_xyz.shape[0], 4), dtype=np.float32)
    data[:, 0:3] = points_xyz.astype(np.float32)
    data[:, 3] = rgb_floats.astype(np.float32)
    msg.data = data.tobytes()
    return msg


def bresenham_2d(x0, y0, x1, y1):
    """Bresenham's line algorithm in 2D for ray-consistency checks."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def safe_popcount(mask: int) -> int:
    try:
        return int(mask.bit_count())
    except Exception:
        return bin(int(mask)).count("1")


def maximum_weight_assignment(scores: np.ndarray) -> List[Tuple[int, int]]:
    """Maximum-weight rectangular assignment with an unmatched option per row.

    This is the O(n^3) Hungarian shortest-augmenting-path algorithm. Extra
    zero-weight columns let an observation remain unmatched instead of forcing
    a bad track association. Non-finite entries are treated as forbidden.
    """
    weights = np.asarray(scores, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] == 0 or weights.shape[1] == 0:
        return []
    rows, real_cols = weights.shape
    finite = np.isfinite(weights)
    work = np.full((rows, real_cols + rows), -1e6, dtype=np.float64)
    work[:, :real_cols] = np.where(finite, weights, -1e6)
    work[:, real_cols:] = 0.0
    cost = -work

    # The implementation below requires rows <= columns, which is guaranteed
    # by the per-row dummy columns above.
    cols = int(cost.shape[1])
    u = np.zeros((rows + 1,), dtype=np.float64)
    v = np.zeros((cols + 1,), dtype=np.float64)
    p = np.zeros((cols + 1,), dtype=np.int32)
    way = np.zeros((cols + 1,), dtype=np.int32)
    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full((cols + 1,), np.inf, dtype=np.float64)
        used = np.zeros((cols + 1,), dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = np.inf
            j1 = 0
            for j in range(1, cols + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if not np.isfinite(delta):
                break
            for j in range(cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0 != 0:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1

    result = []
    for j in range(1, cols + 1):
        row = int(p[j]) - 1
        col = j - 1
        if row < 0 or col < 0 or col >= real_cols:
            continue
        if finite[row, col] and weights[row, col] > 0.0:
            result.append((row, col))
    return result


# -----------------------------
# node
# -----------------------------
class CrossMapsNode(Node):
    """
    CrossMaps v2:
    - Features:
        * internal_frame storage, publish_frame publishing
        * Multi-level confidence: point confidence + cell confidence
        * Multi-scale tile embeddings (coarse+fine)
        * Ray-consistency check against an occupancy grid
        * Soft consistency gating
        * Coherence and visibility-boost in confidence / semantic scoring
        * Debug cloud, semantic cloud, history cloud (warped to publish_frame)
        * Percentile normalization for semantic grids
        * Short-Term Memory (STM): same behavior as before (time-decayed weights)
        * Long-Term Memory (LTM): persistent, noise-filtered, with last_seen timestamp per cell
        * Promotion to LTM: only “confident, multi-view, coherent” STM cells transfer to LTM
        * Contradiction handling: if new promoted evidence contradicts old LTM embedding, replace quickly
    - Subscribes to:
        * /oak/rgb/image_rect/compressed (or as set by rgb_topic param)
        * /oak/stereo/image_raw/compressedDepth (or as set by depth_topic param)
        * /oak/rgb/camera_info (or as set by cam_info_topic param)
        * /map (occupancy grid for ray-consistency, or as set by occ_grid_topic param)
    - Publishes:
        * /crossmaps/stm/semantic_grid
        * /crossmaps/stm/confidence_grid
        * /crossmaps/stm/top1_grid
        * /crossmaps/ltm/semantic_grid
        * /crossmaps/ltm/confidence_grid
        * /crossmaps/ltm/top1_grid
        * /crossmaps/debug_cloud
        * /crossmaps/semantic_cloud
        * /crossmaps/history_cloud
    """

    def __init__(self):
        super().__init__("crossmaps_3_node")
        self.bridge = CvBridge()
        self.frame_counter = 0
        self.publish_counter = 0
        self.inference_lock = threading.Lock()
        self.async_stop_event = threading.Event()
        self.async_worker_thread = None
        self.async_job_seq = 0
        self.async_jobs_submitted = 0
        self.async_jobs_dropped = 0
        self.async_results_dropped = 0
        self.async_apply_timer = None
        self.latest_frame = LatestFrame()
        self.pipeline_health = PipelineHealth()
        self.async_worker_busy = threading.Event()
        self.semantic_job_queue = queue.Queue(maxsize=1)
        self.semantic_result_queue = queue.Queue(maxsize=2)

        # -------------------------
        # Params
        # -------------------------
        # Raw 1280x720 RGB-D over the robot's wireless DDS link can stop
        # delivering complete images despite healthy discovery/odometry.
        # JPEG RGB + lossless uint16 compressedDepth preserve calibration and
        # depth geometry while greatly reducing network fragmentation/load.
        self.declare_parameter("rgb_topic", "/oak/rgb/image_rect/compressed")
        self.declare_parameter("depth_topic", "/oak/stereo/image_raw/compressedDepth")
        self.declare_parameter("cam_info_topic", "/oak/rgb/camera_info")
        self.declare_parameter("rgb_compressed", True)
        self.declare_parameter("depth_compressed", True)
        self.declare_parameter("use_best_effort_sensor_qos", True)
        self.declare_parameter("sensor_qos_depth", 5)

        self.declare_parameter("internal_frame", "odom")
        self.declare_parameter("publish_frame", "map")

        self.declare_parameter("sync_queue_size", 20)
        self.declare_parameter("sync_slop_s", 0.35)

        self.declare_parameter("stride", 8)
        self.declare_parameter("min_depth_m", 0.2)
        self.declare_parameter("max_depth_m", 8.0)

        self.declare_parameter("grid_size_m", 20.0)
        self.declare_parameter("resolution", 0.10)

        self.declare_parameter("query_text", "ring")


        self.declare_parameter("embed_every_n", 10)
        self.declare_parameter("observation_latest_frame", True)
        self.declare_parameter("observation_max_hz", 2.0)
        self.declare_parameter("observation_max_frame_age_s", 2.0)
        self.declare_parameter("pipeline_stale_after_s", 10.0)
        self.declare_parameter("pipeline_processing_timeout_s", 30.0)
        self.declare_parameter("publish_hz", 2.0)
        self.declare_parameter("publish_max_duty_cycle", 0.35)

        # EfficientSAM mask extraction. The old tile parameters are retained
        # for launch-file compatibility, but semantic updates now use masks.
        self.declare_parameter("use_esam_masks", True)
        self.declare_parameter("use_async_semantic_worker", True)
        self.declare_parameter("async_semantic_queue_size", 1)
        self.declare_parameter("async_semantic_result_queue_size", 2)
        self.declare_parameter("async_semantic_max_results_per_publish", 1)
        self.declare_parameter("async_semantic_apply_hz", 0.0)
        self.declare_parameter("esam_variant", "vitt")
        self.declare_parameter("esam_checkpoint", "")
        self.declare_parameter("esam_input_max_side", 640)
        self.declare_parameter("esam_prompt_grid_x", 10)
        self.declare_parameter("esam_prompt_grid_y", 6)
        self.declare_parameter("esam_max_masks", 36)
        self.declare_parameter("esam_min_mask_area_px", 400)
        self.declare_parameter("esam_max_mask_area_frac", 0.75)
        self.declare_parameter("esam_min_iou_pred", 0.65)
        self.declare_parameter("esam_min_stability_score", 0.78)
        self.declare_parameter("esam_stability_logit_offset", 0.05)
        self.declare_parameter("esam_nms_iou", 0.80)
        self.declare_parameter("esam_nms_containment", 0.97)
        self.declare_parameter("esam_nms_containment_min_area_ratio", 0.70)
        self.declare_parameter("esam_min_depth_points", 25)
        self.declare_parameter("esam_clip_batch_size", 8)
        self.declare_parameter("publish_esam_mask_overlay", True)
        self.declare_parameter("esam_mask_overlay_alpha", 0.45)
        self.declare_parameter("publish_dinov2_feature_view", True)
        self.declare_parameter("publish_dinov2_similarity_view", True)
        self.declare_parameter("publish_dinov2_instance_view", True)
        self.declare_parameter("dinov2_feature_view_rgb_alpha", 0.18)
        self.declare_parameter("publish_clip_feature_view", True)
        self.declare_parameter("publish_clip_similarity_view", True)
        self.declare_parameter("clip_feature_view_rgb_alpha", 0.18)
        self.declare_parameter("diagnostic_similarity_x", 0.50)
        self.declare_parameter("diagnostic_similarity_y", 0.50)
        self.declare_parameter("diagnostic_similarity_floor", 0.0)
        self.declare_parameter("diagnostic_similarity_rgb_alpha", 0.12)
        self.declare_parameter("diagnostic_feature_sam_boundaries", True)
        self.declare_parameter("publish_debug_observation_images", True)
        self.declare_parameter("debug_observation_image_hz", 0.0)
        self.declare_parameter("publish_debug_observation_rgb", True)
        self.declare_parameter("publish_debug_observation_depth", True)
        self.declare_parameter("debug_observation_depth_max_m", 6.0)
        self.declare_parameter("publish_debug_compressed_images", True)
        self.declare_parameter("debug_image_jpeg_quality", 82)

        # point cloud publishing (keep off for better performance)
        self.declare_parameter("publish_debug_cloud", False)
        self.declare_parameter("publish_semantic_cloud", False)
        self.declare_parameter("publish_history_cloud", False)

        # STM grid publishing
        self.declare_parameter("compute_2d_grids", True)
        self.declare_parameter("publish_stm_semantic_grid", True)
        self.declare_parameter("publish_stm_confidence_grid", True)
        self.declare_parameter("publish_stm_top1_grid", True)

        # LTM grid publishing
        self.declare_parameter("publish_ltm_semantic_grid", True)
        self.declare_parameter("publish_ltm_confidence_grid", True)
        self.declare_parameter("publish_ltm_top1_grid", True)

        self.declare_parameter("semantic_cloud_gamma", 0.8)
        self.declare_parameter("semantic_cloud_max_points", 15000)

        # 3D semantic voxel memory + RViz heatmap
        self.declare_parameter("use_semantic_voxels", True)
        self.declare_parameter("voxel_size_m", 0.15)
        self.declare_parameter("voxel_z_min_m", -0.5)
        self.declare_parameter("voxel_z_max_m", 2.5)
        self.declare_parameter("voxel_max_count", 250000)
        self.declare_parameter("voxel_fuse_stride", 1)
        self.declare_parameter("voxel_decay_max_per_tick", 5000)
        self.declare_parameter("publish_voxel_heatmap", True)
        self.declare_parameter("publish_voxel_stm_heatmap", True)
        self.declare_parameter("publish_voxel_ltm_heatmap", True)
        self.declare_parameter("publish_voxel_fused_heatmap", True)
        self.declare_parameter("voxel_heatmap_publish_every_n", 1)
        self.declare_parameter("voxel_cloud_max_points", 60000)
        self.declare_parameter("voxel_stm_cloud_max_points", 0)
        self.declare_parameter("voxel_ltm_cloud_max_points", 0)
        self.declare_parameter("voxel_fused_cloud_max_points", 0)
        self.declare_parameter("voxel_cloud_min_score", 0.0)
        self.declare_parameter("voxel_cloud_min_conf", 0.015)
        self.declare_parameter("voxel_cloud_min_coh01", 0.0)
        self.declare_parameter("voxel_cloud_normalize_scores", True)
        self.declare_parameter("voxel_cloud_gamma", 0.8)
        self.declare_parameter("voxel_promote_min_conf", 0.25)
        self.declare_parameter("voxel_promote_min_coh01", 0.15)
        self.declare_parameter("voxel_promote_min_views", 1)
        self.declare_parameter("voxel_promote_max_per_tick", 12000)
        self.declare_parameter("voxel_promote_every_n", 1)
        self.declare_parameter("use_voxel_ray_clearing", True)
        self.declare_parameter("voxel_clear_stride", 4)
        self.declare_parameter("voxel_clear_max_rays_per_frame", 3500)
        self.declare_parameter("voxel_clear_step_m", 0.04)
        self.declare_parameter("voxel_clear_hit_margin_m", 0.12)
        self.declare_parameter("voxel_clear_min_range_m", 0.30)
        self.declare_parameter("voxel_clear_min_observations", 2)
        self.declare_parameter("voxel_clear_stm_decay", 0.35)
        self.declare_parameter("voxel_clear_ltm_decay", 0.82)
        self.declare_parameter("voxel_clear_ltm", True)
        self.declare_parameter("voxel_clear_hard_prune", True)
        self.declare_parameter("voxel_clear_hard_prune_min_observations", 5)
        self.declare_parameter("voxel_clear_hard_prune_max_per_frame", 2500)

        # Entity graph layer. In this experimental node, mask-derived segments
        # are the persistent sparse-voxel entities.
        self.declare_parameter("use_object_abstraction", True)
        self.declare_parameter("entity_track_max_stale_s", 45.0)
        self.declare_parameter("entity_min_support_cells", 3)
        self.declare_parameter("entity_min_support_voxels", 3)
        self.declare_parameter("use_cell_negative_evidence", False)
        self.declare_parameter("cell_free_occ_max", 5)
        self.declare_parameter("cell_negative_decay", 0.90)
        self.declare_parameter("cell_negative_max_per_tick", 2500)
        self.declare_parameter("publish_object_query_voxels", True)
        self.declare_parameter("publish_object_instance_voxels", False)
        self.declare_parameter("publish_segment_voxels", True)
        self.declare_parameter("publish_entity_voxels", True)
        self.declare_parameter("publish_object_unassigned_query_voxels", True)
        self.declare_parameter("object_query_cloud_max_points", 12000)
        self.declare_parameter("object_instance_cloud_max_points", 20000)
        self.declare_parameter("object_unassigned_query_cloud_max_points", 8000)
        self.declare_parameter("object_unassigned_query_min_score01", 0.65)
        self.declare_parameter("publish_object_graph", True)
        self.declare_parameter("publish_object_graph_labels", True)
        self.declare_parameter("object_graph_max_nodes", 18)
        self.declare_parameter("object_graph_max_edges", 80)
        self.declare_parameter("object_graph_min_weight", 0.50)
        self.declare_parameter("object_graph_min_conf", 0.08)
        self.declare_parameter("object_graph_min_observations", 4)
        self.declare_parameter("object_graph_min_support_voxels", 25)
        self.declare_parameter("object_graph_min_view_count", 1)
        self.declare_parameter("object_graph_max_age_s", 60.0)
        self.declare_parameter("object_graph_edge_min_observations", 2)
        self.declare_parameter("object_graph_edge_max_misses", 3)
        self.declare_parameter("object_graph_edge_ema_alpha", 0.35)
        self.declare_parameter("object_graph_node_scale_m", 0.12)
        self.declare_parameter("object_graph_edge_width_m", 0.025)
        self.declare_parameter("object_graph_label_scale_m", 0.12)
        self.declare_parameter("object_graph_near_dist_m", 1.0)
        self.declare_parameter("object_graph_vertical_xy_dist_m", 0.45)
        self.declare_parameter("object_graph_vertical_z_min_m", 0.20)
        self.declare_parameter("object_graph_overlap_xy_min", 0.02)
        self.declare_parameter("object_graph_similar_cos_min", 0.78)
        self.declare_parameter("object_graph_include_overlap_edges", False)
        self.declare_parameter("object_graph_include_similar_edges", False)
        self.declare_parameter("publish_object_graph_descriptors", True)
        self.declare_parameter("object_graph_descriptor_include_embedding", False)
        self.declare_parameter("object_graph_descriptor_embedding_dims", 0)
        self.declare_parameter("object_graph_descriptor_support_hash_sample", 512)

        # Segment memory is the revised FindAnything-inspired entity substrate:
        # persistent mask-derived segments own voxel support, and graph nodes
        # are views over those segments.
        self.declare_parameter("use_segment_memory", True)
        self.declare_parameter("segment_frame_one_to_one_assoc", True)
        self.declare_parameter("segment_weight_max", 200.0)
        self.declare_parameter("segment_conf_weight_full", 2.0)
        self.declare_parameter("segment_support_max_voxels", 12000)
        self.declare_parameter("segment_support_max_cells", 6000)
        self.declare_parameter("segment_min_support_voxels", 3)
        self.declare_parameter("segment_mask_min_depth_points", 3)
        self.declare_parameter("segment_mask_min_support_voxels", 1)
        self.declare_parameter("segment_temporal_consensus", True)
        self.declare_parameter("segment_mask_interior_px", 3)
        self.declare_parameter("segment_mask_vote_dominance", 0.75)
        self.declare_parameter("segment_owner_switch_observations", 3)
        self.declare_parameter("segment_boundary_repair_enabled", True)
        self.declare_parameter("segment_mixed_dino_pool_budget", 12)
        self.declare_parameter("segment_min_support_cells", 3)
        self.declare_parameter("segment_min_observations_to_publish", 2)
        self.declare_parameter("segment_bbox_percentile_low", 5.0)
        self.declare_parameter("segment_bbox_percentile_high", 95.0)
        self.declare_parameter("segment_bbox_padding_m", 0.03)
        self.declare_parameter("segment_update_alpha_min", 0.05)
        self.declare_parameter("segment_assoc_max_center_dist_m", 1.25)
        self.declare_parameter("segment_assoc_bbox_gap_max_m", 0.35)
        self.declare_parameter("segment_assoc_min_score", 0.48)
        self.declare_parameter("segment_assoc_min_sem_cos", -0.08)
        self.declare_parameter("segment_assoc_min_dino_cos", 0.25)
        self.declare_parameter("segment_assoc_min_overlap", 0.02)
        self.declare_parameter("segment_assoc_semantic_weight", 0.12)
        self.declare_parameter("segment_assoc_dino_weight", 0.48)
        self.declare_parameter("segment_assoc_spatial_weight", 0.20)
        self.declare_parameter("segment_assoc_overlap_weight", 0.20)
        self.declare_parameter("segment_assoc_max_age_s", 60.0)
        self.declare_parameter("segment_assoc_global_assignment", True)
        self.declare_parameter("segment_assoc_require_overlap", True)
        self.declare_parameter("segment_assoc_min_bbox_iou", 0.15)
        self.declare_parameter("segment_assoc_ambiguity_margin", 0.06)
        self.declare_parameter("segment_exclusive_point_fusion", True)
        self.declare_parameter("segment_assoc_use_projected_ids", True)
        self.declare_parameter("segment_projected_id_neighbor_radius_vox", 0)
        self.declare_parameter("segment_projected_id_min_segment_observations", 1)
        self.declare_parameter("segment_projected_id_min_pixels", 8)
        self.declare_parameter("segment_projected_id_min_mask_frac", 0.10)
        self.declare_parameter("segment_projected_id_min_visible_frac", 0.25)
        self.declare_parameter("segment_projected_id_min_iou", 0.08)
        self.declare_parameter("segment_projected_id_min_dominance", 0.55)
        self.declare_parameter("segment_projected_id_min_score", 0.30)
        self.declare_parameter("segment_projected_id_min_sem_cos", 0.15)
        self.declare_parameter("segment_projected_id_reject_ambiguous_masks", True)
        self.declare_parameter("segment_projected_id_protect_other_owners", True)
        self.declare_parameter("segment_projected_id_min_filtered_points", 8)
        self.declare_parameter("segment_projected_id_max_unowned_claim_ratio", 0.35)
        self.declare_parameter("segment_frame_assignment_mode", "fine_image")
        self.declare_parameter("segment_frame_assignment_iou_threshold", 0.25)
        self.declare_parameter("segment_frame_assignment_existing_overlap", 0.80)
        self.declare_parameter("segment_frame_assignment_existing_area_scale", 1.20)
        self.declare_parameter("segment_frame_assignment_max_existing_extension_ratio", 1.25)
        self.declare_parameter("segment_frame_assignment_max_depth_frac", 0.75)
        self.declare_parameter("segment_frame_assignment_min_depth_frac", 0.0005)
        self.declare_parameter("segment_frame_assignment_sort_small_first", True)
        self.declare_parameter("segment_projected_fusion_semantic_gate", True)
        self.declare_parameter("segment_projected_fusion_min_sem_cos", 0.25)
        # Appearance disagreement is not proof of changed physical ownership.
        # Retain the legacy carve as an explicit opt-in diagnostic experiment.
        self.declare_parameter("segment_semantic_conflict_carve", False)
        self.declare_parameter("segment_semantic_conflict_min_hits", 2)
        self.declare_parameter("segment_semantic_conflict_min_voxels", 8)
        self.declare_parameter("segment_semantic_conflict_max_voxels_per_frame", 800)
        self.declare_parameter("segment_observation_max_voxels", 6000)
        self.declare_parameter("segment_observation_max_bbox_xy_m", 4.5)
        self.declare_parameter("segment_observation_max_bbox_z_m", 3.2)
        self.declare_parameter("publish_segment_id_projection", True)
        self.declare_parameter("publish_entity_id_projection", True)
        self.declare_parameter("segment_id_projection_dilate_px", 1)
        self.declare_parameter("segment_overlay_fill_masks", True)
        self.declare_parameter("segment_overlay_alpha", 0.78)
        self.declare_parameter("segment_overlay_raw_support", False)
        self.declare_parameter("segment_overlay_dominant_owner", True)
        self.declare_parameter("segment_query_min_score", 0.06)
        self.declare_parameter("segment_query_min_conf", 0.03)
        self.declare_parameter("segment_query_min_support_voxels", 3)
        self.declare_parameter("segment_query_top_k", 30)
        self.declare_parameter("segment_query_color_mode", "instance")
        self.declare_parameter("segment_query_use_support_scores", True)
        self.declare_parameter("segment_query_support_min_score01", 0.65)
        self.declare_parameter("segment_query_support_min_selected_frac", 0.03)
        self.declare_parameter("segment_instance_max_segments", 80)
        self.declare_parameter("segment_instance_max_points_per_segment", 1000)
        self.declare_parameter("segment_instance_min_conf", 0.04)
        self.declare_parameter("segment_instance_min_weight", 0.20)
        self.declare_parameter("segment_instance_min_support_voxels", 8)
        self.declare_parameter("segment_instance_min_view_count", 1)
        self.declare_parameter("segment_use_negative_evidence", True)
        self.declare_parameter("segment_negative_min_observations", 4)
        self.declare_parameter("segment_negative_decay", 0.97)
        self.declare_parameter("segment_negative_max_prune_per_frame", 600)
        self.declare_parameter("segment_negative_drop_empty", True)
        self.declare_parameter("use_segment_maintenance", True)
        self.declare_parameter("segment_maintenance_interval_s", 2.0)
        self.declare_parameter("segment_maintenance_debug_log", True)
        self.declare_parameter("segment_maintenance_debug_interval_s", 8.0)
        self.declare_parameter("segment_retire_tentative_after_s", 30.0)
        self.declare_parameter("segment_retire_stale_after_s", 180.0)
        self.declare_parameter("segment_memory_max_tracks", 512)
        self.declare_parameter("segment_retire_tentative_max_observations", 2)
        self.declare_parameter("segment_split_enabled", True)
        self.declare_parameter("segment_split_connectivity", 26)
        self.declare_parameter("segment_split_gap_vox", 1)
        self.declare_parameter("segment_split_min_support_voxels", 35)
        self.declare_parameter("segment_split_min_component_voxels", 8)
        self.declare_parameter("segment_split_min_secondary_frac", 0.12)
        self.declare_parameter("segment_split_max_components_per_segment", 4)
        self.declare_parameter("segment_split_max_segments_per_tick", 4)
        self.declare_parameter("segment_split_drop_tiny_components", False)
        self.declare_parameter("segment_merge_enabled", False)
        self.declare_parameter("segment_merge_max_pairs_per_tick", 1)
        self.declare_parameter("segment_merge_min_observations", 2)
        self.declare_parameter("segment_merge_min_sem_cos", 0.84)
        self.declare_parameter("segment_merge_adjacent_only_min_sem_cos", 0.90)
        self.declare_parameter("segment_merge_same_frame_min_sem_cos", 0.92)
        self.declare_parameter("segment_merge_min_geometry_score", 0.35)
        self.declare_parameter("segment_merge_max_bbox_gap_m", 0.08)
        self.declare_parameter("segment_merge_adjacency_vox", 1)
        self.declare_parameter("segment_merge_max_center_dist_m", 1.20)
        self.declare_parameter("segment_merge_min_overlap", 0.04)
        self.declare_parameter("segment_merge_min_score", 0.86)
        self.declare_parameter("segment_merge_max_area_growth", 2.0)
        self.declare_parameter("segment_merge_max_volume_growth", 3.0)
        self.declare_parameter("segment_merge_max_z_growth", 2.2)
        self.declare_parameter("segment_merge_split_cooldown_s", 30.0)
        # This now selects the persistent entity view, not greedy clustering.
        self.declare_parameter("use_entity_grouping", True)
        self.declare_parameter("use_object_assembly", True)
        self.declare_parameter("entity_grouping_sensitivity", 0.5)
        self.declare_parameter("entity_assembly_confirm_observations", 3)
        self.declare_parameter("entity_assembly_max_extent_m", 3.5)
        self.declare_parameter("entity_assembly_group_min_dino_cos", 0.55)
        self.declare_parameter("world_model_source_id", "crossmaps_experimental")
        self.declare_parameter("record_identity_replay", False)
        self.declare_parameter("identity_replay_max_frames", 120)
        self.declare_parameter("entity_birth_observations", 3)
        self.declare_parameter("entity_membership_observations", 2)
        self.declare_parameter("entity_reassoc_min_voxels", 8)
        self.declare_parameter("entity_reassoc_min_overlap", 0.65)
        self.declare_parameter("entity_reassoc_ambiguity_margin", 0.20)
        self.declare_parameter("entity_neighborhood_tiebreak", True)
        self.declare_parameter("entity_neighborhood_overlap_gap", 0.10)
        self.declare_parameter("entity_memory_support_limit", 24000)
        self.declare_parameter("entity_dormant_retention_s", 180.0)
        self.declare_parameter("entity_view_cell_m", 0.30)
        self.declare_parameter("instance_color_by_entity", True)
        # Deprecated: accepted so saved parameter sets still load. The former
        # adjacency/semantic clustering thresholds no longer control identity.
        self.declare_parameter("entity_group_min_sem_cos", 0.84)
        self.declare_parameter("entity_group_min_pair_sem_cos", 0.80)
        self.declare_parameter("entity_group_adjacency_vox", 1)
        self.declare_parameter("entity_group_max_bbox_gap_m", 0.10)
        self.declare_parameter("entity_group_min_overlap", 0.03)
        self.declare_parameter("entity_group_min_geometry_score", 0.35)
        self.declare_parameter("entity_group_max_area_growth", 2.2)
        self.declare_parameter("entity_group_max_volume_growth", 3.2)
        self.declare_parameter("entity_group_max_z_growth", 2.3)
        self.declare_parameter("entity_group_candidate_multiplier", 4)
        self.declare_parameter("segment_track_debug_log", True)
        self.declare_parameter("segment_track_debug_interval_s", 5.0)

        # robust fusion / soft gating (STM fusion)
        self.declare_parameter("use_consistency_gating", True)
        self.declare_parameter("gating_cos_min", 0.15)
        self.declare_parameter("gating_reject", False)  # keep False for soft gating
        self.declare_parameter("gating_soft_power", 2.0)
        self.declare_parameter("gating_soft_floor", 0.05)
        self.declare_parameter("gating_soft_neg_penalty", 0.5)

        # per-point confidence model
        self.declare_parameter("conf_d0_m", 2.5)
        self.declare_parameter("conf_min", 0.02)
        self.declare_parameter("conf_fine_boost", 1.0)

        # STM cell state
        self.declare_parameter("cell_weight_max", 50.0)
        self.declare_parameter("decay_half_life_s", 120.0)

        # LTM cell state (separate caps / update gains)
        self.declare_parameter("ltm_weight_max", 200.0)
        self.declare_parameter("ltm_update_gain", 1.0)  # how strongly a promotion adds to LTM
        self.declare_parameter("ltm_replace_cos_thresh", 0.25)  # if cos(new, old) below, replace fast
        self.declare_parameter("ltm_replace_gain", 2.5)  # replacement strength (larger -> quicker overwrite)

        # promotion STM -> LTM filters (noise rejection)
        self.declare_parameter("promote_min_conf", 0.45)   # 0..1 cell confidence used in grids
        self.declare_parameter("promote_min_coh01", 0.35)  # coherence01 threshold
        self.declare_parameter("promote_min_views", 2)     # required distinct view bins

        # coherence
        self.declare_parameter("use_coherence_in_confidence", True)
        self.declare_parameter("use_coherence_in_semantic", True)
        self.declare_parameter("coherence_floor", 0.15)
        self.declare_parameter("coherence_power", 1.5)

        # visibility boost (applied in grid confidence + also used for promotion views)
        self.declare_parameter("use_visibility_boost", True)
        self.declare_parameter("view_bins", 8)
        self.declare_parameter("view_boost_strength", 0.7)
        self.declare_parameter("view_boost_power", 1.0)

        # ray consistency
        self.declare_parameter("use_ray_consistency", True)
        self.declare_parameter("occ_grid_topic", "/map")
        self.declare_parameter("occ_thresh", 50)
        self.declare_parameter("ray_check_stride", 6)
        self.declare_parameter("depth_use_latest_tf", False)
        self.declare_parameter("depth_allow_latest_tf_fallback", False)
        self.declare_parameter("stationary_pose_lock", False)
        self.declare_parameter("depth_tf_timeout_s", 0.03)
        self.declare_parameter("ray_use_latest_tf", True)
        self.declare_parameter("publish_tf_timeout_s", 0.05)

        # normalization for semantic grid
        self.declare_parameter("norm_low_percentile", 65.0)
        self.declare_parameter("norm_high_percentile", 99.0)

        # history cloud
        self.declare_parameter("update_history_cloud", False)
        self.declare_parameter("history_voxel_m", 0.05)
        self.declare_parameter("history_max_points", 300000)

        # persistence
        self.declare_parameter("autoload_map_state", False)
        self.declare_parameter("autosave_map_state", False)
        self.declare_parameter("autosave_interval_s", 3.0)
        self.declare_parameter("map_state_path", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".runtime", "map_state.pkl"))

        # Lightweight stage timing profiler.
        self.declare_parameter("profile_timing", True)
        self.declare_parameter("profile_interval_s", 5.0)
        self.declare_parameter("profile_top_n", 24)
        self.declare_parameter("observation_cycle_average_window", 30)

        # CLIP model selection. ViT-L-14 is stronger; ViT-B-32 is much faster.
        self.declare_parameter("clip_model_name", "ViT-L-14")
        self.declare_parameter("clip_pretrained", "openai")
        self.declare_parameter("clip_text_prompt_ensemble", True)
        self.declare_parameter("clip_mask_embedding_variants", "context")
        self.declare_parameter("clip_context_margin_frac", 0.12)
        self.declare_parameter("clip_context_bg_alpha", 0.35)
        self.declare_parameter("clip_mask_min_crop_px", 32)
        self.declare_parameter("use_clip_dense_features", False)
        self.declare_parameter("clip_dense_point_fusion", False)
        self.declare_parameter("clip_dense_mask_weight", 0.70)
        self.declare_parameter("clip_dense_input_size", 0)

        # DINOv2 supplies appearance correspondence only. Its descriptors stay
        # separate from CLIP, which remains the open-vocabulary semantic space.
        self.declare_parameter("use_dinov2_features", True)
        self.declare_parameter("dinov2_model_name", "dinov2_vits14")
        self.declare_parameter("dinov2_repo", "facebookresearch/dinov2")
        self.declare_parameter("dinov2_local_repo", "")
        self.declare_parameter("dinov2_checkpoint", "")
        self.declare_parameter("dinov2_allow_download", True)
        # One shared, aspect-preserving full-frame pass. 518 is exactly 37
        # ViT/14 patches and gives materially finer masks without per-object
        # crops or additional DINO forwards.
        self.declare_parameter("dinov2_input_size", 518)
        self.declare_parameter("dinov2_mask_min_weight", 0.35)
        self.declare_parameter("dinov2_context_ring_tokens", 1)
        self.declare_parameter("dinov2_context_contrast_weight", 0.35)
        self.declare_parameter("dinov2_min_effective_tokens", 2.0)
        self.declare_parameter("dinov2_assoc_min_quality", 0.25)
        self.declare_parameter("dinov2_track_update_gain", 1.0)
        self.declare_parameter("dinov2_track_ema_alpha", 0.18)
        self.declare_parameter("dinov2_track_memory_weight", 12.0)

        # -------------------------
        # Read params (initial)
        # -------------------------
        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.cam_info_topic = self.get_parameter("cam_info_topic").value
        self.rgb_compressed = bool(self.get_parameter("rgb_compressed").value)
        self.depth_compressed = bool(self.get_parameter("depth_compressed").value)
        self.internal_frame = self.get_parameter("internal_frame").value
        self.publish_frame = self.get_parameter("publish_frame").value

        self.stride = int(self.get_parameter("stride").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)

        self.grid_size_m = float(self.get_parameter("grid_size_m").value)
        self.resolution = float(self.get_parameter("resolution").value)
        self.query_text = str(self.get_parameter("query_text").value)

        self.grid_width = int(self.grid_size_m / self.resolution)
        self.grid_height = int(self.grid_size_m / self.resolution)
        self.origin_x = -self.grid_size_m / 2.0
        self.origin_y = -self.grid_size_m / 2.0
        self.voxel_size = float(self.get_parameter("voxel_size_m").value)
        self.voxel_z_min = float(self.get_parameter("voxel_z_min_m").value)
        self.voxel_z_max = float(self.get_parameter("voxel_z_max_m").value)
        self.voxel_width = max(1, int(math.ceil(self.grid_size_m / max(self.voxel_size, 1e-6))))
        self.voxel_height = max(1, int(math.ceil(self.grid_size_m / max(self.voxel_size, 1e-6))))
        self.voxel_depth = max(1, int(math.ceil((self.voxel_z_max - self.voxel_z_min) / max(self.voxel_size, 1e-6))))
        self.map_state_path = os.path.expanduser(str(self.get_parameter("map_state_path").value))

        # -------------------------
        # STM state (INTERNAL frame grid coords)
        # -------------------------
        self.stm_sum: Dict[Tuple[int, int], np.ndarray] = {}
        self.stm_w: Dict[Tuple[int, int], float] = {}
        self.stm_last: Dict[Tuple[int, int], float] = {}
        self.stm_viewmask: Dict[Tuple[int, int], int] = {}
        self.stm_z_min: Dict[Tuple[int, int], float] = {}
        self.stm_z_max: Dict[Tuple[int, int], float] = {}
        self.stm_z_sum: Dict[Tuple[int, int], float] = {}
        self.stm_z_count: Dict[Tuple[int, int], int] = {}

        # -------------------------
        # LTM state (INTERNAL frame grid coords)
        # -------------------------
        self.ltm_sum: Dict[Tuple[int, int], np.ndarray] = {}
        self.ltm_w: Dict[Tuple[int, int], float] = {}
        self.ltm_last_seen: Dict[Tuple[int, int], float] = {}  # seconds, last evidence accepted
        self.ltm_viewmask: Dict[Tuple[int, int], int] = {}     # optional tracking of view diversity in LTM
        self.ltm_z_min: Dict[Tuple[int, int], float] = {}
        self.ltm_z_max: Dict[Tuple[int, int], float] = {}
        self.ltm_z_sum: Dict[Tuple[int, int], float] = {}
        self.ltm_z_count: Dict[Tuple[int, int], int] = {}

        # -------------------------
        # 3D semantic voxel memory (INTERNAL frame sparse grid)
        # -------------------------
        self.vox_stm_sum: Dict[Tuple[int, int, int], np.ndarray] = {}
        self.vox_stm_w: Dict[Tuple[int, int, int], float] = {}
        self.vox_stm_last: Dict[Tuple[int, int, int], float] = {}
        self.vox_stm_viewmask: Dict[Tuple[int, int, int], int] = {}
        self.vox_ltm_sum: Dict[Tuple[int, int, int], np.ndarray] = {}
        self.vox_ltm_w: Dict[Tuple[int, int, int], float] = {}
        self.vox_ltm_last_seen: Dict[Tuple[int, int, int], float] = {}
        self.vox_ltm_viewmask: Dict[Tuple[int, int, int], int] = {}
        self.vox_clear_hits: Dict[Tuple[int, int, int], int] = {}

        # Segment-centric memory. Segment ids are the stable primitives; object
        # clouds and graph nodes can be rendered directly from these records.
        self.seg_next_id = 1
        self.seg_sum: Dict[int, np.ndarray] = {}
        self.seg_w: Dict[int, float] = {}
        self.seg_first_seen: Dict[int, float] = {}
        self.seg_last_seen: Dict[int, float] = {}
        self.seg_obs_count: Dict[int, int] = {}
        self.seg_last_obs_seq: Dict[int, int] = {}
        self.seg_viewmask: Dict[int, int] = {}
        self.seg_centroid: Dict[int, np.ndarray] = {}
        self.seg_bbox_min: Dict[int, np.ndarray] = {}
        self.seg_bbox_max: Dict[int, np.ndarray] = {}
        self.seg_support_cells: Dict[int, set] = {}
        self.seg_support_voxels: Dict[int, set] = {}
        self.seg_dino_sum: Dict[int, np.ndarray] = {}
        self.seg_dino_w: Dict[int, float] = {}
        self.seg_merge_block_until: Dict[int, float] = {}
        self.voxel_segment_id: Dict[Tuple[int, int, int], int] = {}
        self.segment_voxel_miss_hits: Dict[Tuple[int, int, int], int] = {}
        self.segment_semantic_conflict_hits: Dict[Tuple[int, int, int], int] = {}
        self.segment_semantic_conflict_last_seq: Dict[Tuple[int, int, int], int] = {}
        self.segment_query_support_cache: Dict[int, dict] = {}
        self.segment_query_score_norm_bounds = None
        self.last_segment_track_debug_log_s = 0.0
        self.last_segment_maintenance_s = 0.0
        self.last_segment_maintenance_debug_log_s = 0.0
        self.last_debug_observation_image_pub_s = 0.0
        self.observation_cycle_samples = deque(
            maxlen=max(1, int(self.get_parameter("observation_cycle_average_window").value))
        )
        self.observation_cycle_total_count = 0
        self.observation_cycle_total_ms = 0.0
        self.dinov2_color_projection = None
        self.dinov2_color_projection_dim = 0
        self.clip_color_projection = None
        self.clip_color_projection_dim = 0
        self.feature_color_projections: Dict[str, np.ndarray] = {}
        self.last_segment_assoc_info = {}
        self.last_segment_fuse_created = False
        self._segment_obs_reject_reason = "none"
        self.entity_memory = PersistentEntities()
        self.surface_consensus = SurfaceConsensus()
        self.segment_display_memory = DominantOverlay()
        self.entity_display_memory = DominantOverlay()
        self.boundary_votes = BoundaryOwnershipVotes()
        self.dino_base_dim = 0
        self.seg_last_dino_seq = {}
        self.world_model_id = new_world_id()
        self.observation_source_id = new_world_id()
        self.entity_edges = ObservationEdges()
        self.segment_edges = ObservationEdges()
        self.last_applied_observation_seq = -1
        self.last_accepted_depth_stamp_ns = 0
        self.last_identity_update_ms = 0.0
        self.last_segment_projection_context = None

        # history cloud stored in INTERNAL frame voxel grid
        self.history_vox: Dict[Tuple[int, int, int], Tuple[float, float, float, float]] = {}
        self.white_rgb = pack_rgb(255, 255, 255)
        self.latest_grids = self._empty_grid_bundle()
        self.has_loaded_grid_snapshot = False
        self.state_dirty = False
        self.last_T_publish_internal: Optional[np.ndarray] = None
        self.saved_T_publish_internal: Optional[np.ndarray] = None
        self.profile_stats = {}
        self.profile_last_log_s = self.now_sec()
        self.semantic_job_queue = queue.Queue(
            maxsize=max(1, int(self.get_parameter("async_semantic_queue_size").value))
        )
        self.semantic_result_queue = queue.Queue(
            maxsize=max(1, int(self.get_parameter("async_semantic_result_queue_size").value))
        )

        # Occupancy grid (publish_frame) for ray-consistency
        self.occ: Optional[dict] = None

        # camera intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # CLIP
        #self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.get_logger().info(f"Using device: {self.device}")

        # MODEL SELECTION -> ViT-B-32 is faster but less accurate, ViT-L-14 is slower but better
        model_name = str(self.get_parameter("clip_model_name").value)
        pretrained = str(self.get_parameter("clip_pretrained").value)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.model.to(self.device)

        # EfficientSAM
        self.esam_model = None
        self.esam_unavailable_warned = False
        if bool(self.get_parameter("use_esam_masks").value):
            self.init_esam()

        # DINOv2 appearance model. Initialization failure is non-fatal: CLIP
        # semantics and geometric association continue to operate.
        self.dinov2_model = None
        self.dinov2_unavailable_warned = False
        self.dinov2_patch_size = 14
        if bool(self.get_parameter("use_dinov2_features").value):
            self.init_dinov2()

        # query embedding cache
        self.text_emb: Optional[np.ndarray] = None
        self.text_emb_query: Optional[str] = None

        # cached tiles
        self.tile_stamp = None
        self.tile_coarse = None
        self.tile_fine = None

        # -------------------------
        # ROS pubs/subs
        # -------------------------
        debug_image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # STM grids
        self.pub_stm_sem = self.create_publisher(OccupancyGrid, "/crossmaps/stm/semantic_grid", 1)
        self.pub_stm_conf = self.create_publisher(OccupancyGrid, "/crossmaps/stm/confidence_grid", 1)
        self.pub_stm_top1 = self.create_publisher(OccupancyGrid, "/crossmaps/stm/top1_grid", 1)

        # LTM grids
        self.pub_ltm_sem = self.create_publisher(OccupancyGrid, "/crossmaps/ltm/semantic_grid", 1)
        self.pub_ltm_conf = self.create_publisher(OccupancyGrid, "/crossmaps/ltm/confidence_grid", 1)
        self.pub_ltm_top1 = self.create_publisher(OccupancyGrid, "/crossmaps/ltm/top1_grid", 1)

        # clouds
        self.pub_debug_cloud = self.create_publisher(PointCloud2, "/crossmaps/debug_cloud", 10)
        self.pub_sem_cloud = self.create_publisher(PointCloud2, "/crossmaps/semantic_cloud", 10)
        self.pub_hist_cloud = self.create_publisher(PointCloud2, "/crossmaps/history_cloud", 1)
        self.pub_esam_mask_overlay = self.create_publisher(Image, "/crossmaps/esam/mask_overlay", debug_image_qos)
        self.pub_dinov2_feature_view = self.create_publisher(
            Image, "/crossmaps/dino/feature_view", debug_image_qos
        )
        self.pub_dinov2_similarity_view = self.create_publisher(
            Image, "/crossmaps/dino/similarity_view", debug_image_qos
        )
        self.pub_dinov2_instance_view = self.create_publisher(
            Image, "/crossmaps/dino/instance_view", debug_image_qos
        )
        self.pub_clip_feature_view = self.create_publisher(
            Image, "/crossmaps/clip/feature_view", debug_image_qos
        )
        self.pub_clip_similarity_view = self.create_publisher(
            Image, "/crossmaps/clip/similarity_view", debug_image_qos
        )
        self.pub_segment_id_projection = self.create_publisher(Image, "/crossmaps/debug/segment_id_projection", debug_image_qos)
        self.pub_entity_id_projection = self.create_publisher(Image, "/crossmaps/debug/entity_id_projection", debug_image_qos)
        self.pub_entity_id_projection_compressed = self.create_publisher(
            CompressedImage, "/crossmaps/debug/entity_id_projection/compressed", debug_image_qos
        )
        self.pub_debug_observation_rgb = self.create_publisher(Image, "/crossmaps/debug/last_observation_rgb", debug_image_qos)
        self.pub_debug_observation_depth = self.create_publisher(Image, "/crossmaps/debug/last_observation_depth", debug_image_qos)
        self.pub_esam_mask_overlay_compressed = self.create_publisher(
            CompressedImage, "/crossmaps/esam/mask_overlay/compressed", debug_image_qos
        )
        self.pub_dinov2_feature_view_compressed = self.create_publisher(
            CompressedImage, "/crossmaps/dino/feature_view/compressed", debug_image_qos
        )
        self.pub_dinov2_similarity_view_compressed = self.create_publisher(
            CompressedImage, "/crossmaps/dino/similarity_view/compressed", debug_image_qos
        )
        self.pub_dinov2_instance_view_compressed = self.create_publisher(
            CompressedImage, "/crossmaps/dino/instance_view/compressed", debug_image_qos
        )
        self.pub_clip_feature_view_compressed = self.create_publisher(
            CompressedImage, "/crossmaps/clip/feature_view/compressed", debug_image_qos
        )
        self.pub_clip_similarity_view_compressed = self.create_publisher(
            CompressedImage, "/crossmaps/clip/similarity_view/compressed", debug_image_qos
        )
        self.pub_segment_id_projection_compressed = self.create_publisher(
            CompressedImage, "/crossmaps/debug/segment_id_projection/compressed", debug_image_qos
        )
        self.pub_debug_observation_rgb_compressed = self.create_publisher(
            CompressedImage, "/crossmaps/debug/last_observation_rgb/compressed", debug_image_qos
        )
        self.pub_debug_observation_depth_compressed = self.create_publisher(
            CompressedImage, "/crossmaps/debug/last_observation_depth/compressed", debug_image_qos
        )
        self.pub_obj_query_voxels = self.create_publisher(PointCloud2, "/crossmaps/objects/query_voxels", 1)
        self.pub_obj_instance_voxels = self.create_publisher(PointCloud2, "/crossmaps/objects/instance_voxels", 1)
        self.pub_segment_voxels = self.create_publisher(PointCloud2, "/crossmaps/segments/voxels", 1)
        self.pub_entity_voxels = self.create_publisher(PointCloud2, "/crossmaps/entities/voxels", 1)
        self.entity_correction_sub = self.create_subscription(
            String, "/crossmaps/entities/corrections", self.on_entity_correction, 10
        )
        self.pub_obj_unassigned_query_voxels = self.create_publisher(
            PointCloud2, "/crossmaps/objects/unassigned_query_voxels", 1
        )
        self.pub_obj_graph_nodes = self.create_publisher(MarkerArray, "/crossmaps/object_graph/nodes", 1)
        self.pub_obj_graph_edges = self.create_publisher(MarkerArray, "/crossmaps/object_graph/edges", 1)
        self.pub_obj_graph_descriptors = self.create_publisher(String, "/crossmaps/object_graph/descriptors", 1)
        self.pub_observation_cycle = self.create_publisher(
            String, "/crossmaps/diagnostics/observation_cycle", 10
        )
        self.pub_pipeline_health = self.create_publisher(String, "/crossmaps/diagnostics/pipeline_health", 10)
        self.pub_voxel_stm_heatmap = self.create_publisher(PointCloud2, "/crossmaps/voxels/stm_heatmap", 1)
        self.pub_voxel_ltm_heatmap = self.create_publisher(PointCloud2, "/crossmaps/voxels/ltm_heatmap", 1)
        self.pub_voxel_fused_heatmap = self.create_publisher(PointCloud2, "/crossmaps/voxels/fused_heatmap", 1)

        self.timer = self.create_timer(1.0 / float(self.get_parameter("publish_hz").value), self.on_publish)
        self.autosave_timer = self.create_timer(
            float(self.get_parameter("autosave_interval_s").value),
            self.on_autosave_timer,
        )

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=int(self.get_parameter("sensor_qos_depth").value),
            reliability=(
                ReliabilityPolicy.BEST_EFFORT
                if bool(self.get_parameter("use_best_effort_sensor_qos").value)
                else ReliabilityPolicy.RELIABLE
            ),
            durability=DurabilityPolicy.VOLATILE,
        )

        self.sub_cam = self.create_subscription(CameraInfo, self.cam_info_topic, self.on_cam_info, sensor_qos)

        # occupancy grid subscription (still used by ray-consistency)
        self.sub_occ = self.create_subscription(
            OccupancyGrid, self.get_parameter("occ_grid_topic").value, self.on_occ_grid, 1
        )

        # RGB+Depth sync
        rgb_msg_type = CompressedImage if self.rgb_compressed else Image
        depth_msg_type = CompressedImage if self.depth_compressed else Image
        self.rgb_sub = message_filters.Subscriber(self, rgb_msg_type, self.rgb_topic, qos_profile=sensor_qos)
        self.depth_sub = message_filters.Subscriber(self, depth_msg_type, self.depth_topic, qos_profile=sensor_qos)
        self.rgb_sub.registerCallback(self._note_camera_input, "rgb")
        self.depth_sub.registerCallback(self._note_camera_input, "depth")
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=int(self.get_parameter("sync_queue_size").value),
            slop=float(self.get_parameter("sync_slop_s").value),
            allow_headerless=False,
        )
        self.ts.registerCallback(self.on_rgb_depth)
        self.observation_timer = self.create_timer(0.1, self.on_observation_timer)
        self.pipeline_health_timer = self.create_timer(1.0, self.publish_pipeline_health)

        if bool(self.get_parameter("autoload_map_state").value):
            self.load_map_state()

        if bool(self.get_parameter("use_async_semantic_worker").value):
            self.start_async_semantic_worker()

        self.get_logger().info("CrossMaps v2 node started (surface memory, entity grouping, STM + LTM).")
        self.get_logger().info(f"internal_frame={self.internal_frame}, publish_frame={self.publish_frame}")
        self.get_logger().info(
            "Publishes: "
            "/crossmaps/stm/{semantic_grid,confidence_grid,top1_grid}, "
            "/crossmaps/ltm/{semantic_grid,confidence_grid,top1_grid}, "
            "/crossmaps/debug_cloud, /crossmaps/semantic_cloud, /crossmaps/history_cloud, "
            "/crossmaps/objects/query_voxels, /crossmaps/object_graph/{nodes,edges}"
        )

    # -------------------------
    # Basics
    # -------------------------
    def now_sec(self):
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _profile_enabled(self) -> bool:
        try:
            return bool(self.get_parameter("profile_timing").value)
        except Exception:
            return False

    def _profile_add_ms(self, name: str, ms: float):
        if not self._profile_enabled():
            return
        rec = self.profile_stats.get(name)
        if rec is None:
            rec = {"n": 0, "sum": 0.0, "max": 0.0}
            self.profile_stats[name] = rec
        val = float(ms)
        rec["n"] += 1
        rec["sum"] += val
        rec["max"] = max(float(rec["max"]), val)

    def _profile_add_elapsed(self, name: str, start: float):
        self._profile_add_ms(name, (time.perf_counter() - start) * 1000.0)

    def _profile_add_value(self, name: str, value: float):
        self._profile_add_ms(f"count.{name}", float(value))

    def _profile_report_if_due(self):
        if not self._profile_enabled():
            return
        now_s = self.now_sec()
        interval = float(self.get_parameter("profile_interval_s").value)
        if interval > 0.0 and (now_s - self.profile_last_log_s) < interval:
            return
        self.profile_last_log_s = now_s
        if not self.profile_stats:
            return

        time_items = []
        count_items = []
        for name, rec in self.profile_stats.items():
            n = max(1, int(rec["n"]))
            avg = float(rec["sum"]) / float(n)
            mx = float(rec["max"])
            if name.startswith("count."):
                count_items.append((avg, mx, n, name))
            else:
                time_items.append((avg, mx, n, name))
        time_items.sort(key=lambda x: x[0], reverse=True)
        count_items.sort(key=lambda x: x[0], reverse=True)
        top_n = max(1, int(self.get_parameter("profile_top_n").value))
        parts = []
        for avg, mx, n, name in time_items[:top_n]:
            parts.append(f"{name} avg={avg:.1f}ms max={mx:.1f}ms n={n}")
        for avg, mx, n, name in count_items:
            parts.append(f"{name[6:]} avg={avg:.1f} max={mx:.1f} n={n}")
        self.get_logger().info(
            "profile ms: "
            + " | ".join(parts)
            + f" | sizes stm={len(self.stm_w)} ltm={len(self.ltm_w)} "
            + f"vox_stm={len(self.vox_stm_w)} vox_ltm={len(self.vox_ltm_w)}"
        )
        self.profile_stats = {}

    def _record_observation_cycle(self, start_perf_s: Optional[float], mode: str, seq: int):
        if start_perf_s is None:
            return
        elapsed_ms = max(0.0, (time.perf_counter() - float(start_perf_s)) * 1000.0)
        desired_window = max(
            1, int(self.get_parameter("observation_cycle_average_window").value)
        )
        if self.observation_cycle_samples.maxlen != desired_window:
            self.observation_cycle_samples = deque(
                list(self.observation_cycle_samples)[-desired_window:],
                maxlen=desired_window,
            )
        self.observation_cycle_samples.append(float(elapsed_ms))
        self.observation_cycle_total_count += 1
        self.observation_cycle_total_ms += float(elapsed_ms)
        samples = np.asarray(self.observation_cycle_samples, dtype=np.float64)
        rolling_avg_ms = float(np.mean(samples)) if samples.size else 0.0
        avg_ms = float(self.observation_cycle_total_ms) / float(
            max(self.observation_cycle_total_count, 1)
        )
        p95_ms = float(np.percentile(samples, 95.0)) if samples.size else 0.0
        msg = String()
        msg.data = json.dumps(
            {
                "schema": "crossmaps_observation_cycle_v1",
                "stamp_s": round(self.now_sec(), 3),
                "sequence": int(seq),
                "mode": str(mode),
                "last_ms": round(float(elapsed_ms), 3),
                "average_ms": round(avg_ms, 3),
                "rolling_average_ms": round(rolling_avg_ms, 3),
                "p95_ms": round(p95_ms, 3),
                "window_count": int(samples.size),
                "window_size": int(desired_window),
                "total_count": int(self.observation_cycle_total_count),
                "average_hz": round(1000.0 / avg_ms, 3) if avg_ms > 1e-6 else 0.0,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        self.pub_observation_cycle.publish(msg)
        if hasattr(self, "pipeline_health"):
            self.pipeline_health.complete()
        self._profile_add_ms("observation.full_cycle", elapsed_ms)

    def _empty_grid_bundle(self):
        empty_sem = -np.ones((self.grid_height, self.grid_width), dtype=np.int8)
        empty_conf = -np.ones((self.grid_height, self.grid_width), dtype=np.int8)
        empty_top1 = -np.ones((self.grid_height, self.grid_width), dtype=np.int8)
        return {
            "stm_semantic": empty_sem.copy(),
            "stm_confidence": empty_conf.copy(),
            "stm_top1": empty_top1.copy(),
            "ltm_semantic": empty_sem.copy(),
            "ltm_confidence": empty_conf.copy(),
            "ltm_top1": empty_top1.copy(),
        }

    def _mark_state_dirty(self):
        self.state_dirty = True

    def _serialize_vec_dict(self, data: Dict[Tuple[int, int], np.ndarray]):
        if not data:
            return {"keys": np.zeros((0, 2), dtype=np.int32), "values": np.zeros((0, 0), dtype=np.float32)}
        keys = np.array(list(data.keys()), dtype=np.int32)
        values = np.stack([np.asarray(data[k], dtype=np.float32) for k in data.keys()], axis=0)
        return {"keys": keys, "values": values}

    def _deserialize_vec_dict(self, payload) -> Dict[Tuple[int, int], np.ndarray]:
        keys = np.asarray(payload.get("keys", np.zeros((0, 2), dtype=np.int32)), dtype=np.int32)
        values = np.asarray(payload.get("values", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
        out: Dict[Tuple[int, int], np.ndarray] = {}
        for idx, key_arr in enumerate(keys):
            out[(int(key_arr[0]), int(key_arr[1]))] = values[idx].astype(np.float32)
        return out

    def _serialize_scalar_dict(self, data: Dict[Tuple[int, int], float], dtype):
        if not data:
            return {"keys": np.zeros((0, 2), dtype=np.int32), "values": np.zeros((0,), dtype=dtype)}
        keys = np.array(list(data.keys()), dtype=np.int32)
        values = np.array([data[k] for k in data.keys()], dtype=dtype)
        return {"keys": keys, "values": values}

    def _deserialize_scalar_dict(self, payload, value_cast):
        keys = np.asarray(payload.get("keys", np.zeros((0, 2), dtype=np.int32)), dtype=np.int32)
        values = payload.get("values", np.zeros((0,), dtype=np.float32))
        out = {}
        for idx, key_arr in enumerate(keys):
            out[(int(key_arr[0]), int(key_arr[1]))] = value_cast(values[idx])
        return out

    def _serialize_voxel_vec_dict(self, data: Dict[Tuple[int, int, int], np.ndarray]):
        if not data:
            return {"keys": np.zeros((0, 3), dtype=np.int32), "values": np.zeros((0, 0), dtype=np.float32)}
        keys = np.array(list(data.keys()), dtype=np.int32)
        values = np.stack([np.asarray(data[k], dtype=np.float32) for k in data.keys()], axis=0)
        return {"keys": keys, "values": values}

    def _deserialize_voxel_vec_dict(self, payload) -> Dict[Tuple[int, int, int], np.ndarray]:
        keys = np.asarray(payload.get("keys", np.zeros((0, 3), dtype=np.int32)), dtype=np.int32)
        values = np.asarray(payload.get("values", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
        out: Dict[Tuple[int, int, int], np.ndarray] = {}
        for idx, key_arr in enumerate(keys):
            out[(int(key_arr[0]), int(key_arr[1]), int(key_arr[2]))] = values[idx].astype(np.float32)
        return out

    def _serialize_voxel_scalar_dict(self, data: Dict[Tuple[int, int, int], float], dtype):
        if not data:
            return {"keys": np.zeros((0, 3), dtype=np.int32), "values": np.zeros((0,), dtype=dtype)}
        keys = np.array(list(data.keys()), dtype=np.int32)
        values = np.array([data[k] for k in data.keys()], dtype=dtype)
        return {"keys": keys, "values": values}

    def _deserialize_voxel_scalar_dict(self, payload, value_cast):
        keys = np.asarray(payload.get("keys", np.zeros((0, 3), dtype=np.int32)), dtype=np.int32)
        values = payload.get("values", np.zeros((0,), dtype=np.float32))
        out = {}
        for idx, key_arr in enumerate(keys):
            out[(int(key_arr[0]), int(key_arr[1]), int(key_arr[2]))] = value_cast(values[idx])
        return out

    def _serialize_history_vox(self):
        if not self.history_vox:
            return {"keys": np.zeros((0, 3), dtype=np.int32), "values": np.zeros((0, 4), dtype=np.float32)}
        keys = np.array(list(self.history_vox.keys()), dtype=np.int32)
        values = np.array(list(self.history_vox.values()), dtype=np.float32)
        return {"keys": keys, "values": values}

    def _deserialize_history_vox(self, payload):
        keys = np.asarray(payload.get("keys", np.zeros((0, 3), dtype=np.int32)), dtype=np.int32)
        values = np.asarray(payload.get("values", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32)
        out = {}
        for idx, key_arr in enumerate(keys):
            out[(int(key_arr[0]), int(key_arr[1]), int(key_arr[2]))] = tuple(float(v) for v in values[idx])
        return out

    def _serialize_segment_dino_sums(self, ids: np.ndarray):
        dims = [int(self.seg_dino_sum[int(i)].shape[0]) for i in ids if int(i) in self.seg_dino_sum]
        dim = max(dims) if dims else 0
        out = np.zeros((int(ids.size), dim), dtype=np.float32)
        for idx, sid_arr in enumerate(ids):
            emb = self.seg_dino_sum.get(int(sid_arr))
            if emb is not None:
                out[idx, :emb.shape[0]] = emb.astype(np.float32)
        return out

    def _serialize_segments(self):
        ids = np.array(sorted(self.seg_w.keys()), dtype=np.int64)
        voxel_owner_keys = np.array(list(self.voxel_segment_id.keys()), dtype=np.int32) if self.voxel_segment_id else np.zeros((0, 3), dtype=np.int32)
        voxel_owner_values = (
            np.array([self.voxel_segment_id[k] for k in self.voxel_segment_id.keys()], dtype=np.int64)
            if self.voxel_segment_id
            else np.zeros((0,), dtype=np.int64)
        )
        if ids.size == 0:
            return {
                "next_id": int(self.seg_next_id),
                "ids": ids,
                "sum": np.zeros((0, 0), dtype=np.float32),
                "w": np.zeros((0,), dtype=np.float32),
                "first_seen": np.zeros((0,), dtype=np.float64),
                "last_seen": np.zeros((0,), dtype=np.float64),
                "obs_count": np.zeros((0,), dtype=np.uint32),
                "last_obs_seq": np.zeros((0,), dtype=np.int64),
                "viewmask": np.zeros((0,), dtype=np.uint32),
                "centroid": np.zeros((0, 3), dtype=np.float32),
                "bbox_min": np.zeros((0, 3), dtype=np.float32),
                "bbox_max": np.zeros((0, 3), dtype=np.float32),
                "dino_sum": np.zeros((0, 0), dtype=np.float32),
                "dino_w": np.zeros((0,), dtype=np.float32),
                "support_cells": [],
                "support_voxels": [],
                "voxel_owner_keys": voxel_owner_keys,
                "voxel_owner_values": voxel_owner_values,
            }
        return {
            "next_id": int(self.seg_next_id),
            "ids": ids,
            "sum": np.stack([self.seg_sum[int(i)] for i in ids], axis=0).astype(np.float32),
            "w": np.array([self.seg_w[int(i)] for i in ids], dtype=np.float32),
            "first_seen": np.array([self.seg_first_seen.get(int(i), self.seg_last_seen.get(int(i), 0.0)) for i in ids], dtype=np.float64),
            "last_seen": np.array([self.seg_last_seen.get(int(i), 0.0) for i in ids], dtype=np.float64),
            "obs_count": np.array([self.seg_obs_count.get(int(i), 1) for i in ids], dtype=np.uint32),
            "last_obs_seq": np.array([self.seg_last_obs_seq.get(int(i), -1) for i in ids], dtype=np.int64),
            "viewmask": np.array([self.seg_viewmask.get(int(i), 0) for i in ids], dtype=np.uint32),
            "centroid": np.stack([self.seg_centroid[int(i)] for i in ids], axis=0).astype(np.float32),
            "bbox_min": np.stack([self.seg_bbox_min[int(i)] for i in ids], axis=0).astype(np.float32),
            "bbox_max": np.stack([self.seg_bbox_max[int(i)] for i in ids], axis=0).astype(np.float32),
            "dino_sum": self._serialize_segment_dino_sums(ids),
            "dino_dims": np.array([self.seg_dino_sum[int(i)].size if int(i) in self.seg_dino_sum else 0
                                   for i in ids], dtype=np.int32),
            "dino_w": np.array([self.seg_dino_w.get(int(i), 0.0) for i in ids], dtype=np.float32),
            "support_cells": [
                np.array(sorted(self.seg_support_cells.get(int(i), set())), dtype=np.int32)
                for i in ids
            ],
            "support_voxels": [
                np.array(sorted(self.seg_support_voxels.get(int(i), set())), dtype=np.int32)
                for i in ids
            ],
            "voxel_owner_keys": voxel_owner_keys,
            "voxel_owner_values": voxel_owner_values,
        }

    def _deserialize_segments(self, payload):
        self.seg_next_id = int(payload.get("next_id", 1))
        self.seg_sum = {}
        self.seg_w = {}
        self.seg_first_seen = {}
        self.seg_last_seen = {}
        self.seg_obs_count = {}
        self.seg_last_obs_seq = {}
        self.seg_viewmask = {}
        self.seg_centroid = {}
        self.seg_bbox_min = {}
        self.seg_bbox_max = {}
        self.seg_dino_sum = {}
        self.seg_dino_w = {}
        self.seg_merge_block_until = {}
        self.seg_support_cells = {}
        self.seg_support_voxels = {}
        self.voxel_segment_id = {}
        self.segment_voxel_miss_hits = {}

        ids = np.asarray(payload.get("ids", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        sums = np.asarray(payload.get("sum", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
        weights = np.asarray(payload.get("w", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        firsts = np.asarray(payload.get("first_seen", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
        lasts = np.asarray(payload.get("last_seen", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
        obs_counts = np.asarray(payload.get("obs_count", np.ones((ids.shape[0],), dtype=np.uint32)), dtype=np.uint32)
        last_obs_seqs = np.asarray(payload.get("last_obs_seq", -np.ones((ids.shape[0],), dtype=np.int64)), dtype=np.int64)
        viewmasks = np.asarray(payload.get("viewmask", np.zeros((0,), dtype=np.uint32)), dtype=np.uint32)
        centroids = np.asarray(payload.get("centroid", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        bbox_min = np.asarray(payload.get("bbox_min", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        bbox_max = np.asarray(payload.get("bbox_max", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        dino_sums = np.asarray(payload.get("dino_sum", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
        dino_weights = np.asarray(payload.get("dino_w", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        dino_dims = payload.get("dino_dims")
        support_cells = payload.get("support_cells", [])
        support_voxels = payload.get("support_voxels", [])

        for idx, sid_arr in enumerate(ids):
            sid = int(sid_arr)
            self.seg_sum[sid] = sums[idx].astype(np.float32)
            self.seg_w[sid] = float(weights[idx])
            self.seg_first_seen[sid] = float(firsts[idx]) if idx < firsts.shape[0] and float(firsts[idx]) > 0.0 else float(lasts[idx]) if idx < lasts.shape[0] else 0.0
            self.seg_last_seen[sid] = float(lasts[idx]) if idx < lasts.shape[0] else 0.0
            self.seg_obs_count[sid] = int(obs_counts[idx]) if idx < obs_counts.shape[0] else 1
            self.seg_last_obs_seq[sid] = int(last_obs_seqs[idx]) if idx < last_obs_seqs.shape[0] else -1
            self.seg_viewmask[sid] = int(viewmasks[idx]) if idx < viewmasks.shape[0] else 0
            self.seg_centroid[sid] = centroids[idx].astype(np.float32)
            self.seg_bbox_min[sid] = bbox_min[idx].astype(np.float32)
            self.seg_bbox_max[sid] = bbox_max[idx].astype(np.float32)
            if dino_sums.ndim == 2 and idx < dino_sums.shape[0] and dino_sums.shape[1] > 0:
                dim = int(dino_dims[idx]) if dino_dims is not None else dino_sums.shape[1]
                if not 0 <= dim <= dino_sums.shape[1]:
                    raise ValueError("Invalid saved segment DINO dimension")
                dino_sum = dino_sums[idx, :dim].astype(np.float32)
                if float(np.linalg.norm(dino_sum)) > 1e-6:
                    self.seg_dino_sum[sid] = dino_sum
                    self.seg_dino_w[sid] = (
                        float(dino_weights[idx]) if idx < dino_weights.shape[0] else 1.0
                    )
            cells_arr = np.asarray(support_cells[idx], dtype=np.int32) if idx < len(support_cells) else np.zeros((0, 2), dtype=np.int32)
            self.seg_support_cells[sid] = {(int(c[0]), int(c[1])) for c in cells_arr.reshape((-1, 2))}
            vox_arr = np.asarray(support_voxels[idx], dtype=np.int32) if idx < len(support_voxels) else np.zeros((0, 3), dtype=np.int32)
            self.seg_support_voxels[sid] = {(int(v[0]), int(v[1]), int(v[2])) for v in vox_arr.reshape((-1, 3))}

        owner_keys = np.asarray(payload.get("voxel_owner_keys", np.zeros((0, 3), dtype=np.int32)), dtype=np.int32)
        owner_values = np.asarray(payload.get("voxel_owner_values", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        for idx, key_arr in enumerate(owner_keys.reshape((-1, 3))):
            if idx >= owner_values.shape[0]:
                break
            sid = int(owner_values[idx])
            if sid in self.seg_w:
                self.voxel_segment_id[(int(key_arr[0]), int(key_arr[1]), int(key_arr[2]))] = sid

        if ids.size > 0:
            self.seg_next_id = max(self.seg_next_id, int(ids.max()) + 1)

    def _metadata_snapshot(self):
        return {
            "grid_width": int(self.grid_width),
            "grid_height": int(self.grid_height),
            "grid_size_m": float(self.grid_size_m),
            "resolution": float(self.resolution),
            "origin_x": float(self.origin_x),
            "origin_y": float(self.origin_y),
            "internal_frame": str(self.internal_frame),
            "publish_frame": str(self.publish_frame),
        }

    def _identity_metadata_snapshot(self):
        # Voxel keys and descriptor cosines are meaningful only in the same
        # geometry/feature spaces. Do not silently reinterpret a saved map.
        return {
            "voxel_size": float(self.voxel_size),
            "voxel_z_min": float(self.voxel_z_min),
            "voxel_z_max": float(self.voxel_z_max),
            "clip_model": str(self.get_parameter("clip_model_name").value),
            "clip_pretrained": str(self.get_parameter("clip_pretrained").value),
            "dino_model": str(self.get_parameter("dinov2_model_name").value),
            "dino_checkpoint": str(self.get_parameter("dinov2_checkpoint").value),
        }

    def _grid_bundle_from_payload(self, payload):
        bundle = self._empty_grid_bundle()
        for name in bundle.keys():
            if name not in payload:
                continue
            arr = np.asarray(payload[name], dtype=np.int8)
            if arr.shape != (self.grid_height, self.grid_width):
                raise ValueError(f"Stored grid '{name}' has shape {arr.shape}, expected {(self.grid_height, self.grid_width)}")
            bundle[name] = arr.copy()
        return bundle

    def _serialize_transform(self, T: Optional[np.ndarray]):
        if T is None:
            return None
        arr = np.asarray(T, dtype=np.float32)
        if arr.shape != (4, 4):
            raise ValueError(f"Transform must have shape (4, 4), got {arr.shape}")
        return arr.copy()

    def _deserialize_transform(self, payload) -> Optional[np.ndarray]:
        if payload is None:
            return None
        arr = np.asarray(payload, dtype=np.float32)
        if arr.shape != (4, 4):
            raise ValueError(f"Stored transform must have shape (4, 4), got {arr.shape}")
        return arr.copy()

    def lookup_saved_or_live_T_publish_internal(self, stamp_msg=None):
        T_live = self.lookup_T_publish_internal(stamp_msg)
        if T_live is not None:
            self.last_T_publish_internal = T_live.copy()
            return T_live
        if self.saved_T_publish_internal is not None:
            return self.saved_T_publish_internal.copy()
        return None

    def publish_cached_grids(self):
        bundle = self.latest_grids
        if bool(self.get_parameter("publish_stm_semantic_grid").value):
            self.publish_grid(bundle["stm_semantic"], self.pub_stm_sem)
        if bool(self.get_parameter("publish_stm_confidence_grid").value):
            self.publish_grid(bundle["stm_confidence"], self.pub_stm_conf)
        if bool(self.get_parameter("publish_stm_top1_grid").value):
            self.publish_grid(bundle["stm_top1"], self.pub_stm_top1)
        if bool(self.get_parameter("publish_ltm_semantic_grid").value):
            self.publish_grid(bundle["ltm_semantic"], self.pub_ltm_sem)
        if bool(self.get_parameter("publish_ltm_confidence_grid").value):
            self.publish_grid(bundle["ltm_confidence"], self.pub_ltm_conf)
        if bool(self.get_parameter("publish_ltm_top1_grid").value):
            self.publish_grid(bundle["ltm_top1"], self.pub_ltm_top1)

    def _set_latest_grid(self, name: str, grid: np.ndarray):
        grid_copy = grid.copy()
        if not np.array_equal(self.latest_grids[name], grid_copy):
            self.latest_grids[name] = grid_copy
            self._mark_state_dirty()

    def save_map_state(self):
        try:
            parent = os.path.dirname(self.map_state_path)
            if parent:
                os.makedirs(parent, exist_ok=True)

            payload = {
                "version": 3,
                "world_model_id": self.world_model_id,
                "dino_base_dim": getattr(self, "dino_base_dim", 0),
                "boundary_corrections": self.boundary_votes.to_dict() if hasattr(self, "boundary_votes") else {},
                "metadata": self._metadata_snapshot(),
                "identity_metadata": self._identity_metadata_snapshot(),
                "saved_T_publish_internal": self._serialize_transform(self.last_T_publish_internal),
                "stationary_pose_anchor": self._serialize_transform(getattr(self, "_stationary_pose_anchor", None)),
                "stm_sum": self._serialize_vec_dict(self.stm_sum),
                "stm_w": self._serialize_scalar_dict(self.stm_w, np.float32),
                "stm_last": self._serialize_scalar_dict(self.stm_last, np.float64),
                "stm_viewmask": self._serialize_scalar_dict(self.stm_viewmask, np.uint32),
                "stm_z_min": self._serialize_scalar_dict(self.stm_z_min, np.float32),
                "stm_z_max": self._serialize_scalar_dict(self.stm_z_max, np.float32),
                "stm_z_sum": self._serialize_scalar_dict(self.stm_z_sum, np.float32),
                "stm_z_count": self._serialize_scalar_dict(self.stm_z_count, np.uint32),
                "ltm_sum": self._serialize_vec_dict(self.ltm_sum),
                "ltm_w": self._serialize_scalar_dict(self.ltm_w, np.float32),
                "ltm_last_seen": self._serialize_scalar_dict(self.ltm_last_seen, np.float64),
                "ltm_viewmask": self._serialize_scalar_dict(self.ltm_viewmask, np.uint32),
                "ltm_z_min": self._serialize_scalar_dict(self.ltm_z_min, np.float32),
                "ltm_z_max": self._serialize_scalar_dict(self.ltm_z_max, np.float32),
                "ltm_z_sum": self._serialize_scalar_dict(self.ltm_z_sum, np.float32),
                "ltm_z_count": self._serialize_scalar_dict(self.ltm_z_count, np.uint32),
                "vox_stm_sum": self._serialize_voxel_vec_dict(self.vox_stm_sum),
                "vox_stm_w": self._serialize_voxel_scalar_dict(self.vox_stm_w, np.float32),
                "vox_stm_last": self._serialize_voxel_scalar_dict(self.vox_stm_last, np.float64),
                "vox_stm_viewmask": self._serialize_voxel_scalar_dict(self.vox_stm_viewmask, np.uint32),
                "vox_ltm_sum": self._serialize_voxel_vec_dict(self.vox_ltm_sum),
                "vox_ltm_w": self._serialize_voxel_scalar_dict(self.vox_ltm_w, np.float32),
                "vox_ltm_last_seen": self._serialize_voxel_scalar_dict(self.vox_ltm_last_seen, np.float64),
                "vox_ltm_viewmask": self._serialize_voxel_scalar_dict(self.vox_ltm_viewmask, np.uint32),
                "segments": self._serialize_segments(),
                "entities": self.entity_memory.to_dict(),
                "last_observation_seq": int(self.last_applied_observation_seq),
                "history_vox": self._serialize_history_vox(),
                "latest_grids": {name: grid.copy() for name, grid in self.latest_grids.items()},
                "query_text": str(self.query_text),
            }

            with tempfile.NamedTemporaryFile("wb", dir=parent or ".", delete=False) as tmp_file:
                pickle.dump(payload, tmp_file, protocol=pickle.HIGHEST_PROTOCOL)
                tmp_path = tmp_file.name
            os.replace(tmp_path, self.map_state_path)
            self.state_dirty = False
        except Exception as e:
            self.get_logger().warn(f"Failed to save map state to '{self.map_state_path}': {e}")

    def load_map_state(self):
        if not os.path.exists(self.map_state_path):
            return
        try:
            with open(self.map_state_path, "rb") as f:
                payload = pickle.load(f)

            version = int(payload.get("version", 1))
            if version not in (1, 2, 3):
                raise ValueError(f"Unsupported map-state version {version}")

            restored_anchor = self._deserialize_transform(payload.get("stationary_pose_anchor"))
            if restored_anchor is not None and not bool(self.get_parameter("stationary_pose_lock").value):
                raise ValueError("Stationary-anchored map requires stationary_pose_lock; do not mix it with drifting live poses")

            metadata = payload.get("metadata", {})
            current_meta = self._metadata_snapshot()
            for key, value in current_meta.items():
                if metadata.get(key) != value:
                    raise ValueError(f"Metadata mismatch for '{key}': stored={metadata.get(key)} current={value}")
            if version >= 3:
                identity_metadata = payload.get("identity_metadata", {})
                for key, value in self._identity_metadata_snapshot().items():
                    if identity_metadata.get(key) != value:
                        raise ValueError(f"Identity metadata mismatch for '{key}': stored={identity_metadata.get(key)} current={value}")
            else:
                self.get_logger().warn("Legacy map has no voxel/feature-space metadata; only load it with its original geometry and model settings.")
            restored_entities = PersistentEntities.from_dict(payload.get("entities", {}))
            restored_world_id = validate_world_id(payload["world_model_id"]) if payload.get("world_model_id") else new_world_id()
            restored_boundary_votes = BoundaryOwnershipVotes.from_dict(payload.get("boundary_corrections", {}))

            self.stm_sum = self._deserialize_vec_dict(payload.get("stm_sum", {}))
            self.stm_w = self._deserialize_scalar_dict(payload.get("stm_w", {}), float)
            self.stm_last = self._deserialize_scalar_dict(payload.get("stm_last", {}), float)
            self.stm_viewmask = self._deserialize_scalar_dict(payload.get("stm_viewmask", {}), int)
            self.stm_z_min = self._deserialize_scalar_dict(payload.get("stm_z_min", {}), float)
            self.stm_z_max = self._deserialize_scalar_dict(payload.get("stm_z_max", {}), float)
            self.stm_z_sum = self._deserialize_scalar_dict(payload.get("stm_z_sum", {}), float)
            self.stm_z_count = self._deserialize_scalar_dict(payload.get("stm_z_count", {}), int)
            self.ltm_sum = self._deserialize_vec_dict(payload.get("ltm_sum", {}))
            self.ltm_w = self._deserialize_scalar_dict(payload.get("ltm_w", {}), float)
            self.ltm_last_seen = self._deserialize_scalar_dict(payload.get("ltm_last_seen", {}), float)
            self.ltm_viewmask = self._deserialize_scalar_dict(payload.get("ltm_viewmask", {}), int)
            self.ltm_z_min = self._deserialize_scalar_dict(payload.get("ltm_z_min", {}), float)
            self.ltm_z_max = self._deserialize_scalar_dict(payload.get("ltm_z_max", {}), float)
            self.ltm_z_sum = self._deserialize_scalar_dict(payload.get("ltm_z_sum", {}), float)
            self.ltm_z_count = self._deserialize_scalar_dict(payload.get("ltm_z_count", {}), int)
            self.vox_stm_sum = self._deserialize_voxel_vec_dict(payload.get("vox_stm_sum", {}))
            self.vox_stm_w = self._deserialize_voxel_scalar_dict(payload.get("vox_stm_w", {}), float)
            self.vox_stm_last = self._deserialize_voxel_scalar_dict(payload.get("vox_stm_last", {}), float)
            self.vox_stm_viewmask = self._deserialize_voxel_scalar_dict(payload.get("vox_stm_viewmask", {}), int)
            self.vox_ltm_sum = self._deserialize_voxel_vec_dict(payload.get("vox_ltm_sum", {}))
            self.vox_ltm_w = self._deserialize_voxel_scalar_dict(payload.get("vox_ltm_w", {}), float)
            self.vox_ltm_last_seen = self._deserialize_voxel_scalar_dict(payload.get("vox_ltm_last_seen", {}), float)
            self.vox_ltm_viewmask = self._deserialize_voxel_scalar_dict(payload.get("vox_ltm_viewmask", {}), int)
            self._deserialize_segments(payload.get("segments", {}))
            self.entity_memory = restored_entities
            self._stationary_pose_anchor = restored_anchor
            self.world_model_id = restored_world_id
            self.dino_base_dim = int(payload.get("dino_base_dim", 0))
            self.boundary_votes = restored_boundary_votes
            self.surface_consensus = SurfaceConsensus()  # No stale 2D masks across sessions.
            self.segment_display_memory = DominantOverlay()
            self.entity_display_memory = DominantOverlay()
            self.seg_last_dino_seq = {}
            self.last_applied_observation_seq = max(
                int(payload.get("last_observation_seq", -1)),
                self.entity_memory.last_seq,
                max(self.seg_last_obs_seq.values(), default=-1),
            )
            self.async_job_seq = max(self.async_job_seq, self.last_applied_observation_seq)
            # Relation evidence must be earned again after a restart.
            self.entity_edges = ObservationEdges()
            self.segment_edges = ObservationEdges()
            self.history_vox = self._deserialize_history_vox(payload.get("history_vox", {}))
            self.latest_grids = self._grid_bundle_from_payload(payload.get("latest_grids", {}))
            self.saved_T_publish_internal = self._deserialize_transform(payload.get("saved_T_publish_internal"))
            self.last_T_publish_internal = self.saved_T_publish_internal.copy() if self.saved_T_publish_internal is not None else None
            self.has_loaded_grid_snapshot = True
            self.state_dirty = not bool(payload.get("world_model_id"))
            self.get_logger().info(
                f"Loaded map state from '{self.map_state_path}' "
                f"(STM={len(self.stm_w)}, LTM={len(self.ltm_w)}, "
                f"VoxSTM={len(self.vox_stm_w)}, VoxLTM={len(self.vox_ltm_w)}, history={len(self.history_vox)})"
            )
            if self.saved_T_publish_internal is None:
                self.get_logger().warn(
                    "Loaded map state has no saved publish/internal transform. "
                    "Offline query changes may not work without live TF."
                )
        except Exception as e:
            self.get_logger().warn(f"Failed to load map state from '{self.map_state_path}': {e}")

    def on_autosave_timer(self):
        if not bool(self.get_parameter("autosave_map_state").value):
            return
        if not self.state_dirty:
            return
        self.save_map_state()

    def on_cam_info(self, msg: CameraInfo):
        self._note_camera_input(msg, "camera_info")
        intrinsics = tuple(float(msg.k[i]) for i in (0, 4, 2, 5))
        if not all(np.isfinite(v) for v in intrinsics) or min(intrinsics[:2]) <= 0:
            return
        if intrinsics == getattr(self, "_camera_intrinsics", None):
            return
        self._camera_intrinsics = intrinsics
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])
        self.get_logger().info(
            f"Camera intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}"
        )
        # Keep the tiny calibration subscription alive: diagnostics should
        # measure camera delivery, not a subscription deliberately removed here.

    def on_occ_grid(self, msg: OccupancyGrid):
        try:
            w = int(msg.info.width)
            h = int(msg.info.height)
            res = float(msg.info.resolution)
            ox = float(msg.info.origin.position.x)
            oy = float(msg.info.origin.position.y)
            data = np.array(msg.data, dtype=np.int16).reshape((h, w))
            self.occ = {
                "frame_id": msg.header.frame_id,
                "w": w,
                "h": h,
                "res": res,
                "ox": ox,
                "oy": oy,
                "data": data,
            }
        except Exception as e:
            self.get_logger().warn(f"Failed to parse occupancy grid: {e}")

    # -------------------------
    # CLIP helpers
    # -------------------------
    def init_esam(self):
        variant = str(self.get_parameter("esam_variant").value).lower()
        checkpoint = os.path.expanduser(
            str(self.get_parameter("esam_checkpoint").value).strip()
        )
        if not checkpoint and variant in ("vitt", "vits"):
            checkpoint = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "weights",
                    f"efficient_sam_{variant}.pt",
                )
            )
        checkpoint = checkpoint if checkpoint else None

        try:
            if checkpoint:
                from efficient_sam.efficient_sam import build_efficient_sam

                if variant == "vits":
                    self.esam_model = build_efficient_sam(
                        encoder_patch_embed_dim=384,
                        encoder_num_heads=6,
                        checkpoint=checkpoint,
                    )
                else:
                    self.esam_model = build_efficient_sam(
                        encoder_patch_embed_dim=192,
                        encoder_num_heads=3,
                        checkpoint=checkpoint,
                    )
            else:
                from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits

                if variant == "vits":
                    self.esam_model = build_efficient_sam_vits()
                else:
                    self.esam_model = build_efficient_sam_vitt()

            self.esam_model.eval()
            self.esam_model.to(self.device)
            checkpoint_label = checkpoint if checkpoint else "package default"
            self.get_logger().info(
                f"EfficientSAM enabled: variant={variant} checkpoint={checkpoint_label}"
            )
        except Exception as e:
            self.esam_model = None
            self.get_logger().warn(
                "EfficientSAM masks are enabled but the model could not be initialized. "
                f"Install EfficientSAM and make sure its weights are available. Error: {e}"
            )

    def init_dinov2(self):
        model_name = str(self.get_parameter("dinov2_model_name").value).strip()
        repo = str(self.get_parameter("dinov2_repo").value).strip() or "facebookresearch/dinov2"
        local_repo = os.path.expanduser(str(self.get_parameter("dinov2_local_repo").value).strip())
        checkpoint = os.path.expanduser(str(self.get_parameter("dinov2_checkpoint").value).strip())
        allow_download = bool(self.get_parameter("dinov2_allow_download").value)
        try:
            if local_repo:
                self.dinov2_model = torch.hub.load(
                    local_repo,
                    model_name,
                    source="local",
                    pretrained=not bool(checkpoint),
                )
            elif allow_download:
                self.dinov2_model = torch.hub.load(
                    repo,
                    model_name,
                    pretrained=not bool(checkpoint),
                    trust_repo=True,
                )
            else:
                raise RuntimeError(
                    "no dinov2_local_repo configured and dinov2_allow_download is false"
                )
            if checkpoint:
                state = torch.load(checkpoint, map_location="cpu")
                if isinstance(state, dict):
                    state = state.get("model", state.get("state_dict", state))
                if not isinstance(state, dict):
                    raise ValueError("DINOv2 checkpoint did not contain a state dictionary")
                clean = {
                    str(k).removeprefix("module.").removeprefix("backbone."): v
                    for k, v in state.items()
                }
                missing, unexpected = self.dinov2_model.load_state_dict(clean, strict=False)
                if missing or unexpected:
                    self.get_logger().warn(
                        f"DINOv2 checkpoint loaded non-strictly: missing={len(missing)} "
                        f"unexpected={len(unexpected)}"
                    )
            patch = getattr(self.dinov2_model, "patch_size", 14)
            if isinstance(patch, (tuple, list)):
                patch = max(patch)
            self.dinov2_patch_size = max(1, int(patch))
            self.dinov2_model.eval()
            self.dinov2_model.to(self.device)
            for parameter in self.dinov2_model.parameters():
                parameter.requires_grad_(False)
            self.get_logger().info(
                f"DINOv2 appearance features enabled: model={model_name} "
                f"patch={self.dinov2_patch_size}"
            )
        except Exception as e:
            self.dinov2_model = None
            self.get_logger().warn(
                "DINOv2 appearance association is unavailable; continuing with "
                f"CLIP and geometry. Error: {e}"
            )

    def _dinov2_preprocess_rgb(self, rgb_cv: np.ndarray):
        patch = max(1, int(self.dinov2_patch_size))
        requested = max(patch, int(self.get_parameter("dinov2_input_size").value))
        input_size = max(patch, int(math.ceil(requested / patch)) * patch)
        H, W = rgb_cv.shape[:2]
        scale = float(input_size) / float(max(H, W))
        run_w = max(patch, min(input_size, int(round(W * scale))))
        run_h = max(patch, min(input_size, int(round(H * scale))))
        run_w = max(patch, int(round(run_w / patch)) * patch)
        run_h = max(patch, int(round(run_h / patch)) * patch)
        run_w = min(input_size, run_w)
        run_h = min(input_size, run_h)
        off_x = int((input_size - run_w) // 2)
        off_y = int((input_size - run_h) // 2)

        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        fill = np.clip(mean * 255.0, 0, 255).astype(np.uint8)
        canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        canvas[:, :, :] = fill.reshape((1, 1, 3))
        resized = PILImage.fromarray(rgb_cv).resize((run_w, run_h), PILImage.BICUBIC)
        canvas[off_y : off_y + run_h, off_x : off_x + run_w, :] = np.asarray(
            resized, dtype=np.uint8
        )
        arr = canvas.astype(np.float32) / 255.0
        arr = (arr - mean.reshape((1, 1, 3))) / std.reshape((1, 1, 3))
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor, {
            "input_size": int(input_size),
            "scale_x": float(run_w) / float(max(W, 1)),
            "scale_y": float(run_h) / float(max(H, 1)),
            "offset_x": int(off_x),
            "offset_y": int(off_y),
            "resized_w": int(run_w),
            "resized_h": int(run_h),
            "orig_w": int(W),
            "orig_h": int(H),
            "patch_size": int(patch),
        }

    def compute_dinov2_dense_features(self, rgb_cv: np.ndarray):
        if not bool(self.get_parameter("use_dinov2_features").value):
            return None
        if self.dinov2_model is None:
            if not self.dinov2_unavailable_warned:
                self.get_logger().warn(
                    "DINOv2 was requested but is unavailable; appearance matching is disabled."
                )
                self.dinov2_unavailable_warned = True
            return None
        tensor, meta = self._dinov2_preprocess_rgb(rgb_cv)
        batch = tensor.to(self.device)
        with self.inference_lock:
            with torch.no_grad():
                output = self.dinov2_model.forward_features(batch)
                if isinstance(output, dict):
                    tokens = output.get("x_norm_patchtokens")
                    if tokens is None:
                        tokens = output.get("x_prenorm")
                else:
                    tokens = output
                if tokens is None or tokens.ndim != 3:
                    raise RuntimeError("DINOv2 forward_features returned no patch tokens")
                tokens = tokens[0]
                expected = (
                    int(meta["input_size"]) // max(1, int(meta["patch_size"]))
                ) ** 2
                if int(tokens.shape[0]) > expected:
                    tokens = tokens[-expected:, :]
                if int(tokens.shape[0]) != expected:
                    raise RuntimeError(
                        f"unexpected DINOv2 token count {tokens.shape[0]} (expected {expected})"
                    )
                tokens = tokens / tokens.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                tokens_np = tokens.float().cpu().numpy().astype(np.float32)
        side = int(round(math.sqrt(int(tokens_np.shape[0]))))
        return {
            "features": tokens_np.reshape((side, side, -1)).astype(np.float32),
            "meta": meta,
        }

    def compute_dinov2_mask_embeddings(
        self,
        masks: List[np.ndarray],
        dense_dino,
        return_quality: bool = False,
    ):
        """Pool every mask from the one shared full-frame DINO feature map.

        Token weights are fractional mask coverage, so SAM retains control of
        the boundary instead of a nearest-neighbour token label. The descriptor
        concatenates the object interior with an interior-minus-local-ring
        residual. This adds local context discrimination without another model
        pass or an object crop.
        """
        if dense_dino is None or not masks:
            empty = np.zeros((len(masks), 0), dtype=np.float32)
            quality = np.zeros((len(masks),), dtype=np.float32)
            return (empty, quality) if return_quality else empty
        features = np.asarray(
            dense_dino.get("features", np.zeros((0, 0, 0))), dtype=np.float32
        )
        if features.ndim != 3 or features.shape[0] <= 0 or features.shape[1] <= 0:
            empty = np.zeros((len(masks), 0), dtype=np.float32)
            quality = np.zeros((len(masks),), dtype=np.float32)
            return (empty, quality) if return_quality else empty
        meta = dense_dino.get("meta", {})
        input_size = int(meta.get("input_size", 0))
        run_w = int(meta.get("resized_w", 0))
        run_h = int(meta.get("resized_h", 0))
        off_x = int(meta.get("offset_x", 0))
        off_y = int(meta.get("offset_y", 0))
        if input_size <= 0 or run_w <= 0 or run_h <= 0:
            empty = np.zeros((len(masks), 0), dtype=np.float32)
            quality = np.zeros((len(masks),), dtype=np.float32)
            return (empty, quality) if return_quality else empty
        gh, gw, dim = features.shape
        min_weight = float(
            np.clip(float(self.get_parameter("dinov2_mask_min_weight").value), 0.0, 1.0)
        )
        ring_tokens = max(
            0, int(self.get_parameter("dinov2_context_ring_tokens").value)
        )
        contrast_weight = max(
            0.0,
            float(self.get_parameter("dinov2_context_contrast_weight").value),
        )
        min_effective_tokens = max(
            1e-3,
            float(self.get_parameter("dinov2_min_effective_tokens").value),
        )
        descriptor_dim = int(dim * (2 if contrast_weight > 0.0 else 1))
        result = np.zeros((len(masks), descriptor_dim), dtype=np.float32)
        quality = np.zeros((len(masks),), dtype=np.float32)
        valid_canvas = np.zeros((input_size, input_size), dtype=np.float32)
        valid_canvas[off_y : off_y + run_h, off_x : off_x + run_w] = 1.0
        valid_tokens = cv2.resize(
            valid_canvas, (gw, gh), interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        for idx, mask in enumerate(masks):
            if mask.size == 0 or not np.any(mask):
                continue
            # INTER_AREA preserves the fractional area of a SAM mask covered
            # by each DINO patch. The full square canvas also keeps the
            # letterbox transform exactly aligned with the token grid.
            canvas = np.zeros((input_size, input_size), dtype=np.float32)
            resized = cv2.resize(
                np.asarray(mask, dtype=np.float32),
                (run_w, run_h),
                interpolation=cv2.INTER_AREA,
            )
            canvas[off_y : off_y + run_h, off_x : off_x + run_w] = resized
            coverage = cv2.resize(
                canvas,
                (gw, gh),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32)
            coverage = np.clip(coverage, 0.0, 1.0)
            weights = np.where(coverage >= min_weight, coverage, 0.0).astype(
                np.float32
            )
            # Tiny masks may not own a full high-purity patch. Retain their
            # fractional evidence, but lower quality makes their track update
            # conservative.
            if float(np.sum(weights)) <= 1e-6:
                weights = np.where(coverage > 1e-3, coverage, 0.0).astype(np.float32)
            wsum = float(np.sum(weights, dtype=np.float64))
            if wsum <= 1e-6:
                continue
            interior = np.sum(
                features * weights[:, :, None], axis=(0, 1), dtype=np.float64
            ).astype(np.float32) / wsum
            interior_norm = float(np.linalg.norm(interior))
            if interior_norm <= 1e-6:
                continue
            interior = (interior / interior_norm).astype(np.float32)

            descriptor = interior
            if contrast_weight > 0.0:
                binary_support = (coverage > 1e-3).astype(np.uint8)
                if ring_tokens > 0:
                    kernel_size = int(2 * ring_tokens + 1)
                    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                    expanded = cv2.dilate(binary_support, kernel, iterations=1)
                else:
                    expanded = binary_support
                ring_weights = np.clip(
                    expanded.astype(np.float32) - coverage, 0.0, 1.0
                )
                # Never draw context from padded image tokens.
                ring_weights *= valid_tokens
                ring_sum = float(np.sum(ring_weights, dtype=np.float64))
                residual = np.zeros_like(interior, dtype=np.float32)
                if ring_sum > 1e-6:
                    ring = np.sum(
                        features * ring_weights[:, :, None],
                        axis=(0, 1),
                        dtype=np.float64,
                    ).astype(np.float32) / ring_sum
                    ring_norm = float(np.linalg.norm(ring))
                    if ring_norm > 1e-6:
                        ring = ring / ring_norm
                        residual = (interior - ring).astype(np.float32)
                        residual_norm = float(np.linalg.norm(residual))
                        if residual_norm > 1e-6:
                            residual /= residual_norm
                descriptor = np.concatenate(
                    [interior, contrast_weight * residual], axis=0
                ).astype(np.float32)

            norm = float(np.linalg.norm(descriptor))
            if norm > 1e-6:
                result[idx, :] = (descriptor / norm).astype(np.float32)
                effective_tokens = float(
                    (wsum * wsum)
                    / max(float(np.sum(weights * weights, dtype=np.float64)), 1e-6)
                )
                purity = float(
                    np.sum(weights * coverage, dtype=np.float64) / max(wsum, 1e-6)
                )
                support_quality = float(
                    np.clip(effective_tokens / min_effective_tokens, 0.0, 1.0)
                )
                quality[idx] = float(
                    np.clip(math.sqrt(max(purity * support_quality, 0.0)), 0.05, 1.0)
                )
        return (result, quality) if return_quality else result

    def _prompt(self, text: str) -> str:
        return f"a photo of a {text}"

    def _query_prompts(self, text: str) -> List[str]:
        text = str(text).strip()
        if not text:
            text = "object"
        if not bool(self.get_parameter("clip_text_prompt_ensemble").value):
            return [self._prompt(text)]
        # Keep this deliberately broad: some queries are things ("plant"),
        # some are surfaces ("floor"), and some are places/regions ("carpet").
        templates = [
            "{q}",
            "a photo of {q}",
            "a photo of a {q}",
            "a close-up photo of {q}",
            "a cropped image of {q}",
            "a segment of {q}",
            "a region of {q}",
            "the {q} in an indoor scene",
            "{q} surface",
        ]
        prompts = []
        for tmpl in templates:
            prompt = tmpl.format(q=text)
            if prompt not in prompts:
                prompts.append(prompt)
        return prompts

    def compute_clip_text_embedding(self, text: str) -> np.ndarray:
        prompts = self._query_prompts(text)
        tokens = self.tokenizer(prompts).to(self.device)
        with self.inference_lock:
            with torch.no_grad():
                emb = self.model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                emb = emb.mean(dim=0, keepdim=True)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                return emb.squeeze(0).float().cpu().numpy()

    def _clip_encode_pil_batch(self, images: List[PILImage.Image], batch_size: int) -> np.ndarray:
        if not images:
            return np.zeros((0, 0), dtype=np.float32)
        out = []
        batch_size = max(1, int(batch_size))
        for start in range(0, len(images), batch_size):
            imgs = [self.preprocess(im) for im in images[start : start + batch_size]]
            batch = torch.stack(imgs, dim=0).to(self.device)
            with self.inference_lock:
                with torch.no_grad():
                    emb = self.model.encode_image(batch)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    out.append(emb.float().cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def _clip_visual_input_size(self) -> int:
        override = int(self.get_parameter("clip_dense_input_size").value)
        if override > 0:
            return override
        image_size = getattr(self.model.visual, "image_size", 224)
        if isinstance(image_size, (tuple, list)):
            return int(max(image_size))
        return int(image_size)

    def _clip_dense_preprocess_rgb(self, rgb_cv: np.ndarray):
        input_size = max(16, self._clip_visual_input_size())
        H, W = rgb_cv.shape[:2]
        scale = float(input_size) / float(max(H, W))
        run_w = max(1, int(round(W * scale)))
        run_h = max(1, int(round(H * scale)))
        off_x = int((input_size - run_w) // 2)
        off_y = int((input_size - run_h) // 2)

        mean = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        fill = np.clip(mean * 255.0, 0, 255).astype(np.uint8)
        canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        canvas[:, :, :] = fill.reshape((1, 1, 3))
        resized = PILImage.fromarray(rgb_cv).resize((run_w, run_h), PILImage.BICUBIC)
        canvas[off_y : off_y + run_h, off_x : off_x + run_w, :] = np.asarray(resized, dtype=np.uint8)

        arr = canvas.astype(np.float32) / 255.0
        arr = (arr - mean.reshape((1, 1, 3))) / std.reshape((1, 1, 3))
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        meta = {
            "input_size": int(input_size),
            "scale": float(scale),
            "offset_x": int(off_x),
            "offset_y": int(off_y),
            "resized_w": int(run_w),
            "resized_h": int(run_h),
            "orig_w": int(W),
            "orig_h": int(H),
        }
        return tensor, meta

    def compute_dense_clip_patch_features(self, rgb_cv: np.ndarray):
        if not bool(self.get_parameter("use_clip_dense_features").value):
            return None
        visual = getattr(self.model, "visual", None)
        if visual is None or not all(hasattr(visual, name) for name in ("_embeds", "_pool", "transformer")):
            return None

        tensor, meta = self._clip_dense_preprocess_rgb(rgb_cv)
        batch = tensor.to(self.device)
        try:
            with self.inference_lock:
                with torch.no_grad():
                    x = visual._embeds(batch)
                    x = visual.transformer(x)
                    _, tokens = visual._pool(x)
                    proj = getattr(visual, "proj", None)
                    if proj is not None and tokens.shape[-1] == proj.shape[0]:
                        tokens = tokens @ proj
                    tokens = tokens / tokens.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    tokens_np = tokens.squeeze(0).float().cpu().numpy().astype(np.float32)
        except Exception as e:
            self.get_logger().warn(f"Dense CLIP patch features unavailable, falling back to crop embeddings: {e}")
            return None

        n_tokens = int(tokens_np.shape[0])
        grid_size = getattr(visual, "grid_size", None)
        if isinstance(grid_size, (tuple, list)) and len(grid_size) == 2:
            gh, gw = int(grid_size[0]), int(grid_size[1])
        else:
            side = int(round(math.sqrt(max(1, n_tokens))))
            gh, gw = side, side
        if gh * gw != n_tokens:
            side = int(round(math.sqrt(max(1, n_tokens))))
            if side * side != n_tokens:
                return None
            gh, gw = side, side
        features = tokens_np.reshape((gh, gw, -1)).astype(np.float32)
        return {"features": features, "meta": meta}

    def compute_dense_mask_embeddings(self, masks: List[np.ndarray], dense_clip) -> np.ndarray:
        if dense_clip is None or not masks:
            return np.zeros((0, 0), dtype=np.float32)
        features = np.asarray(dense_clip.get("features", np.zeros((0, 0, 0))), dtype=np.float32)
        if features.ndim != 3 or features.shape[0] <= 0 or features.shape[1] <= 0:
            return np.zeros((len(masks), 0), dtype=np.float32)
        meta = dense_clip.get("meta", {})
        input_size = int(meta.get("input_size", 0))
        run_w = int(meta.get("resized_w", 0))
        run_h = int(meta.get("resized_h", 0))
        off_x = int(meta.get("offset_x", 0))
        off_y = int(meta.get("offset_y", 0))
        if input_size <= 0 or run_w <= 0 or run_h <= 0:
            return np.zeros((len(masks), 0), dtype=np.float32)

        gh, gw, dim = features.shape
        out = np.zeros((len(masks), dim), dtype=np.float32)
        for idx, mask in enumerate(masks):
            if mask.size == 0 or not np.any(mask):
                continue
            canvas = np.zeros((input_size, input_size), dtype=np.uint8)
            mask_img = PILImage.fromarray(mask.astype(np.uint8) * 255)
            resized = mask_img.resize((run_w, run_h), PILImage.NEAREST)
            canvas[off_y : off_y + run_h, off_x : off_x + run_w] = np.asarray(resized, dtype=np.uint8)
            patch_weights = np.asarray(
                PILImage.fromarray(canvas).resize((gw, gh), PILImage.BILINEAR),
                dtype=np.float32,
            ) / 255.0
            wsum = float(np.sum(patch_weights))
            if wsum <= 1e-6:
                continue
            emb = np.sum(features * patch_weights[:, :, None], axis=(0, 1)) / wsum
            n = float(np.linalg.norm(emb))
            if n > 1e-6:
                out[idx, :] = (emb / n).astype(np.float32)
        return out.astype(np.float32)

    def sample_dense_clip_embeddings_for_pixels(self, dense_clip, uu: np.ndarray, vv: np.ndarray):
        if dense_clip is None:
            return None
        features = np.asarray(dense_clip.get("features", np.zeros((0, 0, 0))), dtype=np.float32)
        if features.ndim != 3 or features.shape[0] <= 0 or features.shape[1] <= 0:
            return None
        meta = dense_clip.get("meta", {})
        input_size = float(meta.get("input_size", 0))
        scale = float(meta.get("scale", 0.0))
        off_x = float(meta.get("offset_x", 0.0))
        off_y = float(meta.get("offset_y", 0.0))
        if input_size <= 0.0 or scale <= 0.0:
            return None
        gh, gw, _ = features.shape
        x_in = uu.astype(np.float32) * scale + off_x
        y_in = vv.astype(np.float32) * scale + off_y
        ix = np.floor(x_in / max(input_size, 1e-6) * float(gw)).astype(np.int32)
        iy = np.floor(y_in / max(input_size, 1e-6) * float(gh)).astype(np.int32)
        ix = np.clip(ix, 0, gw - 1)
        iy = np.clip(iy, 0, gh - 1)
        return features[iy, ix, :].astype(np.float32)

    def _combine_unit_embeddings(self, a: np.ndarray, b: np.ndarray, a_weight: float):
        if a is None or a.size == 0:
            return b
        if b is None or b.size == 0:
            return a
        if a.shape != b.shape:
            return a
        w = float(np.clip(a_weight, 0.0, 1.0))
        out = w * a.astype(np.float32) + (1.0 - w) * b.astype(np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        valid = norms[:, 0] > 1e-6
        out[valid, :] /= norms[valid, :]
        return out.astype(np.float32)

    def compute_tile_embeddings_batched(self, rgb_cv: np.ndarray, tiles_x: int, tiles_y: int):
        H, W = rgb_cv.shape[:2]
        crops = []
        for j in range(tiles_y):
            for i in range(tiles_x):
                x0 = int(i * W / tiles_x)
                x1 = int((i + 1) * W / tiles_x)
                y0 = int(j * H / tiles_y)
                y1 = int((j + 1) * H / tiles_y)
                crop = rgb_cv[y0:y1, x0:x1, :]
                crops.append(PILImage.fromarray(crop))

        imgs = [self.preprocess(im) for im in crops]
        batch = torch.stack(imgs, dim=0).to(self.device)

        with self.inference_lock:
            with torch.no_grad():
                emb = self.model.encode_image(batch)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                emb = emb.float().cpu().numpy()

        return emb.reshape((tiles_y, tiles_x, -1)).astype(np.float32)

    def make_esam_grid_prompts(self, width: int, height: int):
        gx = max(1, int(self.get_parameter("esam_prompt_grid_x").value))
        gy = max(1, int(self.get_parameter("esam_prompt_grid_y").value))
        xs = np.linspace(0.5 * width / gx, width - 0.5 * width / gx, gx, dtype=np.float32)
        ys = np.linspace(0.5 * height / gy, height - 0.5 * height / gy, gy, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)

    def _mask_iou(self, a: np.ndarray, b: np.ndarray) -> float:
        inter = int(np.logical_and(a, b).sum())
        if inter <= 0:
            return 0.0
        union = int(np.logical_or(a, b).sum())
        if union <= 0:
            return 0.0
        return float(inter) / float(union)

    def filter_esam_masks(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
        image_area: int,
        stability_scores: Optional[np.ndarray] = None,
    ):
        min_area = int(self.get_parameter("esam_min_mask_area_px").value)
        max_area_frac = float(self.get_parameter("esam_max_mask_area_frac").value)
        min_iou_pred = float(self.get_parameter("esam_min_iou_pred").value)
        min_stability = float(self.get_parameter("esam_min_stability_score").value)
        nms_iou = float(self.get_parameter("esam_nms_iou").value)
        nms_containment = float(self.get_parameter("esam_nms_containment").value)
        containment_min_area_ratio = float(
            self.get_parameter("esam_nms_containment_min_area_ratio").value
        )
        max_masks = int(self.get_parameter("esam_max_masks").value)

        candidates = []
        for idx, mask in enumerate(masks):
            score = float(scores[idx])
            stability = (
                float(stability_scores[idx])
                if stability_scores is not None and idx < int(stability_scores.shape[0])
                else 1.0
            )
            area = int(mask.sum())
            if score < min_iou_pred or stability < min_stability or area < min_area:
                continue
            if (
                image_area > 0
                and 0.0 < max_area_frac < 1.0
                and (float(area) / float(image_area)) > max_area_frac
            ):
                continue
            quality = float(np.clip(score, 0.0, 1.0) * np.clip(stability, 0.0, 1.0))
            candidates.append((quality, area, mask.astype(bool)))

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        kept = []
        kept_scores = []
        for score, area, mask in candidates:
            duplicate = False
            for old in kept:
                inter = int(np.logical_and(mask, old).sum())
                old_area = int(old.sum())
                containment = float(inter / max(min(area, old_area), 1))
                area_ratio = float(min(area, old_area) / max(max(area, old_area), 1))
                if self._mask_iou(mask, old) >= nms_iou or (
                    containment >= nms_containment
                    and area_ratio >= containment_min_area_ratio
                ):
                    duplicate = True
                    break
            if duplicate:
                continue
            kept.append(mask)
            kept_scores.append(score)
            if len(kept) >= max_masks:
                break

        if not kept:
            return [], np.zeros((0,), dtype=np.float32)
        return kept, np.asarray(kept_scores, dtype=np.float32)

    def segment_esam_masks(self, rgb_cv: np.ndarray):
        if self.esam_model is None:
            if not self.esam_unavailable_warned:
                self.get_logger().warn("Skipping mask update because EfficientSAM is unavailable.")
                self.esam_unavailable_warned = True
            return [], np.zeros((0,), dtype=np.float32)

        H, W = rgb_cv.shape[:2]
        max_side = int(self.get_parameter("esam_input_max_side").value)
        if max_side > 0 and max(H, W) > max_side:
            scale = float(max_side) / float(max(H, W))
            run_w = max(1, int(round(W * scale)))
            run_h = max(1, int(round(H * scale)))
            rgb_run = np.asarray(PILImage.fromarray(rgb_cv).resize((run_w, run_h), PILImage.BILINEAR))
        else:
            run_h, run_w = H, W
            rgb_run = rgb_cv

        prompts = self.make_esam_grid_prompts(run_w, run_h)
        if prompts.shape[0] == 0:
            return [], np.zeros((0,), dtype=np.float32)

        img = torch.from_numpy(rgb_run.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        input_points = torch.from_numpy(prompts).view(1, prompts.shape[0], 1, 2).to(self.device)
        input_labels = torch.ones((1, prompts.shape[0], 1), dtype=torch.int64, device=self.device)

        with self.inference_lock:
            with torch.no_grad():
                logits, pred_iou = self.esam_model(img, input_points, input_labels)
                best = torch.argmax(pred_iou, dim=2)
                gather_idx = best[:, :, None, None, None].expand(-1, -1, 1, logits.shape[-2], logits.shape[-1])
                best_logits = torch.gather(logits, 2, gather_idx).squeeze(2)
                best_scores = torch.gather(pred_iou, 2, best[:, :, None]).squeeze(2)

        best_logits_np = best_logits[0].detach().float().cpu().numpy()
        masks = best_logits_np >= 0.0
        stability_offset = max(
            1e-4, float(self.get_parameter("esam_stability_logit_offset").value)
        )
        stable_inner = best_logits_np >= stability_offset
        stable_outer = best_logits_np >= -stability_offset
        stability_scores = np.sum(stable_inner, axis=(1, 2)).astype(np.float32) / np.maximum(
            np.sum(stable_outer, axis=(1, 2)).astype(np.float32), 1.0
        )
        if run_h != H or run_w != W:
            resized_masks = []
            for mask in masks:
                mask_img = PILImage.fromarray(mask.astype(np.uint8) * 255)
                mask_img = mask_img.resize((W, H), PILImage.NEAREST)
                resized_masks.append(np.asarray(mask_img) > 0)
            masks = np.stack(resized_masks, axis=0) if resized_masks else np.zeros((0, H, W), dtype=bool)

        scores = best_scores[0].detach().float().cpu().numpy()
        return self.filter_esam_masks(masks, scores, H * W, stability_scores=stability_scores)

    def exclusive_mask_assignment_for_points(
        self,
        masks: List[np.ndarray],
        scores: np.ndarray,
        uu_i: np.ndarray,
        vv_i: np.ndarray,
        initial: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        count = int(uu_i.shape[0])
        assignment = (
            np.asarray(initial, dtype=np.int32).copy()
            if initial is not None and int(initial.shape[0]) == count
            else np.full((count,), -1, dtype=np.int32)
        )
        items = []
        for idx, mask in enumerate(masks):
            if mask is None or mask.size == 0:
                continue
            area = int(np.count_nonzero(mask))
            quality = float(scores[idx]) if idx < int(scores.shape[0]) else 1.0
            # Smaller high-quality regions get first claim; this preserves
            # object parts without allowing large background masks to absorb
            # every depth point.
            priority = float(area) / max(quality, 0.05)
            items.append((priority, -quality, int(idx)))
        for _, _, idx in sorted(items):
            available = assignment < 0
            if not np.any(available):
                break
            inside = np.asarray(masks[idx], dtype=bool)[vv_i, uu_i]
            assignment[np.logical_and(available, inside)] = int(idx)
        return assignment

    def publisher_has_subscribers(self, pub) -> bool:
        try:
            return int(pub.get_subscription_count()) > 0
        except Exception:
            return True

    def debug_rgb_image_has_subscribers(self, raw_pub, compressed_pub) -> bool:
        if self.publisher_has_subscribers(raw_pub):
            return True
        return (
            compressed_pub is not None
            and bool(self.get_parameter("publish_debug_compressed_images").value)
            and self.publisher_has_subscribers(compressed_pub)
        )

    def publish_debug_rgb_image(self, raw_pub, compressed_pub, rgb_img: np.ndarray, header_msg: Header, compressed_format: str = "jpeg"):
        raw_needed = self.publisher_has_subscribers(raw_pub)
        compressed_needed = (
            compressed_pub is not None
            and bool(self.get_parameter("publish_debug_compressed_images").value)
            and self.publisher_has_subscribers(compressed_pub)
        )
        if not raw_needed and not compressed_needed:
            return

        rgb_u8 = np.clip(rgb_img, 0, 255).astype(np.uint8)
        if raw_needed:
            msg = self.bridge.cv2_to_imgmsg(rgb_u8, encoding="rgb8")
            msg.header = Header(stamp=header_msg.stamp, frame_id=header_msg.frame_id)
            raw_pub.publish(msg)

        if not compressed_needed:
            return
        fmt = str(compressed_format).strip().lower()
        if fmt not in ("png", "jpeg", "jpg"):
            fmt = "jpeg"
        ext = ".png" if fmt == "png" else ".jpg"
        bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        if ext == ".jpg":
            quality = int(np.clip(int(self.get_parameter("debug_image_jpeg_quality").value), 30, 100))
            ok, encoded = cv2.imencode(ext, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            out_format = "jpeg"
        else:
            ok, encoded = cv2.imencode(ext, bgr)
            out_format = "png"
        if not ok:
            return
        cmsg = CompressedImage()
        cmsg.header = Header(stamp=header_msg.stamp, frame_id=header_msg.frame_id)
        cmsg.format = out_format
        cmsg.data = encoded.tobytes()
        compressed_pub.publish(cmsg)

    def publish_esam_mask_overlay(self, rgb_cv: np.ndarray, masks: List[np.ndarray], header_msg: Header):
        if not bool(self.get_parameter("publish_esam_mask_overlay").value):
            return
        if not self.debug_rgb_image_has_subscribers(self.pub_esam_mask_overlay, self.pub_esam_mask_overlay_compressed):
            return
        overlay = rgb_cv.copy().astype(np.float32)
        alpha = float(np.clip(float(self.get_parameter("esam_mask_overlay_alpha").value), 0.0, 1.0))
        palette = np.array(
            [
                [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
                [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
                [210, 245, 60], [250, 190, 190], [0, 128, 128], [230, 190, 255],
            ],
            dtype=np.float32,
        )
        for idx, mask in enumerate(masks):
            if mask.size == 0:
                continue
            color = palette[idx % len(palette)]
            overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color
            edge = mask.copy()
            edge[1:, :] &= mask[:-1, :]
            edge[:-1, :] &= mask[1:, :]
            edge[:, 1:] &= mask[:, :-1]
            edge[:, :-1] &= mask[:, 1:]
            overlay[mask & ~edge] = color
        self.publish_debug_rgb_image(
            self.pub_esam_mask_overlay,
            self.pub_esam_mask_overlay_compressed,
            overlay,
            header_msg,
            compressed_format="jpeg",
        )

    def _valid_dense_feature_crop(self, dense_features):
        if dense_features is None:
            return None
        features = np.asarray(
            dense_features.get("features", np.zeros((0, 0, 0))), dtype=np.float32
        )
        if features.ndim != 3 or min(features.shape) <= 0:
            return None
        meta = dense_features.get("meta", {})
        input_size = float(meta.get("input_size", 0.0))
        off_x = float(meta.get("offset_x", 0.0))
        off_y = float(meta.get("offset_y", 0.0))
        run_w = float(meta.get("resized_w", input_size))
        run_h = float(meta.get("resized_h", input_size))
        if input_size <= 0.0 or run_w <= 0.0 or run_h <= 0.0:
            return features
        gh, gw, _ = features.shape
        x_centers = (np.arange(gw, dtype=np.float32) + 0.5) * input_size / float(gw)
        y_centers = (np.arange(gh, dtype=np.float32) + 0.5) * input_size / float(gh)
        valid_x = np.flatnonzero((x_centers >= off_x) & (x_centers < off_x + run_w))
        valid_y = np.flatnonzero((y_centers >= off_y) & (y_centers < off_y + run_h))
        if valid_x.size <= 0 or valid_y.size <= 0:
            return features
        return features[
            int(valid_y[0]) : int(valid_y[-1]) + 1,
            int(valid_x[0]) : int(valid_x[-1]) + 1,
            :,
        ]

    def _feature_projection_rgb(self, features: np.ndarray, projection_key: str):
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 3 or min(features.shape) <= 0:
            return None
        gh, gw, dim = features.shape
        basis = self.feature_color_projections.get(str(projection_key))
        if basis is None or basis.shape != (int(dim), 3):
            seed = 1337 + int(dim) + sum(ord(ch) for ch in str(projection_key))
            rng = np.random.default_rng(seed)
            random_basis = rng.standard_normal((int(dim), 3)).astype(np.float32)
            basis, _ = np.linalg.qr(random_basis)
            basis = basis[:, :3].astype(np.float32)
            self.feature_color_projections[str(projection_key)] = basis
        centered = features.reshape((-1, dim)).astype(np.float32)
        centered -= np.mean(centered, axis=0, keepdims=True)
        projected = (centered @ basis).reshape((gh, gw, 3))
        color = np.zeros_like(projected, dtype=np.float32)
        for channel in range(3):
            low, high = np.percentile(projected[:, :, channel], [2.0, 98.0])
            if float(high - low) > 1e-6:
                color[:, :, channel] = np.clip(
                    (projected[:, :, channel] - float(low)) / float(high - low),
                    0.0,
                    1.0,
                )
            else:
                color[:, :, channel] = 0.5
        return np.clip(color * 255.0, 0, 255).astype(np.uint8)

    def _sam_boundary_mask(self, masks: List[np.ndarray], image_shape):
        if not bool(self.get_parameter("diagnostic_feature_sam_boundaries").value):
            return None
        h, w = int(image_shape[0]), int(image_shape[1])
        boundary = np.zeros((h, w), dtype=bool)
        for mask in masks:
            if mask is None or mask.shape != (h, w) or not np.any(mask):
                continue
            mask_u8 = np.asarray(mask, dtype=np.uint8)
            eroded = cv2.erode(mask_u8, np.ones((3, 3), dtype=np.uint8), iterations=1)
            boundary |= np.logical_and(mask_u8 > 0, eroded == 0)
        return boundary

    def _draw_feature_boundaries(
        self, view: np.ndarray, masks: List[np.ndarray], boundary=None
    ):
        if boundary is None:
            boundary = self._sam_boundary_mask(masks, view.shape[:2])
        if boundary is not None and np.any(boundary):
            halo = cv2.dilate(boundary.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
            view[halo] = np.array([20, 20, 20], dtype=np.uint8)
            view[boundary] = np.array([245, 245, 245], dtype=np.uint8)
        return view

    def _publish_feature_overview(
        self,
        rgb_cv: np.ndarray,
        masks: List[np.ndarray],
        dense_features,
        projection_key: str,
        rgb_alpha: float,
        raw_pub,
        compressed_pub,
        header_msg: Header,
        boundary=None,
    ):
        if not self.debug_rgb_image_has_subscribers(raw_pub, compressed_pub):
            return
        features = self._valid_dense_feature_crop(dense_features)
        color = self._feature_projection_rgb(features, projection_key) if features is not None else None
        if color is None:
            return
        # Display-only smoothing. Association continues to use the untouched
        # token grid and fractional mask pooling.
        color = cv2.resize(
            color,
            (int(rgb_cv.shape[1]), int(rgb_cv.shape[0])),
            interpolation=cv2.INTER_LINEAR,
        )
        alpha = float(np.clip(rgb_alpha, 0.0, 1.0))
        view = np.clip(
            alpha * rgb_cv.astype(np.float32)
            + (1.0 - alpha) * color.astype(np.float32),
            0,
            255,
        ).astype(np.uint8)
        view = self._draw_feature_boundaries(view, masks, boundary)
        self.publish_debug_rgb_image(
            raw_pub, compressed_pub, view, header_msg, compressed_format="jpeg"
        )

    def _publish_feature_similarity(
        self,
        rgb_cv: np.ndarray,
        masks: List[np.ndarray],
        dense_features,
        raw_pub,
        compressed_pub,
        header_msg: Header,
        boundary=None,
    ):
        if not self.debug_rgb_image_has_subscribers(raw_pub, compressed_pub):
            return
        features = self._valid_dense_feature_crop(dense_features)
        if features is None:
            return
        gh, gw, _ = features.shape
        nx = float(
            np.clip(float(self.get_parameter("diagnostic_similarity_x").value), 0.0, 1.0)
        )
        ny = float(
            np.clip(float(self.get_parameter("diagnostic_similarity_y").value), 0.0, 1.0)
        )
        qx = int(np.clip(math.floor(nx * gw), 0, gw - 1))
        qy = int(np.clip(math.floor(ny * gh), 0, gh - 1))
        flat = features.reshape((-1, features.shape[-1])).astype(np.float32)
        flat /= np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-6)
        query = flat[qy * gw + qx]
        similarity = (flat @ query).reshape((gh, gw))
        floor = float(
            np.clip(
                float(self.get_parameter("diagnostic_similarity_floor").value),
                -1.0,
                0.99,
            )
        )
        normalized = np.clip((similarity - floor) / max(1.0 - floor, 1e-6), 0.0, 1.0)
        heat_u8 = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(heat_u8, getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET))
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
        heat_rgb = cv2.resize(
            heat_rgb,
            (int(rgb_cv.shape[1]), int(rgb_cv.shape[0])),
            interpolation=cv2.INTER_LINEAR,
        )
        alpha = float(
            np.clip(
                float(self.get_parameter("diagnostic_similarity_rgb_alpha").value),
                0.0,
                1.0,
            )
        )
        view = np.clip(
            alpha * rgb_cv.astype(np.float32)
            + (1.0 - alpha) * heat_rgb.astype(np.float32),
            0,
            255,
        ).astype(np.uint8)
        view = self._draw_feature_boundaries(view, masks, boundary)
        px = int(np.clip(round(nx * max(rgb_cv.shape[1] - 1, 0)), 0, rgb_cv.shape[1] - 1))
        py = int(np.clip(round(ny * max(rgb_cv.shape[0] - 1, 0)), 0, rgb_cv.shape[0] - 1))
        cv2.drawMarker(view, (px, py), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.drawMarker(view, (px, py), (15, 15, 15), cv2.MARKER_CROSS, 10, 1)
        cv2.putText(view, "token cosine: red = similar", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(view, "token cosine: red = similar", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
        self.publish_debug_rgb_image(
            raw_pub, compressed_pub, view, header_msg, compressed_format="jpeg"
        )

    def publish_feature_diagnostics(
        self,
        rgb_cv: np.ndarray,
        masks: List[np.ndarray],
        clip_mask_emb: np.ndarray,
        dino_mask_emb: np.ndarray,
        dino_mask_quality: np.ndarray,
        dense_dino,
        header_msg: Header,
    ):
        boundary = self._sam_boundary_mask(masks, rgb_cv.shape[:2])
        if bool(self.get_parameter("publish_dinov2_feature_view").value):
            self._publish_feature_overview(
                rgb_cv,
                masks,
                dense_dino,
                "dino",
                float(self.get_parameter("dinov2_feature_view_rgb_alpha").value),
                self.pub_dinov2_feature_view,
                self.pub_dinov2_feature_view_compressed,
                header_msg,
                boundary,
            )
        if bool(self.get_parameter("publish_dinov2_similarity_view").value):
            self._publish_feature_similarity(
                rgb_cv,
                masks,
                dense_dino,
                self.pub_dinov2_similarity_view,
                self.pub_dinov2_similarity_view_compressed,
                header_msg,
                boundary,
            )
        if bool(self.get_parameter("publish_dinov2_instance_view").value):
            self.publish_dinov2_instance_view(
                rgb_cv, masks, dino_mask_emb, dino_mask_quality, header_msg, boundary
            )
        if bool(self.get_parameter("publish_clip_feature_view").value):
            self._publish_mask_embedding_overview(
                rgb_cv, masks, clip_mask_emb, "clip_masks",
                self.pub_clip_feature_view, self.pub_clip_feature_view_compressed,
                header_msg, boundary,
            )
        if bool(self.get_parameter("publish_clip_similarity_view").value):
            self._publish_mask_embedding_similarity(
                rgb_cv, masks, clip_mask_emb,
                self.pub_clip_similarity_view, self.pub_clip_similarity_view_compressed,
                header_msg, boundary,
            )

    def _publish_mask_embedding_overview(
        self,
        rgb_cv: np.ndarray,
        masks: List[np.ndarray],
        embeddings: np.ndarray,
        projection_key: str,
        raw_pub,
        compressed_pub,
        header_msg: Header,
        boundary=None,
    ):
        if not self.debug_rgb_image_has_subscribers(raw_pub, compressed_pub):
            return
        if (
            not masks
            or embeddings.ndim != 2
            or embeddings.shape[0] != len(masks)
            or embeddings.shape[1] <= 0
        ):
            return
        colors = self._feature_projection_rgb(
            embeddings.reshape((1, len(masks), -1)), projection_key
        )
        if colors is None:
            return
        alpha = float(
            np.clip(
                1.0 - float(self.get_parameter("clip_feature_view_rgb_alpha").value),
                0.0,
                1.0,
            )
        )
        view = (0.18 * rgb_cv.astype(np.float32)).astype(np.uint8)
        order = sorted(
            range(len(masks)),
            key=lambda idx: int(np.count_nonzero(masks[idx])),
            reverse=True,
        )
        for idx in order:
            mask = np.asarray(masks[idx], dtype=bool)
            if mask.shape != rgb_cv.shape[:2] or not np.any(mask):
                continue
            color = colors[0, idx].astype(np.float32)
            view[mask] = np.clip(
                (1.0 - alpha) * rgb_cv[mask].astype(np.float32) + alpha * color,
                0,
                255,
            ).astype(np.uint8)
        view = self._draw_feature_boundaries(view, masks, boundary)
        self.publish_debug_rgb_image(
            raw_pub, compressed_pub, view, header_msg, compressed_format="jpeg"
        )

    def _publish_mask_embedding_similarity(
        self,
        rgb_cv: np.ndarray,
        masks: List[np.ndarray],
        embeddings: np.ndarray,
        raw_pub,
        compressed_pub,
        header_msg: Header,
        boundary=None,
    ):
        if not self.debug_rgb_image_has_subscribers(raw_pub, compressed_pub):
            return
        if (
            not masks
            or embeddings.ndim != 2
            or embeddings.shape[0] != len(masks)
            or embeddings.shape[1] <= 0
        ):
            return
        h, w = rgb_cv.shape[:2]
        nx = float(
            np.clip(float(self.get_parameter("diagnostic_similarity_x").value), 0.0, 1.0)
        )
        ny = float(
            np.clip(float(self.get_parameter("diagnostic_similarity_y").value), 0.0, 1.0)
        )
        px = int(np.clip(round(nx * max(w - 1, 0)), 0, w - 1))
        py = int(np.clip(round(ny * max(h - 1, 0)), 0, h - 1))
        embedding_norms = np.linalg.norm(embeddings, axis=1)
        candidates = [
            idx
            for idx, mask in enumerate(masks)
            if mask.shape == (h, w)
            and bool(mask[py, px])
            and float(embedding_norms[idx]) > 1e-6
        ]
        view = (0.18 * rgb_cv.astype(np.float32)).astype(np.uint8)
        if candidates:
            selected = min(candidates, key=lambda idx: int(np.count_nonzero(masks[idx])))
            unit = np.asarray(embeddings, dtype=np.float32)
            unit /= np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-6)
            similarity = unit @ unit[int(selected)]
            floor = float(
                np.clip(
                    float(self.get_parameter("diagnostic_similarity_floor").value),
                    -1.0,
                    0.99,
                )
            )
            normalized = np.clip(
                (similarity - floor) / max(1.0 - floor, 1e-6), 0.0, 1.0
            )
            heat_u8 = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)[None, :]
            heat_bgr = cv2.applyColorMap(
                heat_u8, getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
            )
            heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)[0]
            order = sorted(
                range(len(masks)),
                key=lambda idx: int(np.count_nonzero(masks[idx])),
                reverse=True,
            )
            for idx in order:
                mask = np.asarray(masks[idx], dtype=bool)
                if mask.shape != (h, w) or not np.any(mask):
                    continue
                view[mask] = heat_rgb[idx]
            label = f"mask {int(selected)} cosine: red = similar"
        else:
            label = "click inside a SAM mask"
        view = self._draw_feature_boundaries(view, masks, boundary)
        cv2.drawMarker(view, (px, py), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.drawMarker(view, (px, py), (15, 15, 15), cv2.MARKER_CROSS, 10, 1)
        cv2.putText(view, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(view, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
        self.publish_debug_rgb_image(
            raw_pub, compressed_pub, view, header_msg, compressed_format="jpeg"
        )

    def publish_dinov2_instance_view(
        self,
        rgb_cv: np.ndarray,
        masks: List[np.ndarray],
        dino_mask_emb: np.ndarray,
        dino_mask_quality: np.ndarray,
        header_msg: Header,
        boundary=None,
    ):
        if not self.debug_rgb_image_has_subscribers(
            self.pub_dinov2_instance_view,
            self.pub_dinov2_instance_view_compressed,
        ):
            return
        if (
            not masks
            or dino_mask_emb.ndim != 2
            or dino_mask_emb.shape[0] != len(masks)
            or dino_mask_emb.shape[1] <= 0
        ):
            return
        colors = self._feature_projection_rgb(
            dino_mask_emb.reshape((1, len(masks), -1)), "dino_instances"
        )
        if colors is None:
            return
        view = (0.18 * rgb_cv.astype(np.float32)).astype(np.uint8)
        order = sorted(
            range(len(masks)),
            key=lambda idx: int(np.count_nonzero(masks[idx])),
            reverse=True,
        )
        for idx in order:
            mask = np.asarray(masks[idx], dtype=bool)
            if (
                mask.shape != rgb_cv.shape[:2]
                or not np.any(mask)
                or float(np.linalg.norm(dino_mask_emb[idx])) <= 1e-6
            ):
                continue
            quality = (
                float(dino_mask_quality[idx])
                if idx < int(dino_mask_quality.shape[0])
                else 0.0
            )
            alpha = float(np.clip(0.35 + 0.65 * quality, 0.35, 1.0))
            color = colors[0, idx].astype(np.float32)
            view[mask] = np.clip(
                (1.0 - alpha) * view[mask].astype(np.float32) + alpha * color,
                0,
                255,
            ).astype(np.uint8)
        view = self._draw_feature_boundaries(view, masks, boundary)
        self.publish_debug_rgb_image(
            self.pub_dinov2_instance_view,
            self.pub_dinov2_instance_view_compressed,
            view,
            header_msg,
            compressed_format="jpeg",
        )

    def publish_debug_observation_images(self, rgb_cv: np.ndarray, rgb_header: Header, depth_msg):
        if not bool(self.get_parameter("publish_debug_observation_images").value):
            return

        hz = max(0.0, float(self.get_parameter("debug_observation_image_hz").value))
        now_s = self.now_sec()
        if hz > 0.0 and (now_s - self.last_debug_observation_image_pub_s) < (1.0 / hz):
            return
        self.last_debug_observation_image_pub_s = now_s

        if bool(self.get_parameter("publish_debug_observation_rgb").value):
            try:
                self.publish_debug_rgb_image(
                    self.pub_debug_observation_rgb,
                    self.pub_debug_observation_rgb_compressed,
                    rgb_cv,
                    rgb_header,
                    compressed_format="jpeg",
                )
            except Exception as e:
                self.get_logger().warn(f"Debug observation RGB publish failed: {e}")

        if depth_msg is None or not bool(self.get_parameter("publish_debug_observation_depth").value):
            return
        depth_raw_needed = self.publisher_has_subscribers(self.pub_debug_observation_depth)
        depth_compressed_needed = (
            bool(self.get_parameter("publish_debug_compressed_images").value)
            and self.publisher_has_subscribers(self.pub_debug_observation_depth_compressed)
        )
        if not depth_raw_needed and not depth_compressed_needed:
            return
        try:
            depth_m = self.decode_depth_msg(depth_msg)
            if depth_m.ndim != 2:
                return
            valid = np.isfinite(depth_m) & (depth_m > 0.0)
            max_m = max(1e-3, float(self.get_parameter("debug_observation_depth_max_m").value))
            norm = np.zeros(depth_m.shape, dtype=np.float32)
            norm[valid] = np.clip(depth_m[valid] / max_m, 0.0, 1.0)
            gray = np.clip((1.0 - norm) * 255.0, 0.0, 255.0).astype(np.uint8)
            gray[~valid] = 0
            color_map = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
            depth_bgr = cv2.applyColorMap(gray, color_map)
            depth_rgb = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)
            self.publish_debug_rgb_image(
                self.pub_debug_observation_depth,
                self.pub_debug_observation_depth_compressed,
                depth_rgb,
                depth_msg.header,
                compressed_format="jpeg",
            )
        except Exception as e:
            self.get_logger().warn(f"Debug observation depth publish failed: {e}")

    def compute_mask_embeddings_batched(self, rgb_cv: np.ndarray, masks: List[np.ndarray], dense_clip=None):
        if not masks:
            return np.zeros((0, 0), dtype=np.float32)
        batch_size = max(1, int(self.get_parameter("esam_clip_batch_size").value))
        dense_weight = float(np.clip(float(self.get_parameter("clip_dense_mask_weight").value), 0.0, 1.0))
        dense_emb = self.compute_dense_mask_embeddings(masks, dense_clip) if dense_clip is not None else np.zeros((0, 0), dtype=np.float32)
        if dense_emb.shape[0] == len(masks) and dense_emb.shape[1] > 0 and dense_weight >= 0.999:
            return dense_emb.astype(np.float32)

        variants_cfg = str(self.get_parameter("clip_mask_embedding_variants").value).strip().lower()
        variants = [v.strip() for v in variants_cfg.split(",") if v.strip()]
        if not variants:
            variants = ["masked"]
        weights_by_variant = {
            "masked": 0.50,
            "mask": 0.50,
            "context": 0.35,
            "full": 0.15,
            "crop": 0.15,
        }
        margin_frac = max(0.0, float(self.get_parameter("clip_context_margin_frac").value))
        bg_alpha = float(np.clip(float(self.get_parameter("clip_context_bg_alpha").value), 0.0, 1.0))
        min_crop = max(1, int(self.get_parameter("clip_mask_min_crop_px").value))
        H, W = rgb_cv.shape[:2]

        crops = []
        owners = []
        weights = []
        for mask_idx, mask in enumerate(masks):
            ys, xs = np.where(mask)
            if xs.size == 0 or ys.size == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            bw = max(1, x1 - x0)
            bh = max(1, y1 - y0)

            if bw < min_crop:
                pad = int(math.ceil(0.5 * (min_crop - bw)))
                x0 -= pad
                x1 += pad
            if bh < min_crop:
                pad = int(math.ceil(0.5 * (min_crop - bh)))
                y0 -= pad
                y1 += pad

            for variant in variants:
                is_context = variant in ("context", "full", "crop")
                mx = int(math.ceil(margin_frac * max(1, x1 - x0))) if is_context else 0
                my = int(math.ceil(margin_frac * max(1, y1 - y0))) if is_context else 0
                cx0 = max(0, x0 - mx)
                cx1 = min(W, x1 + mx)
                cy0 = max(0, y0 - my)
                cy1 = min(H, y1 + my)
                if cx1 <= cx0 or cy1 <= cy0:
                    continue

                crop = rgb_cv[cy0:cy1, cx0:cx1, :].copy()
                crop_mask = mask[cy0:cy1, cx0:cx1]
                if variant in ("masked", "mask"):
                    if np.any(crop_mask):
                        bg = np.mean(crop[crop_mask], axis=0)
                    else:
                        bg = np.mean(crop.reshape((-1, 3)), axis=0)
                    crop[~crop_mask] = np.asarray(bg, dtype=np.uint8)
                elif variant == "context":
                    if np.any(crop_mask):
                        bg = np.mean(crop[crop_mask], axis=0)
                    else:
                        bg = np.mean(crop.reshape((-1, 3)), axis=0)
                    outside = ~crop_mask
                    crop[outside] = np.clip(
                        bg_alpha * crop[outside].astype(np.float32)
                        + (1.0 - bg_alpha) * np.asarray(bg, dtype=np.float32),
                        0,
                        255,
                    ).astype(np.uint8)
                elif variant in ("full", "crop"):
                    pass
                else:
                    continue

                crops.append(PILImage.fromarray(crop))
                owners.append(int(mask_idx))
                weights.append(float(weights_by_variant.get(variant, 0.20)))

        if not crops:
            return dense_emb.astype(np.float32) if dense_emb.shape[0] == len(masks) else np.zeros((0, 0), dtype=np.float32)

        crop_emb = self._clip_encode_pil_batch(crops, batch_size)
        if crop_emb.shape[0] == 0:
            return dense_emb.astype(np.float32) if dense_emb.shape[0] == len(masks) else np.zeros((0, 0), dtype=np.float32)

        dim = int(crop_emb.shape[1])
        out = np.zeros((len(masks), dim), dtype=np.float32)
        wsum = np.zeros((len(masks),), dtype=np.float32)
        for emb, owner, weight in zip(crop_emb, owners, weights):
            out[int(owner), :] += float(weight) * emb.astype(np.float32)
            wsum[int(owner)] += float(weight)
        valid = wsum > 1e-6
        out[valid, :] /= wsum[valid, None]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        valid_norm = norms[:, 0] > 1e-6
        out[valid_norm, :] /= norms[valid_norm, :]
        if dense_emb.shape[0] == len(masks) and dense_emb.shape[1] == out.shape[1]:
            return self._combine_unit_embeddings(dense_emb, out, dense_weight)
        return out.astype(np.float32)

    def ensure_query_embedding(self):
        new_query = str(self.get_parameter("query_text").value)
        if new_query != self.query_text:
            self.query_text = new_query
            self.text_emb = None
            self.text_emb_query = None
            self.object_query_support_cache = {}
            self.object_query_score_norm_bounds = None
            self.segment_query_support_cache = {}
            self.segment_query_score_norm_bounds = None
            self.has_loaded_grid_snapshot = False
            empty = self._empty_grid_bundle()
            for name, grid in empty.items():
                self._set_latest_grid(name, grid)
            self.get_logger().info(f"Updated query_text -> '{self.query_text}'")

        if self.text_emb is None or self.text_emb_query != self.query_text:
            self.text_emb = self.compute_clip_text_embedding(self.query_text)
            self.text_emb_query = self.query_text

    # -------------------------
    # Grid helpers (INTERNAL frame grid)
    # -------------------------
    def xy_to_cell(self, x, y):
        ix = int((x - self.origin_x) / self.resolution)
        iy = int((y - self.origin_y) / self.resolution)
        if ix < 0 or ix >= self.grid_width or iy < 0 or iy >= self.grid_height:
            return None
        return (ix, iy)

    def cell_center_xy(self, ix, iy):
        x = self.origin_x + (ix + 0.5) * self.resolution
        y = self.origin_y + (iy + 0.5) * self.resolution
        return x, y

    # -------------------------
    # STM decay + coherence / embedding
    # -------------------------
    def decay_stm_cell(self, key, now_s: float):
        half_life = float(self.get_parameter("decay_half_life_s").value)
        if half_life <= 0.0:
            return
        last = self.stm_last.get(key, now_s)
        dt = max(0.0, now_s - last)
        if dt <= 1e-6:
            return
        lam = np.log(2.0) / half_life
        decay = float(np.exp(-lam * dt))
        self.stm_w[key] *= decay
        #self.stm_sum[key] *= decay  # this activates decay on the embedding too
        self.stm_last[key] = now_s
        self._mark_state_dirty()
        if self.stm_w[key] < 1e-3:
            self._drop_stm_cell(key)

    def _embedding_and_coherence(self, sum_vec: np.ndarray, weight: float):
        """
        Returns:
            emb_unit (D,), coherence (0..1-ish), weight
        """
        W = float(weight)
        if W <= 1e-6:
            return None, None, None
        S = sum_vec
        mean = S / W
        coh = float(np.linalg.norm(mean))  # 0..1
        nS = float(np.linalg.norm(S))
        if nS <= 1e-6:
            return None, 0.0, W
        emb_unit = S / nS
        return emb_unit.astype(np.float32), coh, W

    def stm_embedding_and_coherence(self, key):
        if key not in self.stm_sum or key not in self.stm_w:
            return None, None, None
        return self._embedding_and_coherence(self.stm_sum[key], self.stm_w[key])

    def ltm_embedding_and_coherence(self, key):
        if key not in self.ltm_sum or key not in self.ltm_w:
            return None, None, None
        return self._embedding_and_coherence(self.ltm_sum[key], self.ltm_w[key])

    def _update_z_stats(
        self,
        z_min_dict: Dict[Tuple[int, int], float],
        z_max_dict: Dict[Tuple[int, int], float],
        z_sum_dict: Dict[Tuple[int, int], float],
        z_count_dict: Dict[Tuple[int, int], int],
        key: Tuple[int, int],
        z: Optional[float],
    ):
        if z is None or not np.isfinite(z):
            return
        zf = float(z)
        if key not in z_count_dict or int(z_count_dict.get(key, 0)) <= 0:
            z_min_dict[key] = zf
            z_max_dict[key] = zf
            z_sum_dict[key] = zf
            z_count_dict[key] = 1
        else:
            z_min_dict[key] = min(float(z_min_dict.get(key, zf)), zf)
            z_max_dict[key] = max(float(z_max_dict.get(key, zf)), zf)
            z_sum_dict[key] = float(z_sum_dict.get(key, 0.0)) + zf
            z_count_dict[key] = int(z_count_dict.get(key, 0)) + 1
        self._mark_state_dirty()

    def _copy_stm_z_to_ltm(self, key: Tuple[int, int], replace: bool = False):
        if key not in self.stm_z_count or int(self.stm_z_count.get(key, 0)) <= 0:
            return
        if replace or key not in self.ltm_z_count or int(self.ltm_z_count.get(key, 0)) <= 0:
            self.ltm_z_min[key] = float(self.stm_z_min.get(key, 0.0))
            self.ltm_z_max[key] = float(self.stm_z_max.get(key, 0.0))
            self.ltm_z_sum[key] = float(self.stm_z_sum.get(key, 0.0))
            self.ltm_z_count[key] = int(self.stm_z_count.get(key, 0))
        else:
            self.ltm_z_min[key] = min(float(self.ltm_z_min.get(key, 0.0)), float(self.stm_z_min.get(key, 0.0)))
            self.ltm_z_max[key] = max(float(self.ltm_z_max.get(key, 0.0)), float(self.stm_z_max.get(key, 0.0)))
            self.ltm_z_sum[key] = float(self.ltm_z_sum.get(key, 0.0)) + float(self.stm_z_sum.get(key, 0.0))
            self.ltm_z_count[key] = int(self.ltm_z_count.get(key, 0)) + int(self.stm_z_count.get(key, 0))
        self._mark_state_dirty()

    def _drop_stm_cell(self, key: Tuple[int, int]):
        self.stm_w.pop(key, None)
        self.stm_sum.pop(key, None)
        self.stm_last.pop(key, None)
        self.stm_viewmask.pop(key, None)
        self.stm_z_min.pop(key, None)
        self.stm_z_max.pop(key, None)
        self.stm_z_sum.pop(key, None)
        self.stm_z_count.pop(key, None)

    def _drop_ltm_cell(self, key: Tuple[int, int]):
        self.ltm_w.pop(key, None)
        self.ltm_sum.pop(key, None)
        self.ltm_last_seen.pop(key, None)
        self.ltm_viewmask.pop(key, None)
        self.ltm_z_min.pop(key, None)
        self.ltm_z_max.pop(key, None)
        self.ltm_z_sum.pop(key, None)
        self.ltm_z_count.pop(key, None)

    # -------------------------
    # 3D semantic voxel memory
    # -------------------------
    def xyz_to_voxel(self, x: float, y: float, z: float):
        if not bool(self.get_parameter("use_semantic_voxels").value):
            return None
        if self.voxel_size <= 1e-6:
            return None
        if z < self.voxel_z_min or z >= self.voxel_z_max:
            return None
        ix = int((x - self.origin_x) / self.voxel_size)
        iy = int((y - self.origin_y) / self.voxel_size)
        iz = int((z - self.voxel_z_min) / self.voxel_size)
        if ix < 0 or ix >= self.voxel_width or iy < 0 or iy >= self.voxel_height or iz < 0 or iz >= self.voxel_depth:
            return None
        return (ix, iy, iz)

    def voxel_center_xyz(self, key: Tuple[int, int, int]):
        x = self.origin_x + (key[0] + 0.5) * self.voxel_size
        y = self.origin_y + (key[1] + 0.5) * self.voxel_size
        z = self.voxel_z_min + (key[2] + 0.5) * self.voxel_size
        return x, y, z

    def vox_stm_embedding_and_coherence(self, key):
        if key not in self.vox_stm_sum or key not in self.vox_stm_w:
            return None, None, None
        return self._embedding_and_coherence(self.vox_stm_sum[key], self.vox_stm_w[key])

    def vox_ltm_embedding_and_coherence(self, key):
        if key not in self.vox_ltm_sum or key not in self.vox_ltm_w:
            return None, None, None
        return self._embedding_and_coherence(self.vox_ltm_sum[key], self.vox_ltm_w[key])

    def _drop_vox_stm_cell(self, key: Tuple[int, int, int]):
        self.vox_stm_w.pop(key, None)
        self.vox_stm_sum.pop(key, None)
        self.vox_stm_last.pop(key, None)
        self.vox_stm_viewmask.pop(key, None)
        if key not in self.vox_ltm_w:
            self.vox_clear_hits.pop(key, None)
            if hasattr(self, "voxel_segment_id"):
                self._drop_segment_voxel_owner_if_empty(key)

    def _drop_vox_ltm_cell(self, key: Tuple[int, int, int]):
        self.vox_ltm_w.pop(key, None)
        self.vox_ltm_sum.pop(key, None)
        self.vox_ltm_last_seen.pop(key, None)
        self.vox_ltm_viewmask.pop(key, None)
        if key not in self.vox_stm_w:
            self.vox_clear_hits.pop(key, None)
            if hasattr(self, "voxel_segment_id"):
                self._drop_segment_voxel_owner_if_empty(key)

    def decay_vox_stm_cell(self, key, now_s: float):
        half_life = float(self.get_parameter("decay_half_life_s").value)
        if half_life <= 0.0:
            return
        last = self.vox_stm_last.get(key, now_s)
        dt = max(0.0, now_s - last)
        if dt <= 1e-6:
            return
        lam = np.log(2.0) / half_life
        decay = float(np.exp(-lam * dt))
        self.vox_stm_w[key] *= decay
        self.vox_stm_last[key] = now_s
        self._mark_state_dirty()
        if self.vox_stm_w[key] < 1e-3:
            self._drop_vox_stm_cell(key)

    def fuse_vox_stm_cell(self, key, new_emb_unit: np.ndarray, conf: float, now_s: float):
        if conf <= 0.0:
            return
        self.vox_clear_hits.pop(key, None)
        self.segment_voxel_miss_hits.pop(key, None)
        if key in self.vox_stm_w:
            self.decay_vox_stm_cell(key, now_s)

        use_gating = bool(self.get_parameter("use_consistency_gating").value)
        gating_cos_min = float(self.get_parameter("gating_cos_min").value)
        gating_reject = bool(self.get_parameter("gating_reject").value)

        if key not in self.vox_stm_sum:
            max_count = int(self.get_parameter("voxel_max_count").value)
            if max_count > 0 and len(self.vox_stm_w) >= max_count:
                return
            self.vox_stm_sum[key] = (conf * new_emb_unit).astype(np.float32)
            self.vox_stm_w[key] = float(conf)
            self.vox_stm_last[key] = now_s
            self.vox_stm_viewmask[key] = self.vox_stm_viewmask.get(key, 0)
            self._mark_state_dirty()
            return

        emb_old, _, _ = self.vox_stm_embedding_and_coherence(key)
        if emb_old is None:
            self.vox_stm_sum[key] = (conf * new_emb_unit).astype(np.float32)
            self.vox_stm_w[key] = float(conf)
            self.vox_stm_last[key] = now_s
            self._mark_state_dirty()
            return

        if use_gating:
            cos = float(np.dot(emb_old, new_emb_unit))
            if cos < gating_cos_min:
                if gating_reject:
                    return
                conf *= self._soft_gating_multiplier(cos, gating_cos_min)

        w_max = float(self.get_parameter("cell_weight_max").value)
        self.vox_stm_sum[key] = (self.vox_stm_sum[key] + conf * new_emb_unit).astype(np.float32)
        self.vox_stm_w[key] = min(float(self.vox_stm_w[key] + conf), w_max)
        self.vox_stm_last[key] = now_s
        self._mark_state_dirty()

    def promote_vox_stm_to_ltm(self, key: Tuple[int, int, int], emb_unit: np.ndarray, coh01: float, conf01: float, now_s: float):
        ltm_w_max = float(self.get_parameter("ltm_weight_max").value)
        upd_gain = float(self.get_parameter("ltm_update_gain").value)
        rep_cos = float(self.get_parameter("ltm_replace_cos_thresh").value)
        rep_gain = float(self.get_parameter("ltm_replace_gain").value)

        ev = float(np.clip(conf01 * coh01, 0.0, 1.0))
        if ev <= 1e-6:
            return
        ev_w = float(upd_gain * ev)

        if key not in self.vox_ltm_sum:
            self.vox_ltm_sum[key] = (ev_w * emb_unit).astype(np.float32)
            self.vox_ltm_w[key] = float(np.clip(ev_w, 0.0, ltm_w_max))
            self.vox_ltm_last_seen[key] = now_s
            self.vox_ltm_viewmask[key] = int(self.vox_stm_viewmask.get(key, 0))
            self._mark_state_dirty()
            return

        old_u, _, old_w = self.vox_ltm_embedding_and_coherence(key)
        if old_u is None or old_w is None:
            self.vox_ltm_sum[key] = (ev_w * emb_unit).astype(np.float32)
            self.vox_ltm_w[key] = float(np.clip(ev_w, 0.0, ltm_w_max))
            self.vox_ltm_last_seen[key] = now_s
            self.vox_ltm_viewmask[key] = int(self.vox_stm_viewmask.get(key, 0))
            self._mark_state_dirty()
            return

        cos = float(np.dot(old_u, emb_unit))
        if cos < rep_cos:
            ev_rep = float(np.clip(rep_gain * ev_w, 0.0, ltm_w_max))
            self.vox_ltm_sum[key] = (ev_rep * emb_unit).astype(np.float32)
            self.vox_ltm_w[key] = float(ev_rep)
            self.vox_ltm_last_seen[key] = now_s
            self.vox_ltm_viewmask[key] = int(self.vox_ltm_viewmask.get(key, 0) | int(self.vox_stm_viewmask.get(key, 0)))
            self._mark_state_dirty()
            return

        self.vox_ltm_sum[key] = (self.vox_ltm_sum[key] + ev_w * emb_unit).astype(np.float32)
        self.vox_ltm_w[key] = float(np.clip(self.vox_ltm_w[key] + ev_w, 0.0, ltm_w_max))
        self.vox_ltm_last_seen[key] = now_s
        self.vox_ltm_viewmask[key] = int(self.vox_ltm_viewmask.get(key, 0) | int(self.vox_stm_viewmask.get(key, 0)))
        self._mark_state_dirty()

    def _decay_voxel_semantic_evidence(self, key: Tuple[int, int, int], stm_decay: float, ltm_decay: float):
        changed = False
        if key in self.vox_stm_w:
            if key in self.vox_stm_sum:
                self.vox_stm_sum[key] = (self.vox_stm_sum[key] * stm_decay).astype(np.float32)
            self.vox_stm_w[key] = float(self.vox_stm_w[key] * stm_decay)
            if self.vox_stm_w[key] < 1e-3:
                self._drop_vox_stm_cell(key)
            changed = True

        if bool(self.get_parameter("voxel_clear_ltm").value) and key in self.vox_ltm_w:
            if key in self.vox_ltm_sum:
                self.vox_ltm_sum[key] = (self.vox_ltm_sum[key] * ltm_decay).astype(np.float32)
            self.vox_ltm_w[key] = float(self.vox_ltm_w[key] * ltm_decay)
            if self.vox_ltm_w[key] < 1e-3:
                self._drop_vox_ltm_cell(key)
            changed = True

        if changed:
            self._mark_state_dirty()
        return changed

    def _drop_voxel_semantic_evidence(self, key: Tuple[int, int, int]):
        changed = key in self.vox_stm_w or key in self.vox_ltm_w
        if not changed:
            self.vox_clear_hits.pop(key, None)
            return False
        if key in self.vox_stm_w:
            self._drop_vox_stm_cell(key)
        if key in self.vox_ltm_w:
            self._drop_vox_ltm_cell(key)
        self.vox_clear_hits.pop(key, None)
        self._mark_state_dirty()
        return True

    def _clear_voxel_candidate_keys(self, cam_int: np.ndarray, pt_int: np.ndarray):
        ray = np.asarray(pt_int, dtype=np.float32) - np.asarray(cam_int, dtype=np.float32)
        dist = float(np.linalg.norm(ray))
        if not np.isfinite(dist):
            return set()
        min_range = float(self.get_parameter("voxel_clear_min_range_m").value)
        hit_margin = float(self.get_parameter("voxel_clear_hit_margin_m").value)
        clear_dist = dist - max(0.0, hit_margin)
        if clear_dist <= min_range:
            return set()

        step = float(self.get_parameter("voxel_clear_step_m").value)
        if step <= 0.0:
            step = max(0.5 * float(self.voxel_size), 0.03)
        direction = ray / max(dist, 1e-6)
        hit_key = self.xyz_to_voxel(float(pt_int[0]), float(pt_int[1]), float(pt_int[2]))

        keys = set()
        t = min_range
        while t < clear_dist:
            p = np.asarray(cam_int, dtype=np.float32) + direction * float(t)
            key = self.xyz_to_voxel(float(p[0]), float(p[1]), float(p[2]))
            if key is not None and key != hit_key:
                keys.add(key)
            t += step
        return keys

    def clear_voxels_along_ray(self, cam_int: np.ndarray, pt_int: np.ndarray):
        stm_decay = float(np.clip(float(self.get_parameter("voxel_clear_stm_decay").value), 0.0, 1.0))
        ltm_decay = float(np.clip(float(self.get_parameter("voxel_clear_ltm_decay").value), 0.0, 1.0))
        if stm_decay >= 1.0 and ltm_decay >= 1.0:
            return 0
        changed = 0
        for key in self._clear_voxel_candidate_keys(cam_int, pt_int):
            if self._decay_voxel_semantic_evidence(key, stm_decay, ltm_decay):
                changed += 1
        return changed

    def clear_voxels_from_depth_rays(self, pts_int: np.ndarray, cam_int: np.ndarray):
        if not bool(self.get_parameter("use_semantic_voxels").value):
            return
        if not bool(self.get_parameter("use_voxel_ray_clearing").value):
            return
        if pts_int is None or pts_int.shape[0] == 0:
            return

        stride = max(1, int(self.get_parameter("voxel_clear_stride").value))
        idxs = np.arange(0, pts_int.shape[0], stride, dtype=np.int32)
        max_rays = int(self.get_parameter("voxel_clear_max_rays_per_frame").value)
        if max_rays > 0 and idxs.size > max_rays:
            sel = np.linspace(0, idxs.size - 1, num=max_rays, dtype=np.int32)
            idxs = idxs[sel]

        candidate_keys = set()
        for idx in idxs:
            candidate_keys.update(self._clear_voxel_candidate_keys(cam_int, pts_int[int(idx)]))

        # Rays through the background can cross the voxel of a foreground
        # endpoint from another pixel. Positive depth in this same observation
        # wins, including one voxel of quantization uncertainty at boundaries.
        endpoints = {self.xyz_to_voxel(*map(float, p)) for p in pts_int}
        protected = set()
        for key in endpoints - {None}:
            protected.update((key[0]+dx,key[1]+dy,key[2]+dz)
                for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1))
        candidate_keys.difference_update(protected)
        for key in protected:
            self.segment_voxel_miss_hits.pop(key, None)
            self.vox_clear_hits.pop(key, None)
        segment_free_hits, segment_pruned = self.prune_segment_support_from_free_space(candidate_keys)
        stm_decay = float(np.clip(float(self.get_parameter("voxel_clear_stm_decay").value), 0.0, 1.0))
        ltm_decay = float(np.clip(float(self.get_parameter("voxel_clear_ltm_decay").value), 0.0, 1.0))
        min_obs = max(1, int(self.get_parameter("voxel_clear_min_observations").value))
        hard_prune = bool(self.get_parameter("voxel_clear_hard_prune").value)
        hard_min_obs = max(min_obs, int(self.get_parameter("voxel_clear_hard_prune_min_observations").value))
        hard_max = int(self.get_parameter("voxel_clear_hard_prune_max_per_frame").value)
        cleared = 0
        confirmed = 0
        hard_pruned = 0
        for key in candidate_keys:
            if key not in self.vox_stm_w and key not in self.vox_ltm_w:
                self.vox_clear_hits.pop(key, None)
                continue
            hits = int(self.vox_clear_hits.get(key, 0)) + 1
            self.vox_clear_hits[key] = hits
            if hits < min_obs:
                continue
            confirmed += 1
            if (
                hard_prune
                and hits >= hard_min_obs
                and (hard_max <= 0 or hard_pruned < hard_max)
            ):
                if self._drop_voxel_semantic_evidence(key):
                    hard_pruned += 1
                    cleared += 1
                continue
            if self._decay_voxel_semantic_evidence(key, stm_decay, ltm_decay):
                cleared += 1
            if key not in self.vox_stm_w and key not in self.vox_ltm_w:
                self.vox_clear_hits.pop(key, None)
        self._profile_add_value("cb.clear_rays", int(idxs.size))
        self._profile_add_value("cb.clear_candidates", int(len(candidate_keys)))
        self._profile_add_value("cb.clear_confirmed", int(confirmed))
        self._profile_add_value("cb.cleared_voxels", int(cleared))
        self._profile_add_value("cb.hard_pruned_voxels", int(hard_pruned))
        self._profile_add_value("cb.segment_free_hits", int(segment_free_hits))
        self._profile_add_value("cb.segment_pruned_voxels", int(segment_pruned))

    # -------------------------
    # visibility bins
    # -------------------------
    def update_viewmask(self, viewmask_dict: Dict[Tuple[int, int], int], key, cam_xy, cell_xy):
        if not bool(self.get_parameter("use_visibility_boost").value):
            return
        bins = int(self.get_parameter("view_bins").value)
        if bins <= 1:
            return
        dx = cam_xy[0] - cell_xy[0]
        dy = cam_xy[1] - cell_xy[1]
        ang = math.atan2(dy, dx)
        b = int(math.floor((ang + math.pi) / (2.0 * math.pi) * bins))
        b = max(0, min(bins - 1, b))
        mask = viewmask_dict.get(key, 0)
        new_mask = mask | (1 << b)
        viewmask_dict[key] = new_mask
        if new_mask != mask:
            self._mark_state_dirty()

    # -------------------------
    # STM fusion: soft gating
    # -------------------------
    def _soft_gating_multiplier(self, cos: float, cos_min: float) -> float:
        soft_pow = float(self.get_parameter("gating_soft_power").value)
        soft_floor = float(self.get_parameter("gating_soft_floor").value)
        neg_pen = float(self.get_parameter("gating_soft_neg_penalty").value)

        t = (cos + 1.0) / max((cos_min + 1.0), 1e-6)
        t = float(np.clip(t, 0.0, 1.0))
        w = float(np.power(t, soft_pow))
        w = float(soft_floor + (1.0 - soft_floor) * w)
        if cos < 0.0:
            w *= float(np.clip(neg_pen, 0.0, 1.0))
        return float(np.clip(w, 0.0, 1.0))

    def fuse_stm_cell(self, key, new_emb_unit: np.ndarray, conf: float, now_s: float, z: Optional[float] = None):
        if conf <= 0.0:
            return

        if key in self.stm_w:
            self.decay_stm_cell(key, now_s)

        use_gating = bool(self.get_parameter("use_consistency_gating").value)
        gating_cos_min = float(self.get_parameter("gating_cos_min").value)
        gating_reject = bool(self.get_parameter("gating_reject").value)

        if key not in self.stm_sum:
            self.stm_sum[key] = (conf * new_emb_unit).astype(np.float32)
            self.stm_w[key] = float(conf)
            self.stm_last[key] = now_s
            self.stm_viewmask[key] = self.stm_viewmask.get(key, 0)
            self._update_z_stats(self.stm_z_min, self.stm_z_max, self.stm_z_sum, self.stm_z_count, key, z)
            self._mark_state_dirty()
            return

        emb_old, _, _ = self.stm_embedding_and_coherence(key)
        if emb_old is None:
            self.stm_sum[key] = (conf * new_emb_unit).astype(np.float32)
            self.stm_w[key] = float(conf)
            self.stm_last[key] = now_s
            self._update_z_stats(self.stm_z_min, self.stm_z_max, self.stm_z_sum, self.stm_z_count, key, z)
            self._mark_state_dirty()
            return

        if use_gating:
            cos = float(np.dot(emb_old, new_emb_unit))
            if cos < gating_cos_min:
                if gating_reject:
                    return
                conf *= self._soft_gating_multiplier(cos, gating_cos_min)

        w_max = float(self.get_parameter("cell_weight_max").value)
        self.stm_sum[key] = (self.stm_sum[key] + conf * new_emb_unit).astype(np.float32)
        self.stm_w[key] = min(float(self.stm_w[key] + conf), w_max)
        self.stm_last[key] = now_s
        self._update_z_stats(self.stm_z_min, self.stm_z_max, self.stm_z_sum, self.stm_z_count, key, z)
        self._mark_state_dirty()

    # -------------------------
    # LTM update: promote & contradiction replace
    # -------------------------
    def promote_stm_to_ltm(self, stm_key: Tuple[int, int], emb_unit: np.ndarray, coh01: float, conf_cell01: float, now_s: float):
        """
        Add/replace LTM for this cell.
        - If contradicts (cos < ltm_replace_cos_thresh): replace quickly.
        - Else: aggregate (average-like) with cap.
        """
        ltm_w_max = float(self.get_parameter("ltm_weight_max").value)
        upd_gain = float(self.get_parameter("ltm_update_gain").value)
        rep_cos = float(self.get_parameter("ltm_replace_cos_thresh").value)
        rep_gain = float(self.get_parameter("ltm_replace_gain").value)

        # Evidence weight for LTM:
        # Use stronger evidence when coherence+confidence are high.
        # (Keeps scale similar to STM but with separate caps.)
        ev = float(np.clip(conf_cell01 * coh01, 0.0, 1.0))
        if ev <= 1e-6:
            return
        ev_w = float(upd_gain * ev)

        if stm_key not in self.ltm_sum:
            self.ltm_sum[stm_key] = (ev_w * emb_unit).astype(np.float32)
            self.ltm_w[stm_key] = float(np.clip(ev_w, 0.0, ltm_w_max))
            self.ltm_last_seen[stm_key] = now_s
            # also bring over view diversity as a hint for future merging
            self.ltm_viewmask[stm_key] = int(self.stm_viewmask.get(stm_key, 0))
            self._copy_stm_z_to_ltm(stm_key, replace=True)
            self._mark_state_dirty()
            return

        old_u, _, old_w = self.ltm_embedding_and_coherence(stm_key)
        if old_u is None or old_w is None:
            self.ltm_sum[stm_key] = (ev_w * emb_unit).astype(np.float32)
            self.ltm_w[stm_key] = float(np.clip(ev_w, 0.0, ltm_w_max))
            self.ltm_last_seen[stm_key] = now_s
            self.ltm_viewmask[stm_key] = int(self.stm_viewmask.get(stm_key, 0))
            self._copy_stm_z_to_ltm(stm_key, replace=True)
            self._mark_state_dirty()
            return

        cos = float(np.dot(old_u, emb_unit))

        if cos < rep_cos:
            # Contradiction: replace fast (acts like “object moved/removed/changed”)
            ev_rep = float(np.clip(rep_gain * ev_w, 0.0, ltm_w_max))
            self.ltm_sum[stm_key] = (ev_rep * emb_unit).astype(np.float32)
            self.ltm_w[stm_key] = float(ev_rep)
            self.ltm_last_seen[stm_key] = now_s
            # update viewmask union
            self.ltm_viewmask[stm_key] = int(self.ltm_viewmask.get(stm_key, 0) | int(self.stm_viewmask.get(stm_key, 0)))
            self._copy_stm_z_to_ltm(stm_key, replace=True)
            self._mark_state_dirty()
            return

        # Consistent: aggregate
        new_sum = (self.ltm_sum[stm_key] + ev_w * emb_unit).astype(np.float32)
        new_w = float(np.clip(self.ltm_w[stm_key] + ev_w, 0.0, ltm_w_max))
        self.ltm_sum[stm_key] = new_sum
        self.ltm_w[stm_key] = new_w
        self.ltm_last_seen[stm_key] = now_s
        self.ltm_viewmask[stm_key] = int(self.ltm_viewmask.get(stm_key, 0) | int(self.stm_viewmask.get(stm_key, 0)))
        self._copy_stm_z_to_ltm(stm_key, replace=False)
        self._mark_state_dirty()

    # -------------------------
    # Object abstraction layer
    # -------------------------
    def _bbox_iou_3d(self, a_min, a_max, b_min, b_max) -> float:
        inter_min = np.maximum(a_min, b_min)
        inter_max = np.minimum(a_max, b_max)
        inter_dims = np.maximum(inter_max - inter_min, 0.0)
        inter = float(np.prod(inter_dims))
        if inter <= 0.0:
            return 0.0
        vol_a = float(np.prod(np.maximum(a_max - a_min, 0.0)))
        vol_b = float(np.prod(np.maximum(b_max - b_min, 0.0)))
        return float(inter / max(vol_a + vol_b - inter, 1e-6))

    def _bbox_gap_3d(self, a_min, a_max, b_min, b_max) -> float:
        a_min = np.asarray(a_min, dtype=np.float32)
        a_max = np.asarray(a_max, dtype=np.float32)
        b_min = np.asarray(b_min, dtype=np.float32)
        b_max = np.asarray(b_max, dtype=np.float32)
        gap = np.maximum(np.maximum(a_min - b_max, b_min - a_max), 0.0)
        return float(np.linalg.norm(gap))

    # -------------------------
    # Segment-centric semantic memory
    # -------------------------
    def segment_embedding_and_coherence(self, sid: int):
        if sid not in self.seg_sum or sid not in self.seg_w:
            return None, None, None
        return self._embedding_and_coherence(self.seg_sum[sid], self.seg_w[sid])

    def segment_dino_embedding(self, sid: int) -> Optional[np.ndarray]:
        sid = int(sid)
        emb_sum = self.seg_dino_sum.get(sid)
        if emb_sum is None or float(self.seg_dino_w.get(sid, 0.0)) <= 1e-6:
            return None
        norm = float(np.linalg.norm(emb_sum))
        if not math.isfinite(norm) or norm <= 1e-6:
            return None
        return (emb_sum / norm).astype(np.float32)

    def _update_segment_dino_memory(
        self, sid: int, embedding: Optional[np.ndarray], observation_weight: float, obs_seq=None
    ) -> None:
        """Confidence-weighted EMA for a segment's multi-view appearance.

        The descriptor remains attached to the 3D segment/voxel support while
        observations arrive from changing viewpoints. Bounded evidence avoids
        an early view becoming effectively immutable after a long run.
        """
        if embedding is None or not math.isfinite(float(observation_weight)) or float(observation_weight) <= 1e-6:
            return
        sid = int(sid)
        emb = np.asarray(embedding, dtype=np.float32).reshape((-1,))
        emb_norm = float(np.linalg.norm(emb))
        if not math.isfinite(emb_norm) or emb_norm <= 1e-6:
            return
        emb = (emb / emb_norm).astype(np.float32)
        if not hasattr(self, "seg_last_dino_seq"):
            self.seg_last_dino_seq = {}
        old = self.segment_dino_embedding(sid)
        max_weight = max(
            1e-3, float(self.get_parameter("dinov2_track_memory_weight").value)
        )
        obs_weight = float(np.clip(observation_weight, 0.0, max_weight))
        if old is None or old.shape != emb.shape:
            stored_weight = min(obs_weight, max_weight)
            self.seg_dino_sum[sid] = (stored_weight * emb).astype(np.float32)
            self.seg_dino_w[sid] = stored_weight
            if obs_seq is not None:
                self.seg_last_dino_seq[sid] = int(obs_seq)
            return

        base_alpha = float(
            np.clip(
                float(self.get_parameter("dinov2_track_ema_alpha").value),
                1e-4,
                1.0,
            )
        )
        alpha = float(1.0 - math.pow(1.0 - base_alpha, obs_weight))
        blended = ((1.0 - alpha) * old + alpha * emb).astype(np.float32)
        blended_norm = float(np.linalg.norm(blended))
        if not math.isfinite(blended_norm) or blended_norm <= 1e-6:
            return
        blended /= blended_norm
        stored_weight = min(
            float(self.seg_dino_w.get(sid, 0.0)) + obs_weight, max_weight
        )
        self.seg_dino_sum[sid] = (stored_weight * blended).astype(np.float32)
        self.seg_dino_w[sid] = stored_weight
        if obs_seq is not None:
            self.seg_last_dino_seq[sid] = int(obs_seq)

    def _segment_semantic_cos(self, sid: int, emb: np.ndarray) -> Optional[float]:
        old_emb, _, _ = self.segment_embedding_and_coherence(int(sid))
        if old_emb is None:
            return None
        emb = np.asarray(emb, dtype=np.float32)
        n = float(np.linalg.norm(emb))
        if n <= 1e-6:
            return None
        return float(np.dot(old_emb, emb / n))

    def _segment_dino_cos(self, sid: int, emb: Optional[np.ndarray]) -> Optional[float]:
        if emb is None:
            return None
        old_emb = self.segment_dino_embedding(int(sid))
        if old_emb is None:
            return None
        emb = np.asarray(emb, dtype=np.float32).reshape((-1,))
        norm = float(np.linalg.norm(emb))
        if not math.isfinite(norm) or norm <= 1e-6 or emb.shape != old_emb.shape:
            return None
        return float(np.dot(old_emb, emb / norm))

    def update_segment_viewmask(self, sid: int, cam_xy, centroid_xy):
        if not bool(self.get_parameter("use_visibility_boost").value):
            return
        bins = int(self.get_parameter("view_bins").value)
        if bins <= 1:
            return
        dx = cam_xy[0] - centroid_xy[0]
        dy = cam_xy[1] - centroid_xy[1]
        ang = math.atan2(dy, dx)
        b = int(math.floor((ang + math.pi) / (2.0 * math.pi) * bins))
        b = max(0, min(bins - 1, b))
        old_mask = int(self.seg_viewmask.get(sid, 0))
        new_mask = int(old_mask | (1 << b))
        self.seg_viewmask[sid] = new_mask
        if new_mask != old_mask:
            self._mark_state_dirty()

    def _drop_segment_record(self, sid: int):
        sid = int(sid)
        for key in list(self.seg_support_voxels.get(sid, set())):
            if int(self.voxel_segment_id.get(key, -1)) == sid:
                self.voxel_segment_id.pop(key, None)
            self.segment_voxel_miss_hits.pop(key, None)
            self.segment_semantic_conflict_hits.pop(key, None)
            self.segment_semantic_conflict_last_seq.pop(key, None)
        self.seg_sum.pop(sid, None)
        self.seg_w.pop(sid, None)
        self.seg_first_seen.pop(sid, None)
        self.seg_last_seen.pop(sid, None)
        self.seg_obs_count.pop(sid, None)
        self.seg_last_obs_seq.pop(sid, None)
        self.seg_viewmask.pop(sid, None)
        self.seg_centroid.pop(sid, None)
        self.seg_bbox_min.pop(sid, None)
        self.seg_bbox_max.pop(sid, None)
        self.seg_dino_sum.pop(sid, None)
        self.seg_dino_w.pop(sid, None)
        getattr(self, "seg_last_dino_seq", {}).pop(sid, None)
        self.seg_merge_block_until.pop(sid, None)
        self.seg_support_cells.pop(sid, None)
        self.seg_support_voxels.pop(sid, None)
        self.segment_query_support_cache.pop(sid, None)

    def _drop_segment_voxel_owner_if_empty(self, key: Tuple[int, int, int]):
        # Semantic STM/LTM cache eviction is not evidence that a surface has
        # disappeared. Geometry has its own bounded support and negative votes.
        owner = self.voxel_segment_id.get(key)
        if owner in self.seg_w and key in self.seg_support_voxels.get(owner, ()):
            return
        if key in self.vox_stm_w or key in self.vox_ltm_w:
            return
        sid = self.voxel_segment_id.pop(key, None)
        self.segment_voxel_miss_hits.pop(key, None)
        self.segment_semantic_conflict_hits.pop(key, None)
        self.segment_semantic_conflict_last_seq.pop(key, None)
        if sid is None:
            return
        support = self.seg_support_voxels.get(int(sid))
        if support is not None:
            support.discard(key)
            self.seg_support_cells[int(sid)] = self._support_cells_from_voxels(support)
            if not support and float(self.seg_w.get(int(sid), 0.0)) < 1e-3:
                self._drop_segment_record(int(sid))

    def _refresh_segment_geometry_from_support(self, sid: int) -> bool:
        sid = int(sid)
        support = {
            key for key in self.seg_support_voxels.get(sid, set())
            if int(self.voxel_segment_id.get(key, -1)) == sid
        }
        self.seg_support_voxels[sid] = support
        self.seg_support_cells[sid] = self._support_cells_from_voxels(support) if support else set()
        if not support:
            return False

        pts = np.array([self.voxel_center_xyz(key) for key in support], dtype=np.float32)
        if pts.size == 0:
            return False
        pad = max(0.5 * float(self.voxel_size), 0.01)
        self.seg_centroid[sid] = np.mean(pts, axis=0).astype(np.float32)
        self.seg_bbox_min[sid] = (np.min(pts, axis=0) - pad).astype(np.float32)
        self.seg_bbox_max[sid] = (np.max(pts, axis=0) + pad).astype(np.float32)
        return True

    def prune_segment_support_from_free_space(self, candidate_keys: set):
        if not bool(self.get_parameter("use_segment_memory").value):
            return 0, 0
        if not bool(self.get_parameter("segment_use_negative_evidence").value):
            return 0, 0
        # Votes must be consecutive observations, not sporadic misses summed
        # over minutes. A missing/occluded ray is never negative evidence.
        for key in list(self.segment_voxel_miss_hits):
            if key not in candidate_keys:
                self.segment_voxel_miss_hits.pop(key, None)
        if not candidate_keys or not self.voxel_segment_id:
            return 0, 0

        min_obs = max(1, int(self.get_parameter("segment_negative_min_observations").value))
        max_prune = int(self.get_parameter("segment_negative_max_prune_per_frame").value)
        decay = float(np.clip(float(self.get_parameter("segment_negative_decay").value), 0.0, 1.0))
        drop_empty = bool(self.get_parameter("segment_negative_drop_empty").value)

        owned_hits = 0
        ready = []
        for key in candidate_keys:
            sid = self.voxel_segment_id.get(key)
            if sid is None:
                self.segment_voxel_miss_hits.pop(key, None)
                self.segment_semantic_conflict_hits.pop(key, None)
                self.segment_semantic_conflict_last_seq.pop(key, None)
                continue
            sid = int(sid)
            support = self.seg_support_voxels.get(sid)
            if sid not in self.seg_w or support is None or key not in support:
                self.voxel_segment_id.pop(key, None)
                self.segment_voxel_miss_hits.pop(key, None)
                self.segment_semantic_conflict_hits.pop(key, None)
                self.segment_semantic_conflict_last_seq.pop(key, None)
                continue
            owned_hits += 1
            hits = int(self.segment_voxel_miss_hits.get(key, 0)) + 1
            self.segment_voxel_miss_hits[key] = hits
            mature = int(self.seg_obs_count.get(sid, 0)) >= 5
            required = max(min_obs, 8) if mature else min_obs
            if hits >= required:
                ready.append((key, sid))

        if max_prune > 0 and len(ready) > max_prune:
            ready = sorted(ready, key=lambda item: item[0])[:max_prune]

        removed_by_sid = {}
        for key, sid in ready:
            if int(self.voxel_segment_id.get(key, -1)) != int(sid):
                continue
            support = self.seg_support_voxels.get(int(sid))
            if support is None or key not in support:
                self.segment_voxel_miss_hits.pop(key, None)
                continue
            support.discard(key)
            self.voxel_segment_id.pop(key, None)
            self.segment_voxel_miss_hits.pop(key, None)
            self.segment_semantic_conflict_hits.pop(key, None)
            self.segment_semantic_conflict_last_seq.pop(key, None)
            removed_by_sid[int(sid)] = int(removed_by_sid.get(int(sid), 0)) + 1

        if not removed_by_sid:
            return int(owned_hits), 0

        min_support = max(1, int(self.get_parameter("segment_min_support_voxels").value))
        removed_total = int(sum(removed_by_sid.values()))
        for sid, removed_n in removed_by_sid.items():
            support = {
                key for key in self.seg_support_voxels.get(sid, set())
                if int(self.voxel_segment_id.get(key, -1)) == sid
            }
            old_n = int(len(support) + removed_n)
            removed_frac = float(removed_n) / float(max(old_n, 1))
            if not support and drop_empty:
                self._drop_segment_record(sid)
                continue

            self.seg_support_voxels[sid] = support
            self.seg_support_cells[sid] = self._support_cells_from_voxels(support) if support else set()
            if support:
                self._refresh_segment_geometry_from_support(sid)
            if decay < 1.0 and sid in self.seg_w:
                self.seg_w[sid] = float(self.seg_w[sid] * (decay ** max(1.0, removed_frac * 4.0)))
            if drop_empty and len(support) < min_support and float(self.seg_w.get(sid, 0.0)) < 1e-3:
                self._drop_segment_record(sid)
            else:
                self.segment_query_support_cache.pop(sid, None)

        self._mark_state_dirty()
        return int(owned_hits), int(removed_total)

    def _valid_projected_segment_owner(self, sid: Optional[int]) -> bool:
        if sid is None:
            return False
        sid = int(sid)
        if sid not in self.seg_w:
            return False
        min_obs = max(1, int(self.get_parameter("segment_projected_id_min_segment_observations").value))
        return int(self.seg_obs_count.get(sid, 0)) >= min_obs

    def _lookup_projected_segment_owner(self, voxel_key: Optional[Tuple[int, int, int]]) -> int:
        if voxel_key is None:
            return 0
        sid = self.voxel_segment_id.get(voxel_key)
        if self._valid_projected_segment_owner(sid):
            return int(sid)

        radius = max(0, int(self.get_parameter("segment_projected_id_neighbor_radius_vox").value))
        if radius <= 0:
            return 0

        candidates = {}
        vx, vy, vz = voxel_key
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    dist = abs(dx) + abs(dy) + abs(dz)
                    if max(abs(dx), abs(dy), abs(dz)) > radius:
                        continue
                    sid_nb = self.voxel_segment_id.get((vx + dx, vy + dy, vz + dz))
                    if not self._valid_projected_segment_owner(sid_nb):
                        continue
                    sid_nb = int(sid_nb)
                    count, best_dist, weight = candidates.get(sid_nb, (0, 999999, 0.0))
                    candidates[sid_nb] = (
                        count + 1,
                        min(best_dist, dist),
                        weight + float(self.seg_w.get(sid_nb, 0.0)),
                    )

        if not candidates:
            return 0
        return int(
            max(
                candidates.items(),
                key=lambda item: (item[1][0], -item[1][1], item[1][2]),
            )[0]
        )

    def build_projected_segment_id_context(
        self,
        pts_int: np.ndarray,
        uu_i: np.ndarray,
        vv_i: np.ndarray,
        image_shape: Tuple[int, int],
    ):
        if not bool(self.get_parameter("segment_assoc_use_projected_ids").value):
            return None
        if pts_int is None or pts_int.shape[0] == 0 or not self.voxel_segment_id:
            return None

        n = int(pts_int.shape[0])
        point_sids = np.zeros((n,), dtype=np.int32)
        visible_counts = {}
        h, w = image_shape
        # Entity rendering also needs seeds when the Segments publisher is off.
        id_img = np.zeros((int(h), int(w)), dtype=np.int32)

        for idx, p in enumerate(pts_int):
            voxel = self.xyz_to_voxel(float(p[0]), float(p[1]), float(p[2]))
            sid = self._lookup_projected_segment_owner(voxel)
            if sid <= 0:
                continue
            point_sids[idx] = int(sid)
            visible_counts[int(sid)] = int(visible_counts.get(int(sid), 0)) + 1
            if id_img is not None:
                id_img[int(vv_i[idx]), int(uu_i[idx])] = int(sid)

        owned_points = int(np.count_nonzero(point_sids))
        if owned_points <= 0:
            return None
        return {
            "point_sids": point_sids,
            "visible_counts": visible_counts,
            "id_image": id_img,
            "owned_points": owned_points,
        }

    def find_projected_segment_match(self, obs, projected_ctx, point_idxs, blocked_sids: Optional[set] = None):
        if projected_ctx is None:
            return None, None
        point_sids = projected_ctx.get("point_sids")
        if point_sids is None or point_idxs is None or int(point_idxs.size) <= 0:
            return None, None

        blocked_sids = blocked_sids or set()
        mask_sids = np.asarray(point_sids[point_idxs], dtype=np.int32)
        owned = mask_sids[mask_sids > 0]
        owned_n = int(owned.size)
        min_pixels = max(1, int(self.get_parameter("segment_projected_id_min_pixels").value))
        if owned_n < min_pixels:
            return None, None

        unique, counts = np.unique(owned, return_counts=True)
        visible_counts = projected_ctx.get("visible_counts", {})
        min_mask_frac = float(self.get_parameter("segment_projected_id_min_mask_frac").value)
        min_visible_frac = float(self.get_parameter("segment_projected_id_min_visible_frac").value)
        min_iou = float(self.get_parameter("segment_projected_id_min_iou").value)
        min_dominance = float(self.get_parameter("segment_projected_id_min_dominance").value)
        min_score = float(self.get_parameter("segment_projected_id_min_score").value)
        min_sem = float(self.get_parameter("segment_projected_id_min_sem_cos").value)

        mask_points = max(1, int(point_idxs.size))
        best_sid = None
        best_score = -1.0
        best_info = None
        for sid_raw, count_raw in zip(unique, counts):
            sid = int(sid_raw)
            if sid in blocked_sids or sid not in self.seg_w:
                continue
            count = int(count_raw)
            if count < min_pixels:
                continue

            visible_n = max(1, int(visible_counts.get(sid, count)))
            obs_frac = float(count / mask_points)
            dominance = float(count / max(owned_n, 1))
            visible_frac = float(count / visible_n)
            iou = float(count / max(mask_points + visible_n - count, 1))
            if dominance < min_dominance:
                continue
            if not (obs_frac >= min_mask_frac or visible_frac >= min_visible_frac or iou >= min_iou):
                continue

            old_emb, _, _ = self.segment_embedding_and_coherence(sid)
            raw_sem = float(np.dot(old_emb, obs["embedding"])) if old_emb is not None else 0.0
            support_overlap = self._segment_support_overlap_with_obs(sid, obs)
            if raw_sem < min_sem and support_overlap <= 0.0 and obs_frac < 0.50:
                continue
            sem01 = float(np.clip((raw_sem + 1.0) * 0.5, 0.0, 1.0))

            score = float(
                0.30 * dominance
                + 0.25 * visible_frac
                + 0.20 * obs_frac
                + 0.15 * iou
                + 0.07 * support_overlap
                + 0.03 * sem01
            )
            if score > best_score:
                best_score = score
                best_sid = sid
                best_info = {
                    "sid": int(sid),
                    "source": "projected_id",
                    "score": float(score),
                    "pixels": int(count),
                    "owned_pixels": int(owned_n),
                    "mask_points": int(mask_points),
                    "visible_points": int(visible_n),
                    "obs_frac": float(obs_frac),
                    "dominance": float(dominance),
                    "visible_frac": float(visible_frac),
                    "iou": float(iou),
                    "raw_sem": float(raw_sem),
                    "overlap": float(support_overlap),
                }

        if best_sid is None or best_score < min_score:
            return None, best_info or {"sid": None, "source": "projected_id", "score": float(best_score)}
        return int(best_sid), best_info

    def owner_filtered_segment_observation(
        self,
        pts_mask: np.ndarray,
        conf: np.ndarray,
        emb_unit: np.ndarray,
        forced_sid: int,
        projected_ctx,
        point_idxs: np.ndarray,
        dino_embedding: Optional[np.ndarray] = None,
        dino_quality: float = 1.0,
    ):
        if projected_ctx is None or point_idxs is None or int(point_idxs.size) <= 0:
            return None
        point_sids = projected_ctx.get("point_sids")
        if point_sids is None:
            return None

        forced_sid = int(forced_sid)
        owners = np.asarray(point_sids[point_idxs], dtype=np.int32)
        same_local = np.flatnonzero(owners == forced_sid)
        unowned_local = np.flatnonzero(owners == 0)
        if same_local.size <= 0:
            return None

        max_ratio = float(self.get_parameter("segment_projected_id_max_unowned_claim_ratio").value)
        if max_ratio >= 0.0 and unowned_local.size > 0:
            max_unowned = int(math.ceil(max_ratio * float(same_local.size)))
            if max_unowned < unowned_local.size:
                if max_unowned > 0:
                    sample = np.linspace(0, unowned_local.size - 1, num=max_unowned, dtype=np.int32)
                    unowned_local = unowned_local[sample]
                else:
                    unowned_local = np.zeros((0,), dtype=np.int64)

        keep_local = np.concatenate([same_local, unowned_local]).astype(np.int64)
        min_points = max(1, int(self.get_parameter("segment_projected_id_min_filtered_points").value))
        if keep_local.size < min_points:
            return None
        keep_local.sort()
        filtered = self.build_segment_observation(
            pts_mask[keep_local],
            emb_unit,
            conf[keep_local],
            dino_embedding=dino_embedding,
            dino_quality=dino_quality,
        )
        if filtered is not None:
            filtered["owner_filtered"] = True
            filtered["filtered_points"] = int(keep_local.size)
            filtered["filtered_same_owner_points"] = int(same_local.size)
            filtered["filtered_unowned_points"] = int(unowned_local.size)
        return filtered

    def publish_segment_id_projection(self, projected_ctx, header_msg: Header):
        # Retain this observation's actual assignment, including when the raw
        # segment image has no subscribers. Entity rendering happens after fusion.
        self.last_segment_projection_context = (projected_ctx, header_msg)
        if not bool(self.get_parameter("publish_segment_id_projection").value):
            return
        if not self.debug_rgb_image_has_subscribers(
            self.pub_segment_id_projection,
            self.pub_segment_id_projection_compressed,
        ):
            return
        if projected_ctx is None:
            return
        id_img = projected_ctx.get("id_image")
        if id_img is None or id_img.size == 0:
            return
        rgb = self._render_id_projection(projected_ctx, self._segment_instance_rgb)
        self.publish_debug_rgb_image(
            self.pub_segment_id_projection,
            self.pub_segment_id_projection_compressed,
            rgb,
            header_msg,
            compressed_format="png",
        )

    def publish_entity_id_projection(self):
        if not bool(self.get_parameter("publish_entity_id_projection").value):
            return
        if not self.debug_rgb_image_has_subscribers(self.pub_entity_id_projection, self.pub_entity_id_projection_compressed):
            return
        if self.last_segment_projection_context is None:
            return
        context, header = self.last_segment_projection_context
        ids = context.get("id_image") if context is not None else None
        if ids is None:
            return
        rgb = self._render_id_projection(context, self._segment_entity_rgb)
        self.publish_debug_rgb_image(self.pub_entity_id_projection, self.pub_entity_id_projection_compressed,
                                     rgb, header, compressed_format="png")

    def _render_id_projection(self, context, color):
        if bool(self.get_parameter("segment_overlay_raw_support").value):
            return render_overlay(context["id_image"], [], context.get("rgb"), color,
                                  float(self.get_parameter("segment_overlay_alpha").value))
        if ("decisions" in context and bool(self.get_parameter("segment_overlay_fill_masks").value)
                and not bool(self.get_parameter("segment_overlay_raw_support").value)):
            entity_view = getattr(color, "__name__", "") == "_segment_entity_rgb"
            def mapper(sid):
                if sid not in self.seg_w:
                    return None
                if entity_view:
                    eid = self.entity_memory.segment_entity.get(sid)
                    return self._effective_entity_id(eid)
                return sid
            dominant = bool(self.get_parameter("segment_overlay_dominant_owner").value)
            decisions = context.get("entity_display_decisions" if entity_view else "segment_display_decisions", []) if dominant else context["decisions"]
            return render_mask_consensus(context["id_image"].shape, context.get("masks", []),
                                         decisions, context.get("rgb"), mapper,
                                         self._segment_instance_rgb,
                                         float(self.get_parameter("segment_overlay_alpha").value),
                                         "E" if entity_view else "S", seeds=context["id_image"],
                                         radius=max(1, int(self.stride)//2), dominant=dominant)
        if "masks" not in context:
            ids = context["id_image"]
            rgb = np.zeros((*ids.shape, 3), dtype=np.uint8)
            for sid in np.unique(ids):
                if sid > 0:
                    rgb[ids == sid] = color(int(sid))
            return rgb
        if "display_ids" not in context:
            seeds = context["id_image"]
            context["display_ids"] = dense_display_ids(
                seeds, context["masks"], max(1, int(self.get_parameter("stride").value) // 2)
            ) if bool(self.get_parameter("segment_overlay_fill_masks").value) else seeds
        return render_overlay(context["display_ids"], context["masks"], context.get("rgb"), color,
                              float(self.get_parameter("segment_overlay_alpha").value))

    def _use_frame_segment_assignment(self) -> bool:
        mode = str(self.get_parameter("segment_frame_assignment_mode").value).strip().lower()
        return mode not in ("", "off", "false", "legacy", "per_mask")

    def _reserve_segment_id(self) -> int:
        sid = int(self.seg_next_id)
        self.seg_next_id += 1
        return sid

    def _add_frame_segment_points(
        self,
        groups: Dict[int, dict],
        assignable: np.ndarray,
        sid: int,
        point_idxs: np.ndarray,
        mask_idx: int,
        mask_emb: np.ndarray,
        mask_score: float,
        dino_emb: Optional[np.ndarray] = None,
        dino_quality: float = 1.0,
    ) -> int:
        if point_idxs is None or int(point_idxs.size) <= 0:
            return 0
        point_idxs = np.asarray(point_idxs, dtype=np.int64)
        point_idxs = point_idxs[assignable[point_idxs]]
        if int(point_idxs.size) <= 0:
            return 0

        sid = int(sid)
        assignable[point_idxs] = False
        group = groups.get(sid)
        if group is None:
            group = {
                "point_chunks": [],
                "point_mask_chunks": [],
                "emb_sum": np.zeros_like(mask_emb, dtype=np.float32),
                "dino_sum": (
                    np.zeros_like(dino_emb, dtype=np.float32)
                    if dino_emb is not None and dino_emb.size > 0
                    else None
                ),
                "weight": 0.0,
                "dino_quality_mass": 0.0,
                "score_sum": 0.0,
                "mask_indices": set(),
            }
            groups[sid] = group
        n = int(point_idxs.size)
        group["point_chunks"].append(point_idxs)
        group["point_mask_chunks"].append(np.full((n,), int(mask_idx), dtype=np.int32))
        group["emb_sum"] = (group["emb_sum"] + float(n) * mask_emb.astype(np.float32)).astype(np.float32)
        if (
            group.get("dino_sum") is not None
            and dino_emb is not None
            and dino_emb.shape == group["dino_sum"].shape
        ):
            dino_quality = float(np.clip(dino_quality, 0.0, 1.0))
            group["dino_sum"] = (
                group["dino_sum"]
                + float(n) * dino_quality * dino_emb.astype(np.float32)
            ).astype(np.float32)
            group["dino_quality_mass"] = float(group["dino_quality_mass"]) + float(
                n
            ) * dino_quality
        group["weight"] = float(group["weight"]) + float(n)
        group["score_sum"] = float(group["score_sum"]) + float(n) * float(mask_score)
        group["mask_indices"].add(int(mask_idx))
        return n

    def _prepare_surface_consensus(self, masks, owners, pts, uu, vv, shape, mask_emb, dino_emb, dino_quality, seq, now):
        if not hasattr(self, "surface_consensus"):
            self.surface_consensus = SurfaceConsensus()
            self.boundary_votes = BoundaryOwnershipVotes()
        self.surface_consensus.config = SurfaceConfig(
            min_samples=max(3, int(self.get_parameter("segment_mask_min_depth_points").value)),
            interior_px=max(0, int(self.get_parameter("segment_mask_interior_px").value)),
            dominance=float(np.clip(self.get_parameter("segment_mask_vote_dominance").value, .6, .98)),
            switch_observations=max(2, int(self.get_parameter("segment_owner_switch_observations").value)),
        )
        clean_masks = [m if m is not None and m.shape == shape else None for m in masks]
        compatibility = {}
        def compatible(sid, index):
            key = (sid, index)
            if key not in compatibility:
                sem = self._segment_semantic_cos(sid, mask_emb[index])
                dino = self._segment_dino_cos(sid, dino_emb[index]) if dino_emb.shape[1] else None
                compatibility[key] = (sem is None or sem >= float(self.get_parameter("segment_projected_fusion_min_sem_cos").value)) and (
                    dino is None or dino_quality[index] < float(self.get_parameter("dinov2_assoc_min_quality").value)
                    or dino >= float(self.get_parameter("segment_assoc_min_dino_cos").value))
            return compatibility[key]
        decisions = self.surface_consensus.propose(clean_masks, owners, pts, uu, vv, now,
                                                    self.voxel_size, set(self.seg_w), compatible)
        measured = owners.copy()
        proposals = []
        if bool(self.get_parameter("segment_boundary_repair_enabled").value):
            protected = {d['winner'] for d in decisions if d['winner'] is not None and d['confidence'] >= .5}
            for item in decisions:
                target = item['winner']
                if (target is None or item['reason'] != 'depth_consensus' or not item['interior']
                        or item['confidence'] < .90):
                    continue
                foreign = {}
                for i in item['core_indices']:
                    old = int(owners[i])
                    if old <= 0 or old == target or old in protected:
                        continue
                    key = self.xyz_to_voxel(*map(float, pts[i]))
                    if key is not None and self.voxel_segment_id.get(key) == old:
                        foreign.setdefault(old, set()).add(key)
                for old, keys in foreign.items():
                    # Do not consume a whole small object or more than 10% of
                    # an existing track. Explicit identity corrections win.
                    if len(keys) > .10*len(self.seg_support_voxels.get(old, ())):
                        continue
                    ea, eb = self.entity_memory.segment_entity.get(old), self.entity_memory.segment_entity.get(target)
                    if (ea is not None and eb is not None and tuple(sorted((ea, eb))) in self.entity_memory.assemblies.blocked):
                        continue
                    if old in self.entity_memory.assembly_exclusions or target in self.entity_memory.assembly_exclusions:
                        continue
                    proposals.extend((key, old, target) for key in keys)
        changes = self.boundary_votes.update(seq, proposals, {s: len(v) for s,v in self.seg_support_voxels.items()})
        changed = set()
        for key, old, target in changes:
            if self.voxel_segment_id.get(key) != old or target not in self.seg_w:
                continue
            self._claim_segment_voxels(target, {key})
            self.seg_support_voxels.setdefault(target, set()).add(key)
            self.seg_support_cells[target] = self._support_cells_from_voxels(self.seg_support_voxels[target])
            self.entity_memory.lineage.append(dict(event="boundary_owner_corrected", voxel=list(key),
                from_segment=old, to_segment=target, seq=seq, basis="repeated_interior_depth_consensus"))
            changed.update((old, target))
        if changed:
            for sid in changed:
                self._refresh_segment_geometry_from_support(sid)
                self.segment_query_support_cache.pop(sid, None)
            measured = np.asarray([int(self.voxel_segment_id.get(self.xyz_to_voxel(*map(float, p)), 0)) for p in pts], dtype=np.int32)
            self._mark_state_dirty()
        guided = measured.copy()
        for item in decisions:
            if item['winner'] is None or item['reason'] not in ('depth_consensus', 'temporal_depth_hold'):
                continue
            indices = item.get('core_indices', np.zeros((0,), dtype=np.int64))
            # Temporal evidence can claim only previously unowned depth support.
            # Foreign ownership needs the separate repeated correction above.
            indices = indices[guided[indices] == 0]
            # A partial mask overlap must not paint all unowned geometry with
            # an old ID (positive feedback that lets one segment eat a room).
            # Limit guidance to one voxel around actual stored support and
            # retain the same extension budget used by normal association.
            support = self.seg_support_voxels.get(item['winner'], set())
            local = []
            for i in indices:
                key = self.xyz_to_voxel(*map(float, pts[i]))
                if key is not None and any((key[0]+dx,key[1]+dy,key[2]+dz) in support
                    for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)):
                    local.append(i)
            anchor = int(np.count_nonzero(measured[item['core_indices']] == item['winner']))
            budget = max(3, int(max(anchor, 1)*max(0., float(self.get_parameter('segment_frame_assignment_max_existing_extension_ratio').value))))
            indices = np.asarray(local[:budget], dtype=np.int64)
            guided[indices] = item['winner']
        return guided, measured, decisions, compatible

    def assign_frame_segments_fine_image(
        self,
        masks: List[np.ndarray],
        mask_emb: np.ndarray,
        mask_scores: np.ndarray,
        dino_mask_emb: np.ndarray,
        dino_mask_quality: np.ndarray,
        pts_int: np.ndarray,
        uu_i: np.ndarray,
        vv_i: np.ndarray,
        image_shape: Tuple[int, int],
        conf_dist: np.ndarray,
        conf_scale: float,
        conf_min: float,
        now_s: float,
        cam_xy_int,
        obs_seq: int,
        projected_ctx,
        header_msg: Header,
    ) -> dict:
        h, w = image_shape
        masks = [np.asarray(m, dtype=bool) if m is not None and np.shape(m) == (h, w) else None
                 for m in masks]
        n_points = int(pts_int.shape[0]) if pts_int is not None else 0
        stats = {
            "attempted": 0,
            "built": 0,
            "new": 0,
            "matched": 0,
            "projected_matched": 0,
            "fallback_matched": 0,
            "owned_points": 0,
            "unique_frame_sids": 0,
            "rejects": {},
            "obs_voxel_counts": [],
            "semantic_conflict_carves": 0,
            "semantic_conflict_voxels": 0,
            "semantic_conflict_ready_voxels": 0,
            "point_mask_indices": np.full((n_points,), -1, dtype=np.int32),
        }
        if n_points <= 0 or not masks:
            if hasattr(self, "surface_consensus"):
                self.surface_consensus.commit(obs_seq, now_s, [], [], set(self.seg_w))
            if hasattr(self, "boundary_votes"):
                self.boundary_votes.update(obs_seq, [], {s: len(v) for s,v in self.seg_support_voxels.items()})
            self.last_mask_attribution = [{"mask": i, "segments": [], "counts": [],
                                           "depth_points": 0, "reason": "no_valid_depth"}
                                          for i in range(len(masks))]
            self.publish_segment_id_projection({"id_image": np.zeros((int(h), int(w)), dtype=np.int32),
                                               "masks": masks, "decisions": [], "rgb": getattr(self, "last_assignment_rgb", None)},
                                              header_msg)
            return stats

        point_sids = np.zeros((n_points,), dtype=np.int32)
        visible_counts = {}
        if projected_ctx is not None:
            old_point_sids = projected_ctx.get("point_sids")
            if old_point_sids is not None and int(old_point_sids.shape[0]) == n_points:
                point_sids = np.asarray(old_point_sids, dtype=np.int32)
                visible_counts = dict(projected_ctx.get("visible_counts", {}))
        stats["owned_points"] = int(np.count_nonzero(point_sids))
        measured_point_sids = point_sids.copy()
        consensus_decisions = []
        if bool(self.get_parameter("segment_temporal_consensus").value):
            point_sids, measured_point_sids, consensus_decisions, consensus_compatible = self._prepare_surface_consensus(
                masks, point_sids, pts_int, uu_i, vv_i, (h,w), mask_emb, dino_mask_emb, dino_mask_quality, obs_seq, now_s)
            ids, counts = np.unique(point_sids[point_sids > 0], return_counts=True)
            visible_counts = {int(s): int(n) for s,n in zip(ids, counts)}
        decisions_by_mask = {d['mask']: d for d in consensus_decisions}
        # Refresh mixed-mask components from their own current projected
        # footprint in the *existing* DINO grid. Never copy one broad mask's
        # descriptor to every owner, and never add neural-model forwards.
        component_dino = {}
        dense_current = getattr(self, "current_assignment_dino", None)
        mixed_sids = sorted({s for d in consensus_decisions if d['reason'] == 'multiple_supported_owners'
                             for s in d['segments']})
        budget = max(0, int(self.get_parameter("segment_mixed_dino_pool_budget").value))
        if dense_current is not None and mixed_sids and budget:
            shift = obs_seq % len(mixed_sids)
            selected = (mixed_sids[shift:] + mixed_sids[:shift])[:budget]
            footprints = []
            radius = max(1, int(self.get_parameter("stride").value)//2)
            kernel = np.ones((2*radius+1, 2*radius+1), dtype=np.uint8)
            for sid in selected:
                footprint = np.zeros((h,w), dtype=np.uint8)
                relevant = np.zeros((h,w), dtype=bool)
                for decision in consensus_decisions:
                    if decision['reason'] == 'multiple_supported_owners' and sid in decision['segments']:
                        relevant |= masks[decision['mask']]
                indices = np.flatnonzero(measured_point_sids == sid)
                footprint[vv_i[indices], uu_i[indices]] = 1
                footprint = (cv2.dilate(footprint, kernel) > 0) & relevant
                foreign = np.flatnonzero((measured_point_sids > 0) & (measured_point_sids != sid))
                footprint[vv_i[foreign], uu_i[foreign]] = False
                footprints.append(footprint)
            vectors, qualities = self.compute_dinov2_mask_embeddings(footprints, dense_current, return_quality=True)
            for idx, sid in enumerate(selected):
                if qualities[idx] >= float(self.get_parameter("dinov2_assoc_min_quality").value):
                    component_dino[sid] = (vectors[idx], float(qualities[idx]))

        min_depth_points = max(1, int(self.get_parameter("segment_mask_min_depth_points").value))
        # Segment attribution is not dense semantic fusion: a 400-pixel SAM
        # mask at stride 8 has only ~6 depth samples, not the old required 25.
        min_points = min_depth_points
        max_depth_frac = float(self.get_parameter("segment_frame_assignment_max_depth_frac").value)
        max_points = int(math.floor(max_depth_frac * float(n_points))) if 0.0 < max_depth_frac < 1.0 else 0

        mask_items = []
        rejects = stats["rejects"]
        for mask_idx, mask in enumerate(masks):
            if mask is None or mask.shape != (h, w):
                rejects["bad_shape"] = int(rejects.get("bad_shape", 0)) + 1
                continue
            in_mask = np.asarray(mask, dtype=bool)[vv_i, uu_i]
            point_idxs = np.flatnonzero(in_mask)
            if int(point_idxs.size) < min_points:
                rejects["small_depth_support"] = int(rejects.get("small_depth_support", 0)) + 1
                continue
            if max_points > 0 and int(point_idxs.size) > max_points:
                rejects["large_depth_support"] = int(rejects.get("large_depth_support", 0)) + 1
                continue
            mask_items.append((int(point_idxs.size), int(mask_idx), point_idxs))

        if bool(self.get_parameter("segment_frame_assignment_sort_small_first").value):
            mask_items.sort(key=lambda item: (item[0], item[1]))

        assignable = np.ones((n_points,), dtype=bool)
        groups: Dict[int, dict] = {}
        conflict_assignments = []
        iou_threshold = float(self.get_parameter("segment_frame_assignment_iou_threshold").value)
        overlap_threshold = float(self.get_parameter("segment_frame_assignment_existing_overlap").value)
        max_existing_extension_ratio = float(
            self.get_parameter("segment_frame_assignment_max_existing_extension_ratio").value
        )
        semantic_gate = bool(self.get_parameter("segment_projected_fusion_semantic_gate").value)
        projected_min_sem = float(self.get_parameter("segment_projected_fusion_min_sem_cos").value)

        # Reserve strong reprojection matches first. Remaining masks are
        # associated jointly so a merely adequate early mask cannot greedily
        # steal the best track from a later observation.
        projected_reserved = set()
        preliminary_obs = {}
        for mask_points, mask_idx, point_idxs in mask_items:
            e_pre = mask_emb[int(mask_idx)].astype(np.float32)
            e_norm = float(np.linalg.norm(e_pre))
            if e_norm <= 1e-6:
                continue
            e_pre = e_pre / e_norm
            dino_pre = (
                dino_mask_emb[int(mask_idx)].astype(np.float32)
                if dino_mask_emb.ndim == 2
                and int(mask_idx) < dino_mask_emb.shape[0]
                and dino_mask_emb.shape[1] > 0
                else None
            )
            dino_quality_pre = (
                float(dino_mask_quality[int(mask_idx)])
                if dino_mask_quality.ndim == 1
                and int(mask_idx) < dino_mask_quality.shape[0]
                else 0.0
            )
            owners_pre = point_sids[point_idxs]
            owned_pre = owners_pre[owners_pre > 0]
            has_projected_anchor = False
            if owned_pre.size > 0:
                unique_pre, counts_pre = np.unique(owned_pre, return_counts=True)
                for sid_raw, count_raw in zip(unique_pre, counts_pre):
                    sid_pre = int(sid_raw)
                    if not self._valid_projected_segment_owner(sid_pre):
                        continue
                    visible_n = max(1, int(visible_counts.get(sid_pre, int(count_raw))))
                    iou = float(count_raw) / float(
                        max(mask_points + visible_n - int(count_raw), 1)
                    )
                    sem = self._segment_semantic_cos(sid_pre, e_pre)
                    dino_cos = self._segment_dino_cos(sid_pre, dino_pre)
                    use_dino_pre = dino_quality_pre >= float(
                        self.get_parameter("dinov2_assoc_min_quality").value
                    )
                    appearance_ok = (
                        not use_dino_pre
                        or
                        dino_cos is None
                        or dino_cos
                        >= float(self.get_parameter("segment_assoc_min_dino_cos").value)
                    )
                    fragment_coverage = float(count_raw) / float(max(mask_points, 1))
                    if (iou >= iou_threshold or fragment_coverage >= overlap_threshold) and (
                        not semantic_gate or sem is None or sem >= projected_min_sem
                    ) and appearance_ok:
                        projected_reserved.add(sid_pre)
                        has_projected_anchor = True
                        break
            if has_projected_anchor:
                continue
            mask_score_pre = (
                float(mask_scores[int(mask_idx)])
                if int(mask_idx) < int(mask_scores.shape[0])
                else 1.0
            )
            conf_pre = np.clip(
                conf_dist[point_idxs] * conf_scale * mask_score_pre, conf_min, 1.0
            ).astype(np.float32)
            obs_pre = self.build_segment_observation(
                pts_int[point_idxs],
                e_pre,
                conf_pre,
                dino_embedding=dino_pre,
                dino_quality=dino_quality_pre,
                min_support_override=int(self.get_parameter("segment_mask_min_support_voxels").value),
            )
            if obs_pre is not None:
                preliminary_obs[int(mask_idx)] = obs_pre
        fallback_matches, fallback_infos = self.match_segment_observations(
            preliminary_obs,
            blocked_sids=projected_reserved,
        )
        fallback_sids = set(int(sid) for sid in fallback_matches.values())

        for mask_points, mask_idx, point_idxs in mask_items:
            stats["attempted"] += 1
            mixed_mask = decisions_by_mask.get(mask_idx, {}).get('reason') == 'multiple_supported_owners'
            e = mask_emb[int(mask_idx)].astype(np.float32)
            en = float(np.linalg.norm(e))
            if en <= 1e-6:
                rejects["bad_embedding"] = int(rejects.get("bad_embedding", 0)) + 1
                continue
            e = e / en
            dino_e = (
                dino_mask_emb[int(mask_idx)].astype(np.float32)
                if dino_mask_emb.ndim == 2
                and int(mask_idx) < dino_mask_emb.shape[0]
                and dino_mask_emb.shape[1] > 0
                else None
            )
            dino_quality = (
                float(dino_mask_quality[int(mask_idx)])
                if dino_mask_quality.ndim == 1
                and int(mask_idx) < dino_mask_quality.shape[0]
                else 0.0
            )
            mask_score = float(mask_scores[int(mask_idx)]) if int(mask_idx) < int(mask_scores.shape[0]) else 1.0
            owners = point_sids[point_idxs] if point_sids.size else np.zeros((point_idxs.size,), dtype=np.int32)
            owned = owners[owners > 0]
            best_sid = None
            best_iou = 0.0
            best_inter_count = 0

            if int(owned.size) > 0:
                unique_sids, counts = np.unique(owned, return_counts=True)
                existing = sorted(
                    [(int(s), int(c)) for s, c in zip(unique_sids, counts) if self._valid_projected_segment_owner(int(s))],
                    key=lambda item: int(visible_counts.get(int(item[0]), item[1])),
                )
                for sid, inter_count in existing:
                    if semantic_gate and not mixed_mask:
                        sem = self._segment_semantic_cos(int(sid), e)
                        dino_cos = self._segment_dino_cos(int(sid), dino_e)
                        dino_conflict = (
                            dino_quality
                            >= float(
                                self.get_parameter("dinov2_assoc_min_quality").value
                            )
                            and
                            dino_cos is not None
                            and dino_cos
                            < float(self.get_parameter("segment_assoc_min_dino_cos").value)
                        )
                        if (sem is not None and sem < projected_min_sem) or dino_conflict:
                            rejects["semantic_conflict_projected"] = int(
                                rejects.get("semantic_conflict_projected", 0)
                            ) + 1
                            sid_points = point_idxs[owners == sid]
                            if int(sid_points.size) > 0:
                                conf_sid = np.clip(
                                    conf_dist[sid_points] * conf_scale * mask_score,
                                    conf_min,
                                    1.0,
                                ).astype(np.float32)
                                carved_sid, conflict_n, ready_n, ready_point_idxs = self.carve_segment_semantic_conflict(
                                    int(sid),
                                    pts_int,
                                    sid_points,
                                    e,
                                    conf_sid,
                                    now_s,
                                    cam_xy_int,
                                    obs_seq,
                                    dino_embedding=dino_e,
                                    dino_quality=dino_quality,
                                )
                                stats["semantic_conflict_voxels"] += int(conflict_n)
                                stats["semantic_conflict_ready_voxels"] += int(ready_n)
                                assignable[sid_points] = False
                                if carved_sid is not None:
                                    stats["semantic_conflict_carves"] += 1
                                    conflict_assignments.append(
                                        (ready_point_idxs, int(carved_sid), int(mask_idx))
                                    )
                            continue
                    visible_n = max(1, int(visible_counts.get(int(sid), inter_count)))
                    if inter_count > 0:
                        # Exact current 3D ownership is enough to retain each
                        # surface under an undersegmented or fragmented mask.
                        sid_points = point_idxs[owners == sid]
                        # An undersegmented mask cannot replace each component's
                        # appearance with the same broad descriptor.
                        update_e = self.segment_embedding_and_coherence(sid)[0] if mixed_mask else e
                        update_dino, update_quality = component_dino.get(sid, (None, 0.0)) if mixed_mask else (dino_e, dino_quality)
                        self._add_frame_segment_points(groups, assignable, sid, sid_points,
                                                       mask_idx, update_e if update_e is not None else e, mask_score,
                                                       update_dino, update_quality)
                    iou = float(inter_count) / float(max(mask_points + visible_n - inter_count, 1))
                    if iou > best_iou:
                        best_iou = iou
                        best_sid = int(sid)
                        best_inter_count = int(inter_count)

            leftover = point_idxs[assignable[point_idxs]]
            if mixed_mask:
                # Preserve measured owners; unknown interior stays unknown.
                continue
            if int(leftover.size) <= 0:
                continue
            # A validated temporal association is an allocation decision, not
            # just an overlay hint. Otherwise its few guided pixels update the
            # incumbent while the same mask's remainder creates a duplicate.
            decision = decisions_by_mask.get(mask_idx, {})
            incumbent = decision.get('winner')
            temporal_reuse = (decision.get('reason') in ('temporal_depth_hold', 'pending_owner_switch')
                              and incumbent in self.seg_w
                              and consensus_compatible(incumbent, mask_idx))
            if temporal_reuse:
                best_sid = int(incumbent)
                best_inter_count = int(np.count_nonzero(owners == best_sid))
            # A small mask may be a well-anchored fragment of a much larger
            # visible surface. IoU alone would give it a new ID every frame.
            fragment_coverage = float(best_inter_count) / float(max(mask_points, 1))
            if best_sid is not None and (temporal_reuse or best_iou >= iou_threshold or fragment_coverage >= overlap_threshold):
                sid = int(best_sid)
                if max_existing_extension_ratio >= 0.0:
                    leftover_owners = point_sids[leftover] if point_sids.size else np.zeros((leftover.size,), dtype=np.int32)
                    same_owner = leftover[leftover_owners == sid]
                    extension = leftover[leftover_owners == 0]
                    if temporal_reuse:
                        support = self.seg_support_voxels.get(sid, set())
                        local = []
                        for i in extension:
                            key = self.xyz_to_voxel(*map(float, pts_int[i]))
                            if key is not None and any((key[0]+dx,key[1]+dy,key[2]+dz) in support
                                for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)):
                                local.append(i)
                        extension = np.asarray(local, dtype=np.int64)
                    anchor_n = max(int(best_inter_count), int(same_owner.size), 1)
                    max_extension = int(math.ceil(max_existing_extension_ratio * float(anchor_n)))
                    if int(extension.size) > max_extension:
                        if max_extension > 0:
                            sample = np.linspace(0, int(extension.size) - 1, num=max_extension, dtype=np.int64)
                            extension = extension[sample]
                        else:
                            extension = np.zeros((0,), dtype=np.int64)
                        rejects["capped_existing_extension"] = int(
                            rejects.get("capped_existing_extension", 0)
                        ) + 1
                    leftover = np.concatenate([same_owner, extension]).astype(np.int64)
                    if int(leftover.size) <= 0:
                        continue
            elif int(mask_idx) in fallback_matches:
                sid = int(fallback_matches[int(mask_idx)])
            else:
                sid = self._reserve_segment_id()
            # An undersegmented SAM mask must not steal another surface's
            # existing support. Preserve each owner even if one mask spans
            # several boxes; only unowned points may extend this segment.
            if bool(self.get_parameter("segment_projected_id_protect_other_owners").value):
                leftover_owners = point_sids[leftover]
                allowed = (leftover_owners == 0) | (leftover_owners == int(sid))
                if not np.all(allowed):
                    rejects["protected_other_owner"] = int(rejects.get("protected_other_owner", 0)) + 1
                leftover = leftover[allowed]
                if leftover.size == 0:
                    continue
            self._add_frame_segment_points(
                groups,
                assignable,
                sid,
                leftover,
                mask_idx,
                e,
                mask_score,
                dino_e,
                dino_quality,
            )

        accepted_point_sid = np.zeros((n_points,), dtype=np.int32)
        accepted_point_mask_idx = np.full((n_points,), -1, dtype=np.int32)
        frame_sids = set()
        for point_idxs_conflict, carved_sid, conflict_mask_idx in conflict_assignments:
            conflict_idxs = np.asarray(point_idxs_conflict, dtype=np.int64)
            accepted_point_sid[conflict_idxs] = int(carved_sid)
            accepted_point_mask_idx[conflict_idxs] = int(conflict_mask_idx)
            frame_sids.add(int(carved_sid))
        for sid, group in groups.items():
            if not group["point_chunks"]:
                continue
            point_idxs = np.concatenate(group["point_chunks"]).astype(np.int64)
            point_mask_idxs = np.concatenate(group["point_mask_chunks"]).astype(np.int32)
            # Neighbor guidance / two SAM boundaries can land in one voxel.
            # Remove only conflicting samples, not an otherwise good whole
            # mask. The final fuse guard still forbids stealing any owner.
            exact_owners = np.asarray([self.voxel_segment_id.get(
                self.xyz_to_voxel(*map(float, pts_int[i])), 0) for i in point_idxs], dtype=np.int32)
            keep = (exact_owners == 0) | (exact_owners == int(sid))
            point_idxs, point_mask_idxs = point_idxs[keep], point_mask_idxs[keep]
            if int(point_idxs.size) < min_points:
                rejects["small_assigned_support"] = int(rejects.get("small_assigned_support", 0)) + 1
                continue
            emb = group["emb_sum"] / max(float(group["weight"]), 1e-6)
            en = float(np.linalg.norm(emb))
            if en <= 1e-6:
                rejects["bad_group_embedding"] = int(rejects.get("bad_group_embedding", 0)) + 1
                continue
            emb = (emb / en).astype(np.float32)
            dino_group = None
            dino_group_quality = 0.0
            if group.get("dino_sum") is not None:
                dino_norm = float(np.linalg.norm(group["dino_sum"]))
                if dino_norm > 1e-6:
                    dino_group = (group["dino_sum"] / dino_norm).astype(np.float32)
                    dino_group_quality = float(
                        np.clip(
                            float(group.get("dino_quality_mass", 0.0))
                            / max(float(group["weight"]), 1e-6),
                            0.0,
                            1.0,
                        )
                    )
            score = float(group["score_sum"]) / max(float(group["weight"]), 1e-6)
            conf = np.clip(conf_dist[point_idxs] * conf_scale * score, conf_min, 1.0).astype(np.float32)
            obs = self.build_segment_observation(
                pts_int[point_idxs],
                emb,
                conf,
                dino_embedding=dino_group,
                dino_quality=dino_group_quality,
                min_support_override=int(self.get_parameter("segment_mask_min_support_voxels").value),
            )
            if obs is None:
                reason = str(self._segment_obs_reject_reason or "unknown")
                rejects[reason] = int(rejects.get(reason, 0)) + 1
                continue

            stats["built"] += 1
            stats["obs_voxel_counts"].append(len(obs.get("support_voxels", set())))
            is_new = int(sid) not in self.seg_w
            assoc_source = (
                "frame_new"
                if is_new
                else "frame_fallback"
                if int(sid) in fallback_sids
                else "frame_projected"
            )
            fused_sid = self.fuse_segment_observation(
                obs,
                now_s,
                cam_xy_int,
                obs_seq=obs_seq,
                forced_sid=int(sid),
                force_new=is_new,
                assoc_source=assoc_source,
                assoc_info={
                    "sid": int(sid),
                    "source": assoc_source,
                    "pixels": int(point_idxs.size),
                    "mask_count": int(len(group["mask_indices"])),
                    "association": (
                        next(
                            (
                                fallback_infos[int(mask_id)]
                                for mask_id in group["mask_indices"]
                                if int(mask_id) in fallback_infos
                            ),
                            {},
                        )
                        if assoc_source == "frame_fallback"
                        else {}
                    ),
                },
            )
            if fused_sid is None:
                rejects["fuse_failed"] = int(rejects.get("fuse_failed", 0)) + 1
                continue
            frame_sids.add(int(fused_sid))
            accepted_point_sid[point_idxs] = int(fused_sid)
            accepted_point_mask_idx[point_idxs] = point_mask_idxs
            if self.last_segment_fuse_created:
                stats["new"] += 1
            else:
                stats["matched"] += 1
                if assoc_source == "frame_fallback":
                    stats["fallback_matched"] += 1
                else:
                    stats["projected_matched"] += 1

        stats["unique_frame_sids"] = int(len(frame_sids))
        # Display current visible owners even if SAM missed them or appearance
        # gating prevented a descriptor update. This is not a fused observation.
        visible_sids = {int(s) for s in point_sids if int(s) in self.seg_w}
        for sid in visible_sids:
            self.seg_last_seen[sid] = max(float(self.seg_last_seen.get(sid, now_s)), now_s)
        visible_point_sid = np.where(accepted_point_sid > 0, accepted_point_sid, measured_point_sids)
        visible_ids, counts = np.unique(visible_point_sid[visible_point_sid > 0], return_counts=True)
        self.last_visible_segment_counts = {int(s): int(c) for s, c in zip(visible_ids, counts)}
        assigned_img = np.zeros((int(h), int(w)), dtype=np.int32)
        valid_assigned = visible_point_sid > 0
        if np.any(valid_assigned):
            assigned_img[vv_i[valid_assigned], uu_i[valid_assigned]] = visible_point_sid[valid_assigned]
        attribution = []
        for index, mask in enumerate(masks):
            if mask is None or mask.shape != (h, w):
                attribution.append({"mask": index, "segments": [], "reason": "bad_shape"})
                continue
            local_ids = visible_point_sid[np.asarray(mask, dtype=bool)[vv_i, uu_i]]
            ids, counts = np.unique(local_ids[local_ids > 0], return_counts=True)
            attribution.append({"mask": index, "segments": [int(s) for s in ids],
                                "counts": [int(c) for c in counts], "depth_points": int(local_ids.size),
                                "reason": "attributed" if len(ids) else "no_valid_depth" if not local_ids.size else
                                "insufficient_depth" if local_ids.size < min_points else "geometry_or_appearance_rejected"})
        self.last_mask_attribution = attribution
        display_decisions = []
        if consensus_decisions:
            display_decisions = self.surface_consensus.propose(masks, visible_point_sid, pts_int, uu_i, vv_i,
                now_s, self.voxel_size, set(self.seg_w), lambda sid, idx: True)
            for item in display_decisions:
                before = decisions_by_mask.get(item['mask'], {})
                if before.get('reason') in ('appearance_conflict', 'pending_owner_switch'):
                    item.update({k: before[k] for k in ('winner', 'segments', 'reason', 'confidence', 'previous', 'switch_hits', 'challenger') if k in before})
                attribution[item['mask']].update(mask_owner=item['winner'], association_basis=item['reason'],
                    association_strength=round(float(item['confidence']), 3))
            self.surface_consensus.commit(obs_seq, now_s, display_decisions, masks, set(self.seg_w))
        if not hasattr(self, 'segment_display_memory'):
            self.segment_display_memory = DominantOverlay()
        segment_display = self.segment_display_memory.update(obs_seq, now_s, display_decisions,
            lambda sid: sid if sid in self.seg_w else None, self.voxel_size)
        for item in segment_display:
            attribution[item['mask']].update(display_owner=item['display_owner'],
                display_provisional=item['display_provisional'], display_confidence=item['display_confidence'])
        self.publish_segment_id_projection({"id_image": assigned_img, "masks": masks,
                                           "observation_seq": obs_seq, "segment_display_decisions": segment_display,
                                           "rgb": getattr(self, "last_assignment_rgb", None),
                                           **({"decisions": display_decisions} if consensus_decisions else {})}, header_msg)
        stats["point_mask_indices"] = accepted_point_mask_idx
        return stats

    def build_segment_observation(
        self,
        pts_mask: np.ndarray,
        emb_unit: np.ndarray,
        conf: np.ndarray,
        dino_embedding: Optional[np.ndarray] = None,
        dino_quality: float = 1.0,
        min_support_override: Optional[int] = None,
    ):
        self._segment_obs_reject_reason = "none"
        if pts_mask.shape[0] == 0:
            self._segment_obs_reject_reason = "empty"
            return None

        cells = set()
        voxels = set()
        for p in pts_mask:
            cell = self.xy_to_cell(float(p[0]), float(p[1]))
            if cell is not None:
                cells.add(cell)
            voxel = self.xyz_to_voxel(float(p[0]), float(p[1]), float(p[2]))
            if voxel is not None:
                voxels.add(voxel)

        min_cells = int(self.get_parameter("segment_min_support_cells").value)
        min_voxels = int(self.get_parameter("segment_min_support_voxels").value)
        if min_support_override is not None:
            min_cells = min_voxels = max(1, int(min_support_override))
        if not voxels:
            self._segment_obs_reject_reason = "outside_voxel_volume"
            return None
        if len(cells) < max(1, min_cells) and len(voxels) < max(1, min_voxels):
            self._segment_obs_reject_reason = "small_support"
            return None

        max_obs_voxels = int(self.get_parameter("segment_observation_max_voxels").value)
        if max_obs_voxels > 0 and len(voxels) > max_obs_voxels:
            self._segment_obs_reject_reason = "large_support"
            return None

        max_cells = int(self.get_parameter("segment_support_max_cells").value)
        if max_cells > 0 and len(cells) > max_cells:
            cells = set(sorted(cells)[:: max(1, int(math.ceil(len(cells) / max_cells)))])
        max_voxels = int(self.get_parameter("segment_support_max_voxels").value)
        if max_voxels > 0 and len(voxels) > max_voxels:
            voxels = set(sorted(voxels)[:: max(1, int(math.ceil(len(voxels) / max_voxels)))])

        q_low = float(np.clip(float(self.get_parameter("segment_bbox_percentile_low").value), 0.0, 49.0))
        q_high = float(np.clip(float(self.get_parameter("segment_bbox_percentile_high").value), 51.0, 100.0))
        if q_high <= q_low:
            q_low, q_high = 5.0, 95.0
        bmin = np.percentile(pts_mask, q_low, axis=0).astype(np.float32)
        bmax = np.percentile(pts_mask, q_high, axis=0).astype(np.float32)
        pad = max(0.0, float(self.get_parameter("segment_bbox_padding_m").value))
        bmin -= pad
        bmax += pad
        extent = np.maximum(bmax - bmin, 0.0)
        max_bbox_xy = float(self.get_parameter("segment_observation_max_bbox_xy_m").value)
        max_bbox_z = float(self.get_parameter("segment_observation_max_bbox_z_m").value)
        if max_bbox_xy > 0.0 and float(max(extent[0], extent[1])) > max_bbox_xy:
            self._segment_obs_reject_reason = "large_bbox_xy"
            return None
        if max_bbox_z > 0.0 and float(extent[2]) > max_bbox_z:
            self._segment_obs_reject_reason = "large_bbox_z"
            return None

        emb = emb_unit.astype(np.float32)
        n = float(np.linalg.norm(emb))
        if n <= 1e-6:
            self._segment_obs_reject_reason = "bad_embedding"
            return None
        emb = emb / n

        mean_conf = float(np.clip(np.mean(conf) if conf.size else 0.0, 0.0, 1.0))
        support_scale = float(np.clip(math.log1p(max(len(voxels), len(cells))) / math.log1p(80.0), 0.35, 1.5))
        self._segment_obs_reject_reason = "accepted"
        observation = {
            "centroid": np.mean(pts_mask, axis=0).astype(np.float32),
            "bbox_min": bmin,
            "bbox_max": bmax,
            "embedding": emb.astype(np.float32),
            "confidence": mean_conf,
            "weight": float(np.clip(mean_conf * support_scale, 0.0, 1.0)),
            "support_cells": cells,
            "support_voxels": voxels,
            "point_count": int(pts_mask.shape[0]),
        }
        if dino_embedding is not None:
            dino = np.asarray(dino_embedding, dtype=np.float32).reshape((-1,))
            dino_norm = float(np.linalg.norm(dino))
            if dino_norm > 1e-6:
                observation["dino_embedding"] = (dino / dino_norm).astype(np.float32)
                observation["dino_quality"] = float(np.clip(dino_quality, 0.0, 1.0))
        return observation

    def _segment_support_overlap_with_obs(self, sid: int, obs) -> float:
        old_vox = self.seg_support_voxels.get(int(sid), set())
        obs_vox = obs.get("support_voxels", set())
        if old_vox or obs_vox:
            inter = len(old_vox & obs_vox)
            if inter > 0:
                obs_frac = float(inter / max(len(obs_vox), 1))
                old_frac = float(inter / max(len(old_vox), 1))
                iou = float(inter / max(len(old_vox | obs_vox), 1))
                return max(iou, min(obs_frac, old_frac), 0.5 * obs_frac)
            # Shared XY cells do not establish shared physical support:
            # vertically stacked objects can have identical footprints.
            return 0.0
        old_cells = self.seg_support_cells.get(int(sid), set())
        obs_cells = obs.get("support_cells", set())
        if old_cells or obs_cells:
            inter = len(old_cells & obs_cells)
            if inter > 0:
                obs_frac = float(inter / max(len(obs_cells), 1))
                old_frac = float(inter / max(len(old_cells), 1))
                iou = float(inter / max(len(old_cells | obs_cells), 1))
                return max(iou, min(obs_frac, old_frac), 0.5 * obs_frac)
        return 0.0

    def _segment_association_metrics(self, obs, sid: int) -> Optional[dict]:
        sid = int(sid)
        if sid not in self.seg_w:
            return None
        max_dist = float(self.get_parameter("segment_assoc_max_center_dist_m").value)
        bbox_gap_max = float(self.get_parameter("segment_assoc_bbox_gap_max_m").value)
        sem_min = float(self.get_parameter("segment_assoc_min_sem_cos").value)
        dino_min = float(self.get_parameter("segment_assoc_min_dino_cos").value)
        min_overlap = float(self.get_parameter("segment_assoc_min_overlap").value)
        sem_w = float(self.get_parameter("segment_assoc_semantic_weight").value)
        dino_w = float(self.get_parameter("segment_assoc_dino_weight").value)
        spatial_w = float(self.get_parameter("segment_assoc_spatial_weight").value)
        overlap_w = float(self.get_parameter("segment_assoc_overlap_weight").value)
        max_age = float(self.get_parameter("segment_assoc_max_age_s").value)
        if max_age > 0.0:
            age = max(0.0, self.now_sec() - float(self.seg_last_seen.get(sid, 0.0)))
            if age > max_age:
                return None

        old_emb, _, _ = self.segment_embedding_and_coherence(sid)
        if old_emb is None:
            return None
        raw_sem = float(np.dot(old_emb, obs["embedding"]))
        if raw_sem < sem_min:
            return None
        sem01 = float(np.clip((raw_sem + 1.0) * 0.5, 0.0, 1.0))

        obs_dino = obs.get("dino_embedding")
        dino_quality = float(np.clip(float(obs.get("dino_quality", 1.0)), 0.0, 1.0))
        min_dino_quality = float(
            np.clip(
                float(self.get_parameter("dinov2_assoc_min_quality").value),
                0.0,
                1.0,
            )
        )
        old_dino = self.segment_dino_embedding(sid)
        dino_cos = None
        if (
            dino_quality >= min_dino_quality
            and obs_dino is not None
            and old_dino is not None
            and obs_dino.shape == old_dino.shape
        ):
            dino_cos = float(np.dot(old_dino, obs_dino))
            if dino_cos < dino_min:
                return None
        dino01 = float(np.clip((float(dino_cos) + 1.0) * 0.5, 0.0, 1.0)) if dino_cos is not None else 0.0

        dist = float(np.linalg.norm(self.seg_centroid[sid] - obs["centroid"]))
        centroid_spatial = (
            float(np.clip(1.0 - dist / max(max_dist, 1e-6), 0.0, 1.0))
            if max_dist > 0.0
            else 0.0
        )
        bbox_gap = self._bbox_gap_3d(
            obs["bbox_min"], obs["bbox_max"], self.seg_bbox_min[sid], self.seg_bbox_max[sid]
        )
        bbox_spatial = (
            float(np.clip(1.0 - bbox_gap / max(bbox_gap_max, 1e-6), 0.0, 1.0))
            if bbox_gap_max > 0.0
            else 0.0
        )
        bbox_overlap = self._bbox_iou_3d(
            obs["bbox_min"], obs["bbox_max"], self.seg_bbox_min[sid], self.seg_bbox_max[sid]
        )
        support_overlap = self._segment_support_overlap_with_obs(sid, obs)
        overlap = max(float(bbox_overlap), float(support_overlap))
        if bool(self.get_parameter("segment_assoc_require_overlap").value):
            # Mere contact between padded boxes is not a correspondence anchor.
            min_bbox_iou = max(1e-6, float(self.get_parameter("segment_assoc_min_bbox_iou").value))
            if support_overlap < max(min_overlap, 1e-6) and bbox_overlap < min_bbox_iou:
                return None
        spatial = max(centroid_spatial, bbox_spatial)
        if overlap > 0.0:
            spatial = max(spatial, min(1.0, 0.45 + 0.55 * overlap))

        # Appearance is deliberately not allowed to teleport an object. A
        # plausible geometric anchor remains mandatory.
        if spatial <= 0.0 and overlap < min_overlap:
            return None
        if overlap < min_overlap and spatial < 0.15 and raw_sem < 0.20:
            return None

        effective_dino_w = dino_w * dino_quality if dino_cos is not None else 0.0
        total_w = max(sem_w + effective_dino_w + spatial_w + overlap_w, 1e-6)
        score = (
            sem_w * sem01
            + effective_dino_w * dino01
            + spatial_w * spatial
            + overlap_w * overlap
        ) / total_w
        return {
            "sid": sid,
            "score": float(score),
            "raw_sem": float(raw_sem),
            "dino_cos": None if dino_cos is None else float(dino_cos),
            "dino_quality": float(dino_quality),
            "spatial": float(spatial),
            "overlap": float(overlap),
            "dist": float(dist),
            "bbox_gap": float(bbox_gap),
        }

    def match_segment_observations(self, observations: Dict[int, dict], blocked_sids: Optional[set] = None):
        if not observations:
            return {}, {}
        blocked_sids = blocked_sids or set()
        obs_ids = sorted(int(idx) for idx in observations.keys())
        track_ids = sorted(int(sid) for sid in self.seg_w.keys() if int(sid) not in blocked_sids)
        if not track_ids:
            return {}, {}
        min_score = float(self.get_parameter("segment_assoc_min_score").value)
        scores = np.full((len(obs_ids), len(track_ids)), -np.inf, dtype=np.float32)
        infos = {}
        for row, obs_id in enumerate(obs_ids):
            for col, sid in enumerate(track_ids):
                info = self._segment_association_metrics(observations[obs_id], sid)
                if info is None or float(info["score"]) < min_score:
                    continue
                scores[row, col] = float(info["score"])
                infos[(row, col)] = info
        # Do not force a choice between near-identical candidates merely
        # because the assignment solver can produce a complete matching.
        margin = max(0.0, float(self.get_parameter("segment_assoc_ambiguity_margin").value))
        if margin > 0.0:
            for row in range(scores.shape[0]):
                valid = np.sort(scores[row, np.isfinite(scores[row])])
                if valid.size > 1 and float(valid[-1] - valid[-2]) < margin:
                    scores[row, :] = -np.inf
        if bool(self.get_parameter("segment_assoc_global_assignment").value):
            pairs = maximum_weight_assignment(scores)
        else:
            candidates = [
                (float(scores[r, c]), r, c)
                for r in range(scores.shape[0])
                for c in range(scores.shape[1])
                if np.isfinite(scores[r, c])
            ]
            pairs = []
            used_rows = set()
            used_cols = set()
            for _, row, col in sorted(candidates, reverse=True):
                if row in used_rows or col in used_cols:
                    continue
                pairs.append((row, col))
                used_rows.add(row)
                used_cols.add(col)
        matches = {}
        match_infos = {}
        for row, col in pairs:
            if not np.isfinite(scores[row, col]) or float(scores[row, col]) < min_score:
                continue
            obs_id = int(obs_ids[row])
            matches[obs_id] = int(track_ids[col])
            match_infos[obs_id] = dict(infos[(row, col)])
        return matches, match_infos

    def find_matching_segment(self, obs, blocked_sids: Optional[set] = None) -> Optional[int]:
        matches, infos = self.match_segment_observations({0: obs}, blocked_sids=blocked_sids)
        best_sid = matches.get(0)
        best_info = infos.get(0)

        if best_sid is None:
            self.last_segment_assoc_info = best_info or {"sid": None, "score": -1.0}
            return None
        self.last_segment_assoc_info = best_info or {"sid": int(best_sid), "score": 1.0}
        return int(best_sid)

    def _claim_segment_voxels(self, sid: int, voxels: set):
        sid = int(sid)
        for key in voxels:
            old_sid = self.voxel_segment_id.get(key)
            if old_sid is not None and int(old_sid) != sid:
                old_support = self.seg_support_voxels.get(int(old_sid))
                if old_support is not None:
                    old_support.discard(key)
                    self.seg_support_cells[int(old_sid)] = self._support_cells_from_voxels(old_support)
            self.segment_voxel_miss_hits.pop(key, None)
            self.segment_semantic_conflict_hits.pop(key, None)
            self.segment_semantic_conflict_last_seq.pop(key, None)
            self.voxel_segment_id[key] = sid

    def _set_segment_support(self, sid: int, voxels: set, cells: set):
        sid = int(sid)
        old_voxels = set(self.seg_support_voxels.get(sid, set()))
        owned_voxels = {key for key in voxels if int(self.voxel_segment_id.get(key, sid)) == sid}
        for key in owned_voxels:
            self.voxel_segment_id[key] = sid
            # Only _claim_segment_voxels (fresh positive support) resets miss
            # votes. Repacking retained historical geometry is not a sighting.
        for key in old_voxels - owned_voxels:
            if int(self.voxel_segment_id.get(key, -1)) == sid:
                self.voxel_segment_id.pop(key, None)
            self.segment_voxel_miss_hits.pop(key, None)
            self.segment_semantic_conflict_hits.pop(key, None)
            self.segment_semantic_conflict_last_seq.pop(key, None)
        self.seg_support_voxels[sid] = owned_voxels
        if owned_voxels:
            self.seg_support_cells[sid] = self._support_cells_from_voxels(owned_voxels)
        else:
            self.seg_support_cells[sid] = set(cells)

    def carve_segment_semantic_conflict(
        self,
        sid: int,
        pts_int: np.ndarray,
        point_idxs: np.ndarray,
        emb_unit: np.ndarray,
        conf: np.ndarray,
        now_s: float,
        cam_xy_int,
        obs_seq: int,
        dino_embedding: Optional[np.ndarray] = None,
        dino_quality: float = 1.0,
    ):
        if not bool(self.get_parameter("segment_semantic_conflict_carve").value):
            return None, 0, 0, np.zeros((0,), dtype=np.int64)
        sid = int(sid)
        if sid not in self.seg_w or point_idxs is None or int(point_idxs.size) <= 0:
            return None, 0, 0, np.zeros((0,), dtype=np.int64)

        key_to_point = {}
        key_to_conf = {}
        key_to_idx = {}
        for local_i, point_idx in enumerate(np.asarray(point_idxs, dtype=np.int64)):
            p = pts_int[int(point_idx)]
            key = self.xyz_to_voxel(float(p[0]), float(p[1]), float(p[2]))
            if key is None or int(self.voxel_segment_id.get(key, -1)) != sid:
                continue
            key_to_point[key] = p.astype(np.float32)
            key_to_idx[key] = int(point_idx)
            c = float(conf[local_i]) if local_i < int(conf.shape[0]) else 1.0
            key_to_conf[key] = float(np.clip(c, 0.0, 1.0))

        if not key_to_point:
            return None, 0, 0, np.zeros((0,), dtype=np.int64)

        min_hits = max(1, int(self.get_parameter("segment_semantic_conflict_min_hits").value))
        ready_keys = []
        for key in key_to_point.keys():
            if int(self.segment_semantic_conflict_last_seq.get(key, -999999999)) != int(obs_seq):
                hits = int(self.segment_semantic_conflict_hits.get(key, 0)) + 1
                self.segment_semantic_conflict_hits[key] = hits
                self.segment_semantic_conflict_last_seq[key] = int(obs_seq)
            else:
                hits = int(self.segment_semantic_conflict_hits.get(key, 0))
            if hits >= min_hits:
                ready_keys.append(key)

        min_voxels = max(1, int(self.get_parameter("segment_semantic_conflict_min_voxels").value))
        if len(ready_keys) < min_voxels:
            return None, len(key_to_point), len(ready_keys), np.zeros((0,), dtype=np.int64)

        max_voxels = int(self.get_parameter("segment_semantic_conflict_max_voxels_per_frame").value)
        if max_voxels > 0 and len(ready_keys) > max_voxels:
            ready_keys = sorted(ready_keys)[:max_voxels]

        pts = np.asarray([key_to_point[key] for key in ready_keys], dtype=np.float32)
        conf_ready = np.asarray([key_to_conf.get(key, 1.0) for key in ready_keys], dtype=np.float32)
        obs = self.build_segment_observation(
            pts,
            emb_unit,
            conf_ready,
            dino_embedding=dino_embedding,
            dino_quality=dino_quality,
        )
        if obs is None:
            return None, len(key_to_point), len(ready_keys), np.zeros((0,), dtype=np.int64)

        new_sid = self._reserve_segment_id()
        carved_sid = self.fuse_segment_observation(
            obs,
            now_s,
            cam_xy_int,
            obs_seq=obs_seq,
            forced_sid=new_sid,
            force_new=True,
            assoc_source="semantic_conflict",
            assoc_info={
                "sid": int(new_sid),
                "source": "semantic_conflict",
                "old_sid": int(sid),
                "voxels": int(len(ready_keys)),
            },
        )
        if carved_sid is None:
            return None, len(key_to_point), len(ready_keys), np.zeros((0,), dtype=np.int64)

        old_support = {
            key for key in self.seg_support_voxels.get(sid, set())
            if int(self.voxel_segment_id.get(key, -1)) == sid
        }
        if old_support:
            self.seg_support_voxels[sid] = old_support
            self.seg_support_cells[sid] = self._support_cells_from_voxels(old_support)
            self._refresh_segment_geometry_from_support(sid)
        else:
            self._drop_segment_record(sid)
        for key in ready_keys:
            self.segment_semantic_conflict_hits.pop(key, None)
            self.segment_semantic_conflict_last_seq.pop(key, None)
        self.segment_query_support_cache.pop(sid, None)
        self._mark_state_dirty()
        ready_point_idxs = np.asarray([key_to_idx[key] for key in ready_keys if key in key_to_idx], dtype=np.int64)
        return int(carved_sid), len(key_to_point), len(ready_keys), ready_point_idxs

    def _segment_component_offsets(self, connectivity: int, gap_vox: int = 0):
        radius = max(1, 1 + max(0, int(gap_vox)))
        connectivity = int(connectivity)
        offsets = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nonzero = int(dx != 0) + int(dy != 0) + int(dz != 0)
                    if connectivity <= 6 and nonzero > 1:
                        continue
                    if connectivity <= 18 and nonzero > 2:
                        continue
                    offsets.append((dx, dy, dz))
        return offsets

    def _segment_connected_components(self, voxels: set):
        if not voxels:
            return []
        connectivity = int(self.get_parameter("segment_split_connectivity").value)
        gap_vox = int(self.get_parameter("segment_split_gap_vox").value)
        offsets = self._segment_component_offsets(connectivity, gap_vox)
        remaining = set(voxels)
        components = []
        while remaining:
            seed = remaining.pop()
            comp = {seed}
            stack = [seed]
            while stack:
                vx, vy, vz = stack.pop()
                for dx, dy, dz in offsets:
                    nb = (vx + dx, vy + dy, vz + dz)
                    if nb not in remaining:
                        continue
                    remaining.remove(nb)
                    comp.add(nb)
                    stack.append(nb)
            components.append(comp)
        components.sort(key=len, reverse=True)
        return components

    def _segment_component_embedding_sum(self, sid: int, keys: set, total_keys: int, base_sum: np.ndarray, base_w: float):
        frac = float(len(keys)) / float(max(total_keys, 1))
        w_max = float(self.get_parameter("segment_weight_max").value)
        new_w = float(np.clip(float(base_w) * frac, 1e-4, w_max))
        acc = np.zeros_like(base_sum, dtype=np.float32)
        acc_w = 0.0
        for key in keys:
            if key in self.vox_ltm_sum:
                acc = (acc + np.asarray(self.vox_ltm_sum[key], dtype=np.float32)).astype(np.float32)
                acc_w += float(self.vox_ltm_w.get(key, 0.0))
            if key in self.vox_stm_sum:
                acc = (acc + np.asarray(self.vox_stm_sum[key], dtype=np.float32)).astype(np.float32)
                acc_w += float(self.vox_stm_w.get(key, 0.0))
        if acc_w > 1e-6 and float(np.linalg.norm(acc)) > 1e-6:
            emb = acc / max(float(np.linalg.norm(acc)), 1e-6)
            return (new_w * emb).astype(np.float32), new_w
        if float(np.linalg.norm(base_sum)) > 1e-6:
            emb = base_sum / max(float(np.linalg.norm(base_sum)), 1e-6)
            return (new_w * emb).astype(np.float32), new_w
        return (base_sum * frac).astype(np.float32), new_w

    def split_disconnected_segments(self):
        if not bool(self.get_parameter("segment_split_enabled").value):
            return {"checked": 0, "split_segments": 0, "new_segments": 0, "pruned_voxels": 0}
        min_support = max(1, int(self.get_parameter("segment_split_min_support_voxels").value))
        min_component = max(1, int(self.get_parameter("segment_split_min_component_voxels").value))
        min_secondary_frac = float(np.clip(float(self.get_parameter("segment_split_min_secondary_frac").value), 0.0, 1.0))
        max_components = max(2, int(self.get_parameter("segment_split_max_components_per_segment").value))
        max_segments = int(self.get_parameter("segment_split_max_segments_per_tick").value)
        drop_tiny = bool(self.get_parameter("segment_split_drop_tiny_components").value)

        checked = 0
        split_segments = 0
        new_segments = 0
        pruned_voxels = 0
        candidates = sorted(
            list(self.seg_w.keys()),
            key=lambda sid: len(self.seg_support_voxels.get(int(sid), set())),
            reverse=True,
        )
        for sid in candidates:
            if max_segments > 0 and checked >= max_segments:
                break
            sid = int(sid)
            if sid not in self.seg_w:
                continue
            support = {
                key for key in self.seg_support_voxels.get(sid, set())
                if int(self.voxel_segment_id.get(key, -1)) == sid
            }
            if len(support) < min_support:
                self.seg_support_voxels[sid] = support
                self.seg_support_cells[sid] = self._support_cells_from_voxels(support) if support else set()
                continue
            checked += 1
            components = self._segment_connected_components(support)
            if len(components) <= 1:
                if support != self.seg_support_voxels.get(sid, set()):
                    self._refresh_segment_geometry_from_support(sid)
                continue

            total_n = int(sum(len(c) for c in components))
            largest = set(components[0])
            secondary = [
                set(comp) for comp in components[1:]
                if len(comp) >= min_component and (float(len(comp)) / float(max(total_n, 1))) >= min_secondary_frac
            ]
            selected_secondary = secondary[: max(0, max_components - 1)]
            selected_keys = set(largest)
            for comp in selected_secondary:
                selected_keys |= comp

            unselected = support - selected_keys
            if not selected_secondary:
                if not drop_tiny or not unselected:
                    continue
                for key in unselected:
                    if int(self.voxel_segment_id.get(key, -1)) == sid:
                        self.voxel_segment_id.pop(key, None)
                    self.segment_voxel_miss_hits.pop(key, None)
                    self.segment_semantic_conflict_hits.pop(key, None)
                    self.segment_semantic_conflict_last_seq.pop(key, None)
                pruned_voxels += len(unselected)
                self._set_segment_support(sid, largest, set())
                self._refresh_segment_geometry_from_support(sid)
                self.segment_query_support_cache.pop(sid, None)
                continue

            keep_support = set(largest)
            if not drop_tiny:
                keep_support |= unselected
            else:
                for key in unselected:
                    if int(self.voxel_segment_id.get(key, -1)) == sid:
                        self.voxel_segment_id.pop(key, None)
                    self.segment_voxel_miss_hits.pop(key, None)
                    self.segment_semantic_conflict_hits.pop(key, None)
                    self.segment_semantic_conflict_last_seq.pop(key, None)
                pruned_voxels += len(unselected)

            base_sum = np.asarray(self.seg_sum[sid], dtype=np.float32).copy()
            base_w = float(self.seg_w.get(sid, 0.0))
            base_first = float(self.seg_first_seen.get(sid, self.seg_last_seen.get(sid, 0.0)))
            base_last = float(self.seg_last_seen.get(sid, 0.0))
            base_obs = int(self.seg_obs_count.get(sid, 1))
            base_dino_sum = self.seg_dino_sum.get(sid)
            base_dino_w = float(self.seg_dino_w.get(sid, 0.0))
            merge_block_until = self.now_sec() + max(
                0.0, float(self.get_parameter("segment_merge_split_cooldown_s").value)
            )
            self.seg_merge_block_until[sid] = merge_block_until

            keep_sum, keep_w = self._segment_component_embedding_sum(sid, keep_support, total_n, base_sum, base_w)
            self.seg_sum[sid] = keep_sum
            self.seg_w[sid] = keep_w
            if base_dino_sum is not None and base_dino_w > 0.0:
                keep_frac = float(len(keep_support)) / float(max(total_n, 1))
                self.seg_dino_sum[sid] = (base_dino_sum * keep_frac).astype(np.float32)
                self.seg_dino_w[sid] = base_dino_w * keep_frac
            self._set_segment_support(sid, keep_support, set())
            self._refresh_segment_geometry_from_support(sid)
            self.segment_query_support_cache.pop(sid, None)

            for comp in selected_secondary:
                new_sid = self._reserve_segment_id()
                comp_sum, comp_w = self._segment_component_embedding_sum(sid, comp, total_n, base_sum, base_w)
                self.seg_sum[new_sid] = comp_sum
                self.seg_w[new_sid] = comp_w
                self.seg_first_seen[new_sid] = base_first
                self.seg_last_seen[new_sid] = base_last
                # Inherit the surface's already measured evidence so ownership
                # remains projectable; the split is NOT a fresh observation.
                # The new entity still earns its own probation from new frames.
                self.seg_obs_count[new_sid] = base_obs
                self.seg_last_obs_seq[new_sid] = -1
                self.seg_viewmask[new_sid] = 0
                self.seg_merge_block_until[new_sid] = merge_block_until
                if base_dino_sum is not None and base_dino_w > 0.0:
                    comp_frac = float(len(comp)) / float(max(total_n, 1))
                    self.seg_dino_sum[new_sid] = (base_dino_sum * comp_frac).astype(np.float32)
                    self.seg_dino_w[new_sid] = base_dino_w * comp_frac
                self.seg_centroid[new_sid] = self.seg_centroid[sid].copy()
                self.seg_bbox_min[new_sid] = self.seg_bbox_min[sid].copy()
                self.seg_bbox_max[new_sid] = self.seg_bbox_max[sid].copy()
                self.seg_support_cells[new_sid] = set()
                self.seg_support_voxels[new_sid] = set()
                self._set_segment_support(new_sid, comp, set())
                self._refresh_segment_geometry_from_support(new_sid)
                self.entity_memory.segment_split(sid, new_sid)
                new_segments += 1

            split_segments += 1

        if split_segments > 0 or pruned_voxels > 0:
            self._mark_state_dirty()
        return {
            "checked": int(checked),
            "split_segments": int(split_segments),
            "new_segments": int(new_segments),
            "pruned_voxels": int(pruned_voxels),
        }

    def _segment_support_overlap(self, a: int, b: int) -> float:
        vox_a = {
            key for key in self.seg_support_voxels.get(int(a), set())
            if int(self.voxel_segment_id.get(key, -1)) == int(a)
        }
        vox_b = {
            key for key in self.seg_support_voxels.get(int(b), set())
            if int(self.voxel_segment_id.get(key, -1)) == int(b)
        }
        if vox_a or vox_b:
            inter = len(vox_a & vox_b)
            if inter > 0:
                return float(inter / max(len(vox_a | vox_b), 1))
        cells_a = self.seg_support_cells.get(int(a), set())
        cells_b = self.seg_support_cells.get(int(b), set())
        if cells_a or cells_b:
            inter = len(cells_a & cells_b)
            if inter > 0:
                return float(inter / max(len(cells_a | cells_b), 1))
        return 0.0

    def _segment_support_near(self, a: int, b: int, radius_vox: int) -> bool:
        radius_vox = max(0, int(radius_vox))
        if radius_vox <= 0:
            return False
        vox_a = {
            key for key in self.seg_support_voxels.get(int(a), set())
            if int(self.voxel_segment_id.get(key, -1)) == int(a)
        }
        vox_b = {
            key for key in self.seg_support_voxels.get(int(b), set())
            if int(self.voxel_segment_id.get(key, -1)) == int(b)
        }
        if not vox_a or not vox_b:
            return False
        if len(vox_a) > len(vox_b):
            vox_a, vox_b = vox_b, vox_a
        offsets = self._segment_component_offsets(26, max(0, radius_vox - 1))
        vox_b = set(vox_b)
        for vx, vy, vz in vox_a:
            for dx, dy, dz in offsets:
                if (vx + dx, vy + dy, vz + dz) in vox_b:
                    return True
        return False

    def _bbox_growth_compatibility(
        self,
        a_min,
        a_max,
        b_min,
        b_max,
        area_limit: Optional[float] = None,
        volume_limit: Optional[float] = None,
        z_limit: Optional[float] = None,
    ):
        a_dims = np.maximum(np.asarray(a_max, dtype=np.float32) - np.asarray(a_min, dtype=np.float32), 0.0)
        b_dims = np.maximum(np.asarray(b_max, dtype=np.float32) - np.asarray(b_min, dtype=np.float32), 0.0)
        u_min = np.minimum(np.asarray(a_min, dtype=np.float32), np.asarray(b_min, dtype=np.float32))
        u_max = np.maximum(np.asarray(a_max, dtype=np.float32), np.asarray(b_max, dtype=np.float32))
        u_dims = np.maximum(u_max - u_min, 0.0)

        def volume(dims):
            return float(max(float(dims[0]) * float(dims[1]) * max(float(dims[2]), 1e-3), 1e-6))

        def area_xy(dims):
            return float(max(float(dims[0]) * float(dims[1]), 1e-6))

        area_growth = float(area_xy(u_dims) / max(area_xy(a_dims), area_xy(b_dims), 1e-6))
        volume_growth = float(volume(u_dims) / max(volume(a_dims), volume(b_dims), 1e-6))
        z_growth = float(max(float(u_dims[2]), 1e-3) / max(float(a_dims[2]), float(b_dims[2]), 1e-3))

        def growth_score(growth: float, limit: float) -> float:
            if limit <= 1.0 or growth <= 1.0:
                return 1.0
            return float(np.clip(1.0 - (growth - 1.0) / (limit - 1.0), 0.0, 1.0))

        if area_limit is None:
            area_limit = float(self.get_parameter("segment_merge_max_area_growth").value)
        if volume_limit is None:
            volume_limit = float(self.get_parameter("segment_merge_max_volume_growth").value)
        if z_limit is None:
            z_limit = float(self.get_parameter("segment_merge_max_z_growth").value)
        area_score = growth_score(area_growth, float(area_limit))
        volume_score = growth_score(volume_growth, float(volume_limit))
        z_score = growth_score(z_growth, float(z_limit))
        score = float(min(area_score, volume_score, z_score))
        return score, {
            "area_growth": area_growth,
            "volume_growth": volume_growth,
            "z_growth": z_growth,
        }

    def _segment_pair_merge_score(self, a: int, b: int):
        a = int(a)
        b = int(b)
        if a == b or a not in self.seg_w or b not in self.seg_w:
            return None
        now_s = self.now_sec()
        if now_s < float(self.seg_merge_block_until.get(a, 0.0)) or now_s < float(
            self.seg_merge_block_until.get(b, 0.0)
        ):
            return None
        min_obs = max(1, int(self.get_parameter("segment_merge_min_observations").value))
        if int(self.seg_obs_count.get(a, 0)) < min_obs or int(self.seg_obs_count.get(b, 0)) < min_obs:
            return None

        emb_a, _, _ = self.segment_embedding_and_coherence(a)
        emb_b, _, _ = self.segment_embedding_and_coherence(b)
        if emb_a is None or emb_b is None:
            return None
        sem = float(np.dot(emb_a, emb_b))
        min_sem = float(self.get_parameter("segment_merge_min_sem_cos").value)
        if sem < min_sem:
            return None
        dino_a = self.segment_dino_embedding(a)
        dino_b = self.segment_dino_embedding(b)
        dino_cos = None
        if dino_a is not None and dino_b is not None and dino_a.shape == dino_b.shape:
            dino_cos = float(np.dot(dino_a, dino_b))
            if dino_cos < float(self.get_parameter("segment_assoc_min_dino_cos").value):
                return None

        bbox_gap = self._bbox_gap_3d(self.seg_bbox_min[a], self.seg_bbox_max[a], self.seg_bbox_min[b], self.seg_bbox_max[b])
        max_gap = float(self.get_parameter("segment_merge_max_bbox_gap_m").value)
        max_center = float(self.get_parameter("segment_merge_max_center_dist_m").value)
        center_dist = float(np.linalg.norm(self.seg_centroid[a] - self.seg_centroid[b]))
        support_overlap = self._segment_support_overlap(a, b)
        bbox_overlap = self._bbox_iou_3d(self.seg_bbox_min[a], self.seg_bbox_max[a], self.seg_bbox_min[b], self.seg_bbox_max[b])
        overlap = max(float(support_overlap), float(bbox_overlap))
        min_overlap = float(self.get_parameter("segment_merge_min_overlap").value)
        adjacent = self._segment_support_near(a, b, int(self.get_parameter("segment_merge_adjacency_vox").value))

        gap_anchor = max_gap > 0.0 and bbox_gap <= max_gap
        overlap_anchor = overlap >= min_overlap
        if not (adjacent or gap_anchor or overlap_anchor):
            return None
        if max_center > 0.0 and center_dist > max_center and not overlap_anchor:
            return None
        if adjacent and not gap_anchor and not overlap_anchor:
            adjacent_min_sem = float(self.get_parameter("segment_merge_adjacent_only_min_sem_cos").value)
            if sem < adjacent_min_sem:
                return None

        same_frame = (
            int(self.seg_last_obs_seq.get(a, -1)) >= 0
            and int(self.seg_last_obs_seq.get(a, -1)) == int(self.seg_last_obs_seq.get(b, -2))
        )
        if same_frame:
            same_frame_min_sem = float(self.get_parameter("segment_merge_same_frame_min_sem_cos").value)
            if sem < same_frame_min_sem and not overlap_anchor:
                return None
            if not adjacent and not overlap_anchor and not gap_anchor:
                return None

        geometry_score, geometry_info = self._bbox_growth_compatibility(
            self.seg_bbox_min[a],
            self.seg_bbox_max[a],
            self.seg_bbox_min[b],
            self.seg_bbox_max[b],
        )
        min_geometry = float(self.get_parameter("segment_merge_min_geometry_score").value)
        if geometry_score < min_geometry and overlap < max(min_overlap, 0.10):
            return None

        gap_score = float(np.clip(1.0 - bbox_gap / max(max_gap, 1e-6), 0.0, 1.0)) if max_gap > 0.0 else 0.0
        dist_score = float(np.clip(1.0 - center_dist / max(max_center, 1e-6), 0.0, 1.0)) if max_center > 0.0 else 0.0
        adjacent_score = 1.0 if adjacent else 0.0
        spatial = max(gap_score, dist_score, adjacent_score)
        overlap_score = float(np.clip(overlap / max(min_overlap, 1e-6), 0.0, 1.0)) if min_overlap > 0.0 else float(np.clip(overlap, 0.0, 1.0))
        if dino_cos is None:
            score = float(
                0.74 * sem + 0.10 * spatial + 0.10 * geometry_score + 0.06 * overlap_score
            )
        else:
            score = float(
                0.34 * sem
                + 0.46 * dino_cos
                + 0.08 * spatial
                + 0.08 * geometry_score
                + 0.04 * overlap_score
            )
        if score < float(self.get_parameter("segment_merge_min_score").value):
            return None
        return {
            "score": score,
            "sem": sem,
            "dino_cos": dino_cos,
            "spatial": spatial,
            "overlap": overlap,
            "geometry": geometry_score,
            "bbox_gap": bbox_gap,
            "center_dist": center_dist,
            "adjacent": bool(adjacent),
            **geometry_info,
        }

    def _merge_segment_records(self, keep_sid: int, drop_sid: int) -> bool:
        keep_sid = int(keep_sid)
        drop_sid = int(drop_sid)
        if keep_sid == drop_sid or keep_sid not in self.seg_w or drop_sid not in self.seg_w:
            return False
        if not self.entity_memory.can_merge_segments(keep_sid, drop_sid):
            return False

        keep_support = {
            key for key in self.seg_support_voxels.get(keep_sid, set())
            if int(self.voxel_segment_id.get(key, -1)) == keep_sid
        }
        drop_support = {
            key for key in self.seg_support_voxels.get(drop_sid, set())
            if int(self.voxel_segment_id.get(key, -1)) == drop_sid
        }
        max_voxels = int(self.get_parameter("segment_support_max_voxels").value)
        merged_support = self._bounded_support_union(keep_support, drop_support, max_voxels)

        for key in drop_support:
            if key in merged_support:
                self.voxel_segment_id[key] = keep_sid
            elif int(self.voxel_segment_id.get(key, -1)) == drop_sid:
                self.voxel_segment_id.pop(key, None)
            self.segment_voxel_miss_hits.pop(key, None)
            self.segment_semantic_conflict_hits.pop(key, None)
            self.segment_semantic_conflict_last_seq.pop(key, None)
        for key in keep_support - merged_support:
            if int(self.voxel_segment_id.get(key, -1)) == keep_sid:
                self.voxel_segment_id.pop(key, None)
            self.segment_voxel_miss_hits.pop(key, None)
            self.segment_semantic_conflict_hits.pop(key, None)
            self.segment_semantic_conflict_last_seq.pop(key, None)

        self.seg_sum[keep_sid] = (self.seg_sum[keep_sid] + self.seg_sum[drop_sid]).astype(np.float32)
        self.seg_w[keep_sid] = float(
            np.clip(
                float(self.seg_w.get(keep_sid, 0.0)) + float(self.seg_w.get(drop_sid, 0.0)),
                0.0,
                float(self.get_parameter("segment_weight_max").value),
            )
        )
        self.seg_first_seen[keep_sid] = min(
            float(self.seg_first_seen.get(keep_sid, self.seg_last_seen.get(keep_sid, 0.0))),
            float(self.seg_first_seen.get(drop_sid, self.seg_last_seen.get(drop_sid, 0.0))),
        )
        self.seg_last_seen[keep_sid] = max(
            float(self.seg_last_seen.get(keep_sid, 0.0)),
            float(self.seg_last_seen.get(drop_sid, 0.0)),
        )
        self.seg_obs_count[keep_sid] = max(
            int(self.seg_obs_count.get(keep_sid, 1)),
            int(self.seg_obs_count.get(drop_sid, 1)),
        )
        self.seg_last_obs_seq[keep_sid] = max(
            int(self.seg_last_obs_seq.get(keep_sid, -1)),
            int(self.seg_last_obs_seq.get(drop_sid, -1)),
        )
        self.seg_viewmask[keep_sid] = int(self.seg_viewmask.get(keep_sid, 0) | self.seg_viewmask.get(drop_sid, 0))
        keep_dino = self.seg_dino_sum.get(keep_sid)
        drop_dino = self.seg_dino_sum.get(drop_sid)
        if drop_dino is not None:
            dino_memory_max = max(
                1e-3,
                float(self.get_parameter("dinov2_track_memory_weight").value),
            )
            if keep_dino is None or keep_dino.shape != drop_dino.shape:
                merged_dino = drop_dino.copy()
                merged_dino_w = min(
                    float(self.seg_dino_w.get(drop_sid, 0.0)), dino_memory_max
                )
            else:
                merged_dino = (keep_dino + drop_dino).astype(np.float32)
                merged_dino_w = min(
                    self.seg_dino_w.get(keep_sid, 0.0)
                    + self.seg_dino_w.get(drop_sid, 0.0),
                    dino_memory_max,
                )
            merged_norm = float(np.linalg.norm(merged_dino))
            if merged_norm > 1e-6 and merged_dino_w > 1e-6:
                self.seg_dino_sum[keep_sid] = (
                    merged_dino_w * merged_dino / merged_norm
                ).astype(np.float32)
                self.seg_dino_w[keep_sid] = float(merged_dino_w)
        self.seg_support_voxels[keep_sid] = merged_support
        self.seg_support_cells[keep_sid] = self._support_cells_from_voxels(merged_support)
        self.entity_memory.segment_merged(keep_sid, drop_sid)
        self._drop_segment_record(drop_sid)
        if merged_support:
            self._refresh_segment_geometry_from_support(keep_sid)
        self.segment_query_support_cache.pop(keep_sid, None)
        self._mark_state_dirty()
        return True

    def merge_compatible_segments(self):
        if not bool(self.get_parameter("segment_merge_enabled").value):
            return {"candidates": 0, "merged": 0}
        max_pairs = int(self.get_parameter("segment_merge_max_pairs_per_tick").value)
        ids = sorted(
            list(self.seg_w.keys()),
            key=lambda sid: (
                float(self.seg_w.get(int(sid), 0.0)),
                len(self.seg_support_voxels.get(int(sid), set())),
            ),
            reverse=True,
        )
        candidates = []
        for idx, a in enumerate(ids):
            for b in ids[idx + 1:]:
                scored = self._segment_pair_merge_score(int(a), int(b))
                if scored is None:
                    continue
                keep, drop = (
                    (int(a), int(b))
                    if float(self.seg_w.get(int(a), 0.0)) >= float(self.seg_w.get(int(b), 0.0))
                    else (int(b), int(a))
                )
                candidates.append((float(scored["score"]), keep, drop, scored))
        if not candidates:
            return {"candidates": 0, "merged": 0}

        candidates.sort(key=lambda item: item[0], reverse=True)
        touched = set()
        merged = 0
        details = []
        for score, keep, drop, info in candidates:
            if max_pairs > 0 and merged >= max_pairs:
                break
            if keep in touched or drop in touched:
                continue
            if keep not in self.seg_w or drop not in self.seg_w:
                continue
            if self._merge_segment_records(keep, drop):
                touched.add(keep)
                touched.add(drop)
                merged += 1
                details.append(
                    f"{keep}<-{drop}@{score:.2f}"
                    f"/sem={float(info.get('sem', 0.0)):.2f}"
                    f"/gap={float(info.get('bbox_gap', 0.0)):.2f}"
                    f"/adj={int(bool(info.get('adjacent', False)))}"
                )
        return {"candidates": int(len(candidates)), "merged": int(merged), "details": details}

    def retire_stale_segments(self, now_s: float):
        tentative_after = float(
            self.get_parameter("segment_retire_tentative_after_s").value
        )
        stale_after = float(self.get_parameter("segment_retire_stale_after_s").value)
        tentative_max_obs = max(
            0, int(self.get_parameter("segment_retire_tentative_max_observations").value)
        )
        retired = []
        for sid in list(self.seg_w.keys()):
            age = max(0.0, float(now_s) - float(self.seg_last_seen.get(int(sid), now_s)))
            observations = int(self.seg_obs_count.get(int(sid), 0))
            # Confirmed geometric tracks become dormant instead of vanishing
            # because a camera looked away or a semantic cache timed out.
            has_support = bool(self.seg_support_voxels.get(int(sid)))
            mature = observations > tentative_max_obs and has_support
            retire = not mature and stale_after > 0.0 and age > stale_after
            if (
                not retire
                and tentative_after > 0.0
                and age > tentative_after
                and observations <= tentative_max_obs
            ):
                retire = True
            if retire:
                retired.append(int(sid))
        for sid in retired:
            self._drop_segment_record(sid)
        limit = max(1, int(self.get_parameter('segment_memory_max_tracks').value))
        if len(self.seg_w) > limit:
            # Explicit capacity eviction, oldest unobserved records first.
            victims = sorted(self.seg_w, key=lambda s: (self.seg_last_seen.get(s, 0.), s))[:len(self.seg_w)-limit]
            for sid in victims:
                self._drop_segment_record(sid)
            retired.extend(victims)
        if retired:
            self._mark_state_dirty()
        return int(len(retired))

    def maintain_segments(self, now_s: Optional[float] = None):
        if not bool(self.get_parameter("use_segment_memory").value):
            return
        if not bool(self.get_parameter("use_segment_maintenance").value):
            return
        now_s = self.now_sec() if now_s is None else float(now_s)
        interval = float(self.get_parameter("segment_maintenance_interval_s").value)
        if interval > 0.0 and (now_s - float(self.last_segment_maintenance_s)) < interval:
            return
        self.last_segment_maintenance_s = now_s

        retired = self.retire_stale_segments(now_s)
        split_stats = self.split_disconnected_segments()
        merge_stats = self.merge_compatible_segments()
        merge_stats.setdefault("details", [])
        self._profile_add_value("segments.maintenance_splits", int(split_stats.get("split_segments", 0)))
        self._profile_add_value("segments.maintenance_new_segments", int(split_stats.get("new_segments", 0)))
        self._profile_add_value("segments.maintenance_pruned_voxels", int(split_stats.get("pruned_voxels", 0)))
        self._profile_add_value("segments.maintenance_merged", int(merge_stats.get("merged", 0)))
        self._profile_add_value("segments.maintenance_retired", retired)
        self._log_segment_maintenance_debug(split_stats, merge_stats, now_s)

    def _log_segment_maintenance_debug(self, split_stats: dict, merge_stats: dict, now_s: float):
        if not bool(self.get_parameter("segment_maintenance_debug_log").value):
            return
        interval = float(self.get_parameter("segment_maintenance_debug_interval_s").value)
        changed = (
            int(split_stats.get("split_segments", 0)) > 0
            or int(split_stats.get("pruned_voxels", 0)) > 0
            or int(merge_stats.get("merged", 0)) > 0
        )
        if not changed and interval > 0.0 and (now_s - float(self.last_segment_maintenance_debug_log_s)) < interval:
            return
        self.last_segment_maintenance_debug_log_s = now_s
        details = ",".join(merge_stats.get("details", [])[:5]) if merge_stats.get("details") else "none"
        self.get_logger().info(
            "segment maintenance: "
            f"checked={int(split_stats.get('checked', 0))} "
            f"splits={int(split_stats.get('split_segments', 0))} "
            f"new={int(split_stats.get('new_segments', 0))} "
            f"pruned_vox={int(split_stats.get('pruned_voxels', 0))} "
            f"merge_candidates={int(merge_stats.get('candidates', 0))} "
            f"merged={int(merge_stats.get('merged', 0))} "
            f"merge_details={details}"
        )

    def fuse_segment_observation(
        self,
        obs,
        now_s: float,
        cam_xy_int,
        obs_seq: int = -1,
        blocked_sids: Optional[set] = None,
        forced_sid: Optional[int] = None,
        force_new: bool = False,
        assoc_source: str = "fallback",
        assoc_info: Optional[dict] = None,
    ):
        if not bool(self.get_parameter("use_segment_memory").value):
            return None
        conf_eff = float(obs.get("weight", obs.get("confidence", 0.0)))
        if conf_eff <= 0.0:
            return None
        w_max = float(self.get_parameter("segment_weight_max").value)
        self.last_segment_fuse_created = False
        blocked_sids = blocked_sids or set()

        sid = None
        forced_new_sid = None
        if forced_sid is not None:
            forced_sid = int(forced_sid)
            if forced_sid in self.seg_w and forced_sid not in blocked_sids:
                sid = forced_sid
                info = dict(assoc_info or {})
                info["sid"] = int(forced_sid)
                info["source"] = str(assoc_source or "projected_id")
                self.last_segment_assoc_info = info
            elif force_new and forced_sid > 0 and forced_sid not in blocked_sids:
                forced_new_sid = forced_sid
        if sid is None and not force_new:
            sid = self.find_matching_segment(obs, blocked_sids=blocked_sids)
            if sid is not None:
                info = dict(self.last_segment_assoc_info or {})
                info["source"] = "fallback"
                self.last_segment_assoc_info = info

        if (
            bool(self.get_parameter("segment_projected_id_protect_other_owners").value)
            and assoc_source != "semantic_conflict"
        ):
            target_sid = sid if sid is not None else forced_new_sid
            conflicting_owner = any(
                self.voxel_segment_id.get(key) in self.seg_w
                and self.voxel_segment_id.get(key) != target_sid
                for key in obs.get("support_voxels", ())
            )
            if conflicting_owner:
                # Last-line ownership protection also covers the legacy path,
                # low-observation owners omitted by projection, and points
                # whose quantization disagrees with the image assignment.
                self._segment_obs_reject_reason = "owned_support_conflict"
                self.last_segment_assoc_info = {"sid": target_sid, "source": "ownership_rejected"}
                return None

        if sid is None:
            if forced_new_sid is not None:
                sid = int(forced_new_sid)
                self.seg_next_id = max(int(self.seg_next_id), int(sid) + 1)
            else:
                sid = int(self.seg_next_id)
                self.seg_next_id += 1
            self.last_segment_fuse_created = True
            self.last_segment_assoc_info = {"sid": int(sid), "source": str(assoc_source or "new"), "score": 1.0}
            self.seg_sum[sid] = (conf_eff * obs["embedding"]).astype(np.float32)
            self.seg_w[sid] = float(np.clip(conf_eff, 0.0, w_max))
            self.seg_first_seen[sid] = now_s
            self.seg_last_seen[sid] = now_s
            self.seg_obs_count[sid] = 1
            self.seg_last_obs_seq[sid] = int(obs_seq)
            self.seg_viewmask[sid] = 0
            self.seg_centroid[sid] = obs["centroid"].astype(np.float32)
            self.seg_bbox_min[sid] = obs["bbox_min"].astype(np.float32)
            self.seg_bbox_max[sid] = obs["bbox_max"].astype(np.float32)
            if obs.get("dino_embedding") is not None:
                dino_gain = max(
                    1e-4,
                    float(self.get_parameter("dinov2_track_update_gain").value),
                )
                dino_w = float(
                    conf_eff
                    * dino_gain
                    * np.clip(float(obs.get("dino_quality", 1.0)), 0.0, 1.0)
                )
                self._update_segment_dino_memory(
                    sid, obs.get("dino_embedding"), dino_w,
                    obs_seq if float(obs.get("dino_quality", 1.0)) >= float(self.get_parameter("dinov2_assoc_min_quality").value) else None
                )
            self._claim_segment_voxels(sid, set(obs.get("support_voxels", set())))
            self._set_segment_support(sid, set(obs.get("support_voxels", set())), set(obs.get("support_cells", set())))
            self.update_segment_viewmask(sid, cam_xy_int, (float(obs["centroid"][0]), float(obs["centroid"][1])))
            self._mark_state_dirty()
            return sid

        old_emb, _, _ = self.segment_embedding_and_coherence(int(sid))
        if old_emb is not None and bool(self.get_parameter("use_consistency_gating").value):
            cos = float(np.dot(old_emb, obs["embedding"]))
            cos_min = float(self.get_parameter("gating_cos_min").value)
            if cos < cos_min:
                if bool(self.get_parameter("gating_reject").value):
                    return sid
                conf_eff *= self._soft_gating_multiplier(cos, cos_min)
        if conf_eff <= 0.0:
            return sid

        old_w = float(self.seg_w[int(sid)])
        alpha = float(conf_eff / max(old_w + conf_eff, 1e-6))
        total_w = old_w + conf_eff
        stored_w = float(np.clip(total_w, 0.0, w_max))
        old_sum = self.seg_sum[int(sid)]
        old_norm = float(np.linalg.norm(old_sum))
        if old_norm > old_w and old_norm > 1e-6:
            old_sum = old_sum * (old_w/old_norm)
        self.seg_sum[int(sid)] = ((old_sum + conf_eff * obs["embedding"]) * (stored_w/max(total_w, 1e-6))).astype(np.float32)
        self.seg_w[int(sid)] = stored_w
        self.seg_last_seen[int(sid)] = now_s
        if int(obs_seq) < 0 or int(self.seg_last_obs_seq.get(int(sid), -999999999)) != int(obs_seq):
            self.seg_obs_count[int(sid)] = int(self.seg_obs_count.get(int(sid), 1)) + 1
            self.seg_last_obs_seq[int(sid)] = int(obs_seq)
        self.seg_centroid[int(sid)] = (
            (1.0 - alpha) * self.seg_centroid[int(sid)] + alpha * obs["centroid"]
        ).astype(np.float32)
        alpha_box = max(alpha, float(np.clip(float(self.get_parameter("segment_update_alpha_min").value), 0.0, 1.0)))
        self.seg_bbox_min[int(sid)] = (
            (1.0 - alpha_box) * self.seg_bbox_min[int(sid)] + alpha_box * obs["bbox_min"]
        ).astype(np.float32)
        self.seg_bbox_max[int(sid)] = (
            (1.0 - alpha_box) * self.seg_bbox_max[int(sid)] + alpha_box * obs["bbox_max"]
        ).astype(np.float32)
        obs_dino = obs.get("dino_embedding")
        if obs_dino is not None:
            dino_gain = max(
                1e-4,
                float(self.get_parameter("dinov2_track_update_gain").value),
            )
            dino_weight = float(
                conf_eff
                * dino_gain
                * np.clip(float(obs.get("dino_quality", 1.0)), 0.0, 1.0)
            )
            self._update_segment_dino_memory(int(sid), obs_dino, dino_weight,
                obs_seq if float(obs.get("dino_quality", 1.0)) >= float(self.get_parameter("dinov2_assoc_min_quality").value) else None)

        obs_voxels = set(obs.get("support_voxels", set()))
        old_voxels = {key for key in self.seg_support_voxels.get(int(sid), set()) if self.voxel_segment_id.get(key) == int(sid)}
        self._claim_segment_voxels(int(sid), obs_voxels)
        max_voxels = int(self.get_parameter("segment_support_max_voxels").value)
        merged_voxels = self._bounded_support_union(old_voxels, obs_voxels, max_voxels)
        self._set_segment_support(int(sid), merged_voxels, set(obs.get("support_cells", set())))
        self.update_segment_viewmask(int(sid), cam_xy_int, (float(self.seg_centroid[int(sid)][0]), float(self.seg_centroid[int(sid)][1])))
        self._mark_state_dirty()
        return int(sid)

    def _segment_metrics(self, sid: int):
        emb_u, coh, w = self.segment_embedding_and_coherence(int(sid))
        if emb_u is None:
            return None
        coh01 = self._coherence01(float(coh))
        score = 0.0
        if self.text_emb is not None:
            score = max(0.0, float(np.dot(emb_u, self.text_emb)))
            if bool(self.get_parameter("use_coherence_in_semantic").value):
                score *= coh01
        conf_full = float(self.get_parameter("segment_conf_weight_full").value)
        if conf_full <= 0.0:
            conf_full = float(self.get_parameter("segment_weight_max").value)
        conf = float(np.clip(float(w) / max(conf_full, 1e-6), 0.0, 1.0))
        if bool(self.get_parameter("use_coherence_in_confidence").value):
            conf *= coh01
        if bool(self.get_parameter("use_visibility_boost").value):
            bins = int(self.get_parameter("view_bins").value)
            if bins > 1:
                nb = safe_popcount(int(self.seg_viewmask.get(int(sid), 0)))
                frac = float(nb) / float(bins)
                boost = 1.0 + float(self.get_parameter("view_boost_strength").value) * (
                    frac ** float(self.get_parameter("view_boost_power").value)
                )
                conf = float(np.clip(conf * boost, 0.0, 1.0))
        return emb_u, float(score), float(conf), float(coh01), float(w)

    def _log_segment_track_debug(
        self,
        attempted: int,
        built: int,
        new: int,
        matched: int,
        projected_matched: int,
        fallback_matched: int,
        projected_owned_points: int,
        unique_frame_sids: int,
        rejects: dict,
        obs_voxel_counts: List[int],
    ):
        if not bool(self.get_parameter("segment_track_debug_log").value):
            return
        now_s = self.now_sec()
        interval = float(self.get_parameter("segment_track_debug_interval_s").value)
        if interval > 0.0 and (now_s - self.last_segment_track_debug_log_s) < interval:
            return
        self.last_segment_track_debug_log_s = now_s
        top = []
        for sid in self.seg_w.keys():
            vox_n = len(self.seg_support_voxels.get(int(sid), set()))
            obs_n = int(self.seg_obs_count.get(int(sid), 0))
            w = float(self.seg_w.get(int(sid), 0.0))
            top.append((vox_n, int(sid), obs_n, w))
        top.sort(reverse=True)
        top_txt = ", ".join(
            f"id={sid}:vox={vox_n}:obs={obs_n}:w={w:.1f}" for vox_n, sid, obs_n, w in top[:6]
        )
        reject_txt = ",".join(f"{k}:{v}" for k, v in sorted(rejects.items())) if rejects else "none"
        if obs_voxel_counts:
            obs_vox_txt = f"obs_vox_avg={float(np.mean(obs_voxel_counts)):.1f} obs_vox_max={int(max(obs_voxel_counts))}"
        else:
            obs_vox_txt = "obs_vox_avg=0.0 obs_vox_max=0"
        self.get_logger().info(
            "segments: "
            f"segments={len(self.seg_w)} "
            f"attempted={attempted} "
            f"built={built} "
            f"new={new} "
            f"matched={matched} "
            f"projected={projected_matched} "
            f"fallback={fallback_matched} "
            f"projected_pts={projected_owned_points} "
            f"unique_frame={unique_frame_sids} "
            f"{obs_vox_txt} "
            f"rejects={reject_txt} "
            f"top=[{top_txt}]"
        )

    # -------------------------
    # Coherence shaping
    # -------------------------
    def _coherence01(self, coh: float) -> float:
        coh_floor = float(self.get_parameter("coherence_floor").value)
        coh_pow = float(self.get_parameter("coherence_power").value)
        coh01 = float(np.clip((coh - coh_floor) / max(1.0 - coh_floor, 1e-6), 0.0, 1.0))
        return float(np.power(coh01, coh_pow))

    # -------------------------
    # Image decoding
    # -------------------------
    def decode_rgb_msg(self, rgb_msg):
        if isinstance(rgb_msg, CompressedImage):
            buf = np.frombuffer(rgb_msg.data, dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("cv2.imdecode returned None for compressed RGB image")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")

    def _decode_compressed_depth_png(self, depth_msg: CompressedImage):
        data = np.frombuffer(depth_msg.data, dtype=np.uint8)
        # compressedDepth transport prepends a small config header before PNG data.
        for offset in (0, 12, 16):
            if data.size <= offset:
                continue
            depth = cv2.imdecode(data[offset:], cv2.IMREAD_UNCHANGED)
            if depth is not None and depth.ndim == 2:
                return depth
        return None

    def decode_depth_msg(self, depth_msg):
        if isinstance(depth_msg, CompressedImage):
            fmt = str(depth_msg.format)
            if "32FC1" in fmt:
                # compressedDepth encodes float depth as quantized inverse
                # depth. Never silently interpret that uint16 PNG as mm.
                raise RuntimeError("32FC1 compressedDepth is not supported; use 16UC1 compressedDepth or raw float depth")
            depth = self._decode_compressed_depth_png(depth_msg)
            if depth is None:
                raise RuntimeError("Could not decode compressed depth image")
            if depth.dtype == np.uint16 or "16UC1" in fmt:
                return depth.astype(np.float32) * 0.001
            if depth.dtype == np.float32 or "32FC1" in fmt:
                # If the compressedDepth header was decoded away by the transport, this is already meters.
                return depth.astype(np.float32)
            return depth.astype(np.float32)

        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth = np.asarray(depth)
        if depth_msg.encoding == "16UC1" or depth.dtype == np.uint16:
            return depth.astype(np.float32) * 0.001
        return depth.astype(np.float32)

    # -------------------------
    # Depth->points in INTERNAL frame
    # -------------------------
    def depth_to_points_internal(self, depth_msg):
        try:
            t0 = time.perf_counter()
            depth_m = self.decode_depth_msg(depth_msg)
            self._profile_add_elapsed("cb.depth_decode", t0)
        except Exception as e:
            self.get_logger().warn(f"Depth decode failed: {e}")
            self.last_depth_rejection = f"depth decode failed: {e}"
            return None

        if depth_m.ndim != 2:
            return None

        t0 = time.perf_counter()
        h, w = depth_m.shape
        v = np.arange(0, h, self.stride)
        u = np.arange(0, w, self.stride)
        uu, vv = np.meshgrid(u, v)

        z = depth_m[vv, uu]
        valid = np.isfinite(z) & (z > self.min_depth_m) & (z < self.max_depth_m)

        uu = uu[valid].astype(np.float32)
        vv = vv[valid].astype(np.float32)
        z = z[valid].astype(np.float32)

        if z.size < 80:
            self.last_depth_rejection = f"insufficient valid depth: {z.size} samples (need 80)"
            return None

        x = (uu - self.cx) * z / self.fx
        y = (vv - self.cy) * z / self.fy
        pts_cam = np.stack([x, y, z], axis=1)
        self._profile_add_elapsed("cb.depth_points_math", t0)

        source_frame = depth_msg.header.frame_id
        if not source_frame:
            return None

        t0 = time.perf_counter()
        tf_timeout = max(0.0, float(self.get_parameter("depth_tf_timeout_s").value))
        use_latest = bool(self.get_parameter("depth_use_latest_tf").value)
        stamp = Time() if use_latest else Time.from_msg(depth_msg.header.stamp)
        try:
            tf = self.tf_buffer.lookup_transform(
                self.internal_frame,
                source_frame,
                stamp,
                timeout=Duration(seconds=tf_timeout),
            )
        except Exception as e:
            if use_latest or not bool(self.get_parameter("depth_allow_latest_tf_fallback").value):
                self.last_depth_rejection = f"depth TF unavailable: {e}"
                self.get_logger().warn(f"TF lookup failed {self.internal_frame} <- {source_frame}: {e}")
                self._profile_add_elapsed("cb.depth_tf_lookup", t0)
                return None
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.internal_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=tf_timeout),
                )
            except Exception as e2:
                self.last_depth_rejection = f"depth TF unavailable: {e2}"
                self.get_logger().warn(
                    f"TF lookup failed {self.internal_frame} <- {source_frame}: {e}; latest fallback: {e2}"
                )
                self._profile_add_elapsed("cb.depth_tf_lookup", t0)
                return None
        self._profile_add_elapsed("cb.depth_tf_lookup", t0)

        t0 = time.perf_counter()
        T = transform_to_matrix(tf)
        self.last_depth_transform = T.copy()
        pts_h = np.ones((pts_cam.shape[0], 4), dtype=np.float32)
        pts_h[:, :3] = pts_cam.astype(np.float32)
        pts_out = (T @ pts_h.T).T[:, :3]
        cam_pos = T[:3, 3].copy()
        self._profile_add_elapsed("cb.depth_transform_points", t0)

        return pts_out, cam_pos, uu, vv, z, (h, w)

    # -----------------------------
    # Ray-consistency check (publish_frame occupancy grid)
    # -----------------------------
    def ray_is_occluded(self, cam_xy_map, pt_xy_map):
        if self.occ is None:
            return False
        if self.occ["frame_id"] != self.publish_frame:
            return False

        data = self.occ["data"]
        w = self.occ["w"]
        h = self.occ["h"]
        res = self.occ["res"]
        ox = self.occ["ox"]
        oy = self.occ["oy"]
        occ_thresh = int(self.get_parameter("occ_thresh").value)

        def to_ij(x, y):
            j = int((x - ox) / res)
            i = int((y - oy) / res)
            return i, j

        i0, j0 = to_ij(cam_xy_map[0], cam_xy_map[1])
        i1, j1 = to_ij(pt_xy_map[0], pt_xy_map[1])

        if not (0 <= i0 < h and 0 <= j0 < w and 0 <= i1 < h and 0 <= j1 < w):
            return False

        first = True
        for (j, i) in bresenham_2d(j0, i0, j1, i1):
            if first:
                first = False
                continue
            if (j == j1 and i == i1):
                break
            occ = int(data[i, j])
            if occ >= occ_thresh:
                return True
        return False

    # -------------------------
    # INTERNAL->PUBLISH transform
    # -------------------------
    def lookup_T_publish_internal(self, stamp_msg=None):
        try:
            timeout_s = max(0.0, float(self.get_parameter("publish_tf_timeout_s").value))
            if stamp_msg is None:
                tf = self.tf_buffer.lookup_transform(
                    self.publish_frame, self.internal_frame, Time(), timeout=Duration(seconds=timeout_s)
                )
            else:
                tf = self.tf_buffer.lookup_transform(
                    self.publish_frame,
                    self.internal_frame,
                    Time.from_msg(stamp_msg),
                    timeout=Duration(seconds=timeout_s),
                )
            T = transform_to_matrix(tf)
            self.last_T_publish_internal = T.copy()
            return T
        except Exception:
            return None

    # -------------------------
    # History cloud (INTERNAL voxel downsample; publish warped)
    # -------------------------
    def update_history_cloud(self, pts_internal: np.ndarray):
        voxel = float(self.get_parameter("history_voxel_m").value)
        if voxel <= 0.0:
            return
        max_pts = int(self.get_parameter("history_max_points").value)
        if len(self.history_vox) >= max_pts:
            return
        inv = 1.0 / voxel
        added = False
        for p in pts_internal:
            vx = int(np.floor(p[0] * inv))
            vy = int(np.floor(p[1] * inv))
            vz = int(np.floor(p[2] * inv))
            k = (vx, vy, vz)
            if k in self.history_vox:
                continue
            self.history_vox[k] = (float(p[0]), float(p[1]), float(p[2]), self.white_rgb)
            added = True
            if len(self.history_vox) >= max_pts:
                break
        if added:
            self._mark_state_dirty()

    def publish_history_cloud(self):
        if not bool(self.get_parameter("publish_history_cloud").value):
            return
        if len(self.history_vox) < 10:
            return
        T = self.lookup_T_publish_internal()
        if T is None:
            return
        arr = np.array(list(self.history_vox.values()), dtype=np.float32)  # Nx4
        pts = arr[:, :3]
        rgb = arr[:, 3]
        pts_h = np.ones((pts.shape[0], 4), dtype=np.float32)
        pts_h[:, :3] = pts
        pts_pub = (T @ pts_h.T).T[:, :3]
        msg = make_cloud_xyzrgb(pts_pub, rgb, self.publish_frame, self.get_clock().now().to_msg())
        self.pub_hist_cloud.publish(msg)

    # -------------------------
    # Async semantic worker
    # -------------------------
    def start_async_semantic_worker(self):
        if self.async_worker_thread is not None and self.async_worker_thread.is_alive():
            return
        self.async_stop_event.clear()
        self.async_worker_thread = threading.Thread(
            target=self.semantic_worker_loop,
            name="crossmaps_semantic_worker",
            daemon=True,
        )
        self.async_worker_thread.start()
        apply_hz = float(self.get_parameter("async_semantic_apply_hz").value)
        if apply_hz > 0.0 and self.async_apply_timer is None:
            self.async_apply_timer = self.create_timer(1.0 / apply_hz, self.on_async_apply_timer)
        self.get_logger().info("Async semantic worker started.")

    def stop_async_semantic_worker(self):
        self.async_stop_event.set()
        if self.async_worker_thread is None:
            return
        try:
            self.semantic_job_queue.put_nowait(None)
        except queue.Full:
            try:
                self.semantic_job_queue.get_nowait()
                self.semantic_job_queue.put_nowait(None)
            except Exception:
                pass
        self.async_worker_thread.join(timeout=2.0)
        self.async_worker_thread = None

    def on_async_apply_timer(self):
        t0 = time.perf_counter()
        self.apply_finished_semantic_jobs()
        self._profile_add_elapsed("async.timer_apply", t0)
        self._profile_report_if_due()

    def _copy_header(self, header_msg: Header) -> Header:
        return Header(stamp=header_msg.stamp, frame_id=header_msg.frame_id)

    def submit_async_semantic_job(
        self,
        rgb_cv: np.ndarray,
        rgb_header: Header,
        depth_header: Header,
        pts_int: np.ndarray,
        cam_int: np.ndarray,
        uu: np.ndarray,
        vv: np.ndarray,
        z_cam: np.ndarray,
        image_shape: Tuple[int, int],
        cycle_start_perf_s: Optional[float] = None,
    ) -> bool:
        self.async_job_seq += 1
        job = {
            "seq": int(self.async_job_seq),
            "queued_perf_s": time.perf_counter(),
            "cycle_start_perf_s": (
                time.perf_counter()
                if cycle_start_perf_s is None
                else float(cycle_start_perf_s)
            ),
            "obs_time_s": self.now_sec(),
            "rgb_cv": rgb_cv,
            "rgb_header": self._copy_header(rgb_header),
            "depth_header": self._copy_header(depth_header),
            "pts_int": pts_int,
            "cam_int": cam_int,
            "uu": uu,
            "vv": vv,
            "z_cam": z_cam,
            "image_shape": image_shape,
        }
        try:
            self.semantic_job_queue.put_nowait(job)
            self.async_jobs_submitted += 1
            return True
        except queue.Full:
            try:
                self.semantic_job_queue.get_nowait()
                self.async_jobs_dropped += 1
                self._profile_add_value("async.dropped_jobs", 1)
            except queue.Empty:
                pass
            try:
                self.semantic_job_queue.put_nowait(job)
                self.async_jobs_submitted += 1
                return True
            except queue.Full:
                self.async_jobs_dropped += 1
                self._profile_add_value("async.dropped_jobs", 1)
                return False

    def _enqueue_async_result(self, result: dict):
        while not self.async_stop_event.is_set():
            try:
                self.semantic_result_queue.put_nowait(result)
                return
            except queue.Full:
                try:
                    self.semantic_result_queue.get_nowait()
                    self.async_results_dropped += 1
                except queue.Empty:
                    return

    def semantic_worker_loop(self):
        while not self.async_stop_event.is_set():
            try:
                job = self.semantic_job_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if job is None:
                break

            self.async_worker_busy.set()
            self.pipeline_health.working("eSAM", job["seq"])
            result = dict(job)
            timings_ms = {}
            worker_t0 = time.perf_counter()
            try:
                t0 = time.perf_counter()
                masks, mask_scores = self.segment_esam_masks(job["rgb_cv"])
                timings_ms["async.esam_segment"] = (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()
                try:
                    self.pipeline_health.working("DINO", job["seq"])
                    dense_dino = self.compute_dinov2_dense_features(job["rgb_cv"])
                except Exception as dino_error:
                    dense_dino = None
                    self.dinov2_model = None
                    if not self.dinov2_unavailable_warned:
                        self.get_logger().warn(
                            "DINOv2 inference failed; disabling appearance features "
                            f"and using geometry/CLIP: {dino_error}"
                        )
                        self.dinov2_unavailable_warned = True
                timings_ms["async.dinov2_dense"] = (time.perf_counter() - t0) * 1000.0

                if masks:
                    self.pipeline_health.working("DINO pooling", job["seq"])
                    t0 = time.perf_counter()
                    dino_mask_emb, dino_mask_quality = self.compute_dinov2_mask_embeddings(
                        masks, dense_dino, return_quality=True
                    )
                    timings_ms["async.dinov2_mask_pool"] = (time.perf_counter() - t0) * 1000.0
                    dense_clip = None
                    self.pipeline_health.working("CLIP", job["seq"])
                    if bool(self.get_parameter("use_clip_dense_features").value):
                        t0 = time.perf_counter()
                        dense_clip = self.compute_dense_clip_patch_features(job["rgb_cv"])
                        timings_ms["async.clip_dense_embed"] = (time.perf_counter() - t0) * 1000.0
                    t0 = time.perf_counter()
                    mask_emb = self.compute_mask_embeddings_batched(job["rgb_cv"], masks, dense_clip=dense_clip)
                    timings_ms["async.clip_mask_embed"] = (time.perf_counter() - t0) * 1000.0
                else:
                    dense_clip = None
                    dino_mask_emb = np.zeros((0, 0), dtype=np.float32)
                    dino_mask_quality = np.zeros((0,), dtype=np.float32)
                    mask_emb = np.zeros((0, 0), dtype=np.float32)

                result.update(
                    {
                        "masks": masks,
                        "mask_scores": mask_scores,
                        "mask_emb": mask_emb,
                        "dino_mask_emb": dino_mask_emb,
                        "dino_mask_quality": dino_mask_quality,
                        "dense_clip": dense_clip,
                        "dense_dino": dense_dino,
                        "error": None,
                    }
                )
            except Exception as e:
                result.update(
                    {
                        "masks": [],
                        "mask_scores": np.zeros((0,), dtype=np.float32),
                        "mask_emb": np.zeros((0, 0), dtype=np.float32),
                        "dino_mask_emb": np.zeros((0, 0), dtype=np.float32),
                        "dino_mask_quality": np.zeros((0,), dtype=np.float32),
                        "dense_clip": None,
                        "dense_dino": None,
                        "error": str(e),
                    }
                )
            result["ready_perf_s"] = time.perf_counter()
            timings_ms["async.worker_total"] = (result["ready_perf_s"] - worker_t0) * 1000.0
            result["timings_ms"] = timings_ms
            self.pipeline_health.working("awaiting fusion", job["seq"])
            self._enqueue_async_result(result)
            self.async_worker_busy.clear()

    def apply_finished_semantic_jobs(self):
        if not bool(self.get_parameter("use_async_semantic_worker").value):
            return
        max_results = max(1, int(self.get_parameter("async_semantic_max_results_per_publish").value))
        applied = 0
        attempted = 0
        while attempted < max_results:
            try:
                result = self.semantic_result_queue.get_nowait()
            except queue.Empty:
                break
            attempted += 1

            for name, value_ms in result.get("timings_ms", {}).items():
                self._profile_add_ms(name, float(value_ms))

            now_perf = time.perf_counter()
            queued_perf = result.get("queued_perf_s")
            ready_perf = result.get("ready_perf_s")
            if queued_perf is not None:
                self._profile_add_ms("async.end_to_end_latency", (now_perf - float(queued_perf)) * 1000.0)
            if ready_perf is not None:
                self._profile_add_ms("async.ready_to_apply_latency", (now_perf - float(ready_perf)) * 1000.0)

            if result.get("error"):
                self.pipeline_health.idle(result['error'])
                self._profile_add_value("async.worker_errors", 1)
                self.get_logger().warn(f"Async semantic worker failed: {result['error']}")
                continue

            t0 = time.perf_counter()
            self.pipeline_health.working("fusion", result.get("seq"))
            try:
                accepted = self.apply_semantic_result(result, "async")
            except Exception as error:
                self.pipeline_health.idle(f"Fusion failed: {error}")
                self.get_logger().error(f"Semantic result {result.get('seq')} failed during fusion: {error}")
                continue
            self._profile_add_elapsed("async.apply_total", t0)
            if not accepted:
                self.pipeline_health.reject("duplicate, old or invalid semantic result")
                self.pipeline_health.idle()
                continue
            self._record_observation_cycle(
                result.get("cycle_start_perf_s", result.get("queued_perf_s")),
                "async",
                int(result.get("seq", -1)),
            )
            applied += 1

        if applied > 0:
            self._profile_add_value("async.results_applied", applied)
        if self.async_results_dropped > 0:
            self._profile_add_value("async.dropped_results", self.async_results_dropped)
            self.async_results_dropped = 0

    def apply_semantic_result(self, result: dict, profile_prefix: str):
        obs_seq = int(result.get("seq", -1))
        if obs_seq <= self.last_applied_observation_seq:
            self._profile_add_value(f"{profile_prefix}.duplicate_or_old_observations", 1)
            return False
        depth_header = result["depth_header"]
        stamp = depth_header.stamp
        source_stamp_ns = int(getattr(stamp, "sec", 0)) * 1000000000 + int(getattr(stamp, "nanosec", 0))
        if source_stamp_ns > 0 and source_stamp_ns <= self.last_accepted_depth_stamp_ns:
            self._profile_add_value(f"{profile_prefix}.duplicate_or_old_sensor_stamps", 1)
            return False  # Do not clear camera/attribution state for rejected input.
        self.last_segment_projection_context = None
        masks = result.get("masks", [])
        mask_scores = result.get("mask_scores", np.zeros((0,), dtype=np.float32))
        mask_emb = result.get("mask_emb", np.zeros((0, 0), dtype=np.float32))
        dino_mask_emb = result.get("dino_mask_emb", np.zeros((0, 0), dtype=np.float32))
        dino_mask_quality = np.asarray(
            result.get("dino_mask_quality", np.zeros((0,), dtype=np.float32)),
            dtype=np.float32,
        )
        dense_dino = result.get("dense_dino", None)
        self.current_assignment_dino = dense_dino
        rgb_cv = result["rgb_cv"]
        self.last_assignment_rgb = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)
        self.last_mask_attribution = []
        self.last_visible_segment_counts = {}
        rgb_header = result["rgb_header"]
        if dense_dino is not None:
            features = np.asarray(dense_dino.get("features", []))
            if features.ndim == 3:
                self.dino_base_dim = int(features.shape[-1])
        pts_int = result["pts_int"]
        cam_int = result["cam_int"]
        uu = result["uu"]
        vv = result["vv"]
        z_cam = result["z_cam"]
        h, w = result["image_shape"]

        self._profile_add_value(f"{profile_prefix}.masks", len(masks))
        t0 = time.perf_counter()
        self.publish_esam_mask_overlay(rgb_cv, masks, rgb_header)
        self._profile_add_elapsed(f"{profile_prefix}.mask_overlay", t0)
        if not masks:
            t0 = time.perf_counter()
            self.publish_feature_diagnostics(
                rgb_cv,
                [],
                np.zeros((0, 0), dtype=np.float32),
                np.zeros((0, 0), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                dense_dino,
                rgb_header,
            )
            self._profile_add_elapsed(f"{profile_prefix}.feature_diagnostics", t0)
            ctx = self.build_projected_segment_id_context(pts_int, np.asarray(uu, dtype=np.int32),
                                                          np.asarray(vv, dtype=np.int32), (h, w))
            ctx = ctx or {"id_image": np.zeros((int(h), int(w)), dtype=np.int32)}
            ctx.update(masks=[], decisions=[], rgb=self.last_assignment_rgb)
            if hasattr(self, "surface_consensus"):
                self.surface_consensus.commit(obs_seq, float(result.get("obs_time_s", self.now_sec())), [], [], set(self.seg_w))
                self.boundary_votes.update(obs_seq, [], {s: len(v) for s,v in self.seg_support_voxels.items()})
            self.last_visible_segment_counts = dict(ctx.get("visible_counts", {}))
            for sid in self.last_visible_segment_counts:
                if sid in self.seg_w:
                    self.seg_last_seen[sid] = max(self.seg_last_seen.get(sid, 0.0),
                                                  float(result.get("obs_time_s", self.now_sec())))
            self.publish_segment_id_projection(ctx, rgb_header)
            self.last_applied_observation_seq = obs_seq
            self.last_accepted_depth_stamp_ns = max(self.last_accepted_depth_stamp_ns, source_stamp_ns)
            self._update_identity_graph(obs_seq, float(result.get("obs_time_s", self.now_sec())), cam_int)
            return True

        if mask_emb.shape[0] != len(masks):
            self.get_logger().warn("Mask embedding count did not match mask count; skipping frame.")
            self.publish_segment_id_projection({"id_image": np.zeros((int(h), int(w)), dtype=np.int32)}, rgb_header)
            return False
        if dino_mask_emb.ndim != 2 or dino_mask_emb.shape[0] != len(masks):
            dino_mask_emb = np.zeros((len(masks), 0), dtype=np.float32)
        if dino_mask_quality.ndim != 1 or dino_mask_quality.shape[0] != len(masks):
            dino_mask_quality = np.where(
                np.linalg.norm(dino_mask_emb, axis=1) > 1e-6,
                1.0,
                0.0,
            ).astype(np.float32)
        valid_dino_quality = dino_mask_quality[
            np.linalg.norm(dino_mask_emb, axis=1) > 1e-6
        ]
        if valid_dino_quality.size > 0:
            self._profile_add_value(
                f"{profile_prefix}.dinov2_quality_mean",
                float(np.mean(valid_dino_quality)),
            )
            self._profile_add_value(
                f"{profile_prefix}.dinov2_low_quality_masks",
                int(
                    np.count_nonzero(
                        valid_dino_quality
                        < float(self.get_parameter("dinov2_assoc_min_quality").value)
                    )
                ),
            )
        t0 = time.perf_counter()
        self.publish_feature_diagnostics(
            rgb_cv,
            masks,
            mask_emb,
            dino_mask_emb,
            dino_mask_quality,
            dense_dino,
            rgb_header,
        )
        self._profile_add_elapsed(f"{profile_prefix}.feature_diagnostics", t0)

        # point confidence
        t0 = time.perf_counter()
        d0 = float(self.get_parameter("conf_d0_m").value)
        conf_min = float(self.get_parameter("conf_min").value)
        conf_dist = np.exp(-np.clip(z_cam, 0.0, 50.0) / max(d0, 1e-3)).astype(np.float32)
        conf_scale = float(self.get_parameter("conf_fine_boost").value)
        self._profile_add_elapsed(f"{profile_prefix}.confidence_prepare", t0)

        # Ray consistency needs map coords
        t0 = time.perf_counter()
        use_ray = bool(self.get_parameter("use_ray_consistency").value)
        ray_stride = int(self.get_parameter("ray_check_stride").value)
        ray_stamp = None if bool(self.get_parameter("ray_use_latest_tf").value) else depth_header.stamp
        T_pub_int = self.lookup_T_publish_internal(ray_stamp) if use_ray else None
        self._profile_add_elapsed(f"{profile_prefix}.ray_tf_lookup", t0)

        # fuse into STM (INTERNAL grid)
        now_s = float(result.get("obs_time_s", self.now_sec()))
        obs_seq = int(result.get("seq", -1))
        cam_xy_int = (float(cam_int[0]), float(cam_int[1]))
        uu_i = np.clip(uu.astype(np.int32), 0, w - 1)
        vv_i = np.clip(vv.astype(np.int32), 0, h - 1)
        dense_point_emb = None
        dense_clip = result.get("dense_clip", None)
        if bool(self.get_parameter("clip_dense_point_fusion").value) and dense_clip is not None:
            dense_point_emb = self.sample_dense_clip_embeddings_for_pixels(dense_clip, uu, vv)
            if dense_point_emb is not None and dense_point_emb.shape[0] != pts_int.shape[0]:
                dense_point_emb = None
        min_depth_points = int(self.get_parameter("esam_min_depth_points").value)
        use_voxels = bool(self.get_parameter("use_semantic_voxels").value)
        voxel_fuse_stride = max(1, int(self.get_parameter("voxel_fuse_stride").value))
        sem_cloud_pts = []
        sem_cloud_embs = []
        accepted_masks = 0
        processed_mask_points = 0
        fused_cells = 0
        fused_voxels = 0
        ray_checks = 0
        ray_rejects = 0
        use_segments = bool(self.get_parameter("use_segment_memory").value)
        segment_obs_attempted = 0
        segment_obs_built = 0
        segment_obs_new = 0
        segment_obs_matched = 0
        segment_obs_rejects = {}
        segment_obs_voxel_counts = []
        segment_frame_sids = set()
        segment_one_to_one = bool(self.get_parameter("segment_frame_one_to_one_assoc").value)
        segment_projected_ctx = None
        segment_projected_owned_points = 0
        segment_projected_matched = 0
        segment_fallback_matched = 0
        segment_frame_unique_count = 0
        segment_semantic_conflict_carves = 0
        segment_semantic_conflict_voxels = 0
        segment_semantic_conflict_ready_voxels = 0
        use_frame_assignment = use_segments and self._use_frame_segment_assignment()
        exclusive_point_mask_indices = None
        if not use_segments:
            self.publish_segment_id_projection({"id_image": np.zeros((int(h), int(w)), dtype=np.int32)}, rgb_header)
        if use_segments:
            t_proj = time.perf_counter()
            segment_projected_ctx = self.build_projected_segment_id_context(pts_int, uu_i, vv_i, (h, w))
            if segment_projected_ctx is not None:
                segment_projected_owned_points = int(segment_projected_ctx.get("owned_points", 0))
                if not use_frame_assignment:
                    self.publish_segment_id_projection(segment_projected_ctx, rgb_header)
            elif not use_frame_assignment:
                self.publish_segment_id_projection({"id_image": np.zeros((int(h), int(w)), dtype=np.int32)}, rgb_header)
            self._profile_add_elapsed(f"{profile_prefix}.segment_id_projection", t_proj)
            self._profile_add_value(f"{profile_prefix}.segment_projected_owned_points", segment_projected_owned_points)
            if use_frame_assignment:
                t_assign = time.perf_counter()
                assignment_stats = self.assign_frame_segments_fine_image(
                    masks,
                    mask_emb,
                    mask_scores,
                    dino_mask_emb,
                    dino_mask_quality,
                    pts_int,
                    uu_i,
                    vv_i,
                    (h, w),
                    conf_dist,
                    conf_scale,
                    conf_min,
                    now_s,
                    cam_xy_int,
                    obs_seq,
                    segment_projected_ctx,
                    rgb_header,
                )
                self._profile_add_elapsed(f"{profile_prefix}.segment_frame_assignment", t_assign)
                segment_obs_attempted = int(assignment_stats.get("attempted", 0))
                segment_obs_built = int(assignment_stats.get("built", 0))
                segment_obs_new = int(assignment_stats.get("new", 0))
                segment_obs_matched = int(assignment_stats.get("matched", 0))
                segment_projected_matched = int(assignment_stats.get("projected_matched", 0))
                segment_fallback_matched = int(assignment_stats.get("fallback_matched", 0))
                segment_projected_owned_points = int(assignment_stats.get("owned_points", segment_projected_owned_points))
                segment_frame_unique_count = int(assignment_stats.get("unique_frame_sids", 0))
                segment_obs_rejects = dict(assignment_stats.get("rejects", {}))
                segment_obs_voxel_counts = list(assignment_stats.get("obs_voxel_counts", []))
                segment_semantic_conflict_carves = int(assignment_stats.get("semantic_conflict_carves", 0))
                segment_semantic_conflict_voxels = int(assignment_stats.get("semantic_conflict_voxels", 0))
                segment_semantic_conflict_ready_voxels = int(assignment_stats.get("semantic_conflict_ready_voxels", 0))
                exclusive_point_mask_indices = assignment_stats.get("point_mask_indices")

        if bool(self.get_parameter("segment_exclusive_point_fusion").value):
            exclusive_point_mask_indices = self.exclusive_mask_assignment_for_points(
                masks,
                mask_scores,
                uu_i,
                vv_i,
                initial=exclusive_point_mask_indices,
            )

        t0 = time.perf_counter()
        for mask_idx, (mask, e, mask_score) in enumerate(zip(masks, mask_emb, mask_scores)):
            in_mask = mask[vv_i, uu_i]
            if (
                bool(self.get_parameter("segment_exclusive_point_fusion").value)
                and exclusive_point_mask_indices is not None
                and int(exclusive_point_mask_indices.shape[0]) == int(in_mask.shape[0])
            ):
                in_mask = np.logical_and(
                    in_mask,
                    np.asarray(exclusive_point_mask_indices, dtype=np.int32) == int(mask_idx),
                )
            point_idxs = np.flatnonzero(in_mask)
            if point_idxs.size < min_depth_points:
                continue
            accepted_masks += 1
            processed_mask_points += int(point_idxs.size)

            e = e.astype(np.float32)
            dino_e = (
                dino_mask_emb[int(mask_idx)].astype(np.float32)
                if dino_mask_emb.ndim == 2
                and int(mask_idx) < dino_mask_emb.shape[0]
                and dino_mask_emb.shape[1] > 0
                else None
            )
            dino_quality = (
                float(dino_mask_quality[int(mask_idx)])
                if int(mask_idx) < dino_mask_quality.shape[0]
                else 0.0
            )
            conf = np.clip(conf_dist[point_idxs] * conf_scale * float(mask_score), conf_min, 1.0).astype(np.float32)
            pts_mask = pts_int[point_idxs]
            point_embs = dense_point_emb[point_idxs] if dense_point_emb is not None else None

            if use_segments and use_frame_assignment:
                pass
            elif use_segments:
                segment_obs_attempted += 1
                seg_obs = self.build_segment_observation(
                    pts_mask,
                    e,
                    conf,
                    dino_embedding=dino_e,
                    dino_quality=dino_quality,
                )
                if seg_obs is not None:
                    segment_obs_built += 1
                    blocked = segment_frame_sids if segment_one_to_one else None
                    forced_sid, projected_info = self.find_projected_segment_match(
                        seg_obs,
                        segment_projected_ctx,
                        point_idxs,
                        blocked_sids=blocked,
                    )
                    if forced_sid is None and bool(self.get_parameter("segment_projected_id_reject_ambiguous_masks").value):
                        owned_pixels = int((projected_info or {}).get("owned_pixels", 0))
                        min_pixels = max(1, int(self.get_parameter("segment_projected_id_min_pixels").value))
                        if owned_pixels >= min_pixels:
                            segment_obs_rejects["ambiguous_projected"] = int(
                                segment_obs_rejects.get("ambiguous_projected", 0)
                            ) + 1
                            continue
                    if forced_sid is not None and bool(self.get_parameter("segment_projected_id_protect_other_owners").value):
                        filtered_obs = self.owner_filtered_segment_observation(
                            pts_mask,
                            conf,
                            e,
                            int(forced_sid),
                            segment_projected_ctx,
                            point_idxs,
                            dino_embedding=dino_e,
                            dino_quality=dino_quality,
                        )
                        if filtered_obs is None:
                            segment_obs_rejects["owner_filter_empty"] = int(
                                segment_obs_rejects.get("owner_filter_empty", 0)
                            ) + 1
                            continue
                        seg_obs = filtered_obs
                    segment_obs_voxel_counts.append(len(seg_obs.get("support_voxels", set())))
                    sid = self.fuse_segment_observation(
                        seg_obs,
                        now_s,
                        cam_xy_int,
                        obs_seq=obs_seq,
                        blocked_sids=blocked,
                        forced_sid=forced_sid,
                        assoc_source="projected_id" if forced_sid is not None else "fallback",
                        assoc_info=projected_info,
                    )
                    if sid is not None:
                        segment_frame_sids.add(int(sid))
                        if self.last_segment_fuse_created:
                            segment_obs_new += 1
                        else:
                            segment_obs_matched += 1
                            source = str((self.last_segment_assoc_info or {}).get("source", "fallback"))
                            if source == "projected_id":
                                segment_projected_matched += 1
                            else:
                                segment_fallback_matched += 1
                else:
                    reason = str(self._segment_obs_reject_reason or "unknown")
                    segment_obs_rejects[reason] = int(segment_obs_rejects.get(reason, 0)) + 1
            for local_idx, (pt, c) in enumerate(zip(pts_mask, conf)):
                e_point = point_embs[local_idx].astype(np.float32) if point_embs is not None else e
                x, y, z = pt
                cell = self.xy_to_cell(float(x), float(y))
                if cell is None:
                    continue

                # visibility bins (STM)
                cx_cell, cy_cell = self.cell_center_xy(cell[0], cell[1])
                self.update_viewmask(self.stm_viewmask, cell, cam_xy_int, (cx_cell, cy_cell))

                # ray consistency (sparse)
                if (
                    use_ray
                    and (ray_stride > 0)
                    and (((mask_idx * 1000003) + local_idx) % ray_stride == 0)
                    and (self.occ is not None)
                    and (T_pub_int is not None)
                ):
                    ray_checks += 1
                    cam_h = np.array([cam_int[0], cam_int[1], cam_int[2], 1.0], dtype=np.float32)
                    pt_h = np.array([x, y, z, 1.0], dtype=np.float32)
                    cam_pub = (T_pub_int @ cam_h)[:3]
                    pt_pub = (T_pub_int @ pt_h)[:3]
                    if self.ray_is_occluded((float(cam_pub[0]), float(cam_pub[1])), (float(pt_pub[0]), float(pt_pub[1]))):
                        ray_rejects += 1
                        continue

                self.fuse_stm_cell(cell, e_point, float(c), now_s, z=float(z))
                fused_cells += 1

                if use_voxels and (local_idx % voxel_fuse_stride == 0):
                    voxel = self.xyz_to_voxel(float(x), float(y), float(z))
                    if voxel is not None:
                        vx, vy, vz = self.voxel_center_xyz(voxel)
                        self.update_viewmask(self.vox_stm_viewmask, voxel, cam_xy_int, (vx, vy))
                        self.fuse_vox_stm_cell(voxel, e_point, float(c), now_s)
                        fused_voxels += 1

            if bool(self.get_parameter("publish_semantic_cloud").value):
                sem_cloud_pts.append(pts_mask)
                sem_cloud_embs.append(point_embs if point_embs is not None else e)
        self._profile_add_elapsed(f"{profile_prefix}.fusion_loop", t0)
        self._profile_add_value(f"{profile_prefix}.accepted_masks", accepted_masks)
        self._profile_add_value(f"{profile_prefix}.mask_points", processed_mask_points)
        self._profile_add_value(f"{profile_prefix}.fused_cells", fused_cells)
        self._profile_add_value(f"{profile_prefix}.fused_voxels", fused_voxels)
        self._profile_add_value(f"{profile_prefix}.ray_checks", ray_checks)
        self._profile_add_value(f"{profile_prefix}.ray_rejects", ray_rejects)
        if use_segments:
            self._profile_add_value(f"{profile_prefix}.segments_built", segment_obs_built)
            self._profile_add_value(f"{profile_prefix}.segments_new", segment_obs_new)
            self._profile_add_value(f"{profile_prefix}.segments_matched", segment_obs_matched)
            self._profile_add_value(f"{profile_prefix}.segments_projected_matched", segment_projected_matched)
            self._profile_add_value(f"{profile_prefix}.segments_fallback_matched", segment_fallback_matched)
            self._profile_add_value(f"{profile_prefix}.segments_semantic_conflict_carves", segment_semantic_conflict_carves)
            self._profile_add_value(f"{profile_prefix}.segments_semantic_conflict_voxels", segment_semantic_conflict_voxels)
            self._profile_add_value(f"{profile_prefix}.segments_semantic_conflict_ready_voxels", segment_semantic_conflict_ready_voxels)
            self._log_segment_track_debug(
                attempted=segment_obs_attempted,
                built=segment_obs_built,
                new=segment_obs_new,
                matched=segment_obs_matched,
                projected_matched=segment_projected_matched,
                fallback_matched=segment_fallback_matched,
                projected_owned_points=segment_projected_owned_points,
                unique_frame_sids=segment_frame_unique_count if use_frame_assignment else len(segment_frame_sids),
                rejects=segment_obs_rejects,
                obs_voxel_counts=segment_obs_voxel_counts,
            )

        # Maintenance and evidence updates belong to the accepted observation
        # cycle, not the display timer. This also includes their cost in Cycle.
        if use_segments:
            self.maintain_segments(now_s)
        self.last_applied_observation_seq = obs_seq
        self.last_accepted_depth_stamp_ns = max(self.last_accepted_depth_stamp_ns, source_stamp_ns)
        if use_segments:
            self._update_identity_graph(obs_seq, now_s, cam_int)

        # publish semantic cloud (publish frame)
        if bool(self.get_parameter("publish_semantic_cloud").value):
            t0 = time.perf_counter()
            self.ensure_query_embedding()
            if self.text_emb is None:
                self._profile_add_elapsed(f"{profile_prefix}.semantic_cloud", t0)
                return True
            T = self.lookup_T_publish_internal(depth_header.stamp)
            if T is not None and sem_cloud_pts:
                max_pts = int(self.get_parameter("semantic_cloud_max_points").value)
                gamma = float(self.get_parameter("semantic_cloud_gamma").value)
                pts_all = np.concatenate(sem_cloud_pts, axis=0)
                sims = []
                for pts_mask, e in zip(sem_cloud_pts, sem_cloud_embs):
                    e_arr = np.asarray(e, dtype=np.float32)
                    if e_arr.ndim == 2 and e_arr.shape[0] == pts_mask.shape[0]:
                        s = np.maximum(0.0, np.clip(e_arr @ self.text_emb, -1.0, 1.0)).astype(np.float32)
                        sims.append(s)
                    else:
                        s = max(0.0, float(np.clip(np.dot(e_arr.reshape((-1,)), self.text_emb), -1.0, 1.0)))
                        sims.append(np.full((pts_mask.shape[0],), s, dtype=np.float32))
                sim_pos = np.concatenate(sims, axis=0)
                N = pts_all.shape[0]
                if N > max_pts:
                    idxs = np.random.choice(N, size=max_pts, replace=False)
                    pts_sel = pts_all[idxs]
                    s_sel = sim_pos[idxs]
                else:
                    pts_sel = pts_all
                    s_sel = sim_pos

                pts_h = np.ones((pts_sel.shape[0], 4), dtype=np.float32)
                pts_h[:, :3] = pts_sel.astype(np.float32)
                pts_pub = (T @ pts_h.T).T[:, :3]

                s01 = np.power(np.clip(s_sel, 0.0, 1.0), gamma)
                rgb = np.zeros((pts_pub.shape[0],), dtype=np.float32)
                for i, v01 in enumerate(s01):
                    r, g, b = blue_to_red(float(v01))
                    rgb[i] = pack_rgb(r, g, b)
                self.pub_sem_cloud.publish(make_cloud_xyzrgb(pts_pub, rgb, self.publish_frame, depth_header.stamp))
            self._profile_add_elapsed(f"{profile_prefix}.semantic_cloud", t0)

        return True

    # -------------------------
    # Sync callback (updates STM)
    # -------------------------
    def _note_camera_input(self, msg, kind):
        stamp = msg.header.stamp
        stamp_ns = int(stamp.sec)*1000000000+int(stamp.nanosec)
        self.pipeline_health.input(kind, stamp_ns if stamp_ns else None)

    def on_rgb_depth(self, rgb_msg, depth_msg):
        self.pipeline_health.input("sync")
        if bool(self.get_parameter("observation_latest_frame").value):
            self.latest_frame.offer(rgb_msg, depth_msg, time.monotonic())
            return
        self.frame_counter += 1
        if self.frame_counter % max(1, int(self.get_parameter("embed_every_n").value)) != 0:
            return
        self._process_rgb_depth(rgb_msg, depth_msg)

    def on_observation_timer(self):
        self.apply_finished_semantic_jobs()
        if not bool(self.get_parameter("observation_latest_frame").value):
            self.latest_frame.pending = None
            return
        busy = (self.async_worker_busy.is_set() or not self.semantic_job_queue.empty()
                or not self.semantic_result_queue.empty())
        frame = self.latest_frame.take(time.monotonic(),
            max(.01, float(self.get_parameter("observation_max_hz").value)),
            max(.1, float(self.get_parameter("observation_max_frame_age_s").value)), busy)
        if frame is None:
            return
        self.pipeline_health.working("preprocessing")
        try:
            self._process_rgb_depth(frame[0], frame[1])
        except Exception as error:
            self.pipeline_health.idle(f"Observation preparation failed: {error}")
            self.get_logger().error(f"Observation preparation failed: {error}")
        finally:
            if not self.async_worker_busy.is_set() and self.semantic_job_queue.empty() and self.semantic_result_queue.empty():
                self.pipeline_health.idle()

    def publish_pipeline_health(self):
        payload = self.pipeline_health.snapshot(
            max(1., float(self.get_parameter("pipeline_stale_after_s").value)),
            max(5., float(self.get_parameter("pipeline_processing_timeout_s").value)))
        payload.update(worker_alive=bool(self.async_worker_thread and self.async_worker_thread.is_alive()),
            perception_publishers=self.count_publishers('/crossmaps/diagnostics/pipeline_health'),
            pose_basis='fixed_stationary_anchor' if bool(self.get_parameter('stationary_pose_lock').value) else 'external_tf',
            stationary_pose_lock=bool(self.get_parameter('stationary_pose_lock').value),
            camera_transport=dict(rgb_topic=self.rgb_topic, depth_topic=self.depth_topic,
                rgb_compressed=self.rgb_compressed, depth_compressed=self.depth_compressed),
            pending_latest_frame=self.latest_frame.pending is not None,
            replaced_frames=self.latest_frame.replaced, expired_frames=self.latest_frame.expired,
            job_queue=self.semantic_job_queue.qsize(), result_queue=self.semantic_result_queue.qsize(),
            latest_frame_mode=bool(self.get_parameter("observation_latest_frame").value),
            publish_last_ms=round(getattr(self, "publish_last_ms", 0.), 2),
            publish_skipped_ticks=getattr(self, "publish_skipped_ticks", 0))
        if bool(self.get_parameter("use_async_semantic_worker").value) and not payload['worker_alive']:
            payload['state'] = 'worker_not_running'
        msg = String()
        msg.data = json.dumps(payload, allow_nan=False)
        self.pub_pipeline_health.publish(msg)

    def _process_rgb_depth(self, rgb_msg, depth_msg):

        cb_t0 = time.perf_counter()

        # RGB
        try:
            t0 = time.perf_counter()
            rgb_cv = self.decode_rgb_msg(rgb_msg)
            self._profile_add_elapsed("cb.rgb_decode", t0)
        except Exception as e:
            self.pipeline_health.reject(f"RGB decode failed: {e}")
            self.get_logger().warn(f"RGB conversion failed: {e}")
            self._profile_add_elapsed("cb.total", cb_t0)
            self._profile_report_if_due()
            return

        t0 = time.perf_counter()
        self.publish_debug_observation_images(rgb_cv, rgb_msg.header, depth_msg)
        self._profile_add_elapsed("cb.debug_observation_images", t0)
        if self.fx is None:
            self.pipeline_health.reject("waiting for camera calibration")
            return
        self.last_depth_rejection = "invalid depth shape or missing frame"
        t0 = time.perf_counter()
        out = self.depth_to_points_internal(depth_msg)
        self._profile_add_elapsed("cb.depth_project", t0)
        if out is None:
            self.pipeline_health.reject(self.last_depth_rejection)
            self._profile_add_elapsed("cb.total", cb_t0)
            self._profile_report_if_due()
            return
        pts_int, cam_int, uu, vv, z_cam, (h, w) = out
        pts_int, cam_int = self._apply_stationary_pose_anchor(pts_int, cam_int)
        self._profile_add_value("cb.depth_points", int(pts_int.shape[0]))

        # update history
        t0 = time.perf_counter()
        if bool(self.get_parameter("update_history_cloud").value) or bool(self.get_parameter("publish_history_cloud").value):
            self.update_history_cloud(pts_int)
        self._profile_add_elapsed("cb.history_update", t0)
        t0 = time.perf_counter()
        self.clear_voxels_from_depth_rays(pts_int, cam_int)
        self._profile_add_elapsed("cb.voxel_ray_clear", t0)

        # debug cloud (publish-frame warped)
        if bool(self.get_parameter("publish_debug_cloud").value):
            t0 = time.perf_counter()
            T = self.lookup_T_publish_internal(depth_msg.header.stamp)
            if T is not None:
                pts_h = np.ones((pts_int.shape[0], 4), dtype=np.float32)
                pts_h[:, :3] = pts_int.astype(np.float32)
                pts_pub = (T @ pts_h.T).T[:, :3]
                self.pub_debug_cloud.publish(make_cloud_xyz(pts_pub, self.publish_frame, depth_msg.header.stamp))
            self._profile_add_elapsed("cb.debug_cloud", t0)

        if bool(self.get_parameter("use_async_semantic_worker").value):
            t0 = time.perf_counter()
            queued = self.submit_async_semantic_job(
                rgb_cv=rgb_cv,
                rgb_header=rgb_msg.header,
                depth_header=depth_msg.header,
                pts_int=pts_int,
                cam_int=cam_int,
                uu=uu,
                vv=vv,
                z_cam=z_cam,
                image_shape=(h, w),
                cycle_start_perf_s=cb_t0,
            )
            if queued:
                self._profile_add_value("async.queued_jobs", 1)
            else:
                self._profile_add_value("async.queue_submit_failed", 1)
            self._profile_add_elapsed("cb.async_submit", t0)
            self._profile_add_elapsed("cb.total", cb_t0)
            self._profile_report_if_due()
            return

        self.pipeline_health.working("eSAM")
        t0 = time.perf_counter()
        masks, mask_scores = self.segment_esam_masks(rgb_cv)
        self._profile_add_elapsed("cb.esam_segment", t0)

        self.pipeline_health.working("DINO")
        t0 = time.perf_counter()
        try:
            dense_dino = self.compute_dinov2_dense_features(rgb_cv)
        except Exception as dino_error:
            dense_dino = None
            self.dinov2_model = None
            if not self.dinov2_unavailable_warned:
                self.get_logger().warn(
                    "DINOv2 inference failed; disabling appearance features "
                    f"and using geometry/CLIP: {dino_error}"
                )
                self.dinov2_unavailable_warned = True
        self._profile_add_elapsed("cb.dinov2_dense", t0)

        if masks:
            self.pipeline_health.working("DINO pooling")
            t0 = time.perf_counter()
            dino_mask_emb, dino_mask_quality = self.compute_dinov2_mask_embeddings(
                masks, dense_dino, return_quality=True
            )
            self._profile_add_elapsed("cb.dinov2_mask_pool", t0)
            self.pipeline_health.working("CLIP")
            dense_clip = None
            if bool(self.get_parameter("use_clip_dense_features").value):
                t0 = time.perf_counter()
                dense_clip = self.compute_dense_clip_patch_features(rgb_cv)
                self._profile_add_elapsed("cb.clip_dense_embed", t0)
            t0 = time.perf_counter()
            mask_emb = self.compute_mask_embeddings_batched(rgb_cv, masks, dense_clip=dense_clip)
            self._profile_add_elapsed("cb.clip_mask_embed", t0)
        else:
            dense_clip = None
            dino_mask_emb = np.zeros((0, 0), dtype=np.float32)
            dino_mask_quality = np.zeros((0,), dtype=np.float32)
            mask_emb = np.zeros((0, 0), dtype=np.float32)

        self.async_job_seq += 1
        self.pipeline_health.working("fusion", self.async_job_seq)
        accepted = self.apply_semantic_result(
            {
                "seq": int(self.async_job_seq),
                "obs_time_s": self.now_sec(),
                "rgb_cv": rgb_cv,
                "rgb_header": self._copy_header(rgb_msg.header),
                "depth_header": self._copy_header(depth_msg.header),
                "pts_int": pts_int,
                "cam_int": cam_int,
                "uu": uu,
                "vv": vv,
                "z_cam": z_cam,
                "image_shape": (h, w),
                "masks": masks,
                "mask_scores": mask_scores,
                "mask_emb": mask_emb,
                "dino_mask_emb": dino_mask_emb,
                "dino_mask_quality": dino_mask_quality,
                "dense_clip": dense_clip,
                "dense_dino": dense_dino,
            },
            "cb",
        )
        if accepted:
            self._record_observation_cycle(cb_t0, "synchronous", int(self.async_job_seq))

        self._profile_add_elapsed("cb.total", cb_t0)
        self._profile_report_if_due()

    # -------------------------
    # Memory -> published grids (shared logic)
    # -------------------------
    def _apply_stationary_pose_anchor(self, points, camera):
        """Explicit stationary test mode, never inferred physical stationarity."""
        if not bool(self.get_parameter('stationary_pose_lock').value):
            self._stationary_pose_anchor = None
            return points, camera
        current = self.last_depth_transform
        if getattr(self, '_stationary_pose_anchor', None) is None:
            self._stationary_pose_anchor = current.copy()
            self.get_logger().warn('Stationary pose lock enabled: keep robot/camera still; disable before moving.')
        correction = self._stationary_pose_anchor @ np.linalg.inv(current)
        points = (points @ correction[:3,:3].T + correction[:3,3]).astype(np.float32)
        return points, self._stationary_pose_anchor[:3,3].copy()

    def _compute_cell_scores_conf(
        self,
        keys: List[Tuple[int, int]],
        sum_dict: Dict[Tuple[int, int], np.ndarray],
        w_dict: Dict[Tuple[int, int], float],
        viewmask_dict: Dict[Tuple[int, int], int],
        weight_max: float,
        apply_view_boost: bool,
        use_coh_conf: bool,
        use_coh_sem: bool,
    ):
        """
        For each key:
            returns emb_unit, coh01, score (semantic raw, incl optional coh), conf01 (incl optional coh+view), view_bins_count
        """
        out_emb = []
        out_coh01 = []
        out_score = []
        out_conf01 = []
        out_views = []

        bins = int(self.get_parameter("view_bins").value)
        vb_strength = float(self.get_parameter("view_boost_strength").value)
        vb_pow = float(self.get_parameter("view_boost_power").value)

        for key in keys:
            S = sum_dict.get(key, None)
            W = float(w_dict.get(key, 0.0))
            if S is None or W <= 1e-6:
                continue

            emb_u, coh, w = self._embedding_and_coherence(S, W)
            if emb_u is None:
                continue

            s = float(np.dot(emb_u, self.text_emb))
            s = max(0.0, s)

            coh01 = self._coherence01(float(coh))
            if use_coh_sem:
                s *= coh01

            c = float(np.clip(float(w) / max(weight_max, 1e-6), 0.0, 1.0))
            if use_coh_conf:
                c *= coh01

            nb = 0
            if apply_view_boost and bins > 1:
                mask = int(viewmask_dict.get(key, 0))
                nb = safe_popcount(mask)
                frac = float(nb) / float(bins)
                boost = 1.0 + vb_strength * (frac ** vb_pow)
                c = float(np.clip(c * boost, 0.0, 1.5))
                c = float(np.clip(c, 0.0, 1.0))
            else:
                mask = int(viewmask_dict.get(key, 0))
                nb = safe_popcount(mask) if bins > 1 else (1 if mask != 0 else 0)

            out_emb.append(emb_u)
            out_coh01.append(float(coh01))
            out_score.append(float(s))
            out_conf01.append(float(c))
            out_views.append(int(nb))

        if len(out_emb) == 0:
            return None

        return (
            np.stack(out_emb, axis=0).astype(np.float32),
            np.array(out_coh01, dtype=np.float32),
            np.array(out_score, dtype=np.float32),
            np.array(out_conf01, dtype=np.float32),
            np.array(out_views, dtype=np.int32),
        )

    def _compute_keyed_scores_conf(self, keys, sum_dict, w_dict, viewmask_dict, weight_max,
                                   apply_view_boost, use_coh_conf, use_coh_sem):
        options = (float(weight_max), bool(apply_view_boost), bool(use_coh_conf), bool(use_coh_sem))
        context = getattr(self, "_entity_score_context", None)
        bundle = None if context is None else context.get("bundle")
        shared = bundle is not None and sum_dict is bundle[1] and w_dict is bundle[2]
        def compute(selected):
            return score_voxels(selected, sum_dict, w_dict, viewmask_dict, self.text_emb, weight_max,
                int(self.get_parameter("view_bins").value), float(self.get_parameter("view_boost_strength").value),
                float(self.get_parameter("view_boost_power").value), float(self.get_parameter("coherence_floor").value),
                float(self.get_parameter("coherence_power").value), apply_view_boost, use_coh_conf, use_coh_sem)
        if not shared:
            return compute(keys)
        cache = context.setdefault("scores", {})
        if options not in cache:
            result = compute(bundle[0])
            cache[options] = (result, {} if result is None else {k:i for i,k in enumerate(result[0])})
        result, index = cache[options]
        if result is None:
            return None
        selected = [k for k in keys if k in index]
        if not selected:
            return None
        indices = np.asarray([index[k] for k in selected], dtype=np.int64)
        return (selected, *(array[indices] for array in result[1:]))

    def _compute_keyed_scores_conf_scalar(
        self,
        keys,
        sum_dict,
        w_dict,
        viewmask_dict,
        weight_max: float,
        apply_view_boost: bool,
        use_coh_conf: bool,
        use_coh_sem: bool,
    ):
        out_keys = []
        out_emb = []
        out_coh01 = []
        out_score = []
        out_conf01 = []
        out_views = []

        bins = int(self.get_parameter("view_bins").value)
        vb_strength = float(self.get_parameter("view_boost_strength").value)
        vb_pow = float(self.get_parameter("view_boost_power").value)

        for key in keys:
            S = sum_dict.get(key, None)
            W = float(w_dict.get(key, 0.0))
            if S is None or W <= 1e-6:
                continue
            emb_u, coh, w = self._embedding_and_coherence(S, W)
            if emb_u is None:
                continue

            s = 0.0
            if self.text_emb is not None:
                s = max(0.0, float(np.dot(emb_u, self.text_emb)))
            coh01 = self._coherence01(float(coh))
            if use_coh_sem:
                s *= coh01

            c = float(np.clip(float(w) / max(weight_max, 1e-6), 0.0, 1.0))
            if use_coh_conf:
                c *= coh01

            mask = int(viewmask_dict.get(key, 0))
            nb = safe_popcount(mask) if bins > 1 else (1 if mask != 0 else 0)
            if apply_view_boost and bins > 1:
                frac = float(nb) / float(bins)
                boost = 1.0 + vb_strength * (frac ** vb_pow)
                c = float(np.clip(c * boost, 0.0, 1.0))

            out_keys.append(key)
            out_emb.append(emb_u)
            out_coh01.append(float(coh01))
            out_score.append(float(s))
            out_conf01.append(float(c))
            out_views.append(int(nb))

        if not out_keys:
            return None
        return (
            out_keys,
            np.stack(out_emb, axis=0).astype(np.float32),
            np.array(out_coh01, dtype=np.float32),
            np.array(out_score, dtype=np.float32),
            np.array(out_conf01, dtype=np.float32),
            np.array(out_views, dtype=np.int32),
        )

    def _project_internal_cells_to_publish_cells(self, T_pub_int: np.ndarray, keys: List[Tuple[int, int]]):
        """
        Project INTERNAL cell centers to publish frame, then map back into THIS node's grid index space.
        """
        cells_pub = []
        keys_keep = []
        for key in keys:
            cx_i, cy_i = self.cell_center_xy(key[0], key[1])
            p_int = np.array([cx_i, cy_i, 0.0, 1.0], dtype=np.float32)
            p_pub = (T_pub_int @ p_int)[:3]
            cell_pub = self.xy_to_cell(float(p_pub[0]), float(p_pub[1]))
            if cell_pub is None:
                continue
            cells_pub.append(cell_pub)
            keys_keep.append(key)
        return cells_pub, keys_keep

    def _build_sem_conf_top1_grids(self, cells_pub: List[Tuple[int, int]], scores_np: np.ndarray, confs_np: np.ndarray):
        sem_grid = -np.ones((self.grid_height, self.grid_width), dtype=np.int8)
        conf_grid = -np.ones((self.grid_height, self.grid_width), dtype=np.int8)
        top1_grid = -np.ones((self.grid_height, self.grid_width), dtype=np.int8)

        if len(cells_pub) < 10:
            return sem_grid, conf_grid, top1_grid

        # normalize semantic using percentiles
        p_low = float(self.get_parameter("norm_low_percentile").value)
        p_high = float(self.get_parameter("norm_high_percentile").value)
        lo = float(np.percentile(scores_np, p_low))
        hi = float(np.percentile(scores_np, p_high))
        den = max(hi - lo, 1e-6)

        for (ix, iy), s, c in zip(cells_pub, scores_np, confs_np):
            sem01 = float(np.clip((s - lo) / den, 0.0, 1.0))
            val_sem = int(sem01 * 100)
            val_conf = int(np.clip(c, 0.0, 1.0) * 100)

            if sem_grid[iy, ix] < val_sem:
                sem_grid[iy, ix] = val_sem
            if conf_grid[iy, ix] < val_conf:
                conf_grid[iy, ix] = val_conf

        # top1
        rank = scores_np * confs_np
        best_i = int(np.argmax(rank))
        bx, by = cells_pub[best_i]
        top1_grid[by, bx] = 100

        return sem_grid, conf_grid, top1_grid

    def _voxel_memory_bundle(self, memory: str):
        memory = str(memory).strip().lower()
        context = getattr(self, "_entity_score_context", None)
        if memory == "stm_ltm" and context is not None and "bundle" in context:
            return context["bundle"]
        if memory in ("stm", "short", "short_term"):
            return (
                list(self.vox_stm_w.keys()),
                self.vox_stm_sum,
                self.vox_stm_w,
                self.vox_stm_viewmask,
                float(self.get_parameter("cell_weight_max").value),
            )
        if memory in ("ltm", "long", "long_term"):
            return (
                list(self.vox_ltm_w.keys()),
                self.vox_ltm_sum,
                self.vox_ltm_w,
                self.vox_ltm_viewmask,
                float(self.get_parameter("ltm_weight_max").value),
            )

        sum_dict = {}
        w_dict = {}
        viewmask_dict = {}
        for key in self.vox_ltm_w.keys():
            if key in self.vox_ltm_sum:
                sum_dict[key] = self.vox_ltm_sum[key]
                w_dict[key] = self.vox_ltm_w[key]
                viewmask_dict[key] = self.vox_ltm_viewmask.get(key, 0)
        for key in self.vox_stm_w.keys():
            if key in self.vox_stm_sum:
                sum_dict[key] = self.vox_stm_sum[key]
                w_dict[key] = self.vox_stm_w[key]
                viewmask_dict[key] = self.vox_stm_viewmask.get(key, 0)
        bundle = (list(w_dict.keys()), sum_dict, w_dict, viewmask_dict, float(self.get_parameter("cell_weight_max").value))
        if memory == "stm_ltm" and context is not None:
            context["bundle"] = bundle
        return bundle

    def _project_internal_voxels_to_publish_points(self, T_pub_int: np.ndarray, keys: List[Tuple[int, int, int]]):
        if not keys:
            return np.zeros((0, 3), dtype=np.float32)
        keys_np = np.asarray(keys, dtype=np.float32)
        pts = np.empty((keys_np.shape[0], 3), dtype=np.float32)
        pts[:, 0] = self.origin_x + (keys_np[:, 0] + 0.5) * self.voxel_size
        pts[:, 1] = self.origin_y + (keys_np[:, 1] + 0.5) * self.voxel_size
        pts[:, 2] = self.voxel_z_min + (keys_np[:, 2] + 0.5) * self.voxel_size
        pts_h = np.ones((pts.shape[0], 4), dtype=np.float32)
        pts_h[:, :3] = pts
        return (T_pub_int @ pts_h.T).T[:, :3].astype(np.float32)

    def voxel_cloud_max_points_for_memory(self, memory: str) -> int:
        memory = str(memory).strip().lower()
        if memory in ("stm", "short", "short_term"):
            specific = int(self.get_parameter("voxel_stm_cloud_max_points").value)
        elif memory in ("ltm", "long", "long_term"):
            specific = int(self.get_parameter("voxel_ltm_cloud_max_points").value)
        else:
            specific = int(self.get_parameter("voxel_fused_cloud_max_points").value)
        if specific > 0:
            return specific
        return int(self.get_parameter("voxel_cloud_max_points").value)

    def publish_empty_voxel_heatmap(self, publisher):
        msg = make_cloud_xyzrgb(
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            self.publish_frame,
            self.get_clock().now().to_msg(),
        )
        publisher.publish(msg)

    def publish_voxel_heatmap_for_memory(self, T_pub_int: np.ndarray, memory: str, publisher):
        if not bool(self.get_parameter("use_semantic_voxels").value):
            self.publish_empty_voxel_heatmap(publisher)
            return

        keys, sum_dict, w_dict, viewmask_dict, weight_max = self._voxel_memory_bundle(memory)
        if len(keys) == 0:
            self.publish_empty_voxel_heatmap(publisher)
            return

        res = self._compute_keyed_scores_conf(
            keys=keys,
            sum_dict=sum_dict,
            w_dict=w_dict,
            viewmask_dict=viewmask_dict,
            weight_max=weight_max,
            apply_view_boost=bool(self.get_parameter("use_visibility_boost").value),
            use_coh_conf=bool(self.get_parameter("use_coherence_in_confidence").value),
            use_coh_sem=bool(self.get_parameter("use_coherence_in_semantic").value),
        )
        if res is None:
            self.publish_empty_voxel_heatmap(publisher)
            return

        scored_keys, _, coh01s, scores_np, confs_np, _ = res
        keep = (
            (scores_np >= float(self.get_parameter("voxel_cloud_min_score").value))
            & (confs_np >= float(self.get_parameter("voxel_cloud_min_conf").value))
            & (coh01s >= float(self.get_parameter("voxel_cloud_min_coh01").value))
        )
        idxs = np.flatnonzero(keep)
        if idxs.size == 0:
            self.publish_empty_voxel_heatmap(publisher)
            return

        scores_keep = scores_np[idxs]
        confs_keep = confs_np[idxs]
        if bool(self.get_parameter("voxel_cloud_normalize_scores").value) and scores_keep.size >= 5:
            p_low = float(self.get_parameter("norm_low_percentile").value)
            p_high = float(self.get_parameter("norm_high_percentile").value)
            lo = float(np.percentile(scores_keep, p_low))
            hi = float(np.percentile(scores_keep, p_high))
            score01 = np.clip((scores_keep - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        else:
            score01 = np.clip(scores_keep, 0.0, 1.0)
        gamma = max(1e-3, float(self.get_parameter("voxel_cloud_gamma").value))
        color01 = np.power(score01, gamma)
        rank = color01 * np.clip(confs_keep, 0.0, 1.0)

        max_pts = self.voxel_cloud_max_points_for_memory(memory)
        if max_pts > 0 and idxs.size > max_pts:
            top_local = np.argpartition(rank, -max_pts)[-max_pts:]
            idxs = idxs[top_local]
            color01 = color01[top_local]
            confs_keep = confs_keep[top_local]

        selected_keys = [scored_keys[int(i)] for i in idxs]
        pts_pub = self._project_internal_voxels_to_publish_points(T_pub_int, selected_keys)
        brightness = np.clip(0.25 + 0.75 * confs_keep, 0.25, 1.0).astype(np.float32)
        vals = np.clip(color01, 0.0, 1.0).astype(np.float32)
        rr = np.clip((255.0 * vals) * brightness, 0, 255).astype(np.uint32)
        gg = np.clip((40.0 * (1.0 - np.abs(vals - 0.5) * 2.0)) * brightness, 0, 255).astype(np.uint32)
        bb = np.clip((255.0 * (1.0 - vals)) * brightness, 0, 255).astype(np.uint32)
        rgb_uint32 = (rr << 16) | (gg << 8) | bb
        rgb = rgb_uint32.astype(np.uint32).view(np.float32)

        msg = make_cloud_xyzrgb(pts_pub, rgb, self.publish_frame, self.get_clock().now().to_msg())
        publisher.publish(msg)

    def publish_voxel_heatmaps(self, T_pub_int: np.ndarray):
        if not bool(self.get_parameter("publish_voxel_heatmap").value):
            return
        every_n = max(1, int(self.get_parameter("voxel_heatmap_publish_every_n").value))
        if every_n > 1 and ((self.publish_counter - 1) % every_n) != 0:
            return
        if bool(self.get_parameter("publish_voxel_stm_heatmap").value):
            t0 = time.perf_counter()
            self.publish_voxel_heatmap_for_memory(T_pub_int, "stm", self.pub_voxel_stm_heatmap)
            self._profile_add_elapsed("pub.voxel_cloud_stm", t0)
        if bool(self.get_parameter("publish_voxel_ltm_heatmap").value):
            t0 = time.perf_counter()
            self.publish_voxel_heatmap_for_memory(T_pub_int, "ltm", self.pub_voxel_ltm_heatmap)
            self._profile_add_elapsed("pub.voxel_cloud_ltm", t0)
        if bool(self.get_parameter("publish_voxel_fused_heatmap").value):
            t0 = time.perf_counter()
            self.publish_voxel_heatmap_for_memory(T_pub_int, "stm_ltm", self.pub_voxel_fused_heatmap)
            self._profile_add_elapsed("pub.voxel_cloud_fused", t0)

    def publish_voxel_heatmap(self, T_pub_int: np.ndarray):
        # Backward-compatible wrapper: now publishes the configured set of voxel heatmaps.
        self.publish_voxel_heatmaps(T_pub_int)

    def promote_voxel_memory(self, now_s: float):
        if not bool(self.get_parameter("use_semantic_voxels").value):
            return
        keys = list(self.vox_stm_w.keys())
        if not keys:
            return
        max_promote = int(self.get_parameter("voxel_promote_max_per_tick").value)
        if max_promote > 0 and len(keys) > max_promote:
            keys = keys[:max_promote]
        res = self._compute_keyed_scores_conf(
            keys=keys,
            sum_dict=self.vox_stm_sum,
            w_dict=self.vox_stm_w,
            viewmask_dict=self.vox_stm_viewmask,
            weight_max=float(self.get_parameter("cell_weight_max").value),
            apply_view_boost=bool(self.get_parameter("use_visibility_boost").value),
            use_coh_conf=bool(self.get_parameter("use_coherence_in_confidence").value),
            use_coh_sem=bool(self.get_parameter("use_coherence_in_semantic").value),
        )
        if res is None:
            return
        scored_keys, emb_u, coh01s, _, confs_np, views_np = res
        min_conf = float(self.get_parameter("voxel_promote_min_conf").value)
        min_coh01 = float(self.get_parameter("voxel_promote_min_coh01").value)
        min_views = int(self.get_parameter("voxel_promote_min_views").value)
        for key, e_u, coh01, c01, nb in zip(scored_keys, emb_u, coh01s, confs_np, views_np):
            if c01 < min_conf or coh01 < min_coh01 or nb < min_views:
                continue
            self.promote_vox_stm_to_ltm(key, e_u, float(coh01), float(c01), now_s)

    def _query_voxel_score01_bundle(self):
        if self.text_emb is None:
            return None
        keys, sum_dict, w_dict, viewmask_dict, weight_max = self._voxel_memory_bundle("stm_ltm")
        if not keys:
            return None
        res = self._compute_keyed_scores_conf(
            keys=keys,
            sum_dict=sum_dict,
            w_dict=w_dict,
            viewmask_dict=viewmask_dict,
            weight_max=weight_max,
            apply_view_boost=bool(self.get_parameter("use_visibility_boost").value),
            use_coh_conf=bool(self.get_parameter("use_coherence_in_confidence").value),
            use_coh_sem=bool(self.get_parameter("use_coherence_in_semantic").value),
        )
        if res is None:
            return None

        scored_keys, _, coh01s, scores_np, confs_np, _ = res
        keep = (
            (scores_np >= float(self.get_parameter("voxel_cloud_min_score").value))
            & (confs_np >= float(self.get_parameter("voxel_cloud_min_conf").value))
            & (coh01s >= float(self.get_parameter("voxel_cloud_min_coh01").value))
        )
        idxs = np.flatnonzero(keep)
        if idxs.size == 0:
            return [], np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        scores_keep = scores_np[idxs]
        confs_keep = confs_np[idxs]
        if bool(self.get_parameter("voxel_cloud_normalize_scores").value) and scores_keep.size >= 5:
            p_low = float(self.get_parameter("norm_low_percentile").value)
            p_high = float(self.get_parameter("norm_high_percentile").value)
            lo = float(np.percentile(scores_keep, p_low))
            hi = float(np.percentile(scores_keep, p_high))
            score01 = np.clip((scores_keep - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        else:
            score01 = np.clip(scores_keep, 0.0, 1.0)
        keys_keep = [scored_keys[int(i)] for i in idxs]
        return keys_keep, score01.astype(np.float32), confs_keep.astype(np.float32)

    def _support_cells_from_voxels(self, voxels: set):
        cells = set()
        for key in voxels:
            x, y, _ = self.voxel_center_xyz(key)
            cell = self.xy_to_cell(float(x), float(y))
            if cell is not None:
                cells.add(cell)
        return cells

    def _bounded_support_union(self, old_support: set, new_support: set, max_items: int) -> set:
        old_support = set(old_support)
        new_support = set(new_support)
        if max_items <= 0:
            return old_support | new_support
        if len(new_support) >= max_items:
            new_sorted = sorted(new_support)
            step = max(1, int(math.ceil(len(new_sorted) / max_items)))
            return set(new_sorted[::step][:max_items])
        budget_old = max_items - len(new_support)
        old_only = sorted(old_support - new_support)
        if len(old_only) > budget_old:
            step = max(1, int(math.ceil(len(old_only) / max(budget_old, 1))))
            old_only = old_only[::step][:budget_old]
        return set(new_support) | set(old_only)

    def _point_msg(self, xyz):
        p = Point()
        p.x = float(xyz[0])
        p.y = float(xyz[1])
        p.z = float(xyz[2])
        return p

    def _transform_point_internal_to_publish(self, T_pub_int: np.ndarray, xyz):
        p_int = np.array([float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0], dtype=np.float32)
        return (T_pub_int @ p_int)[:3].astype(np.float32)

    def _transform_bbox_internal_to_publish(self, T_pub_int: np.ndarray, bmin, bmax):
        bmin = np.asarray(bmin, dtype=np.float32)
        bmax = np.asarray(bmax, dtype=np.float32)
        corners = np.array(
            [
                [x, y, z, 1.0]
                for x in (bmin[0], bmax[0])
                for y in (bmin[1], bmax[1])
                for z in (bmin[2], bmax[2])
            ],
            dtype=np.float32,
        )
        transformed = (T_pub_int @ corners.T).T[:, :3]
        return (
            np.min(transformed, axis=0).astype(np.float32),
            np.max(transformed, axis=0).astype(np.float32),
        )

    def _entity_dynamic_state(self, age_s: float, conf: float, support_count: int, obs_count: int):
        stale_after = float(self.get_parameter("entity_track_max_stale_s").value)
        min_support = max(
            int(self.get_parameter("entity_min_support_voxels").value),
            int(self.get_parameter("entity_min_support_cells").value),
        )
        min_obs = int(self.get_parameter("object_graph_min_observations").value)
        if stale_after > 0.0 and age_s > stale_after:
            return "stale"
        if obs_count < max(1, min_obs) or support_count < max(1, min_support) or conf < 0.08:
            return "tentative"
        return "active"

    def _object_shape_traits(self, bmin: np.ndarray, bmax: np.ndarray, support_count: int):
        dims = np.maximum(np.asarray(bmax, dtype=np.float32) - np.asarray(bmin, dtype=np.float32), 0.0)
        xy_long = float(max(dims[0], dims[1]))
        xy_short = float(max(min(dims[0], dims[1]), 1e-6))
        footprint = float(dims[0] * dims[1])
        volume = float(max(footprint * max(float(dims[2]), 1e-6), 0.0))
        return {
            "dims_m": [round(float(v), 4) for v in dims],
            "footprint_m2": round(footprint, 5),
            "volume_m3": round(volume, 5),
            "xy_elongation": round(float(xy_long / xy_short), 4),
            "z_to_xy_long": round(float(dims[2] / max(xy_long, 1e-6)), 4),
            "support_density": round(float(support_count / max(volume, 1e-6)), 4),
        }

    def _bbox_xy_overlap_min_frac(self, a: dict, b: dict):
        a_min = a["bbox_min"]
        a_max = a["bbox_max"]
        b_min = b["bbox_min"]
        b_max = b["bbox_max"]
        inter_x = max(0.0, min(float(a_max[0]), float(b_max[0])) - max(float(a_min[0]), float(b_min[0])))
        inter_y = max(0.0, min(float(a_max[1]), float(b_max[1])) - max(float(a_min[1]), float(b_min[1])))
        inter = inter_x * inter_y
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, float(a_max[0] - a_min[0])) * max(0.0, float(a_max[1] - a_min[1]))
        area_b = max(0.0, float(b_max[0] - b_min[0])) * max(0.0, float(b_max[1] - b_min[1]))
        return float(inter / max(min(area_a, area_b), 1e-6))

    def _raw_object_graph_edges(self, nodes: List[dict]):
        near_dist = float(self.get_parameter("object_graph_near_dist_m").value)
        vertical_xy = float(self.get_parameter("object_graph_vertical_xy_dist_m").value)
        vertical_z = float(self.get_parameter("object_graph_vertical_z_min_m").value)
        overlap_min = float(self.get_parameter("object_graph_overlap_xy_min").value)
        similar_min = float(self.get_parameter("object_graph_similar_cos_min").value)
        include_overlap = bool(self.get_parameter("object_graph_include_overlap_edges").value)
        include_similar = bool(self.get_parameter("object_graph_include_similar_edges").value)

        edges = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a = nodes[i]
                b = nodes[j]
                ca = a["centroid"]
                cb = b["centroid"]
                delta = cb - ca
                dxy = float(np.linalg.norm(delta[:2]))
                dist = float(np.linalg.norm(delta))
                dz = float(delta[2])
                overlap_xy = self._bbox_xy_overlap_min_frac(a, b)
                sem = float(np.dot(a["embedding"], b["embedding"]))

                relations = []
                if abs(dz) >= vertical_z and dxy <= vertical_xy:
                    if dz > 0.0:
                        start = b
                        end = a
                    else:
                        start = a
                        end = b
                    strength = float(np.clip(abs(dz) / max(1.0, vertical_z), 0.0, 1.0))
                    relations.append((start, end, "above", strength, 3))
                if near_dist > 0.0 and dist <= near_dist:
                    strength = float(np.clip(1.0 - dist / max(near_dist, 1e-6), 0.0, 1.0))
                    relations.append((a, b, "near", strength, 2))
                if include_overlap and overlap_xy >= overlap_min:
                    strength = float(np.clip(overlap_xy, 0.0, 1.0))
                    relations.append((a, b, "overlap_xy", strength, 1))
                if include_similar and sem >= similar_min:
                    strength = float(np.clip((sem - similar_min) / max(1.0 - similar_min, 1e-6), 0.0, 1.0))
                    relations.append((a, b, "similar", strength, 1))

                for start, end, relation, strength, priority in relations:
                    edges.append(
                        {
                            "a": start,
                            "b": end,
                            "relation": relation,
                            "strength": float(strength),
                            "priority": int(priority),
                            "distance": float(dist),
                            "semantic_cos": float(sem),
                        }
                    )

        edges.sort(key=lambda e: (int(e["priority"]), float(e["strength"])), reverse=True)
        # The display edge budget must not suppress evidence or create misses.
        return edges

    def _object_graph_edges(self, nodes: List[dict]):
        """Read-only rendering: publication never manufactures observations."""
        memory = self.entity_edges if bool(self.get_parameter("use_entity_grouping").value) else self.segment_edges
        by_id = {int(node["id"]): node for node in nodes}
        edges = []
        for item in memory.snapshot(
            by_id, max(1, int(self.get_parameter("object_graph_edge_min_observations").value))
        ):
            edges.append({
                "a": by_id[item["from"]], "b": by_id[item["to"]],
                "relation": item["relation"], "strength": item["strength"],
                "priority": item["priority"], "distance": item["distance"],
                "semantic_cos": item["semantic_cos"], "observations": item["hits"],
                "last_observation_seq": item["last_seq"], "misses": item["misses"],
            })
        edges.sort(key=lambda edge: (edge["priority"], edge["strength"]), reverse=True)
        limit = int(self.get_parameter("object_graph_max_edges").value)
        return edges[:limit] if limit > 0 else edges

    def _relation_color(self, relation: str):
        if relation == "overlap_xy":
            return 0.95, 0.35, 1.0, 0.85
        if relation == "above":
            return 1.0, 0.78, 0.20, 0.85
        if relation == "near":
            return 0.25, 0.95, 0.75, 0.75
        if relation == "similar":
            return 0.55, 0.55, 1.0, 0.65
        return 0.8, 0.8, 0.8, 0.65

    def _json_vec(self, arr, ndigits: int = 4):
        return [round(float(v), ndigits) for v in np.asarray(arr).reshape((-1,))]

    def _descriptor_embedding(self, emb: np.ndarray):
        if not bool(self.get_parameter("object_graph_descriptor_include_embedding").value):
            return None
        dims = int(self.get_parameter("object_graph_descriptor_embedding_dims").value)
        emb_np = np.asarray(emb, dtype=np.float32).reshape((-1,))
        if dims > 0:
            emb_np = emb_np[:dims]
        return [round(float(v), 6) for v in emb_np]

    def _object_graph_descriptor_payload(
        self,
        nodes: List[dict],
        edges: List[dict],
        T_pub_int: np.ndarray,
    ):
        now_s = self.now_sec()
        node_payload = []
        for node in nodes:
            oid = int(node["id"])
            centroid_pub = self._transform_point_internal_to_publish(
                T_pub_int, node["centroid"]
            )
            bbox_min_pub, bbox_max_pub = self._transform_bbox_internal_to_publish(
                T_pub_int, node["bbox_min"], node["bbox_max"]
            )
            node_d = {
                "id": oid,
                "node_type": "object_hypothesis" if "segment_ids" in node else "surface_segment",
                "entity_id": node.get("entity_id"),
                "objectness": node.get("objectness", "unverified"),
                "membership_basis": node.get("membership_basis"),
                "last_observation_seq": int(node.get("last_observation_seq", -1)),
                "state": str(node.get("state", "active")),
                "support_hash": str(node.get("support_hash", "")),
                "support_count": int(node.get("support_count", 0)),
                "centroid_m": self._json_vec(centroid_pub),
                "bbox_min_m": self._json_vec(bbox_min_pub),
                "bbox_max_m": self._json_vec(bbox_max_pub),
                "shape": self._object_shape_traits(
                    bbox_min_pub,
                    bbox_max_pub,
                    int(node.get("support_count", 0)),
                ),
                "confidence": round(float(node.get("conf", 0.0)), 4),
                "coherence01": round(float(node.get("coh01", 0.0)), 4),
                "query_score": round(float(node.get("score", 0.0)), 4),
                "weight": round(float(node.get("weight", 0.0)), 4),
                "observations": int(node.get("observations", 1)),
                "view_count": int(node.get("view_count", 0)),
                "appearance_descriptor": bool(node.get("dino_embedding") is not None),
                "age_s": round(float(node.get("age_s", 0.0)), 3),
                "first_seen_s": round(float(node.get("first_seen_s", 0.0)), 3),
                "last_seen_s": round(float(node.get("last_seen_s", 0.0)), 3),
            }
            if "segment_ids" in node:
                node_d["segment_ids"] = [int(sid) for sid in node.get("segment_ids", [])]
                node_d["component_entity_ids"] = [int(eid) for eid in node.get("component_entity_ids", [])]
                node_d["entity_size"] = int(node.get("entity_size", len(node_d["segment_ids"])))
                node_d["semantic_consistency"] = round(float(node.get("semantic_consistency", 1.0)), 4)
            emb = self._descriptor_embedding(node["embedding"])
            if emb is not None:
                node_d["embedding"] = emb
            node_payload.append(node_d)

        kind = "entity" if bool(self.get_parameter("use_entity_grouping").value) else "segment"
        for node_d in node_payload:
            node_d["uid"] = node_uid(self.world_model_id, kind, node_d["id"])
            node_d["segment_uids"] = [node_uid(self.world_model_id, "segment", sid) for sid in node_d.get("segment_ids", [])]
            node_d["component_entity_uids"] = [node_uid(self.world_model_id, "entity", eid) for eid in node_d.get("component_entity_ids", [])]
            node_d["semantic_labels"] = []  # Query scores are not class labels.
            node_d["confidence_semantics"] = "uncalibrated_evidence_strength_not_object_probability"
            node_d["surface_role"] = getattr(self, "entity_surface_roles", {}).get(node_d["id"], "unknown") if kind == "entity" else "unknown"
        edge_payload = []
        for edge in edges:
            edge_payload.append(
                {
                    "from": int(edge["a"]["id"]),
                    "to": int(edge["b"]["id"]),
                    "from_uid": node_uid(self.world_model_id, kind, edge["a"]["id"]),
                    "to_uid": node_uid(self.world_model_id, kind, edge["b"]["id"]),
                    "evidence_kind": "repeated_geometric_relation_hypothesis",
                    "relation": str(edge["relation"]),
                    "strength": round(float(edge.get("strength", 0.0)), 4),
                    "distance_m": round(float(edge.get("distance", 0.0)), 4),
                    "semantic_cos": round(float(edge.get("semantic_cos", 0.0)), 4),
                    "observations": int(edge.get("observations", 1)),
                }
            )

        return {
            "schema": "crossmaps_hierarchical_scene_graph_v3",
            "world_model": {
                "id": self.world_model_id, "source_id": str(self.get_parameter("world_model_source_id").value),
                "observation_source_id": getattr(self, "observation_source_id", self.world_model_id),
                "observation_seq": int(self.last_applied_observation_seq),
                "sensor_depth_stamp_ns": int(self.last_accepted_depth_stamp_ns),
                "snapshot_semantics": "filtered_view_not_complete_map", "absence_is_negative_evidence": False,
                "segment_count_total": len(self.seg_w), "base_entity_count_total": len(self.entity_memory.records),
                "pose_quality": "external_tf_not_validated", "cross_map_alignment": "required_not_inferred",
                "pose_basis": "fixed_stationary_anchor" if bool(self.get_parameter('stationary_pose_lock').value) else "external_tf",
                "feature_spaces": self._identity_metadata_snapshot(),
                "grouping_descriptor": "dino_interior", "semantic_classification": "not_inferred_from_query",
            },
            "entity_type": "entity" if bool(self.get_parameter("use_entity_grouping").value) else "segment",
            "stamp_s": round(float(now_s), 3),
            "frame_id": str(self.publish_frame),
            "internal_frame": str(self.internal_frame),
            "query_text": str(self.query_text),
            "voxel_size_m": float(self.voxel_size),
            "node_count": int(len(node_payload)),
            "edge_count": int(len(edge_payload)),
            "nodes": node_payload,
            "edges": edge_payload,
            "identity": {**self.entity_memory.diagnostics(), "update_ms": round(self.last_identity_update_ms, 3)},
            "mask_attribution": getattr(self, "last_mask_attribution", []),
            "surface_tracking": self.surface_consensus.diagnostics() if hasattr(self, "surface_consensus") else {},
            "identity_recording": self.evidence_recorder.status(bool(self.get_parameter("record_identity_replay").value))
                if getattr(self, "evidence_recorder", None) else None,
            "group_aliases": [{"from_uid": node_uid(self.world_model_id, "entity", eid),
                               "to_uid": node_uid(self.world_model_id, "entity", root),
                               "reversible": True, "basis": "dino_interior_contact_hypothesis"}
                              for eid, root in sorted(self.entity_memory.assemblies.aliases.items())
                              if bool(self.get_parameter("use_object_assembly").value)],
            "memberships": [
                {"from_type": "surface_segment", "from": sid, "to_type": "object_hypothesis",
                 "from_uid": node_uid(self.world_model_id, "segment", sid),
                 "to_uid": node_uid(self.world_model_id, "entity", self._effective_entity_id(eid)),
                 "base_entity_uid": node_uid(self.world_model_id, "entity", eid),
                 "to": self._effective_entity_id(eid), "relation": "member_of", "basis": "surface_continuity_or_assembly"}
                for sid, eid in sorted(self.entity_memory.segment_entity.items())
                if (self._effective_entity_id(eid) if bool(self.get_parameter("use_entity_grouping").value) else sid)
                in {int(n["id"]) for n in nodes}
            ],
            "pending_memberships": [dict(segment_id=sid, **pending)
                                    for sid, pending in sorted(self.entity_memory.pending.items())[:100]],
            "lineage": list(self.entity_memory.lineage)[-32:],
            "areas": [],
        }

    def publish_object_graph_descriptors(
        self,
        nodes: List[dict],
        edges: List[dict],
        T_pub_int: np.ndarray,
    ):
        if not bool(self.get_parameter("publish_object_graph_descriptors").value):
            return
        msg = String()
        msg.data = json.dumps(
            self._object_graph_descriptor_payload(nodes, edges, T_pub_int),
            separators=(",", ":"),
            sort_keys=True,
        )
        self.pub_obj_graph_descriptors.publish(msg)

    def publish_object_graph(self, T_pub_int: np.ndarray):
        markers_enabled = bool(self.get_parameter("publish_object_graph").value)
        if not markers_enabled and not bool(self.get_parameter("publish_object_graph_descriptors").value):
            return

        now_msg = self.get_clock().now().to_msg()
        publish_labels = bool(self.get_parameter("publish_object_graph_labels").value)
        node_scale = float(self.get_parameter("object_graph_node_scale_m").value)
        edge_width = float(self.get_parameter("object_graph_edge_width_m").value)
        label_scale = float(self.get_parameter("object_graph_label_scale_m").value)
        nodes = self._segment_graph_candidates()
        edges = self._object_graph_edges(nodes)
        self.publish_object_graph_descriptors(nodes, edges, T_pub_int)
        if not markers_enabled:
            return
        use_groups = bool(self.get_parameter("use_entity_grouping").value)
        label_prefix = "ent" if use_groups else "seg"

        node_arr = MarkerArray()
        clear_nodes = Marker()
        clear_nodes.header = Header(stamp=now_msg, frame_id=self.publish_frame)
        clear_nodes.action = Marker.DELETEALL
        node_arr.markers.append(clear_nodes)

        for node in nodes:
            oid = int(node["id"])
            center_pub = self._transform_point_internal_to_publish(T_pub_int, node["centroid"])
            r, g, b = self._segment_instance_rgb(oid)
            scale = float(np.clip(node_scale * (0.7 + 0.8 * float(node["conf"])), 0.05, node_scale * 1.8))

            sphere = Marker()
            sphere.header = Header(stamp=now_msg, frame_id=self.publish_frame)
            sphere.ns = "crossmaps_object_graph_node"
            sphere.id = oid
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(center_pub[0])
            sphere.pose.position.y = float(center_pub[1])
            sphere.pose.position.z = float(center_pub[2])
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = scale
            sphere.scale.y = scale
            sphere.scale.z = scale
            sphere.color.r = float(r) / 255.0
            sphere.color.g = float(g) / 255.0
            sphere.color.b = float(b) / 255.0
            sphere.color.a = float(np.clip(0.35 + 0.55 * float(node["conf"]), 0.35, 0.95))
            node_arr.markers.append(sphere)

            if publish_labels:
                label = Marker()
                label.header = Header(stamp=now_msg, frame_id=self.publish_frame)
                label.ns = "crossmaps_object_graph_node_label"
                label.id = oid
                label.type = Marker.TEXT_VIEW_FACING
                label.action = Marker.ADD
                label.pose.position.x = float(center_pub[0])
                label.pose.position.y = float(center_pub[1])
                label.pose.position.z = float(center_pub[2] + 0.16)
                label.pose.orientation.w = 1.0
                label.scale.z = label_scale
                label.color.r = 1.0
                label.color.g = 1.0
                label.color.b = 1.0
                label.color.a = 0.9
                size_txt = f" n={int(node.get('entity_size', 1))}" if use_groups else ""
                label.text = f"{label_prefix} {oid}{size_txt} q={float(node['score']):.2f} c={float(node['conf']):.2f}"
                node_arr.markers.append(label)

        edge_arr = MarkerArray()
        clear_edges = Marker()
        clear_edges.header = Header(stamp=now_msg, frame_id=self.publish_frame)
        clear_edges.action = Marker.DELETEALL
        edge_arr.markers.append(clear_edges)

        for eid, edge in enumerate(edges):
            a_pub = self._transform_point_internal_to_publish(T_pub_int, edge["a"]["centroid"])
            b_pub = self._transform_point_internal_to_publish(T_pub_int, edge["b"]["centroid"])
            cr, cg, cb, ca = self._relation_color(str(edge["relation"]))

            line = Marker()
            line.header = Header(stamp=now_msg, frame_id=self.publish_frame)
            line.ns = "crossmaps_object_graph_edge"
            line.id = int(eid)
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.pose.orientation.w = 1.0
            line.scale.x = edge_width
            line.color.r = cr
            line.color.g = cg
            line.color.b = cb
            line.color.a = ca
            line.points = [self._point_msg(a_pub), self._point_msg(b_pub)]
            edge_arr.markers.append(line)

            if publish_labels:
                mid = 0.5 * (a_pub + b_pub)
                label = Marker()
                label.header = Header(stamp=now_msg, frame_id=self.publish_frame)
                label.ns = "crossmaps_object_graph_edge_label"
                label.id = int(eid)
                label.type = Marker.TEXT_VIEW_FACING
                label.action = Marker.ADD
                label.pose.position.x = float(mid[0])
                label.pose.position.y = float(mid[1])
                label.pose.position.z = float(mid[2] + 0.06)
                label.pose.orientation.w = 1.0
                label.scale.z = max(0.07, 0.85 * label_scale)
                label.color.r = cr
                label.color.g = cg
                label.color.b = cb
                label.color.a = 0.9
                label.text = str(edge["relation"])
                edge_arr.markers.append(label)

        self.pub_obj_graph_nodes.publish(node_arr)
        self.pub_obj_graph_edges.publish(edge_arr)

    def _segment_instance_rgb(self, sid: int):
        return id_color(sid)

    def _segment_support_hash(self, sid: int):
        voxels = sorted(self.seg_support_voxels.get(int(sid), set()))
        max_sample = int(self.get_parameter("object_graph_descriptor_support_hash_sample").value)
        h = hashlib.blake2b(digest_size=12)
        h.update(str(len(voxels)).encode("ascii"))
        if voxels:
            sample = voxels
            if max_sample > 0 and len(sample) > max_sample:
                step = max(1, int(math.ceil(len(sample) / max_sample)))
                sample = sample[::step]
            h.update(np.asarray(sample, dtype=np.int32).tobytes())
        return h.hexdigest()

    def _segment_support_points_internal(self, sid: int):
        voxels = [
            key for key in self.seg_support_voxels.get(int(sid), set())
            if int(self.voxel_segment_id.get(key, -1)) == int(sid)
        ]
        if not voxels:
            return np.zeros((0, 3), dtype=np.float32)
        pts = np.zeros((len(voxels), 3), dtype=np.float32)
        for idx, key in enumerate(voxels):
            pts[idx, :] = np.asarray(self.voxel_center_xyz(key), dtype=np.float32)
        return pts

    def _segment_instance_candidate_ids(self):
        min_obs = int(self.get_parameter("segment_min_observations_to_publish").value)
        max_segments = int(self.get_parameter("segment_instance_max_segments").value)
        min_conf = float(self.get_parameter("segment_instance_min_conf").value)
        min_weight = float(self.get_parameter("segment_instance_min_weight").value)
        min_support = int(self.get_parameter("segment_instance_min_support_voxels").value)
        min_views = int(self.get_parameter("segment_instance_min_view_count").value)
        ids = []
        for sid in list(self.seg_w.keys()):
            if int(self.seg_obs_count.get(int(sid), 0)) < max(1, min_obs):
                continue
            metrics = self._segment_metrics(int(sid))
            if metrics is None:
                continue
            _, _, conf, coh01, weight = metrics
            if float(weight) < min_weight:
                continue
            if float(conf) < min_conf:
                continue
            support_n = len(self.seg_support_voxels.get(int(sid), set()))
            if support_n < max(1, min_support):
                continue
            view_count = safe_popcount(int(self.seg_viewmask.get(int(sid), 0)))
            if view_count < max(1, min_views):
                continue
            rank = (
                float(weight)
                * max(float(conf), 1e-3)
                * max(float(coh01), 0.10)
                * math.log1p(float(support_n))
                * (1.0 + 0.15 * float(view_count))
            )
            ids.append((rank, int(sid)))
        ids.sort(reverse=True)
        if max_segments > 0:
            ids = ids[:max_segments]
        return [sid for _, sid in ids]

    def publish_segment_instance_voxels(self, T_pub_int: np.ndarray):
        """One geometry pass, independent surface/entity colors; no inference.

        Keep the optional legacy topic for external consumers. New streams do
        not depend on its switch, graph display mode, or graph node budget.
        """
        outputs = []
        for parameter, publisher, entity_colors in (
            ("publish_segment_voxels", self.pub_segment_voxels, False),
            ("publish_entity_voxels", self.pub_entity_voxels, True),
            ("publish_object_instance_voxels", self.pub_obj_instance_voxels,
             bool(self.get_parameter("instance_color_by_entity").value)),
        ):
            if bool(self.get_parameter(parameter).value) and publisher.get_subscription_count() > 0:
                outputs.append((publisher, entity_colors))
        if not outputs:
            return
        max_pts = int(self.get_parameter("object_instance_cloud_max_points").value)
        max_per_seg = int(self.get_parameter("segment_instance_max_points_per_segment").value)
        pts_pub_all = []
        color_modes = {entity_colors: [] for _, entity_colors in outputs}
        point_count = 0
        candidates = self._segment_instance_candidate_ids() if bool(self.get_parameter("use_segment_memory").value) else []
        for sid in candidates:
            pts_int = self._segment_support_points_internal(int(sid))
            if pts_int.shape[0] == 0:
                continue
            if max_per_seg > 0 and pts_int.shape[0] > max_per_seg:
                step = max(1, int(math.ceil(pts_int.shape[0] / max_per_seg)))
                pts_int = pts_int[::step, :][:max_per_seg, :]
            if max_pts > 0 and (point_count + pts_int.shape[0]) > max_pts:
                keep = max(0, max_pts - point_count)
                if keep <= 0:
                    break
                pts_int = pts_int[:keep, :]
            pts_h = np.concatenate([pts_int, np.ones((pts_int.shape[0], 1), dtype=np.float32)], axis=1)
            pts_pub = (T_pub_int @ pts_h.T).T[:, :3].astype(np.float32)
            pts_pub_all.append(pts_pub)
            for entity_colors, colors in color_modes.items():
                color = self._segment_entity_rgb(sid) if entity_colors else self._segment_instance_rgb(sid)
                colors.append(np.full((pts_pub.shape[0],), pack_rgb(*color), dtype=np.float32))
            point_count += int(pts_pub.shape[0])

        stamp = self.get_clock().now().to_msg()
        points = np.concatenate(pts_pub_all, axis=0) if pts_pub_all else np.zeros((0, 3), dtype=np.float32)
        messages = {
            mode: make_cloud_xyzrgb(
                points, np.concatenate(colors) if colors else np.zeros((0,), dtype=np.float32),
                self.publish_frame, stamp,
            )
            for mode, colors in color_modes.items()
        }
        for publisher, mode in outputs:
            publisher.publish(messages[mode])

    def _segment_entity_rgb(self, sid: int):
        """Shared camera/3D palette: stable entity, unassigned, ambiguous."""
        if self.entity_memory.pending.get(int(sid), {}).get("reason") == "ambiguous_support":
            return (255, 180, 50)
        eid = self.entity_memory.segment_entity.get(int(sid))
        return (175, 175, 175) if eid is None else self._segment_instance_rgb(self._effective_entity_id(eid))

    def _effective_entity_id(self, eid):
        if eid is None or not bool(self.get_parameter("use_object_assembly").value):
            return eid
        return self.entity_memory.assemblies.canonical(eid)

    def _segment_query_score_norm_bounds(self, all_keys, sum_dict, w_dict, viewmask_dict, weight_max):
        if self.segment_query_score_norm_bounds is not None:
            return self.segment_query_score_norm_bounds
        all_res = self._compute_keyed_scores_conf(
            keys=all_keys,
            sum_dict=sum_dict,
            w_dict=w_dict,
            viewmask_dict=viewmask_dict,
            weight_max=weight_max,
            apply_view_boost=bool(self.get_parameter("use_visibility_boost").value),
            use_coh_conf=bool(self.get_parameter("use_coherence_in_confidence").value),
            use_coh_sem=bool(self.get_parameter("use_coherence_in_semantic").value),
        )
        if all_res is None:
            return None
        _, _, all_coh01s, all_scores, all_confs, _ = all_res
        all_keep = (
            (all_scores >= float(self.get_parameter("voxel_cloud_min_score").value))
            & (all_confs >= float(self.get_parameter("voxel_cloud_min_conf").value))
            & (all_coh01s >= float(self.get_parameter("voxel_cloud_min_coh01").value))
        )
        ref_scores = all_scores[all_keep]
        if ref_scores.size < 5:
            return None
        p_low = float(self.get_parameter("norm_low_percentile").value)
        p_high = float(self.get_parameter("norm_high_percentile").value)
        lo = float(np.percentile(ref_scores, p_low))
        hi = float(np.percentile(ref_scores, p_high))
        self.segment_query_score_norm_bounds = (lo, hi)
        return self.segment_query_score_norm_bounds

    def _segment_query_support_score(self, sid: int):
        keys = [
            key for key in self.seg_support_voxels.get(int(sid), set())
            if int(self.voxel_segment_id.get(key, -1)) == int(sid)
        ]
        if not keys or self.text_emb is None:
            return {"selected_keys": [], "score": 0.0, "conf": 0.0}
        all_keys, sum_dict, w_dict, viewmask_dict, weight_max = self._voxel_memory_bundle("stm_ltm")
        keys = [key for key in keys if key in sum_dict and key in w_dict]
        if not keys:
            return {"selected_keys": [], "score": 0.0, "conf": 0.0}
        res = self._compute_keyed_scores_conf(
            keys=keys,
            sum_dict=sum_dict,
            w_dict=w_dict,
            viewmask_dict=viewmask_dict,
            weight_max=weight_max,
            apply_view_boost=bool(self.get_parameter("use_visibility_boost").value),
            use_coh_conf=bool(self.get_parameter("use_coherence_in_confidence").value),
            use_coh_sem=bool(self.get_parameter("use_coherence_in_semantic").value),
        )
        if res is None:
            return {"selected_keys": [], "score": 0.0, "conf": 0.0}
        scored_keys, _, _, scores_np, confs_np, _ = res
        bounds = self._segment_query_score_norm_bounds(all_keys, sum_dict, w_dict, viewmask_dict, weight_max)
        if bounds is not None:
            lo, hi = bounds
            score01 = np.clip((scores_np - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        else:
            score01 = np.clip(scores_np, 0.0, 1.0)
        min_score01 = float(self.get_parameter("segment_query_support_min_score01").value)
        min_conf = float(self.get_parameter("segment_query_min_conf").value)
        keep = (score01 >= min_score01) & (confs_np >= min_conf)
        if not np.any(keep):
            best_idx = int(np.argmax(score01)) if score01.size else 0
            return {
                "selected_keys": [],
                "score": float(score01[best_idx]) if score01.size else 0.0,
                "conf": float(confs_np[best_idx]) if confs_np.size else 0.0,
            }
        selected_keys = [key for key, keep_i in zip(scored_keys, keep) if bool(keep_i)]
        min_selected_frac = float(np.clip(float(self.get_parameter("segment_query_support_min_selected_frac").value), 0.0, 1.0))
        if min_selected_frac > 0.0 and (float(len(selected_keys)) / float(max(len(keys), 1))) < min_selected_frac:
            best_idx = int(np.argmax(score01)) if score01.size else 0
            return {
                "selected_keys": [],
                "score": float(score01[best_idx]) if score01.size else 0.0,
                "conf": float(confs_np[best_idx]) if confs_np.size else 0.0,
            }
        selected_scores = score01[keep]
        selected_confs = confs_np[keep]
        order = np.argsort(selected_scores * np.maximum(selected_confs, 1e-3))[::-1]
        top_n = max(1, min(16, int(math.ceil(0.25 * order.size))))
        top = order[:top_n]
        return {
            "selected_keys": selected_keys,
            "score": float(np.mean(selected_scores[top])),
            "conf": float(np.mean(selected_confs[top])),
        }

    def _ranked_query_segments(self):
        self.segment_query_support_cache = {}
        self.segment_query_score_norm_bounds = None
        min_score = float(self.get_parameter("segment_query_min_score").value)
        min_conf = float(self.get_parameter("segment_query_min_conf").value)
        min_obs = int(self.get_parameter("segment_min_observations_to_publish").value)
        min_support = int(self.get_parameter("segment_query_min_support_voxels").value)
        use_support = bool(self.get_parameter("segment_query_use_support_scores").value)
        ranked = []
        for sid in list(self.seg_w.keys()):
            if int(self.seg_obs_count.get(int(sid), 0)) < max(1, min_obs):
                continue
            support_n = len(self.seg_support_voxels.get(int(sid), set()))
            if support_n < max(1, min_support):
                continue
            metrics = self._segment_metrics(int(sid))
            if metrics is None:
                continue
            _, emb_score, emb_conf, coh01, _ = metrics
            score = float(emb_score)
            conf = float(emb_conf)
            if use_support:
                support = self._segment_query_support_score(int(sid))
                self.segment_query_support_cache[int(sid)] = support
                selected_n = len(support.get("selected_keys", []))
                if selected_n < max(1, min_support):
                    continue
                score = float(support.get("score", score))
                conf = float(max(conf, support.get("conf", 0.0)))
            else:
                self.segment_query_support_cache[int(sid)] = {
                    "selected_keys": list(self.seg_support_voxels.get(int(sid), set())),
                    "score": score,
                    "conf": conf,
                }
            if score < min_score or conf < min_conf:
                continue
            ranked.append((float(score * max(conf, 1e-3)), int(sid), float(score), float(conf), float(coh01)))
        ranked.sort(reverse=True)
        top_k = int(self.get_parameter("segment_query_top_k").value)
        if top_k > 0:
            ranked = ranked[:top_k]
        return ranked

    def publish_segment_query_voxels(self, T_pub_int: np.ndarray):
        if (not bool(self.get_parameter("publish_object_query_voxels").value)
                or not self.publisher_has_subscribers(self.pub_obj_query_voxels)):
            return
        max_pts = int(self.get_parameter("object_query_cloud_max_points").value)
        color_mode = str(self.get_parameter("segment_query_color_mode").value).strip().lower()
        pts_pub_all = []
        rgb_all = []
        point_count = 0
        for _, sid, score, conf, _ in self._ranked_query_segments():
            selected_keys = list(self.segment_query_support_cache.get(int(sid), {}).get("selected_keys", []))
            if not selected_keys:
                continue
            pts_pub = self._project_internal_voxels_to_publish_points(T_pub_int, selected_keys)
            if pts_pub.shape[0] == 0:
                continue
            if max_pts > 0 and (point_count + pts_pub.shape[0]) > max_pts:
                keep = max(0, max_pts - point_count)
                if keep <= 0:
                    break
                pts_pub = pts_pub[:keep, :]
            if color_mode in ("score", "query", "heat"):
                r, g, b = blue_to_red(float(score) * float(conf))
            else:
                r, g, b = self._segment_instance_rgb(int(sid))
            pts_pub_all.append(pts_pub)
            rgb_all.append(np.full((pts_pub.shape[0],), pack_rgb(r, g, b), dtype=np.float32))
            point_count += int(pts_pub.shape[0])

        stamp = self.get_clock().now().to_msg()
        if not pts_pub_all:
            self.pub_obj_query_voxels.publish(
                make_cloud_xyzrgb(
                    np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    self.publish_frame,
                    stamp,
                )
            )
            return
        self.pub_obj_query_voxels.publish(
            make_cloud_xyzrgb(
                np.concatenate(pts_pub_all, axis=0).astype(np.float32),
                np.concatenate(rgb_all, axis=0).astype(np.float32),
                self.publish_frame,
                stamp,
            )
        )

    def publish_segment_unassigned_query_voxels(self, T_pub_int: np.ndarray):
        if (not bool(self.get_parameter("publish_object_unassigned_query_voxels").value)
                or not self.publisher_has_subscribers(self.pub_obj_unassigned_query_voxels)):
            return
        stamp = self.get_clock().now().to_msg()
        empty = make_cloud_xyzrgb(
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            self.publish_frame,
            stamp,
        )
        bundle = self._query_voxel_score01_bundle()
        if bundle is None:
            self.pub_obj_unassigned_query_voxels.publish(empty)
            return
        keys, score01, confs = bundle
        if not keys:
            self.pub_obj_unassigned_query_voxels.publish(empty)
            return
        min_score01 = float(self.get_parameter("object_unassigned_query_min_score01").value)
        idxs = [i for i, key in enumerate(keys) if key not in self.voxel_segment_id and float(score01[i]) >= min_score01]
        if not idxs:
            self.pub_obj_unassigned_query_voxels.publish(empty)
            return
        idxs_np = np.asarray(idxs, dtype=np.int32)
        rank = score01[idxs_np] * np.maximum(confs[idxs_np], 1e-3)
        max_pts = int(self.get_parameter("object_unassigned_query_cloud_max_points").value)
        if max_pts > 0 and idxs_np.size > max_pts:
            top = np.argpartition(rank, -max_pts)[-max_pts:]
            idxs_np = idxs_np[top]
        selected_keys = [keys[int(i)] for i in idxs_np]
        pts_pub = self._project_internal_voxels_to_publish_points(T_pub_int, selected_keys)
        rgb = np.zeros((pts_pub.shape[0],), dtype=np.float32)
        for out_i, src_i in enumerate(idxs_np):
            r, g, b = blue_to_red(float(score01[int(src_i)]))
            rgb[out_i] = pack_rgb(r, g, b)
        self.pub_obj_unassigned_query_voxels.publish(make_cloud_xyzrgb(pts_pub.astype(np.float32), rgb, self.publish_frame, stamp))


    def _aggregate_entity_group(self, children: List[dict], entity_id: int) -> dict:
        if not children:
            return {}
        weights = np.array([max(1.0, float(c.get("support_count", 1))) for c in children], dtype=np.float32)
        weight_sum = float(np.sum(weights))
        emb_sum = np.zeros_like(children[0]["embedding"], dtype=np.float32)
        centroid = np.zeros((3,), dtype=np.float32)
        for child, w in zip(children, weights):
            emb_sum = (emb_sum + float(w) * np.asarray(child["embedding"], dtype=np.float32)).astype(np.float32)
            centroid = (centroid + float(w) * np.asarray(child["centroid"], dtype=np.float32)).astype(np.float32)
        emb_norm = float(np.linalg.norm(emb_sum))
        emb = (emb_sum / emb_norm).astype(np.float32) if emb_norm > 1e-6 else children[0]["embedding"].astype(np.float32)
        dino_children = [
            (child, float(w))
            for child, w in zip(children, weights)
            if child.get("dino_embedding") is not None
        ]
        dino_embedding = None
        if dino_children:
            dino_sum = np.zeros_like(
                np.asarray(dino_children[0][0]["dino_embedding"], dtype=np.float32)
            )
            dino_weight = 0.0
            for child, weight in dino_children:
                child_dino = np.asarray(child["dino_embedding"], dtype=np.float32)
                if child_dino.shape != dino_sum.shape:
                    continue
                dino_sum += float(weight) * child_dino
                dino_weight += float(weight)
            dino_norm = float(np.linalg.norm(dino_sum))
            if dino_weight > 0.0 and dino_norm > 1e-6:
                dino_embedding = (dino_sum / dino_norm).astype(np.float32)
        centroid = (centroid / max(weight_sum, 1e-6)).astype(np.float32)
        bmin = np.min(np.stack([c["bbox_min"] for c in children], axis=0), axis=0).astype(np.float32)
        bmax = np.max(np.stack([c["bbox_max"] for c in children], axis=0), axis=0).astype(np.float32)
        segment_ids = sorted(int(c["id"]) for c in children)

        pair_sems = []
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                pair_sems.append(float(np.dot(children[i]["embedding"], children[j]["embedding"])))
        consistency = float(min(pair_sems)) if pair_sems else 1.0
        mean_conf = float(np.mean([float(c.get("conf", 0.0)) for c in children]))
        support_count = int(sum(int(c.get("support_count", 0)) for c in children))
        obs_count = int(sum(int(c.get("observations", 1)) for c in children))
        now_s = self.now_sec()
        last_seen = max(float(c.get("last_seen_s", 0.0)) for c in children)
        first_seen = min(float(c.get("first_seen_s", last_seen)) for c in children)
        age_s = max(0.0, now_s - last_seen)
        h = hashlib.blake2b(digest_size=12)
        for sid in segment_ids:
            h.update(str(sid).encode("ascii"))
            h.update(b",")
        return {
            "id": int(entity_id),
            "entity_id": int(entity_id),
            "segment_ids": segment_ids,
            "entity_size": int(len(children)),
            "embedding": emb,
            "dino_embedding": dino_embedding,
            # Different-looking surfaces do not make identity less reliable.
            "conf": float(np.clip(mean_conf, 0.0, 1.0)),
            "coh01": float(np.mean([float(c.get("coh01", 0.0)) for c in children])),
            "score": float(max(float(c.get("score", 0.0)) for c in children)),
            "weight": float(sum(float(c.get("weight", 0.0)) for c in children)),
            "centroid": centroid,
            "bbox_min": bmin,
            "bbox_max": bmax,
            "support_count": support_count,
            "support_hash": h.hexdigest(),
            "age_s": float(age_s),
            "first_seen_s": float(first_seen),
            "last_seen_s": float(last_seen),
            "observations": obs_count,
            "view_count": int(max(int(c.get("view_count", 0)) for c in children)),
            "state": self._entity_dynamic_state(age_s, mean_conf, support_count, obs_count),
            "shape": self._object_shape_traits(bmin, bmax, support_count),
            "semantic_consistency": float(consistency),
            "children": children,
        }

    def _entity_group_graph_candidates(self, segment_nodes: List[dict], assembled=True) -> List[dict]:
        """Aggregate persistent memberships, never infer identity in a renderer."""
        by_sid = {int(node["id"]): node for node in segment_nodes}
        entities = []
        groups = {}
        for eid, record in self.entity_memory.records.items():
            root = self._effective_entity_id(eid) if assembled else eid
            groups.setdefault(root, []).append(record)
        for eid, records in groups.items():
            members = set().union(*(record.members for record in records))
            children = [by_sid[sid] for sid in sorted(members) if sid in by_sid]
            if not children:
                continue
            node = self._aggregate_entity_group(children, eid)
            node.update(
                id=eid, entity_id=eid, observations=max(r.observations for r in records),
                first_seen_s=min(r.first_seen for r in records), last_seen_s=max(r.last_seen for r in records),
                age_s=max(0.0, self.now_sec() - max(r.last_seen for r in records)),
                view_count=len(set().union(*(r.views for r in records))), last_observation_seq=max(r.last_seq for r in records),
                objectness="unverified", membership_basis="dino_geometry_assembly" if len(records) > 1 else "surface_continuity",
                component_entity_ids=sorted(r.id for r in records),
            )
            node["state"] = "grouped" if len(records) > 1 else "tracked" if node["observations"] >= 2 else "tentative"
            entities.append(node)
        return entities

    def _segment_graph_candidates(self, *, entity_view=None, for_evidence=False):
        if entity_view is None:
            entity_view = bool(self.get_parameter("use_entity_grouping").value)
        nodes = []
        for sid in sorted(self.seg_w):
            metrics = self._segment_metrics(int(sid))
            if metrics is None:
                continue
            emb_u, score, conf, coh01, weight = metrics
            pts = self._segment_support_points_internal(int(sid))
            if pts.shape[0] == 0:
                continue
            pad = max(0.5 * float(self.voxel_size), 0.01)
            bmin = np.min(pts, axis=0).astype(np.float32) - pad
            bmax = np.max(pts, axis=0).astype(np.float32) + pad
            last = float(self.seg_last_seen.get(sid, 0.0))
            nodes.append({
                "id": sid, "entity_id": self._effective_entity_id(self.entity_memory.segment_entity.get(sid)),
                "embedding": emb_u.astype(np.float32),
                "dino_embedding": self.segment_dino_embedding(sid),
                "conf": float(conf), "score": float(score), "coh01": float(coh01),
                "weight": float(weight), "centroid": np.mean(pts, axis=0).astype(np.float32),
                "bbox_min": bmin, "bbox_max": bmax, "support_count": int(pts.shape[0]),
                "support_hash": self._segment_support_hash(sid),
                "age_s": max(0.0, self.now_sec() - last),
                "first_seen_s": float(self.seg_first_seen.get(sid, last)), "last_seen_s": last,
                "observations": int(self.seg_obs_count.get(sid, 0)),
                "last_observation_seq": int(self.seg_last_obs_seq.get(sid, -1)),
                "view_count": safe_popcount(int(self.seg_viewmask.get(sid, 0))),
                "state": "tracked" if sid in self.entity_memory.segment_entity else "unassigned",
                "shape": self._object_shape_traits(bmin, bmax, int(pts.shape[0])),
            })
        if entity_view:
            nodes = self._entity_group_graph_candidates(nodes)
        if for_evidence:
            return nodes
        # Presentation filters are applied *after* entity aggregation. Hiding a
        # child or lowering the node budget must never change object identity.
        min_obs = max(int(self.get_parameter("object_graph_min_observations").value),
                      int(self.get_parameter("segment_min_observations_to_publish").value))
        max_age = float(self.get_parameter("object_graph_max_age_s").value)
        nodes = [n for n in nodes if
                 n["observations"] >= min_obs and
                 n["weight"] >= float(self.get_parameter("object_graph_min_weight").value) and
                 n["conf"] >= float(self.get_parameter("object_graph_min_conf").value) and
                 n["support_count"] >= int(self.get_parameter("object_graph_min_support_voxels").value) and
                 n["view_count"] >= int(self.get_parameter("object_graph_min_view_count").value) and
                 (max_age <= 0.0 or n["age_s"] <= max_age)]
        nodes.sort(key=lambda n: (-n["conf"] * math.log1p(n["support_count"]), n["id"]))
        limit = int(self.get_parameter("object_graph_max_nodes").value)
        return nodes[:limit] if limit > 0 else nodes

    def on_entity_correction(self, msg):
        """Explicit non-destructive correction; never guesses semantic part-of."""
        try:
            command = json.loads(msg.data)
            if not isinstance(command, dict):
                raise ValueError("Expected a correction object")
            action = command.get("action")
            if action in ("separate_entities", "allow_entity_group"):
                ids = command.get("entity_ids")
                if not isinstance(ids, list) or len(ids) != 2 or ids[0] == ids[1] or any(
                    type(e) is not int or e not in self.entity_memory.records for e in ids
                ):
                    raise ValueError("entity_ids must contain two distinct base entity IDs")
                method = self.entity_memory.assemblies.separate if action == "separate_entities" else self.entity_memory.assemblies.allow
                method(*ids)
                self._mark_state_dirty()
                return
            if action != "detach_segment":
                raise ValueError("Unknown entity correction action")
            sid = command.get("segment_id")
            if type(sid) is not int or sid not in self.seg_w:
                raise ValueError("segment_id must identify an existing surface")
            if self.entity_memory.detach_segment(sid):
                self._mark_state_dirty()
                self.get_logger().info(f"Detached segment {sid}; geometry preserved, new membership will remain conservative.")
        except (ValueError, TypeError) as error:
            self.get_logger().warn(f"Entity correction rejected: {error}")

    def _update_identity_graph(self, obs_seq, now_s, cam_int):
        """Called once after fusion/maintenance, never by the publishing timer."""
        started = time.perf_counter()
        self.entity_memory.config = IdentityConfig(
            birth_observations=max(1, int(self.get_parameter("entity_birth_observations").value)),
            membership_observations=max(1, int(self.get_parameter("entity_membership_observations").value)),
            min_overlap_voxels=max(1, int(self.get_parameter("entity_reassoc_min_voxels").value)),
            min_overlap_fraction=float(np.clip(self.get_parameter("entity_reassoc_min_overlap").value, 0.0, 1.0)),
            ambiguity_margin=float(np.clip(self.get_parameter("entity_reassoc_ambiguity_margin").value, 0.0, 1.0)),
            min_dino_cos=float(self.get_parameter("segment_assoc_min_dino_cos").value),
            support_limit=max(8, int(self.get_parameter("entity_memory_support_limit").value)),
            dormant_retention_s=max(0.0, float(self.get_parameter("entity_dormant_retention_s").value)),
            view_cell_m=max(0.01, float(self.get_parameter("entity_view_cell_m").value)),
            neighborhood_tiebreak=bool(self.get_parameter("entity_neighborhood_tiebreak").value),
            neighborhood_overlap_gap=float(np.clip(self.get_parameter("entity_neighborhood_overlap_gap").value, 0., .15)),
            neighborhood_min_dino=0.92-0.24*float(np.clip(self.get_parameter("entity_grouping_sensitivity").value, 0., 1.)),
            voxel_size_m=self.voxel_size,
            neighborhood_max_extent_m=max(.1, float(self.get_parameter("entity_assembly_max_extent_m").value)),
        )
        segments = {}
        observed = set()
        for sid in self.seg_w:
            support = {key for key in self.seg_support_voxels.get(sid, ())
                       if self.voxel_segment_id.get(key) == sid}
            dino = self.segment_dino_embedding(sid)
            segments[sid] = {"support": support, "dino": () if dino is None else tuple(float(v) for v in dino)}
            if self.seg_last_obs_seq.get(sid, -1) == obs_seq:
                observed.add(sid)
                interior = interior_descriptor(dino, getattr(self, 'dino_base_dim', 0))
                clip = self.segment_embedding_and_coherence(sid)[0]
                points = np.array([self.voxel_center_xyz(k) for k in sorted(support)], dtype=np.float32)
                role = surface_role(points, self.voxel_size)
                if role == 'vertical_structural_surface' and len(points) and np.ptp(points[:, 2])+self.voxel_size < 1.2:
                    role = 'object_panel_candidate'
                segments[sid].update(group_dino=() if interior is None else tuple(float(v) for v in interior),
                    clip=() if clip is None else tuple(float(v) for v in clip), surface_role=role,
                    fresh_dino=getattr(self, 'seg_last_dino_seq', {}).get(sid) == obs_seq)
        self.entity_memory.update(obs_seq, now_s, segments, observed, cam_int)
        segment_nodes = self._segment_graph_candidates(entity_view=False, for_evidence=True)
        self._update_object_assemblies(obs_seq, now_s, cam_int, segment_nodes, observed)
        entity_nodes = self._entity_group_graph_candidates(segment_nodes)
        alpha = float(np.clip(self.get_parameter("object_graph_edge_ema_alpha").value, 0.0, 1.0))
        max_misses = max(0, int(self.get_parameter("object_graph_edge_max_misses").value))
        for nodes, memory, alive in (
            (segment_nodes, self.segment_edges, set(self.seg_w)),
            (entity_nodes, self.entity_edges, {node["id"] for node in entity_nodes}),
        ):
            evaluated = [node for node in nodes if node["last_observation_seq"] == obs_seq]
            raw = self._raw_object_graph_edges(evaluated)
            edges = [{"from": edge["a"]["id"], "to": edge["b"]["id"],
                      **{key: value for key, value in edge.items() if key not in ("a", "b")}}
                     for edge in raw]
            memory.update(obs_seq, edges, {node["id"] for node in evaluated}, alive, max_misses, alpha)
        self.last_identity_update_ms = (time.perf_counter() - started) * 1000.0
        self._profile_add_ms("identity.update", self.last_identity_update_ms)
        self._profile_add_value("identity.pending_segments", len(self.entity_memory.pending))
        self._update_entity_display(obs_seq, now_s)
        self.publish_entity_id_projection()
        self._mark_state_dirty()

    def _update_entity_display(self, seq, now_s):
        projection = getattr(self, 'last_segment_projection_context', None)
        if projection is None or projection[0].get('observation_seq') != seq:
            return
        context = projection[0]
        if not hasattr(self, 'entity_display_memory'):
            self.entity_display_memory = DominantOverlay()
        def mapper(sid):
            eid = self.entity_memory.segment_entity.get(sid) if sid in self.seg_w else None
            return self._effective_entity_id(eid)
        context['entity_display_decisions'] = self.entity_display_memory.update(
            seq, now_s, context.get('decisions', []), mapper, self.voxel_size)

    def _update_object_assemblies(self, seq, now_s, camera, segment_nodes, observed):
        assembly = self.entity_memory.assemblies
        assembly.config = AssemblyConfig(
            enabled=bool(self.get_parameter("use_object_assembly").value),
            sensitivity=float(np.clip(self.get_parameter("entity_grouping_sensitivity").value, 0.0, 1.0)),
            confirm_observations=max(2, int(self.get_parameter("entity_assembly_confirm_observations").value)),
            max_extent_m=max(0.1, float(self.get_parameter("entity_assembly_max_extent_m").value)),
            group_min_dino=float(np.clip(self.get_parameter("entity_assembly_group_min_dino_cos").value, .4, .9)),
        )
        nodes = {}
        for node in self._entity_group_graph_candidates(segment_nodes, assembled=False):
            eid = node["id"]
            support = set().union(*(self.seg_support_voxels.get(s, set()) for s in node["segment_ids"]))
            full_support_count = len(support)
            support, geometry_reliable = dominant_surface_core(support)
            dino = interior_descriptor(node.get("dino_embedding"), getattr(self, "dino_base_dim", 0))
            dino_mass = sum(float(self.seg_dino_w.get(s, 0.0)) for s in node["segment_ids"])
            pts = np.array([self.voxel_center_xyz(key) for key in sorted(support)], dtype=np.float32)
            pad = .5*self.voxel_size
            core_min = np.min(pts, axis=0)-pad if len(pts) else node['bbox_min']
            core_max = np.max(pts, axis=0)+pad if len(pts) else node['bbox_max']
            role = surface_role(pts, self.voxel_size)
            if role == 'vertical_structural_surface' and core_max[2]-core_min[2] < 1.2:
                role = 'object_panel_candidate'
            nodes[eid] = {"min": tuple(float(v) for v in core_min),
                          "max": tuple(float(v) for v in core_max), "support": support,
                          "geometry_reliable": geometry_reliable,
                          "geometry_basis": "dominant_connected_surface",
                          "excluded_outlier_voxels": full_support_count-len(support),
                          "dino": () if dino is None or dino_mass < 0.1 else tuple(float(v) for v in dino),
                          "surface_role": role,
                          "fresh_dino": any(getattr(self, "seg_last_dino_seq", {}).get(s) == seq for s in node["segment_ids"])}
        visible = {}
        for sid, count in getattr(self, "last_visible_segment_counts", {}).items():
            eid = self.entity_memory.segment_entity.get(sid)
            if eid is not None:
                visible[eid] = visible.get(eid, 0) + count
        shared = set()
        for mask in getattr(self, "last_mask_attribution", []):
            counts = {}
            for sid, count in zip(mask.get("segments", []), mask.get("counts", [])):
                eid = self.entity_memory.segment_entity.get(sid)
                if eid is not None:
                    counts[eid] = counts.get(eid, 0) + count
            covered = sorted(e for e, count in counts.items() if count >= 3 and count/max(1, visible.get(e, 0)) >= 0.65)
            shared.update((a, b) for i, a in enumerate(covered) for b in covered[i+1:])
        exclusions = [(self.entity_memory.segment_entity[sid], old)
                      for sid, old in self.entity_memory.assembly_exclusions.items()
                      if sid in self.entity_memory.segment_entity and self.entity_memory.segment_entity[sid] != old]
        observed_entities = {self.entity_memory.segment_entity[s] for s in observed if s in self.entity_memory.segment_entity}
        recording = bool(self.get_parameter("record_identity_replay").value)
        if recording and not getattr(self, "evidence_record_was_enabled", False):
            self.evidence_recorder = None
            self.evidence_record_error_reported = False
        self.evidence_record_was_enabled = recording
        recorder = getattr(self, "evidence_recorder", None)
        recording = recording and (recorder is None or (recorder.frames < recorder.max_frames and not recorder.error))
        initial = assembly.to_dict() if recording and not getattr(self, "evidence_recorder", None) else None
        assembly.update(seq, now_s, nodes, observed_entities,
                        shared, self.voxel_size, camera, exclusions)
        if recording:
            try:
                if not getattr(self, "evidence_recorder", None):
                    self.evidence_recorder = EvidenceRecorder(self.world_model_id,
                        int(self.get_parameter("identity_replay_max_frames").value))
                self.evidence_recorder.record(dict(seq=int(seq), time_s=float(now_s), world_model_id=self.world_model_id,
                    source_id=str(self.get_parameter("world_model_source_id").value),
                    observation_source_id=getattr(self, "observation_source_id", self.world_model_id),
                    frame_id=str(self.internal_frame),
                    sensor_depth_stamp_ns=int(self.last_accepted_depth_stamp_ns), voxel_size=float(self.voxel_size),
                    camera_position=[float(v) for v in camera] if camera is not None else None,
                    assembly_config=assembly.to_dict()['config'],
                    nodes={str(e): {**n, 'support': sorted(n['support'])} for e,n in nodes.items()},
                    observed=sorted(observed_entities), shared_masks=sorted(shared), exclusions=exclusions,
                    expected_aliases=dict(assembly.aliases), mask_attribution=getattr(self, 'last_mask_attribution', []),
                    surface_tracking=self.surface_consensus.diagnostics() if hasattr(self, 'surface_consensus') else {}), initial)
            except (OSError, ValueError, TypeError) as error:
                if getattr(self, "evidence_recorder", None):
                    self.evidence_recorder.error = str(error)
                if not getattr(self, "evidence_record_error_reported", False):
                    self.get_logger().warn(f"Identity evidence recording stopped: {error}")
                    self.evidence_record_error_reported = True
        self.entity_surface_roles = {}
        for eid, node in nodes.items():
            root = self._effective_entity_id(eid)
            previous = self.entity_surface_roles.get(root)
            role = node['surface_role']
            self.entity_surface_roles[root] = role if previous in (None, role) else 'mixed_surface_roles'

    def publish_entity_views(self, T_pub_int: np.ndarray):
        # Scores are shared only within this serialized callback. Drop the cache
        # before accepting another observation, query, parameter or decay update.
        self._entity_score_context = {}
        try:
            for name, callback in (("instances", self.publish_segment_instance_voxels),
                                   ("query", self.publish_segment_query_voxels),
                                   ("unassigned", self.publish_segment_unassigned_query_voxels),
                                   ("graph", self.publish_object_graph)):
                if name != "instances" and not bool(self.get_parameter("use_object_abstraction").value):
                    break
                started = time.perf_counter()
                callback(T_pub_int)
                self._profile_add_elapsed(f"pub.entity_{name}", started)
        finally:
            self._entity_score_context = None

    def _occ_value_at_publish_xy(self, x: float, y: float) -> Optional[int]:
        if self.occ is None or self.occ["frame_id"] != self.publish_frame:
            return None
        j = int((x - self.occ["ox"]) / self.occ["res"])
        i = int((y - self.occ["oy"]) / self.occ["res"])
        if not (0 <= i < self.occ["h"] and 0 <= j < self.occ["w"]):
            return None
        return int(self.occ["data"][i, j])

    def _decay_semantic_cell_for_free_space(self, key, sum_dict, w_dict, drop_fn, decay: float):
        if key not in w_dict:
            return
        if key in sum_dict:
            sum_dict[key] = (sum_dict[key] * decay).astype(np.float32)
        w_dict[key] = float(w_dict[key] * decay)
        if w_dict[key] < 1e-3:
            drop_fn(key)
        self._mark_state_dirty()

    def apply_cell_negative_evidence(self, T_pub_int: np.ndarray):
        if not bool(self.get_parameter("use_cell_negative_evidence").value):
            return
        if self.occ is None or self.occ["frame_id"] != self.publish_frame:
            return

        free_occ_max = int(self.get_parameter("cell_free_occ_max").value)
        decay = float(np.clip(float(self.get_parameter("cell_negative_decay").value), 0.0, 1.0))
        max_cells = int(self.get_parameter("cell_negative_max_per_tick").value)
        if decay >= 1.0 or max_cells <= 0:
            return

        checked = 0
        for source in ("stm", "ltm"):
            keys = list(self.stm_w.keys()) if source == "stm" else list(self.ltm_w.keys())
            for key in keys:
                if checked >= max_cells:
                    return
                checked += 1
                cx_i, cy_i = self.cell_center_xy(key[0], key[1])
                p_int = np.array([cx_i, cy_i, 0.0, 1.0], dtype=np.float32)
                p_pub = (T_pub_int @ p_int)[:3]
                occ = self._occ_value_at_publish_xy(float(p_pub[0]), float(p_pub[1]))
                if occ is None or occ < 0 or occ > free_occ_max:
                    continue
                if source == "stm":
                    self._decay_semantic_cell_for_free_space(
                        key, self.stm_sum, self.stm_w, self._drop_stm_cell, decay
                    )
                else:
                    self._decay_semantic_cell_for_free_space(
                        key, self.ltm_sum, self.ltm_w, self._drop_ltm_cell, decay
                    )

    # -------------------------
    # Publish grids (STM + LTM) + promotions
    # -------------------------
    def on_publish(self):
        started = time.perf_counter()
        if started < getattr(self, "_next_publish_perf", 0.):
            self.publish_skipped_ticks = getattr(self, "publish_skipped_ticks", 0) + 1
            return
        try:
            self._publish_cycle()
        finally:
            elapsed = time.perf_counter()-started
            self.publish_last_ms = elapsed*1000.
            duty = float(np.clip(self.get_parameter("publish_max_duty_cycle").value, .05, .9))
            self._next_publish_perf = started + elapsed/duty

    def _publish_cycle(self):
        self.publish_counter += 1
        pub_t0 = time.perf_counter()
        t0 = time.perf_counter()
        self.ensure_query_embedding()
        self.publish_history_cloud()
        self._profile_add_elapsed("pub.query_history", t0)

        t0 = time.perf_counter()
        self.apply_finished_semantic_jobs()
        self._profile_add_elapsed("pub.async_apply", t0)

        # empty defaults
        empty_sem = -np.ones((self.grid_height, self.grid_width), dtype=np.int8)
        empty_conf = -np.ones((self.grid_height, self.grid_width), dtype=np.int8)
        empty_top1 = -np.ones((self.grid_height, self.grid_width), dtype=np.int8)

        if self.text_emb is None:
            if self.has_loaded_grid_snapshot:
                self.publish_cached_grids()
            else:
                self._set_latest_grid("stm_semantic", empty_sem)
                self._set_latest_grid("stm_confidence", empty_conf)
                self._set_latest_grid("stm_top1", empty_top1)
                self._set_latest_grid("ltm_semantic", empty_sem)
                self._set_latest_grid("ltm_confidence", empty_conf)
                self._set_latest_grid("ltm_top1", empty_top1)
                self.publish_cached_grids()
            self._profile_add_elapsed("pub.total", pub_t0)
            self._profile_report_if_due()
            return

        # decay some STM cells
        now_s = self.now_sec()
        t0 = time.perf_counter()
        stm_keys = list(self.stm_w.keys())
        for k in stm_keys[: min(3000, len(stm_keys))]:
            self.decay_stm_cell(k, now_s)
        vox_decay_max = int(self.get_parameter("voxel_decay_max_per_tick").value)
        if bool(self.get_parameter("use_semantic_voxels").value) and vox_decay_max != 0:
            vox_stm_keys = list(self.vox_stm_w.keys())
            decay_keys = vox_stm_keys if vox_decay_max < 0 else vox_stm_keys[: min(vox_decay_max, len(vox_stm_keys))]
            for k in decay_keys:
                self.decay_vox_stm_cell(k, now_s)
        self._profile_add_elapsed("pub.decay", t0)

        # need transform internal->publish for both STM and LTM grids
        t0 = time.perf_counter()
        T = self.lookup_saved_or_live_T_publish_internal()
        self._profile_add_elapsed("pub.tf_lookup", t0)
        if T is None:
            if self.has_loaded_grid_snapshot:
                self.publish_cached_grids()
            self._profile_add_elapsed("pub.total", pub_t0)
            self._profile_report_if_due()
            return

        use_view = bool(self.get_parameter("use_visibility_boost").value)
        use_coh_conf = bool(self.get_parameter("use_coherence_in_confidence").value)
        use_coh_sem = bool(self.get_parameter("use_coherence_in_semantic").value)
        t0 = time.perf_counter()
        voxel_promote_every_n = max(1, int(self.get_parameter("voxel_promote_every_n").value))
        if ((self.publish_counter - 1) % voxel_promote_every_n) == 0:
            self.promote_voxel_memory(now_s)
        self._profile_add_elapsed("pub.voxel_promote", t0)
        t0 = time.perf_counter()
        self.publish_voxel_heatmap(T)
        self._profile_add_elapsed("pub.voxel_clouds", t0)

        t0 = time.perf_counter()
        self.apply_cell_negative_evidence(T)
        self._profile_add_elapsed("pub.cell_negative", t0)

        t0 = time.perf_counter()
        self.publish_entity_views(T)
        self._profile_add_elapsed("pub.entity_views", t0)

        if not bool(self.get_parameter("compute_2d_grids").value):
            self._profile_add_elapsed("pub.total", pub_t0)
            self._profile_report_if_due()
            return

        # -------------------------
        # STM -> compute and publish
        # -------------------------
        t0 = time.perf_counter()
        stm_keys = list(self.stm_w.keys())
        if len(stm_keys) < 20:
            self._set_latest_grid("stm_semantic", empty_sem)
            self._set_latest_grid("stm_confidence", empty_conf)
            self._set_latest_grid("stm_top1", empty_top1)
            if bool(self.get_parameter("publish_stm_semantic_grid").value):
                self.publish_grid(empty_sem, self.pub_stm_sem)
            if bool(self.get_parameter("publish_stm_confidence_grid").value):
                self.publish_grid(empty_conf, self.pub_stm_conf)
            if bool(self.get_parameter("publish_stm_top1_grid").value):
                self.publish_grid(empty_top1, self.pub_stm_top1)
        else:
            stm_cells_pub, stm_keys_keep = self._project_internal_cells_to_publish_cells(T, stm_keys)

            res_stm = self._compute_cell_scores_conf(
                keys=stm_keys_keep,
                sum_dict=self.stm_sum,
                w_dict=self.stm_w,
                viewmask_dict=self.stm_viewmask,
                weight_max=float(self.get_parameter("cell_weight_max").value),
                apply_view_boost=use_view,
                use_coh_conf=use_coh_conf,
                use_coh_sem=use_coh_sem,
            )

            if res_stm is None or len(stm_cells_pub) < 10:
                self._set_latest_grid("stm_semantic", empty_sem)
                self._set_latest_grid("stm_confidence", empty_conf)
                self._set_latest_grid("stm_top1", empty_top1)
                if bool(self.get_parameter("publish_stm_semantic_grid").value):
                    self.publish_grid(empty_sem, self.pub_stm_sem)
                if bool(self.get_parameter("publish_stm_confidence_grid").value):
                    self.publish_grid(empty_conf, self.pub_stm_conf)
                if bool(self.get_parameter("publish_stm_top1_grid").value):
                    self.publish_grid(empty_top1, self.pub_stm_top1)
            else:
                emb_u, coh01s, scores_np, confs_np, views_np = res_stm

                sem_grid, conf_grid, top1_grid = self._build_sem_conf_top1_grids(
                    stm_cells_pub, scores_np, confs_np
                )
                self._set_latest_grid("stm_semantic", sem_grid)
                self._set_latest_grid("stm_confidence", conf_grid)
                self._set_latest_grid("stm_top1", top1_grid)

                if bool(self.get_parameter("publish_stm_semantic_grid").value):
                    self.publish_grid(sem_grid, self.pub_stm_sem)
                if bool(self.get_parameter("publish_stm_confidence_grid").value):
                    self.publish_grid(conf_grid, self.pub_stm_conf)
                if bool(self.get_parameter("publish_stm_top1_grid").value):
                    self.publish_grid(top1_grid, self.pub_stm_top1)

                # -------------------------
                # Promotion STM -> LTM
                # -------------------------
                promote_min_conf = float(self.get_parameter("promote_min_conf").value)
                promote_min_coh01 = float(self.get_parameter("promote_min_coh01").value)
                promote_min_views = int(self.get_parameter("promote_min_views").value)

                # We promote using INTERNAL keys (stm_keys_keep) and their corresponding computed stats.
                # This means we can promote “facts about the world” independent of the publish_frame warp.
                for key_i, e_u, coh01, c01, nb in zip(stm_keys_keep, emb_u, coh01s, confs_np, views_np):
                    if c01 < promote_min_conf:
                        continue
                    if coh01 < promote_min_coh01:
                        continue
                    if nb < promote_min_views:
                        continue
                    self.promote_stm_to_ltm(key_i, e_u, float(coh01), float(c01), now_s)
        self._profile_add_elapsed("pub.stm_grids_promote", t0)

        # -------------------------
        # LTM -> compute and publish
        # -------------------------
        t0 = time.perf_counter()
        ltm_keys = list(self.ltm_w.keys())
        if len(ltm_keys) < 20:
            self._set_latest_grid("ltm_semantic", empty_sem)
            self._set_latest_grid("ltm_confidence", empty_conf)
            self._set_latest_grid("ltm_top1", empty_top1)
            if bool(self.get_parameter("publish_ltm_semantic_grid").value):
                self.publish_grid(empty_sem, self.pub_ltm_sem)
            if bool(self.get_parameter("publish_ltm_confidence_grid").value):
                self.publish_grid(empty_conf, self.pub_ltm_conf)
            if bool(self.get_parameter("publish_ltm_top1_grid").value):
                self.publish_grid(empty_top1, self.pub_ltm_top1)
            self._profile_add_elapsed("pub.ltm_grids", t0)
            self._profile_add_elapsed("pub.total", pub_t0)
            self._profile_report_if_due()
            return

        ltm_cells_pub, ltm_keys_keep = self._project_internal_cells_to_publish_cells(T, ltm_keys)

        res_ltm = self._compute_cell_scores_conf(
            keys=ltm_keys_keep,
            sum_dict=self.ltm_sum,
            w_dict=self.ltm_w,
            viewmask_dict=self.ltm_viewmask,
            weight_max=float(self.get_parameter("ltm_weight_max").value),
            apply_view_boost=use_view,
            use_coh_conf=use_coh_conf,
            use_coh_sem=use_coh_sem,
        )

        if res_ltm is None or len(ltm_cells_pub) < 10:
            self._set_latest_grid("ltm_semantic", empty_sem)
            self._set_latest_grid("ltm_confidence", empty_conf)
            self._set_latest_grid("ltm_top1", empty_top1)
            if bool(self.get_parameter("publish_ltm_semantic_grid").value):
                self.publish_grid(empty_sem, self.pub_ltm_sem)
            if bool(self.get_parameter("publish_ltm_confidence_grid").value):
                self.publish_grid(empty_conf, self.pub_ltm_conf)
            if bool(self.get_parameter("publish_ltm_top1_grid").value):
                self.publish_grid(empty_top1, self.pub_ltm_top1)
            self._profile_add_elapsed("pub.ltm_grids", t0)
            self._profile_add_elapsed("pub.total", pub_t0)
            self._profile_report_if_due()
            return

        _, _, scores_ltm, confs_ltm, _ = res_ltm
        sem_ltm, conf_ltm, top1_ltm = self._build_sem_conf_top1_grids(ltm_cells_pub, scores_ltm, confs_ltm)
        self._set_latest_grid("ltm_semantic", sem_ltm)
        self._set_latest_grid("ltm_confidence", conf_ltm)
        self._set_latest_grid("ltm_top1", top1_ltm)

        if bool(self.get_parameter("publish_ltm_semantic_grid").value):
            self.publish_grid(sem_ltm, self.pub_ltm_sem)
        if bool(self.get_parameter("publish_ltm_confidence_grid").value):
            self.publish_grid(conf_ltm, self.pub_ltm_conf)
        if bool(self.get_parameter("publish_ltm_top1_grid").value):
            self.publish_grid(top1_ltm, self.pub_ltm_top1)
        self._profile_add_elapsed("pub.ltm_grids", t0)
        self._profile_add_elapsed("pub.total", pub_t0)
        self._profile_report_if_due()

    # -------------------------
    # Grid publisher
    # -------------------------
    def publish_grid(self, grid: np.ndarray, publisher):
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.publish_frame

        info = MapMetaData()
        info.resolution = float(self.resolution)
        info.width = int(self.grid_width)
        info.height = int(self.grid_height)
        info.origin.position.x = float(self.origin_x)
        info.origin.position.y = float(self.origin_y)
        info.origin.position.z = 0.0
        info.origin.orientation.w = 1.0
        msg.info = info

        msg.data = grid.flatten().tolist()
        publisher.publish(msg)

    def destroy_node(self):
        self.stop_async_semantic_worker()
        return super().destroy_node()


def main():
    lock_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.runtime',
                             'perception_domain_' + str(int(os.environ.get('ROS_DOMAIN_ID', '0'))) + '.lock')
    with PipelineInstance(lock_path):
        rclpy.init()
        probe = rclpy.create_node('crossmaps_perception_startup_probe')
        try:
            check_existing_publishers(probe, rclpy.spin_once)
        except Exception:
            if rclpy.ok():
                rclpy.shutdown()
            raise
        finally:
            probe.destroy_node()
        node = None
        try:
            node = CrossMapsNode()
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            if node is not None:
                if bool(node.get_parameter("autosave_map_state").value):
                    node.save_map_state()
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
