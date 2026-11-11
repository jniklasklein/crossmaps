#!/usr/bin/env python3
"""Browser dashboard for CrossMaps.

This runs as a separate ROS 2 node. It subscribes to CrossMaps visualization
topics, exposes a small HTTP API, serves the static dashboard UI, and writes
parameter updates back to the CrossMaps node through ROS parameter services.
"""

import argparse
import json
import math
import mimetypes
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - dashboard still starts, image endpoints degrade.
    cv2 = None

ROS_IMPORT_ERROR = None
try:
    import rclpy
    from rcl_interfaces.msg import ParameterEvent, ParameterType
    from rcl_interfaces.srv import GetParameters, SetParameters
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import BatteryState, CompressedImage, Image, PointCloud2, PointField
    from nav_msgs.msg import OccupancyGrid, Odometry
    from std_msgs.msg import String
    from visualization_msgs.msg import MarkerArray
except Exception as exc:  # pragma: no cover - useful error when run outside ROS.
    ROS_IMPORT_ERROR = exc
    Node = object


APP_DIR = Path(__file__).resolve().parent / "crossmaps_dashboard"
DEFAULT_PRESETS_PATH = os.path.expanduser("~/.crossmaps/crossmaps_dashboard_presets.json")


CLOUD_LAYERS = [
    {
        "id": "voxel_fused",
        "label": "Fused Voxels",
        "topic": "/crossmaps/voxels/fused_heatmap",
        "color": [255, 80, 120],
        "defaultVisible": True,
        "pointBudget": 50000,
    },
    {
        "id": "voxel_stm",
        "label": "STM Voxels",
        "topic": "/crossmaps/voxels/stm_heatmap",
        "color": [70, 180, 255],
        "defaultVisible": False,
        "pointBudget": 35000,
    },
    {
        "id": "voxel_ltm",
        "label": "LTM Voxels",
        "topic": "/crossmaps/voxels/ltm_heatmap",
        "color": [130, 230, 120],
        "defaultVisible": False,
        "pointBudget": 35000,
    },
    {
        "id": "object_query",
        "label": "Query Objects",
        "topic": "/crossmaps/objects/query_voxels",
        "color": [255, 220, 90],
        "defaultVisible": True,
        "pointBudget": 25000,
    },
    {
        "id": "segment_voxels",
        "label": "Segments",
        "topic": "/crossmaps/segments/voxels",
        "publishParam": "publish_segment_voxels",
        "requiresParams": ["use_segment_memory"],
        "description": "Stable surface IDs. Selecting this layer enables its publisher and hides other overlapping voxel layers.",
        "color": [190, 130, 255],
        "defaultVisible": False,
        "pointBudget": 30000,
    },
    {
        "id": "entity_voxels",
        "label": "Entities",
        "topic": "/crossmaps/entities/voxels",
        "publishParam": "publish_entity_voxels",
        "requiresParams": ["use_segment_memory"],
        "description": "Same geometry, colored by persistent entity/group hypothesis. Gray: unassigned; amber: ambiguous. Tune DINO grouping under Identity; groups are not verified physical objects.",
        "color": [242, 193, 78],
        "defaultVisible": False,
        "pointBudget": 30000,
    },
    {
        "id": "unassigned_query",
        "label": "Unassigned Query",
        "topic": "/crossmaps/objects/unassigned_query_voxels",
        "color": [255, 140, 60],
        "defaultVisible": False,
        "pointBudget": 18000,
    },
    {
        "id": "semantic_cloud",
        "label": "Semantic Cloud",
        "topic": "/crossmaps/semantic_cloud",
        "color": [90, 210, 190],
        "defaultVisible": False,
        "pointBudget": 20000,
    },
    {
        "id": "history_cloud",
        "label": "History Cloud",
        "topic": "/crossmaps/history_cloud",
        "color": [210, 210, 210],
        "defaultVisible": False,
        "pointBudget": 30000,
    },
    {
        "id": "debug_cloud",
        "label": "Debug Cloud",
        "topic": "/crossmaps/debug_cloud",
        "color": [245, 245, 245],
        "defaultVisible": False,
        "pointBudget": 20000,
    },
]


IMAGE_LAYERS = [
    {
        "id": "camera_rgb",
        "label": "RGB",
        "topic": "/crossmaps/debug/last_observation_rgb/compressed",
        "topics": [
            {"topic": "/crossmaps/debug/last_observation_rgb/compressed", "type": "compressed"},
        ],
        "direct_topics": [
            {"topic": "/oak/rgb/image_rect", "type": "image"},
            {"topic": "/oak/rgb/image_raw", "type": "image"},
            {"topic": "/oak/rgb/image_rect/compressed", "type": "compressed"},
            {"topic": "/oak/rgb/image_raw/compressed", "type": "compressed"},
            {"topic": "/camera/color/image_raw", "type": "image"},
            {"topic": "/camera/color/image_raw/compressed", "type": "compressed"},
        ],
        "defaultVisible": True,
    },
    {
        "id": "esam_overlay",
        "label": "SAM",
        "topic": "/crossmaps/esam/mask_overlay/compressed",
        "topics": [{"topic": "/crossmaps/esam/mask_overlay/compressed", "type": "compressed"}],
        "defaultVisible": False,
    },
    {
        "id": "dinov2_features",
        "label": "DINO",
        "topic": "/crossmaps/dino/feature_view/compressed",
        "topics": [
            {"topic": "/crossmaps/dino/feature_view/compressed", "type": "compressed"},
        ],
        "defaultVisible": False,
    },
    {
        "id": "dinov2_similarity",
        "label": "DINO Sim",
        "topic": "/crossmaps/dino/similarity_view/compressed",
        "topics": [
            {"topic": "/crossmaps/dino/similarity_view/compressed", "type": "compressed"},
        ],
        "defaultVisible": False,
    },
    {
        "id": "dinov2_instances",
        "label": "DINO Inst",
        "topic": "/crossmaps/dino/instance_view/compressed",
        "topics": [
            {"topic": "/crossmaps/dino/instance_view/compressed", "type": "compressed"},
        ],
        "defaultVisible": False,
    },
    {
        "id": "clip_features",
        "label": "CLIP",
        "topic": "/crossmaps/clip/feature_view/compressed",
        "topics": [
            {"topic": "/crossmaps/clip/feature_view/compressed", "type": "compressed"},
        ],
        "defaultVisible": False,
    },
    {
        "id": "clip_similarity",
        "label": "CLIP Sim",
        "topic": "/crossmaps/clip/similarity_view/compressed",
        "topics": [
            {"topic": "/crossmaps/clip/similarity_view/compressed", "type": "compressed"},
        ],
        "defaultVisible": False,
    },
    {
        "id": "segment_projection",
        "label": "Segments",
        "topic": "/crossmaps/debug/segment_id_projection/compressed",
        "topics": [{"topic": "/crossmaps/debug/segment_id_projection/compressed", "type": "compressed"}],
        "defaultVisible": False,
    },
    {
        "id": "entity_projection",
        "label": "Entities",
        "topic": "/crossmaps/debug/entity_id_projection/compressed",
        "topics": [{"topic": "/crossmaps/debug/entity_id_projection/compressed", "type": "compressed"}],
        "defaultVisible": False,
    },
    {
        "id": "depth",
        "label": "Depth",
        "topic": "/crossmaps/debug/last_observation_depth/compressed",
        "topics": [
            {"topic": "/crossmaps/debug/last_observation_depth/compressed", "type": "compressed"},
        ],
        "direct_topics": [
            {"topic": "/oak/stereo/image_raw", "type": "image"},
            {"topic": "/oak/stereo/image_raw/compressedDepth", "type": "compressed"},
            {"topic": "/camera/depth/image_raw", "type": "image"},
            {"topic": "/camera/depth/image_raw/compressedDepth", "type": "compressed"},
        ],
        "defaultVisible": False,
    },
]


GRID_LAYERS = [
    {"id": "stm_semantic", "label": "STM Sem", "topic": "/crossmaps/stm/semantic_grid"},
    {"id": "stm_confidence", "label": "STM Conf", "topic": "/crossmaps/stm/confidence_grid"},
    {"id": "stm_top1", "label": "STM Top1", "topic": "/crossmaps/stm/top1_grid"},
    {"id": "ltm_semantic", "label": "LTM Sem", "topic": "/crossmaps/ltm/semantic_grid"},
    {"id": "ltm_confidence", "label": "LTM Conf", "topic": "/crossmaps/ltm/confidence_grid"},
    {"id": "ltm_top1", "label": "LTM Top1", "topic": "/crossmaps/ltm/top1_grid"},
]


BATTERY_TOPICS = [
    "/battery_state",
    "/battery",
    "/battery/status",
    "/power/battery_state",
    "/ugv/battery_state",
]


ROVER_POSE_TOPICS = [
    "/odom",
    "/odom_raw",
    "/odometry/filtered",
    "/wheel/odom",
    "/robot/odom",
]
ROVER_POSE_STALE_S = 2.0


PARAM_GROUPS = [
    {
        "id": "query",
        "label": "Query",
        "params": [
            {"name": "query_text", "label": "Query Text", "type": "str", "control": "text"},
            {"name": "publish_hz", "label": "Publish Hz", "type": "float", "min": 0.2, "max": 10.0, "step": 0.1},
            {"name": "observation_latest_frame", "label": "Latest-Frame Scheduling", "description": "Process the newest synchronized pair when the worker is available. No every-N-frame starvation or inference backlog.", "type": "bool"},
            {"name": "stationary_pose_lock", "label": "Stationary Pose Lock", "description": "Stationary diagnostic only. Fix the camera pose while testing a stationary robot with drifting odometry. Disable BEFORE moving; this is not SLAM or an automatic drift correction.", "type": "bool"},
            {"name": "observation_max_hz", "label": "Observation Rate Cap (Hz)", "type": "float", "min": 0.1, "max": 10.0, "step": 0.1},
            {"name": "observation_max_frame_age_s", "label": "Pending Frame Max Age (s)", "type": "float", "min": 0.1, "max": 10.0, "step": 0.1},
            {"name": "publish_max_duty_cycle", "label": "Publishing CPU Time Budget", "description": "Maximum target fraction of the serialized callback loop spent publishing. Heavy ticks automatically reduce publication cadence, not model quality.", "type": "float", "min": 0.05, "max": 0.9, "step": 0.05},
            {"name": "embed_every_n", "label": "Legacy Embed Every N", "description": "Only used when Latest-Frame Scheduling is disabled.", "type": "int", "min": 1, "max": 60, "step": 1},
            {"name": "pipeline_stale_after_s", "label": "Input Stale Threshold (s)", "type": "float", "min": 1.0, "max": 60.0, "step": 1.0},
            {"name": "pipeline_processing_timeout_s", "label": "Processing Stall Warning (s)", "description": "Diagnostic threshold only; never launches a second model worker.", "type": "float", "min": 5.0, "max": 180.0, "step": 5.0},
            {"name": "profile_timing", "label": "Profile Timing", "type": "bool"},
            {"name": "profile_interval_s", "label": "Profile Interval", "type": "float", "min": 1.0, "max": 30.0, "step": 0.5},
            {"name": "observation_cycle_average_window", "label": "Cycle Avg Window", "type": "int", "min": 1, "max": 300, "step": 1},
        ],
    },
    {
        "id": "sam",
        "label": "SAM",
        "params": [
            {"name": "use_esam_masks", "label": "Use Masks", "type": "bool"},
            {"name": "use_async_semantic_worker", "label": "Async Inference", "type": "bool"},
            {"name": "publish_esam_mask_overlay", "label": "Publish Overlay", "type": "bool"},
            {"name": "esam_input_max_side", "label": "Input Max Side", "type": "int", "min": 160, "max": 1280, "step": 16},
            {"name": "esam_prompt_grid_x", "label": "Prompt Grid X", "type": "int", "min": 1, "max": 20, "step": 1},
            {"name": "esam_prompt_grid_y", "label": "Prompt Grid Y", "type": "int", "min": 1, "max": 16, "step": 1},
            {"name": "esam_max_masks", "label": "Max Masks", "type": "int", "min": 1, "max": 80, "step": 1},
            {"name": "esam_min_mask_area_px", "label": "Min Area Px", "type": "int", "min": 100, "max": 20000, "step": 100},
            {"name": "esam_max_mask_area_frac", "label": "Max Area Frac", "type": "float", "min": 0.05, "max": 1.0, "step": 0.01},
            {"name": "esam_min_iou_pred", "label": "Min IoU Pred", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "esam_min_stability_score", "label": "Min Stability", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "esam_nms_iou", "label": "NMS IoU", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "esam_nms_containment", "label": "NMS Containment", "type": "float", "min": 0.5, "max": 1.0, "step": 0.01},
            {"name": "esam_nms_containment_min_area_ratio", "label": "NMS Area Ratio", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "esam_min_depth_points", "label": "Min Depth Points", "type": "int", "min": 0, "max": 300, "step": 1},
            {"name": "esam_clip_batch_size", "label": "CLIP Batch", "type": "int", "min": 1, "max": 32, "step": 1},
            {"name": "esam_mask_overlay_alpha", "label": "Overlay Alpha", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
        ],
    },
    {
        "id": "dino",
        "label": "DINOv2",
        "params": [
            {"name": "use_dinov2_features", "label": "Appearance Features", "type": "bool"},
            {"name": "dinov2_model_name", "label": "Model", "type": "str", "control": "select", "choices": ["dinov2_vits14", "dinov2_vitb14"]},
            {"name": "dinov2_input_size", "label": "Input Size", "type": "int", "min": 224, "max": 728, "step": 14},
            {"name": "dinov2_mask_min_weight", "label": "Interior Coverage", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "dinov2_context_ring_tokens", "label": "Context Ring Tokens", "type": "int", "min": 0, "max": 4, "step": 1},
            {"name": "dinov2_context_contrast_weight", "label": "Context Contrast", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "dinov2_min_effective_tokens", "label": "Full Quality Tokens", "type": "float", "min": 0.25, "max": 12.0, "step": 0.25},
            {"name": "dinov2_assoc_min_quality", "label": "Assoc Min Quality", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "dinov2_track_update_gain", "label": "Track Update Gain", "type": "float", "min": 0.05, "max": 4.0, "step": 0.05},
            {"name": "dinov2_track_ema_alpha", "label": "Track EMA Alpha", "type": "float", "min": 0.01, "max": 1.0, "step": 0.01},
            {"name": "dinov2_track_memory_weight", "label": "Track Memory Weight", "type": "float", "min": 1.0, "max": 100.0, "step": 1.0},
            {"name": "publish_dinov2_feature_view", "label": "Publish Camera View", "type": "bool"},
            {"name": "publish_dinov2_similarity_view", "label": "Publish Similarity", "type": "bool"},
            {"name": "publish_dinov2_instance_view", "label": "Publish Instances", "type": "bool"},
            {"name": "dinov2_feature_view_rgb_alpha", "label": "Camera RGB Alpha", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "diagnostic_similarity_x", "label": "Similarity X", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "diagnostic_similarity_y", "label": "Similarity Y", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "diagnostic_similarity_floor", "label": "Similarity Floor", "type": "float", "min": -1.0, "max": 0.95, "step": 0.01},
            {"name": "diagnostic_similarity_rgb_alpha", "label": "Similarity RGB", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "diagnostic_feature_sam_boundaries", "label": "SAM Boundaries", "type": "bool"},
        ],
    },
    {
        "id": "clip",
        "label": "CLIP",
        "params": [
            {"name": "publish_clip_feature_view", "label": "Publish Mask View", "type": "bool"},
            {"name": "publish_clip_similarity_view", "label": "Publish Similarity", "type": "bool"},
            {"name": "clip_feature_view_rgb_alpha", "label": "Camera RGB Alpha", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "use_clip_dense_features", "label": "Optional Dense Fusion", "type": "bool"},
        ],
    },
    {
        "id": "voxels",
        "label": "Voxels",
        "params": [
            {"name": "use_semantic_voxels", "label": "Use Voxels", "type": "bool"},
            {"name": "publish_voxel_heatmap", "label": "Publish Heatmaps", "type": "bool"},
            {"name": "publish_voxel_stm_heatmap", "label": "STM Heatmap", "type": "bool"},
            {"name": "publish_voxel_ltm_heatmap", "label": "LTM Heatmap", "type": "bool"},
            {"name": "publish_voxel_fused_heatmap", "label": "Fused Heatmap", "type": "bool"},
            {"name": "voxel_size_m", "label": "Voxel Size", "type": "float", "min": 0.03, "max": 0.6, "step": 0.01},
            {"name": "voxel_z_min_m", "label": "Z Min", "type": "float", "min": -3.0, "max": 2.0, "step": 0.05},
            {"name": "voxel_z_max_m", "label": "Z Max", "type": "float", "min": 0.0, "max": 6.0, "step": 0.05},
            {"name": "voxel_cloud_min_score", "label": "Min Score", "type": "float", "min": -1.0, "max": 1.0, "step": 0.01},
            {"name": "voxel_cloud_min_conf", "label": "Min Conf", "type": "float", "min": 0.0, "max": 1.0, "step": 0.005},
            {"name": "voxel_cloud_min_coh01", "label": "Min Coherence", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "voxel_cloud_gamma", "label": "Color Gamma", "type": "float", "min": 0.1, "max": 3.0, "step": 0.05},
            {"name": "voxel_heatmap_publish_every_n", "label": "Publish Every N", "type": "int", "min": 1, "max": 30, "step": 1},
            {"name": "voxel_cloud_max_points", "label": "Max Points", "type": "int", "min": 1000, "max": 200000, "step": 1000},
            {"name": "use_voxel_ray_clearing", "label": "Ray Clearing", "type": "bool"},
            {"name": "voxel_clear_stride", "label": "Clear Stride", "type": "int", "min": 1, "max": 32, "step": 1},
            {"name": "voxel_clear_max_rays_per_frame", "label": "Max Clear Rays", "type": "int", "min": 0, "max": 12000, "step": 100},
            {"name": "voxel_clear_stm_decay", "label": "STM Clear Decay", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "voxel_clear_ltm_decay", "label": "LTM Clear Decay", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
        ],
    },
    {
        "id": "segments",
        "label": "Segments",
        "params": [
            {"name": "use_segment_memory", "label": "Segment Memory", "type": "bool"},
            {"name": "publish_object_query_voxels", "label": "Query Voxels", "type": "bool"},
            {"name": "publish_segment_voxels", "label": "Segments 3D", "type": "bool"},
            {"name": "publish_entity_voxels", "label": "Entities 3D", "type": "bool"},
            {"name": "segment_mask_min_depth_points", "label": "Mask Min Depth Samples", "type": "int", "min": 3, "max": 100, "step": 1},
            {"name": "segment_mask_min_support_voxels", "label": "Mask Min Support Voxels", "type": "int", "min": 1, "max": 50, "step": 1},
            {"name": "segment_overlay_fill_masks", "label": "SAM-Aligned ID View", "type": "bool"},
            {"name": "segment_overlay_raw_support", "label": "Raw Support Debug View", "description": "Show measured/projected depth IDs instead of mask-level identity consensus. This is sparse geometry, not a mask boundary visualization.", "type": "bool"},
            {"name": "segment_overlay_dominant_owner", "label": "Stable Dominant Overlay", "description": "One temporally stable best-supported color per SAM mask. Dashed amber outline and ? mark provisional display choices, not committed ownership. Turn off for mixed ownership diagnostics; Raw Support overrides both.", "type": "bool"},
            {"name": "segment_overlay_alpha", "label": "Camera Color Opacity", "type": "float", "min": 0.1, "max": 1.0, "step": 0.05},
            {"name": "segment_temporal_consensus", "label": "Temporal Mask Consensus", "type": "bool"},
            {"name": "segment_mask_interior_px", "label": "Reliable Interior Margin (px)", "type": "int", "min": 0, "max": 12, "step": 1},
            {"name": "segment_mask_vote_dominance", "label": "Mask Owner Dominance", "type": "float", "min": 0.6, "max": 0.98, "step": 0.01},
            {"name": "segment_owner_switch_observations", "label": "Owner Switch Confirmation", "type": "int", "min": 2, "max": 10, "step": 1},
            {"name": "segment_boundary_repair_enabled", "label": "Conservative Boundary Repair", "description": "Four repeated interior-depth votes can correct a small foreign boundary patch; never more than 10% of a source track's initial repair budget. Whole-track merges remain disabled.", "type": "bool"},
            {"name": "segment_mixed_dino_pool_budget", "label": "Mixed-Part DINO Pool Budget", "description": "Maximum components re-pooled from the existing full-frame DINO grid per observation. Adds no model passes; zero disables refresh under mixed masks.", "type": "int", "min": 0, "max": 36, "step": 1},
            {"name": "publish_object_instance_voxels", "label": "Legacy Instance Topic", "type": "bool"},
            {"name": "object_instance_cloud_max_points", "label": "3D Total Point Limit", "type": "int", "min": 1000, "max": 100000, "step": 1000},
            {"name": "segment_instance_min_support_voxels", "label": "3D Min Support", "type": "int", "min": 1, "max": 200, "step": 1},
            {"name": "segment_min_observations_to_publish", "label": "3D Min Observations", "type": "int", "min": 1, "max": 50, "step": 1},
            {"name": "publish_object_unassigned_query_voxels", "label": "Unassigned Query", "type": "bool"},
            {"name": "object_query_cloud_max_points", "label": "Query Max Points", "type": "int", "min": 1000, "max": 100000, "step": 1000},
            {"name": "segment_query_min_score", "label": "Query Min Score", "type": "float", "min": -1.0, "max": 1.0, "step": 0.01},
            {"name": "segment_query_min_conf", "label": "Query Min Conf", "type": "float", "min": 0.0, "max": 1.0, "step": 0.005},
            {"name": "segment_query_min_support_voxels", "label": "Min Support Voxels", "type": "int", "min": 0, "max": 200, "step": 1},
            {"name": "segment_query_top_k", "label": "Top K", "type": "int", "min": 1, "max": 200, "step": 1},
            {
                "name": "segment_query_color_mode",
                "label": "Color Mode",
                "type": "str",
                "control": "select",
                "choices": ["instance", "score", "query", "heat"],
            },
            {"name": "segment_query_use_support_scores", "label": "Support Scores", "type": "bool"},
            {"name": "segment_query_support_min_score01", "label": "Support Min Score", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "segment_assoc_min_score", "label": "Assoc Min Score", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "segment_assoc_min_sem_cos", "label": "Assoc Sem Cos", "type": "float", "min": -1.0, "max": 1.0, "step": 0.01},
            {"name": "segment_assoc_min_dino_cos", "label": "Assoc DINO Cos", "type": "float", "min": -1.0, "max": 1.0, "step": 0.01},
            {"name": "segment_assoc_semantic_weight", "label": "Semantic Weight", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "segment_assoc_dino_weight", "label": "DINO Weight", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "segment_assoc_spatial_weight", "label": "Spatial Weight", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "segment_assoc_overlap_weight", "label": "Overlap Weight", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "segment_assoc_global_assignment", "label": "Global Assignment", "type": "bool"},
            {"name": "segment_assoc_require_overlap", "label": "Require 3D Overlap", "type": "bool"},
            {"name": "segment_assoc_min_bbox_iou", "label": "Fallback Min Box IoU", "type": "float", "min": 0.01, "max": 0.8, "step": 0.01},
            {"name": "segment_assoc_ambiguity_margin", "label": "Association Margin", "type": "float", "min": 0.0, "max": 0.5, "step": 0.01},
            {"name": "segment_exclusive_point_fusion", "label": "Exclusive Fusion", "type": "bool"},
            {"name": "segment_split_enabled", "label": "Split Segments", "type": "bool"},
            {"name": "segment_merge_enabled", "label": "Legacy Merge (destructive)", "type": "bool"},
            {"name": "segment_merge_split_cooldown_s", "label": "Split Cooldown", "type": "float", "min": 0.0, "max": 300.0, "step": 5.0},
            {"name": "instance_color_by_entity", "label": "Legacy Topic Entity Colors", "type": "bool"},
        ],
    },
    {
        "id": "identity",
        "label": "Identity",
        "params": [
            {"name": "use_entity_grouping", "label": "Entity Graph (off: segments)", "type": "bool"},
            {"name": "use_object_assembly", "label": "DINO Object Grouping", "type": "bool"},
            {"name": "entity_grouping_sensitivity", "label": "Grouping Sensitivity", "description": "Higher accepts lower DINO similarity and larger spatial gaps. Default 0.5: DINO ≥ 0.80, gap ≤ 15 cm. Repeated evidence/contact safeguards remain active. See Graph → Grouping evidence after the next observation.", "type": "float", "min": 0.0, "max": 1.0, "step": 0.05},
            {"name": "entity_assembly_confirm_observations", "label": "Grouping Confirm Observations", "type": "int", "min": 2, "max": 20, "step": 1},
            {"name": "entity_assembly_max_extent_m", "label": "Grouping Max Extent (m)", "type": "float", "min": 0.1, "max": 10.0, "step": 0.1},
            {"name": "entity_assembly_group_min_dino_cos", "label": "Group Compatibility Floor", "description": "Minimum interior-DINO compatibility across the whole group. Joining edges still need the stronger sensitivity-controlled threshold.", "type": "float", "min": 0.4, "max": 0.9, "step": 0.01},
            {"name": "world_model_source_id", "label": "World Model Source Name", "type": "str"},
            {"name": "record_identity_replay", "label": "Record Identity Replay", "description": "Opt-in local recording of geometry, features and decisions, without RGB. Bounded by frames and 32 MiB; files stay under this experimental directory. Toggle off/on for a new trace.", "type": "bool"},
            {"name": "identity_replay_max_frames", "label": "Replay Frame Limit", "type": "int", "min": 10, "max": 1000, "step": 10},
            {"name": "publish_entity_id_projection", "label": "Entities Camera View", "type": "bool"},
            {"name": "entity_birth_observations", "label": "Birth Observations", "type": "int", "min": 1, "max": 20, "step": 1},
            {"name": "entity_membership_observations", "label": "Membership Observations", "type": "int", "min": 1, "max": 20, "step": 1},
            {"name": "entity_neighborhood_tiebreak", "label": "Neighborhood Tie-Break", "description": "Resolve close geometric candidates using fresh DINO agreement and contact with an established observed part. Requires at least three consecutive observations; cannot override exclusions or reassign established segments.", "type": "bool"},
            {"name": "entity_neighborhood_overlap_gap", "label": "Tie-Break Overlap Window", "description": "Maximum overlap-score deficit eligible for neighborhood evidence. This is a score difference, not a probability. Only candidates above the normal overlap minimum qualify.", "type": "float", "min": 0.0, "max": 0.15, "step": 0.01},
            {"name": "entity_reassoc_min_voxels", "label": "Reassociation Min Voxels", "type": "int", "min": 1, "max": 200, "step": 1},
            {"name": "entity_reassoc_min_overlap", "label": "Reassociation Overlap", "type": "float", "min": 0.1, "max": 1.0, "step": 0.01},
            {"name": "entity_reassoc_ambiguity_margin", "label": "Reassociation Margin", "type": "float", "min": 0.0, "max": 0.8, "step": 0.01},
            {"name": "entity_dormant_retention_s", "label": "Dormant Memory (s)", "type": "float", "min": 0.0, "max": 900.0, "step": 10.0},
            {"name": "entity_view_cell_m", "label": "View Diversity Cell (m)", "type": "float", "min": 0.05, "max": 2.0, "step": 0.05},
        ],
    },
    {
        "id": "graph",
        "label": "Graph",
        "params": [
            {"name": "use_object_abstraction", "label": "Object Layer", "type": "bool"},
            {"name": "publish_object_graph", "label": "Graph Markers", "type": "bool"},
            {"name": "publish_object_graph_labels", "label": "Graph Labels", "type": "bool"},
            {"name": "publish_object_graph_descriptors", "label": "Descriptors", "type": "bool"},
            {"name": "object_graph_max_nodes", "label": "Max Nodes", "type": "int", "min": 1, "max": 120, "step": 1},
            {"name": "object_graph_max_edges", "label": "Max Edges", "type": "int", "min": 0, "max": 400, "step": 1},
            {"name": "object_graph_min_weight", "label": "Min Weight", "type": "float", "min": 0.0, "max": 10.0, "step": 0.05},
            {"name": "object_graph_min_conf", "label": "Min Conf", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "object_graph_min_observations", "label": "Min Obs", "type": "int", "min": 1, "max": 50, "step": 1},
            {"name": "object_graph_min_view_count", "label": "Min Distinct Views", "type": "int", "min": 1, "max": 20, "step": 1},
            {"name": "object_graph_min_support_voxels", "label": "Min Support", "type": "int", "min": 0, "max": 500, "step": 1},
            {"name": "object_graph_near_dist_m", "label": "Near Dist", "type": "float", "min": 0.05, "max": 5.0, "step": 0.05},
            {"name": "object_graph_max_age_s", "label": "Max Track Age", "type": "float", "min": 0.0, "max": 600.0, "step": 5.0},
            {"name": "object_graph_edge_min_observations", "label": "Edge Min Obs", "type": "int", "min": 1, "max": 20, "step": 1},
            {"name": "object_graph_edge_max_misses", "label": "Edge Max Misses", "type": "int", "min": 0, "max": 30, "step": 1},
            {"name": "object_graph_include_overlap_edges", "label": "Overlap Edges", "type": "bool"},
            {"name": "object_graph_include_similar_edges", "label": "Similar Edges", "type": "bool"},
        ],
    },
    {
        "id": "memory",
        "label": "Memory",
        "params": [
            {"name": "use_consistency_gating", "label": "Consistency Gating", "type": "bool"},
            {"name": "gating_cos_min", "label": "Gating Cos Min", "type": "float", "min": -1.0, "max": 1.0, "step": 0.01},
            {"name": "gating_soft_power", "label": "Gating Power", "type": "float", "min": 0.2, "max": 6.0, "step": 0.05},
            {"name": "gating_soft_floor", "label": "Gating Floor", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "cell_weight_max", "label": "STM Weight Max", "type": "float", "min": 1.0, "max": 300.0, "step": 1.0},
            {"name": "ltm_weight_max", "label": "LTM Weight Max", "type": "float", "min": 1.0, "max": 500.0, "step": 1.0},
            {"name": "ltm_update_gain", "label": "LTM Update Gain", "type": "float", "min": 0.0, "max": 8.0, "step": 0.05},
            {"name": "ltm_replace_cos_thresh", "label": "Replace Cos", "type": "float", "min": -1.0, "max": 1.0, "step": 0.01},
            {"name": "ltm_replace_gain", "label": "Replace Gain", "type": "float", "min": 0.0, "max": 10.0, "step": 0.05},
            {"name": "promote_min_conf", "label": "Promote Conf", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "promote_min_coh01", "label": "Promote Coherence", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "promote_min_views", "label": "Promote Views", "type": "int", "min": 1, "max": 16, "step": 1},
            {"name": "use_visibility_boost", "label": "Visibility Boost", "type": "bool"},
            {"name": "view_boost_strength", "label": "View Boost", "type": "float", "min": 0.0, "max": 3.0, "step": 0.05},
            {"name": "decay_half_life_s", "label": "Decay Half Life", "type": "float", "min": 1.0, "max": 12000000.0, "step": 1.0},
        ],
    },
    {
        "id": "debug",
        "label": "Debug",
        "params": [
            {"name": "publish_debug_cloud", "label": "Debug Cloud", "type": "bool"},
            {"name": "publish_semantic_cloud", "label": "Semantic Cloud", "type": "bool"},
            {"name": "publish_history_cloud", "label": "History Cloud", "type": "bool"},
            {"name": "update_history_cloud", "label": "Update History", "type": "bool"},
            {"name": "publish_segment_id_projection", "label": "Segment Projection", "type": "bool"},
            {"name": "publish_debug_observation_images", "label": "Observation Images", "type": "bool"},
            {"name": "debug_observation_image_hz", "label": "Obs Max Hz", "type": "float", "min": 0.0, "max": 10.0, "step": 0.25},
            {"name": "publish_debug_observation_rgb", "label": "Observation RGB", "type": "bool"},
            {"name": "publish_debug_observation_depth", "label": "Observation Depth", "type": "bool"},
            {"name": "debug_observation_depth_max_m", "label": "Depth Max M", "type": "float", "min": 0.5, "max": 12.0, "step": 0.25},
            {"name": "publish_debug_compressed_images", "label": "Compressed Images", "type": "bool"},
            {"name": "debug_image_jpeg_quality", "label": "JPEG Quality", "type": "int", "min": 30, "max": 100, "step": 1},
            {"name": "compute_2d_grids", "label": "2D Grids", "type": "bool"},
            {"name": "publish_stm_semantic_grid", "label": "STM Semantic", "type": "bool"},
            {"name": "publish_ltm_semantic_grid", "label": "LTM Semantic", "type": "bool"},
            {"name": "autosave_map_state", "label": "Autosave State", "type": "bool"},
            {"name": "autosave_interval_s", "label": "Autosave Interval", "type": "float", "min": 1.0, "max": 120.0, "step": 1.0},
        ],
    },
]


PARAM_BY_NAME = {
    param["name"]: param
    for group in PARAM_GROUPS
    for param in group["params"]
}


def stamp_to_sec(stamp: Any) -> float:
    if stamp is None:
        return 0.0
    return float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9


def now_sec() -> float:
    return time.time()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return default


def parameter_value_to_python(value: Any) -> Any:
    value_type = int(getattr(value, "type", 0))
    if value_type == int(ParameterType.PARAMETER_BOOL):
        return bool(value.bool_value)
    if value_type == int(ParameterType.PARAMETER_INTEGER):
        return int(value.integer_value)
    if value_type == int(ParameterType.PARAMETER_DOUBLE):
        return float(value.double_value)
    if value_type == int(ParameterType.PARAMETER_STRING):
        return str(value.string_value)
    if value_type == int(ParameterType.PARAMETER_BYTE_ARRAY):
        return list(value.byte_array_value)
    if value_type == int(ParameterType.PARAMETER_BOOL_ARRAY):
        return list(value.bool_array_value)
    if value_type == int(ParameterType.PARAMETER_INTEGER_ARRAY):
        return list(value.integer_array_value)
    if value_type == int(ParameterType.PARAMETER_DOUBLE_ARRAY):
        return list(value.double_array_value)
    if value_type == int(ParameterType.PARAMETER_STRING_ARRAY):
        return list(value.string_array_value)
    return None


def coerce_param_value(spec: Dict[str, Any], value: Any) -> Any:
    param_type = spec.get("type", "str")
    if param_type == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if param_type == "int":
        return int(round(float(value)))
    if param_type == "float":
        return float(value)
    return str(value)


def parameter_from_json(name: str, value: Any) -> "Parameter":
    spec = PARAM_BY_NAME.get(name, {"type": "str"})
    coerced = coerce_param_value(spec, value)
    param_type = spec.get("type", "str")
    if param_type == "bool":
        return Parameter(name, Parameter.Type.BOOL, coerced)
    if param_type == "int":
        return Parameter(name, Parameter.Type.INTEGER, coerced)
    if param_type == "float":
        return Parameter(name, Parameter.Type.DOUBLE, coerced)
    return Parameter(name, Parameter.Type.STRING, coerced)


@dataclass(frozen=True)
class CloudSnapshot:
    positions: np.ndarray
    colors: np.ndarray
    frame_id: str
    source_stamp_s: float
    received_s: float
    original_count: int
    decoded_count: int


@dataclass(frozen=True)
class ImageSnapshot:
    encoded: bytes
    mime: str
    width: int
    height: int
    encoding: str
    frame_id: str
    source_topic: str
    source_stamp_s: float
    received_s: float


@dataclass(frozen=True)
class GridSnapshot:
    data: np.ndarray
    width: int
    height: int
    resolution: float
    origin: Tuple[float, float, float]
    frame_id: str
    source_stamp_s: float
    received_s: float


@dataclass(frozen=True)
class MarkerSnapshot:
    items: Dict[str, Any]
    frame_id: str
    source_stamp_s: float
    received_s: float


@dataclass(frozen=True)
class BatterySnapshot:
    percentage: Optional[float]
    voltage: Optional[float]
    current: Optional[float]
    power_supply_status: int
    source_topic: str
    source_stamp_s: float
    received_s: float


@dataclass(frozen=True)
class RobotPoseSnapshot:
    position: Tuple[float, float, float]
    yaw_rad: float
    frame_id: str
    child_frame_id: str
    source_topic: str
    source_stamp_s: float
    received_s: float


@dataclass
class DashboardState:
    started_s: float = field(default_factory=now_sec)
    lock: threading.RLock = field(default_factory=threading.RLock)
    clouds: Dict[str, CloudSnapshot] = field(default_factory=dict)
    images: Dict[str, ImageSnapshot] = field(default_factory=dict)
    grids: Dict[str, GridSnapshot] = field(default_factory=dict)
    graph_parts: Dict[str, MarkerSnapshot] = field(default_factory=dict)
    graph_descriptor: Dict[str, Any] = field(default_factory=dict)
    observation_cycle: Dict[str, Any] = field(default_factory=dict)
    pipeline_health: Dict[str, Any] = field(default_factory=dict)
    battery: Optional[BatterySnapshot] = None
    robot_pose: Optional[RobotPoseSnapshot] = None
    params: Dict[str, Any] = field(default_factory=dict)
    param_errors: Dict[str, str] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    request_count: int = 0

    def add_event(self, level: str, message: str, **extra: Any) -> None:
        item = {"t": round(now_sec(), 3), "level": level, "message": message}
        item.update(extra)
        with self.lock:
            self.events.append(item)
            self.events = self.events[-80:]


def point_cloud_default_color(layer_id: str) -> Tuple[int, int, int]:
    for layer in CLOUD_LAYERS:
        if layer["id"] == layer_id:
            return tuple(int(v) for v in layer["color"])
    return 230, 230, 230


def point_field_dtype(field: Any, bigendian: bool) -> Optional[np.dtype]:
    prefix = ">" if bigendian else "<"
    datatype = int(field.datatype)
    if datatype == int(PointField.INT8):
        return np.dtype("i1")
    if datatype == int(PointField.UINT8):
        return np.dtype("u1")
    if datatype == int(PointField.INT16):
        return np.dtype(prefix + "i2")
    if datatype == int(PointField.UINT16):
        return np.dtype(prefix + "u2")
    if datatype == int(PointField.INT32):
        return np.dtype(prefix + "i4")
    if datatype == int(PointField.UINT32):
        return np.dtype(prefix + "u4")
    if datatype == int(PointField.FLOAT32):
        return np.dtype(prefix + "f4")
    if datatype == int(PointField.FLOAT64):
        return np.dtype(prefix + "f8")
    return None


def strided_cloud_array(msg: "PointCloud2", field: Any, count: int) -> Optional[np.ndarray]:
    dtype = point_field_dtype(field, bool(msg.is_bigendian))
    if dtype is None:
        return None
    needed = int(field.offset) + max(0, count - 1) * int(msg.point_step) + dtype.itemsize
    if needed > len(msg.data):
        return None
    return np.ndarray(
        shape=(count,),
        dtype=dtype,
        buffer=msg.data,
        offset=int(field.offset),
        strides=(int(msg.point_step),),
    )


def decode_cloud(msg: "PointCloud2", layer_id: str, budget: int) -> Optional[CloudSnapshot]:
    fields = {field.name: field for field in msg.fields}
    if "x" not in fields or "y" not in fields or "z" not in fields:
        return None

    count = int(msg.width) * int(msg.height)
    if count <= 0 or int(msg.point_step) <= 0:
        return CloudSnapshot(
            positions=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.uint8),
            frame_id=str(msg.header.frame_id),
            source_stamp_s=stamp_to_sec(msg.header.stamp),
            received_s=now_sec(),
            original_count=0,
            decoded_count=0,
        )

    x = strided_cloud_array(msg, fields["x"], count)
    y = strided_cloud_array(msg, fields["y"], count)
    z = strided_cloud_array(msg, fields["z"], count)
    if x is None or y is None or z is None:
        return None

    points = np.stack([x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)], axis=1)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]

    color = np.asarray(point_cloud_default_color(layer_id), dtype=np.uint8)
    colors = np.repeat(color.reshape(1, 3), points.shape[0], axis=0)
    rgb_field = fields.get("rgb") or fields.get("rgba")
    if rgb_field is not None:
        rgb_arr = strided_cloud_array(msg, rgb_field, count)
        if rgb_arr is not None:
            rgb_arr = np.asarray(rgb_arr)[finite]
            if np.issubdtype(rgb_arr.dtype, np.floating):
                rgb_u = rgb_arr.astype(np.float32, copy=False).copy().view(np.uint32)
            else:
                rgb_u = rgb_arr.astype(np.uint32, copy=False)
            colors = np.empty((rgb_u.shape[0], 3), dtype=np.uint8)
            colors[:, 0] = ((rgb_u >> 16) & 255).astype(np.uint8)
            colors[:, 1] = ((rgb_u >> 8) & 255).astype(np.uint8)
            colors[:, 2] = (rgb_u & 255).astype(np.uint8)

    if budget > 0 and points.shape[0] > budget:
        idx = np.linspace(0, points.shape[0] - 1, budget, dtype=np.int64)
        points = points[idx]
        colors = colors[idx]

    return CloudSnapshot(
        positions=np.asarray(points, dtype=np.float32),
        colors=np.asarray(colors, dtype=np.uint8),
        frame_id=str(msg.header.frame_id),
        source_stamp_s=stamp_to_sec(msg.header.stamp),
        received_s=now_sec(),
        original_count=count,
        decoded_count=int(points.shape[0]),
    )


def decode_image(msg: "Image", source_topic: str = "", quality: int = 88) -> Optional[ImageSnapshot]:
    if cv2 is None:
        return None
    encoding = str(msg.encoding).lower()
    height = int(msg.height)
    width = int(msg.width)
    if height <= 0 or width <= 0:
        return None

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    try:
        if encoding in ("rgb8", "bgr8"):
            view = raw.reshape((height, int(msg.step)))[:, : width * 3]
            img = view.reshape((height, width, 3))
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if encoding == "rgb8" else img
        elif encoding in ("rgba8", "bgra8"):
            view = raw.reshape((height, int(msg.step)))[:, : width * 4]
            img = view.reshape((height, width, 4))
            code = cv2.COLOR_RGBA2BGR if encoding == "rgba8" else cv2.COLOR_BGRA2BGR
            bgr = cv2.cvtColor(img, code)
        elif encoding in ("mono8", "8uc1"):
            view = raw.reshape((height, int(msg.step)))[:, :width]
            bgr = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
        elif encoding in ("16uc1", "mono16"):
            raw16 = np.frombuffer(msg.data, dtype=np.uint16)
            view = raw16.reshape((height, int(msg.step) // 2))[:, :width]
            scaled = cv2.normalize(view, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            bgr = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
        elif encoding in ("32fc1",):
            raw32 = np.frombuffer(msg.data, dtype=np.float32)
            view = raw32.reshape((height, int(msg.step) // 4))[:, :width]
            finite = np.isfinite(view)
            scaled = np.zeros_like(view, dtype=np.uint8)
            if np.any(finite):
                vals = view[finite]
                lo = float(np.percentile(vals, 2.0))
                hi = float(np.percentile(vals, 98.0))
                norm = np.clip((view - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
                scaled = np.clip(255.0 * norm, 0, 255).astype(np.uint8)
            color_map = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
            bgr = cv2.applyColorMap(scaled, color_map)
        else:
            return None
        ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
    except Exception:
        return None

    return ImageSnapshot(
        encoded=bytes(encoded),
        mime="image/jpeg",
        width=width,
        height=height,
        encoding=str(msg.encoding),
        frame_id=str(msg.header.frame_id),
        source_topic=str(source_topic),
        source_stamp_s=stamp_to_sec(msg.header.stamp),
        received_s=now_sec(),
    )


def decode_compressed_image(msg: "CompressedImage", source_topic: str = "", quality: int = 88) -> Optional[ImageSnapshot]:
    if cv2 is None:
        return None
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None and raw.size > 12:
        # compressedDepth messages often contain a small header before the PNG.
        img = cv2.imdecode(raw[12:], cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        if img.dtype == np.uint16:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            color_map = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
            bgr = cv2.applyColorMap(img, color_map)
        else:
            bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        bgr = img
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return ImageSnapshot(
        encoded=bytes(encoded),
        mime="image/jpeg",
        width=int(bgr.shape[1]),
        height=int(bgr.shape[0]),
        encoding=str(msg.format),
        frame_id=str(msg.header.frame_id),
        source_topic=str(source_topic),
        source_stamp_s=stamp_to_sec(msg.header.stamp),
        received_s=now_sec(),
    )


def yaw_from_quaternion(q: Any) -> float:
    x = float(q.x)
    y = float(q.y)
    z = float(q.z)
    w = float(q.w)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def decode_robot_pose(msg: "Odometry", source_topic: str) -> RobotPoseSnapshot:
    pos = msg.pose.pose.position
    return RobotPoseSnapshot(
        position=(round(float(pos.x), 4), round(float(pos.y), 4), round(float(pos.z), 4)),
        yaw_rad=round(float(yaw_from_quaternion(msg.pose.pose.orientation)), 5),
        frame_id=str(msg.header.frame_id),
        child_frame_id=str(msg.child_frame_id),
        source_topic=str(source_topic),
        source_stamp_s=stamp_to_sec(msg.header.stamp),
        received_s=now_sec(),
    )


def decode_battery(msg: "BatteryState", source_topic: str) -> BatterySnapshot:
    percentage = safe_float(msg.percentage, -1.0)
    if percentage < 0.0:
        percentage_out = None
    else:
        percentage_out = float(np.clip(percentage, 0.0, 1.0))
    voltage = safe_float(msg.voltage, float("nan"))
    current = safe_float(msg.current, float("nan"))
    return BatterySnapshot(
        percentage=percentage_out,
        voltage=None if not math.isfinite(voltage) else voltage,
        current=None if not math.isfinite(current) else current,
        power_supply_status=int(msg.power_supply_status),
        source_topic=str(source_topic),
        source_stamp_s=stamp_to_sec(msg.header.stamp),
        received_s=now_sec(),
    )


def decode_grid(msg: "OccupancyGrid") -> Optional[GridSnapshot]:
    width = int(msg.info.width)
    height = int(msg.info.height)
    if width <= 0 or height <= 0:
        return None
    data = np.asarray(msg.data, dtype=np.int16)
    if data.size != width * height:
        return None
    origin = msg.info.origin.position
    return GridSnapshot(
        data=data.reshape((height, width)),
        width=width,
        height=height,
        resolution=float(msg.info.resolution),
        origin=(float(origin.x), float(origin.y), float(origin.z)),
        frame_id=str(msg.header.frame_id),
        source_stamp_s=stamp_to_sec(msg.header.stamp),
        received_s=now_sec(),
    )


def grid_to_png(snapshot: GridSnapshot, layer_id: str) -> Optional[bytes]:
    if cv2 is None:
        return None
    data = np.flipud(snapshot.data)
    unknown = data < 0
    clipped = np.clip(data, 0, 100).astype(np.float32) / 100.0
    if "semantic" in layer_id or "top1" in layer_id:
        heat = np.zeros((snapshot.height, snapshot.width, 3), dtype=np.uint8)
        heat[:, :, 0] = np.clip(255.0 * clipped, 0, 255).astype(np.uint8)
        heat[:, :, 1] = np.clip(70.0 + 120.0 * (1.0 - np.abs(clipped - 0.5) * 2.0), 0, 255).astype(np.uint8)
        heat[:, :, 2] = np.clip(255.0 * (1.0 - clipped), 0, 255).astype(np.uint8)
        heat[unknown] = (45, 45, 45)
        bgr = cv2.cvtColor(heat, cv2.COLOR_RGB2BGR)
    else:
        gray = np.clip(255.0 * clipped, 0, 255).astype(np.uint8)
        gray[unknown] = 40
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ok, encoded = cv2.imencode(".png", bgr)
    return bytes(encoded) if ok else None


def marker_color(marker: Any) -> List[int]:
    color = marker.color
    alpha = safe_float(color.a, 1.0)
    brightness = 1.0 if alpha <= 0.0 else min(1.0, max(0.25, alpha))
    return [
        int(np.clip(float(color.r) * 255.0 * brightness, 0, 255)),
        int(np.clip(float(color.g) * 255.0 * brightness, 0, 255)),
        int(np.clip(float(color.b) * 255.0 * brightness, 0, 255)),
    ]


def point_list(points: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for point in points:
        out.extend([round(float(point.x), 4), round(float(point.y), 4), round(float(point.z), 4)])
    return out


def decode_marker_array(msg: "MarkerArray", part: str) -> MarkerSnapshot:
    nodes: List[Dict[str, Any]] = []
    labels: List[Dict[str, Any]] = []
    lines: List[Dict[str, Any]] = []
    frame_id = ""
    stamp = 0.0

    for marker in msg.markers:
        frame_id = frame_id or str(marker.header.frame_id)
        stamp = stamp or stamp_to_sec(marker.header.stamp)
        marker_type = int(marker.type)
        action = int(marker.action)
        if action in (2, 3):
            continue
        if marker_type in (1, 2, 3):
            pos = marker.pose.position
            nodes.append(
                {
                    "ns": str(marker.ns),
                    "id": int(marker.id),
                    "position": [round(float(pos.x), 4), round(float(pos.y), 4), round(float(pos.z), 4)],
                    "scale": [
                        round(float(marker.scale.x), 4),
                        round(float(marker.scale.y), 4),
                        round(float(marker.scale.z), 4),
                    ],
                    "color": marker_color(marker),
                }
            )
        elif marker_type == 9:
            pos = marker.pose.position
            labels.append(
                {
                    "ns": str(marker.ns),
                    "id": int(marker.id),
                    "text": str(marker.text),
                    "position": [round(float(pos.x), 4), round(float(pos.y), 4), round(float(pos.z), 4)],
                    "color": marker_color(marker),
                }
            )
        elif marker_type in (4, 5, 6, 7, 8):
            lines.append(
                {
                    "ns": str(marker.ns),
                    "id": int(marker.id),
                    "type": marker_type,
                    "points": point_list(marker.points),
                    "color": marker_color(marker),
                    "scale": round(float(marker.scale.x), 4),
                }
            )

    return MarkerSnapshot(
        items={"part": part, "nodes": nodes, "labels": labels, "lines": lines},
        frame_id=frame_id,
        source_stamp_s=stamp,
        received_s=now_sec(),
    )


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class CrossMapsDashboard(Node):
    def __init__(
        self,
        target_node: str,
        host: str,
        port: int,
        static_dir: Path,
        presets_path: str,
        cloud_budget: int,
        enable_camera_streams: bool,
        robot_odom_topic: str,
    ):
        super().__init__("crossmaps_dashboard")
        self.target_node = target_node if target_node.startswith("/") else f"/{target_node}"
        self.host = host
        self.port = int(port)
        self.static_dir = static_dir
        self.presets_path = os.path.expanduser(presets_path)
        self.cloud_budget = max(1000, int(cloud_budget))
        self.camera_streams_enabled = bool(enable_camera_streams)
        self.robot_odom_topic = (robot_odom_topic or "auto").strip() or "auto"
        if self.robot_odom_topic != "auto" and not self.robot_odom_topic.startswith("/"):
            self.robot_odom_topic = f"/{self.robot_odom_topic}"
        self.robot_pose_source_topic: Optional[str] = None
        self.robot_pose_topic_priority = {topic: idx for idx, topic in enumerate(ROVER_POSE_TOPICS)}
        self.image_decode_warned = set()
        self.state = DashboardState()
        self.get_param_client = self.create_client(GetParameters, f"{self.target_node}/get_parameters")
        self.set_param_client = self.create_client(SetParameters, f"{self.target_node}/set_parameters")
        self.httpd: Optional[ReusableThreadingHTTPServer] = None
        self.http_thread: Optional[threading.Thread] = None

        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )

        self.dashboard_subscriptions = []
        for layer in CLOUD_LAYERS:
            self.dashboard_subscriptions.append(
                self.create_subscription(
                    PointCloud2,
                    layer["topic"],
                    self._cloud_callback(layer["id"], int(layer.get("pointBudget", self.cloud_budget))),
                    self.sensor_qos,
                )
            )
        for layer in IMAGE_LAYERS:
            self._create_image_layer_subscriptions(layer, self.dashboard_subscriptions)
        for layer in GRID_LAYERS:
            self.dashboard_subscriptions.append(
                self.create_subscription(
                    OccupancyGrid,
                    layer["topic"],
                    self._grid_callback(layer["id"]),
                    self.sensor_qos,
                )
            )
        self.dashboard_subscriptions.append(
            self.create_subscription(
                MarkerArray,
                "/crossmaps/object_graph/nodes",
                self._marker_callback("nodes"),
                self.sensor_qos,
            )
        )
        self.dashboard_subscriptions.append(
            self.create_subscription(
                MarkerArray,
                "/crossmaps/object_graph/edges",
                self._marker_callback("edges"),
                self.sensor_qos,
            )
        )
        self.dashboard_subscriptions.append(
            self.create_subscription(
                String,
                "/crossmaps/object_graph/descriptors",
                self.on_graph_descriptors,
                10,
            )
        )
        self.dashboard_subscriptions.append(
            self.create_subscription(
                String,
                "/crossmaps/diagnostics/observation_cycle",
                self.on_observation_cycle,
                10,
            )
        )
        self.dashboard_subscriptions.append(
            self.create_subscription(
                ParameterEvent,
                "/parameter_events",
                self.on_parameter_event,
                10,
            )
        )
        self.dashboard_subscriptions.append(self.create_subscription(
            String, "/crossmaps/diagnostics/pipeline_health", self.on_pipeline_health, 10))
        for topic in BATTERY_TOPICS:
            self.dashboard_subscriptions.append(
                self.create_subscription(
                    BatteryState,
                    topic,
                    self._battery_callback(topic),
                    self.sensor_qos,
                )
            )
        rover_pose_topics = ROVER_POSE_TOPICS if self.robot_odom_topic == "auto" else [self.robot_odom_topic]
        for topic in rover_pose_topics:
            self.dashboard_subscriptions.append(
                self.create_subscription(
                    Odometry,
                    topic,
                    self._robot_pose_callback(topic),
                    self.sensor_qos,
                )
            )

        self.direct_image_subscriptions = []
        if self.camera_streams_enabled:
            self._set_camera_streams_enabled(True)

        self.start_http_server()
        self.state.add_event("info", "Dashboard started", url=self.url)

    @property
    def url(self) -> str:
        host = "localhost" if self.host in ("0.0.0.0", "::", "") else self.host
        return f"http://{host}:{self.port}"

    def _create_image_layer_subscriptions(
        self,
        layer: Dict[str, Any],
        target_list: List[Any],
        sources: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        sources = sources or layer.get("topics") or [{"topic": layer["topic"], "type": "image"}]
        for source in sources:
            topic = str(source["topic"])
            source_type = str(source.get("type", "image")).lower()
            is_compressed = "compressed" in source_type
            msg_type = CompressedImage if is_compressed else Image
            callback = (
                self._compressed_image_callback(layer["id"], topic)
                if is_compressed
                else self._image_callback(layer["id"], topic)
            )
            target_list.append(
                self.create_subscription(
                    msg_type,
                    topic,
                    callback,
                    self.sensor_qos,
                )
            )

    def _set_camera_streams_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self.camera_streams_enabled and self.direct_image_subscriptions:
            return
        if not enabled:
            for sub in list(self.direct_image_subscriptions):
                self.destroy_subscription(sub)
            self.direct_image_subscriptions = []
            with self.state.lock:
                for layer in IMAGE_LAYERS:
                    if layer.get("direct_topics"):
                        snap = self.state.images.get(layer["id"])
                        direct_topics = {str(src["topic"]) for src in layer.get("direct_topics", [])}
                        if snap is not None and snap.source_topic in direct_topics:
                            self.state.images.pop(layer["id"], None)
            self.camera_streams_enabled = False
            self.state.add_event("info", "Direct camera streams disabled")
            return

        if self.direct_image_subscriptions:
            return
        for layer in IMAGE_LAYERS:
            direct_sources = layer.get("direct_topics", [])
            if direct_sources:
                self._create_image_layer_subscriptions(layer, self.direct_image_subscriptions, direct_sources)
        self.camera_streams_enabled = True
        self.state.add_event("info", "Direct camera streams enabled")

    def set_camera_streams_enabled(self, enabled: bool) -> Dict[str, Any]:
        self._set_camera_streams_enabled(bool(enabled))
        return {"ok": True, "enabled": self.camera_streams_enabled}

    def _cloud_callback(self, layer_id: str, point_budget: int):
        budget = min(self.cloud_budget, max(1, int(point_budget)))

        def callback(msg: "PointCloud2") -> None:
            try:
                snapshot = decode_cloud(msg, layer_id, budget)
                if snapshot is not None:
                    with self.state.lock:
                        self.state.clouds[layer_id] = snapshot
            except Exception as exc:
                self.state.add_event("error", f"Failed to decode cloud {layer_id}", error=str(exc))

        return callback

    def _image_callback(self, layer_id: str, source_topic: str):
        def callback(msg: "Image") -> None:
            try:
                snapshot = decode_image(msg, source_topic)
                if snapshot is not None:
                    with self.state.lock:
                        self.state.images[layer_id] = snapshot
                else:
                    key = ("image", layer_id, source_topic, str(getattr(msg, "encoding", "")))
                    if key not in self.image_decode_warned:
                        self.image_decode_warned.add(key)
                        self.state.add_event(
                            "warn",
                            f"Unsupported image encoding for {layer_id}",
                            topic=source_topic,
                            encoding=str(getattr(msg, "encoding", "")),
                        )
            except Exception as exc:
                self.state.add_event("error", f"Failed to decode image {layer_id}", error=str(exc))

        return callback

    def _compressed_image_callback(self, layer_id: str, source_topic: str):
        def callback(msg: "CompressedImage") -> None:
            try:
                snapshot = decode_compressed_image(msg, source_topic)
                if snapshot is not None:
                    with self.state.lock:
                        self.state.images[layer_id] = snapshot
                else:
                    key = ("compressed", layer_id, source_topic, str(getattr(msg, "format", "")))
                    if key not in self.image_decode_warned:
                        self.image_decode_warned.add(key)
                        self.state.add_event(
                            "warn",
                            f"Could not decode compressed image for {layer_id}",
                            topic=source_topic,
                            encoding=str(getattr(msg, "format", "")),
                        )
            except Exception as exc:
                self.state.add_event("error", f"Failed to decode compressed image {layer_id}", error=str(exc))

        return callback

    def _grid_callback(self, layer_id: str):
        def callback(msg: "OccupancyGrid") -> None:
            try:
                snapshot = decode_grid(msg)
                if snapshot is not None:
                    with self.state.lock:
                        self.state.grids[layer_id] = snapshot
            except Exception as exc:
                self.state.add_event("error", f"Failed to decode grid {layer_id}", error=str(exc))

        return callback

    def _battery_callback(self, source_topic: str):
        def callback(msg: "BatteryState") -> None:
            try:
                snapshot = decode_battery(msg, source_topic)
                with self.state.lock:
                    self.state.battery = snapshot
            except Exception as exc:
                self.state.add_event("error", "Failed to decode battery state", error=str(exc))

        return callback

    def _should_accept_robot_pose_locked(self, source_topic: str) -> bool:
        if self.robot_odom_topic != "auto":
            if source_topic != self.robot_odom_topic:
                return False
            if self.robot_pose_source_topic != source_topic:
                self.robot_pose_source_topic = source_topic
                self.state.add_event("info", "Robot pose source selected", topic=source_topic)
            return True

        if self.robot_pose_source_topic is None:
            self.robot_pose_source_topic = source_topic
            self.state.add_event("info", "Robot pose source selected", topic=source_topic)
            return True

        if source_topic == self.robot_pose_source_topic:
            return True

        current = self.state.robot_pose
        current_age = float("inf") if current is None else now_sec() - float(current.received_s)
        current_priority = self.robot_pose_topic_priority.get(self.robot_pose_source_topic, len(ROVER_POSE_TOPICS))
        new_priority = self.robot_pose_topic_priority.get(source_topic, len(ROVER_POSE_TOPICS))
        if current_age > ROVER_POSE_STALE_S or new_priority < current_priority:
            self.robot_pose_source_topic = source_topic
            self.state.add_event("info", "Robot pose source selected", topic=source_topic)
            return True
        return False

    def _robot_pose_callback(self, source_topic: str):
        def callback(msg: "Odometry") -> None:
            try:
                snapshot = decode_robot_pose(msg, source_topic)
                with self.state.lock:
                    if self._should_accept_robot_pose_locked(source_topic):
                        self.state.robot_pose = snapshot
            except Exception as exc:
                self.state.add_event("error", "Failed to decode robot odometry", error=str(exc))

        return callback

    def _marker_callback(self, part: str):
        def callback(msg: "MarkerArray") -> None:
            try:
                snapshot = decode_marker_array(msg, part)
                with self.state.lock:
                    self.state.graph_parts[part] = snapshot
            except Exception as exc:
                self.state.add_event("error", f"Failed to decode graph {part}", error=str(exc))

        return callback

    def on_graph_descriptors(self, msg: "String") -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            payload = {"raw": msg.data}
        payload["_received_s"] = now_sec()
        with self.state.lock:
            self.state.graph_descriptor = payload

    def on_observation_cycle(self, msg: "String") -> None:
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                return
        except Exception as exc:
            self.state.add_event("error", "Failed to decode observation timing", error=str(exc))
            return
        payload["_received_s"] = now_sec()
        with self.state.lock:
            self.state.observation_cycle = payload

    def on_pipeline_health(self, msg: "String") -> None:
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                return
        except (ValueError, TypeError):
            return
        payload["_received_s"] = now_sec()
        with self.state.lock:
            self.state.pipeline_health = payload

    def on_parameter_event(self, event: "ParameterEvent") -> None:
        if str(event.node) != self.target_node:
            return
        updates = {}
        for param in list(event.new_parameters) + list(event.changed_parameters):
            updates[str(param.name)] = parameter_value_to_python(param.value)
        with self.state.lock:
            self.state.params.update(updates)
            for param in event.deleted_parameters:
                self.state.params.pop(str(param.name), None)

    def start_http_server(self) -> None:
        handler = make_handler(self)
        self.httpd = ReusableThreadingHTTPServer((self.host, self.port), handler)
        self.port = int(self.httpd.server_address[1])
        self.http_thread = threading.Thread(target=self.httpd.serve_forever, name="crossmaps-dashboard-http", daemon=True)
        self.http_thread.start()
        self.get_logger().info(f"CrossMaps dashboard: {self.url}")

    def shutdown_dashboard(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None

    def wait_future(self, future: Any, timeout_s: float = 3.0) -> Any:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if future.done():
                return future.result()
            time.sleep(0.01)
        raise TimeoutError(f"Timed out waiting for {self.target_node} parameter service")

    def refresh_parameters(self, names: Optional[List[str]] = None) -> Dict[str, Any]:
        names = names or sorted(PARAM_BY_NAME.keys())
        if not self.get_param_client.wait_for_service(timeout_sec=0.5):
            raise TimeoutError(f"Parameter service unavailable: {self.target_node}/get_parameters")
        request = GetParameters.Request()
        request.names = list(names)
        future = self.get_param_client.call_async(request)
        response = self.wait_future(future, timeout_s=3.0)
        values = getattr(response, "values", [])
        params = {}
        for name, value in zip(names, values):
            if hasattr(value, "value"):
                params[name] = value.value
            else:
                params[name] = parameter_value_to_python(value)
        with self.state.lock:
            self.state.params.update(params)
            self.state.param_errors.clear()
        return params

    def set_parameters_from_json(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        params = [parameter_from_json(name, value) for name, value in updates.items()]
        if not self.set_param_client.wait_for_service(timeout_sec=0.5):
            raise TimeoutError(f"Parameter service unavailable: {self.target_node}/set_parameters")
        request = SetParameters.Request()
        request.parameters = [param.to_parameter_msg() for param in params]
        future = self.set_param_client.call_async(request)
        response = self.wait_future(future, timeout_s=4.0)
        results = getattr(response, "results", [])
        output = {}
        accepted_updates = {}
        errors = {}
        for param, result in zip(params, results):
            ok = bool(getattr(result, "successful", False))
            reason = str(getattr(result, "reason", ""))
            output[param.name] = {"ok": ok, "reason": reason}
            if ok:
                accepted_updates[param.name] = param.value
            else:
                errors[param.name] = reason or "rejected"
        with self.state.lock:
            self.state.params.update(accepted_updates)
            self.state.param_errors.update(errors)
        if accepted_updates:
            self.state.add_event("info", "Parameters updated", names=", ".join(sorted(accepted_updates.keys())))
        if errors:
            self.state.add_event("warn", "Some parameters were rejected", names=", ".join(sorted(errors.keys())))
        return output

    def load_presets(self) -> Dict[str, Any]:
        try:
            with open(self.presets_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                payload.setdefault("presets", [])
                return payload
        except FileNotFoundError:
            pass
        except Exception as exc:
            self.state.add_event("warn", "Could not read presets", error=str(exc))
        return {"presets": []}

    def save_presets(self, payload: Dict[str, Any]) -> None:
        path = Path(self.presets_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)

    def status_payload(self, light: bool = False) -> Dict[str, Any]:
        t = now_sec()
        with self.state.lock:
            topics: Dict[str, Any] = {}
            for layer in CLOUD_LAYERS:
                snap = self.state.clouds.get(layer["id"])
                topics[layer["id"]] = {
                    "kind": "cloud",
                    "label": layer["label"],
                    "topic": layer["topic"],
                    "age_s": None if snap is None else round(t - snap.received_s, 3),
                    "count": 0 if snap is None else int(snap.decoded_count),
                    "original_count": 0 if snap is None else int(snap.original_count),
                    "frame_id": "" if snap is None else snap.frame_id,
                }
            for layer in IMAGE_LAYERS:
                snap = self.state.images.get(layer["id"])
                topics[layer["id"]] = {
                    "kind": "image",
                    "label": layer["label"],
                    "topic": layer["topic"],
                    "topics": layer.get("topics", [{"topic": layer["topic"], "type": "image"}]),
                    "age_s": None if snap is None else round(t - snap.received_s, 3),
                    "width": 0 if snap is None else int(snap.width),
                    "height": 0 if snap is None else int(snap.height),
                    "frame_id": "" if snap is None else snap.frame_id,
                    "source_topic": "" if snap is None else snap.source_topic,
                    "source_stamp_s": None if snap is None else snap.source_stamp_s,
                }
            graph_descriptor = dict(self.state.graph_descriptor)
            observation_cycle = dict(self.state.observation_cycle)
            pipeline_health = dict(self.state.pipeline_health)
            params = dict(self.state.params)
            events = list(self.state.events)
            graph_parts = dict(self.state.graph_parts)
            battery = self.state.battery
            robot_pose = self.state.robot_pose
            robot_pose_source_topic = self.robot_pose_source_topic
        graph_node_count = int(graph_descriptor.get("node_count", 0)) if graph_descriptor else 0
        graph_edge_count = int(graph_descriptor.get("edge_count", 0)) if graph_descriptor else 0
        if not graph_descriptor:
            graph_node_count = len(graph_parts.get("nodes", MarkerSnapshot({}, "", 0, 0)).items.get("nodes", []))
            graph_edge_count = sum(len(part.items.get("lines", [])) for part in graph_parts.values())
        payload = {
            "ok": True,
            "target_node": self.target_node,
            "dashboard_node": self.get_name(),
            "uptime_s": round(t - self.state.started_s, 1),
            "url": self.url,
            "cv2_available": cv2 is not None,
            "camera_streams_enabled": bool(self.camera_streams_enabled),
            "pipeline_health": None if not pipeline_health else {
                **{k:v for k,v in pipeline_health.items() if not k.startswith("_")},
                "age_s": round(t-float(pipeline_health.get("_received_s", t)), 3)},
            "active_query": params.get("query_text", graph_descriptor.get("query_text", "")),
            "graph": {
                "schema": graph_descriptor.get("schema", ""),
                "entity_type": graph_descriptor.get("entity_type", ""),
                "node_count": graph_node_count,
                "edge_count": graph_edge_count,
                "age_s": None if not graph_descriptor else round(t - float(graph_descriptor.get("_received_s", t)), 3),
            },
            "observation_cycle": None if not observation_cycle else {
                "last_ms": observation_cycle.get("last_ms"),
                "average_ms": observation_cycle.get("average_ms"),
                "rolling_average_ms": observation_cycle.get("rolling_average_ms"),
                "p95_ms": observation_cycle.get("p95_ms"),
                "average_hz": observation_cycle.get("average_hz"),
                "window_count": observation_cycle.get("window_count", 0),
                "window_size": observation_cycle.get("window_size", 0),
                "total_count": observation_cycle.get("total_count", 0),
                "mode": observation_cycle.get("mode", ""),
                "age_s": round(t - float(observation_cycle.get("_received_s", t)), 3),
            },
            "battery": None if battery is None else {
                "percentage": battery.percentage,
                "voltage": battery.voltage,
                "current": battery.current,
                "status": battery.power_supply_status,
                "topic": battery.source_topic,
                "age_s": round(t - battery.received_s, 3),
            },
            "robot": {
                "available": robot_pose is not None,
                "position": [0.0, 0.0, 0.0] if robot_pose is None else list(robot_pose.position),
                "yaw_rad": 0.0 if robot_pose is None else robot_pose.yaw_rad,
                "frame_id": "" if robot_pose is None else robot_pose.frame_id,
                "child_frame_id": "" if robot_pose is None else robot_pose.child_frame_id,
                "topic": "" if robot_pose is None else robot_pose.source_topic,
                "selected_topic": "" if robot_pose_source_topic is None else robot_pose_source_topic,
                "configured_topic": self.robot_odom_topic,
                "age_s": None if robot_pose is None else round(t - robot_pose.received_s, 3),
            },
            "topics": topics,
            "events": events[-12:] if light else events,
        }
        return payload

    def clouds_payload(self, max_points: int = 0) -> Dict[str, Any]:
        with self.state.lock:
            snapshots = dict(self.state.clouds)
            params = dict(self.state.params)
            graph_parts = dict(self.state.graph_parts)
            robot_pose = self.state.robot_pose
            robot_pose_source_topic = self.robot_pose_source_topic

        layers = []
        for layer in CLOUD_LAYERS:
            snap = snapshots.get(layer["id"])
            disabled = [name for name in [layer.get("publishParam"), *layer.get("requiresParams", [])]
                        if name and params.get(name) is False]
            state = ("disabled" if disabled else "waiting" if snap is None else
                     "stale" if now_sec() - snap.received_s > 10.0 else
                     "empty" if snap.decoded_count == 0 else "ready")
            diagnostics = {"state": state, "disabled_by": disabled}
            if snap is None or disabled:
                layers.append(
                    {
                        **diagnostics,
                        "id": layer["id"],
                        "label": layer["label"],
                        "topic": layer["topic"],
                        "available": False,
                        "positions": [],
                        "colors": [],
                        "count": 0,
                        "age_s": None,
                    }
                )
                continue
            positions = snap.positions
            colors = snap.colors
            if max_points > 0 and positions.shape[0] > max_points:
                idx = np.linspace(0, positions.shape[0] - 1, max_points, dtype=np.int64)
                positions = positions[idx]
                colors = colors[idx]
            layers.append(
                {
                    **diagnostics,
                    "id": layer["id"],
                    "label": layer["label"],
                    "topic": layer["topic"],
                    "available": True,
                    "positions": np.round(positions.reshape(-1), 4).tolist(),
                    "colors": colors.reshape(-1).astype(int).tolist(),
                    "count": int(positions.shape[0]),
                    "decoded_count": int(snap.decoded_count),
                    "original_count": int(snap.original_count),
                    "frame_id": snap.frame_id,
                    "age_s": round(now_sec() - snap.received_s, 3),
                }
            )

        graph = {"nodes": [], "labels": [], "lines": []}
        for part in graph_parts.values():
            for key in ("nodes", "labels", "lines"):
                graph[key].extend(part.items.get(key, []))

        robot = {
            "available": robot_pose is not None,
            "position": [0.0, 0.0, 0.0] if robot_pose is None else list(robot_pose.position),
            "yaw_rad": 0.0 if robot_pose is None else robot_pose.yaw_rad,
            "frame_id": "" if robot_pose is None else robot_pose.frame_id,
            "child_frame_id": "" if robot_pose is None else robot_pose.child_frame_id,
            "topic": "" if robot_pose is None else robot_pose.source_topic,
            "selected_topic": "" if robot_pose_source_topic is None else robot_pose_source_topic,
            "configured_topic": self.robot_odom_topic,
            "age_s": None if robot_pose is None else round(now_sec() - robot_pose.received_s, 3),
        }

        return {"layers": layers, "graph": graph, "robot": robot, "stamp_s": round(now_sec(), 3)}

    def graph_payload(self) -> Dict[str, Any]:
        with self.state.lock:
            descriptor = dict(self.state.graph_descriptor)
            graph_parts = dict(self.state.graph_parts)
        graph = {"nodes": [], "labels": [], "lines": []}
        for part in graph_parts.values():
            for key in ("nodes", "labels", "lines"):
                graph[key].extend(part.items.get(key, []))
        descriptor.setdefault("nodes", [])
        descriptor.setdefault("edges", [])
        return {"descriptor": descriptor, "markers": graph}

    def config_payload(self) -> Dict[str, Any]:
        return {
            "cloudLayers": CLOUD_LAYERS,
            "imageLayers": IMAGE_LAYERS,
            "gridLayers": GRID_LAYERS,
            "paramGroups": PARAM_GROUPS,
            "targetNode": self.target_node,
            "presetsPath": self.presets_path,
            "cameraStreamsEnabled": bool(self.camera_streams_enabled),
            "robotOdomTopic": self.robot_odom_topic,
        }


def make_handler(dashboard: CrossMapsDashboard):
    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "CrossMapsDashboard/1.0"

        def handle(self) -> None:
            # Normal browser reloads/tab switches can reset a keep-alive socket
            # while BaseHTTPRequestHandler is reading the next request line.
            # Catch only disconnects; programming/server errors must still surface.
            try:
                super().handle()
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                self.close_connection = True

        def log_message(self, fmt: str, *args: Any) -> None:
            dashboard.get_logger().debug(fmt % args)

        def end_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            super().end_headers()

        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()

        def do_GET(self) -> None:
            try:
                parsed = urlparse(self.path)
                path = unquote(parsed.path)
                query = parse_qs(parsed.query)
                if path == "/":
                    self.send_static_file("index.html")
                elif path.startswith("/static/"):
                    self.send_static_file(path[len("/static/") :])
                elif path == "/api/config":
                    self.send_json(dashboard.config_payload())
                elif path == "/api/status":
                    self.send_json(dashboard.status_payload(light=query.get("light", ["0"])[0] == "1"))
                elif path == "/api/clouds":
                    max_points = int(query.get("max_points", ["0"])[0] or "0")
                    self.send_json(dashboard.clouds_payload(max_points=max_points))
                elif path == "/api/graph":
                    self.send_json(dashboard.graph_payload())
                elif path == "/api/params":
                    refresh = query.get("refresh", ["0"])[0] == "1"
                    names = query.get("name", [])
                    if refresh:
                        try:
                            params = dashboard.refresh_parameters(names or None)
                        except Exception as exc:
                            self.send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_GATEWAY)
                            return
                    else:
                        with dashboard.state.lock:
                            params = dict(dashboard.state.params)
                    self.send_json({"ok": True, "params": params})
                elif path == "/api/presets":
                    self.send_json(dashboard.load_presets())
                elif path == "/api/camera_streams":
                    self.send_json({"ok": True, "enabled": bool(dashboard.camera_streams_enabled)})
                elif path.startswith("/api/image/"):
                    layer_id = path.rsplit("/", 1)[-1]
                    with dashboard.state.lock:
                        snap = dashboard.state.images.get(layer_id)
                    if snap is None:
                        self.send_json({"ok": False, "error": "image unavailable"}, status=HTTPStatus.NOT_FOUND)
                    else:
                        self.send_bytes(snap.encoded, snap.mime)
                elif path.startswith("/api/grid/"):
                    layer_id = path.rsplit("/", 1)[-1]
                    with dashboard.state.lock:
                        snap = dashboard.state.grids.get(layer_id)
                    if snap is None:
                        self.send_json({"ok": False, "error": "grid unavailable"}, status=HTTPStatus.NOT_FOUND)
                    else:
                        png = grid_to_png(snap, layer_id)
                        if png is None:
                            self.send_json({"ok": False, "error": "grid encoding unavailable"}, status=HTTPStatus.NOT_FOUND)
                        else:
                            self.send_bytes(png, "image/png")
                elif path == "/api/events":
                    self.stream_events()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except Exception:
                dashboard.state.add_event("error", "HTTP GET failed", traceback=traceback.format_exc(limit=2))
                self.send_json({"ok": False, "error": "request failed"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self) -> None:
            try:
                parsed = urlparse(self.path)
                path = unquote(parsed.path)
                body = self.read_json()
                if path == "/api/params":
                    updates = body.get("updates")
                    if updates is None and "name" in body:
                        updates = {body["name"]: body.get("value")}
                    if not isinstance(updates, dict) or not updates:
                        self.send_json({"ok": False, "error": "missing updates"}, status=HTTPStatus.BAD_REQUEST)
                        return
                    results = dashboard.set_parameters_from_json(updates)
                    self.send_json({"ok": True, "results": results})
                elif path == "/api/presets":
                    name = str(body.get("name", "")).strip()
                    values = body.get("values")
                    if not name or not isinstance(values, dict):
                        self.send_json({"ok": False, "error": "preset name and values required"}, status=HTTPStatus.BAD_REQUEST)
                        return
                    payload = dashboard.load_presets()
                    presets = [preset for preset in payload.get("presets", []) if preset.get("name") != name]
                    presets.append({"name": name, "values": values, "updated_s": round(now_sec(), 3)})
                    payload["presets"] = sorted(presets, key=lambda item: item.get("name", ""))
                    dashboard.save_presets(payload)
                    dashboard.state.add_event("info", "Preset saved", name=name)
                    self.send_json(payload)
                elif path == "/api/presets/apply":
                    name = str(body.get("name", "")).strip()
                    payload = dashboard.load_presets()
                    preset = next((item for item in payload.get("presets", []) if item.get("name") == name), None)
                    if preset is None:
                        self.send_json({"ok": False, "error": "preset not found"}, status=HTTPStatus.NOT_FOUND)
                        return
                    results = dashboard.set_parameters_from_json(dict(preset.get("values", {})))
                    self.send_json({"ok": True, "results": results, "preset": preset})
                elif path == "/api/camera_streams":
                    enabled = bool(body.get("enabled", False))
                    self.send_json(dashboard.set_camera_streams_enabled(enabled))
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as exc:
                dashboard.state.add_event("error", "HTTP POST failed", error=str(exc))
                self.send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_DELETE(self) -> None:
            try:
                parsed = urlparse(self.path)
                path = unquote(parsed.path)
                query = parse_qs(parsed.query)
                if path != "/api/presets":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                name = str(query.get("name", [""])[0]).strip()
                payload = dashboard.load_presets()
                payload["presets"] = [preset for preset in payload.get("presets", []) if preset.get("name") != name]
                dashboard.save_presets(payload)
                dashboard.state.add_event("info", "Preset deleted", name=name)
                self.send_json(payload)
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def read_json(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(raw.decode("utf-8"))

        def send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
            data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_bytes(self, payload: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def send_static_file(self, rel_path: str) -> None:
            rel_path = rel_path.strip("/") or "index.html"
            path = (dashboard.static_dir / rel_path).resolve()
            root = dashboard.static_dir.resolve()
            if root not in path.parents and path != root:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if not path.exists() or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type, _ = mimetypes.guess_type(str(path))
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def stream_events(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            for _ in range(3600):
                try:
                    payload = dashboard.status_payload(light=True)
                    data = f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode("utf-8")
                    self.wfile.write(data)
                    self.wfile.flush()
                    time.sleep(1.0)
                except (BrokenPipeError, ConnectionResetError):
                    break

    return DashboardHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CrossMaps browser dashboard")
    parser.add_argument("--target-node", default="/crossmaps_3_node", help="CrossMaps ROS node name")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address")
    parser.add_argument("--port", default=8765, type=int, help="HTTP port, use 0 for an ephemeral port")
    parser.add_argument("--static-dir", default=str(APP_DIR), help="Dashboard static asset directory")
    parser.add_argument("--presets", default=DEFAULT_PRESETS_PATH, help="Parameter preset JSON path")
    parser.add_argument("--cloud-budget", default=60000, type=int, help="Maximum decoded points per cloud layer")
    parser.add_argument(
        "--robot-odom-topic",
        default="auto",
        help="Odometry topic for the rover marker, or 'auto' to choose a stable source",
    )
    parser.add_argument(
        "--enable-camera-streams",
        action="store_true",
        help="Subscribe directly to raw/compressed camera and depth topics at startup",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if ROS_IMPORT_ERROR is not None:
        print(f"ROS 2 Python imports failed: {ROS_IMPORT_ERROR}")
        print("Run this inside the same ROS 2 environment you use for CrossMaps.")
        return 2

    static_dir = Path(args.static_dir).expanduser().resolve()
    if not static_dir.exists():
        print(f"Dashboard static directory not found: {static_dir}")
        return 2

    rclpy.init()
    node = CrossMapsDashboard(
        target_node=args.target_node,
        host=args.host,
        port=args.port,
        static_dir=static_dir,
        presets_path=args.presets,
        cloud_budget=args.cloud_budget,
        enable_camera_streams=args.enable_camera_streams,
        robot_odom_topic=args.robot_odom_topic,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_dashboard()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
