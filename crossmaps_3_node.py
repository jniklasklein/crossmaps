#!/usr/bin/env python3
# crossmaps_3_node.py
import os
import math
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

import tf2_ros

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Header

from cv_bridge import CvBridge

import torch
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


# -----------------------------
# node
# -----------------------------
class CrossMaps3Node(Node):
    """
    CrossMaps v3:
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
        * /oak/rgb/image_rect (or as set by rgb_topic param)
        * /oak/stereo/image_raw (or as set by depth_topic param)
        * /oak/rgb/camera_info (or as set by cam_info_topic param)
        * /map (occupancy grid for ray-consistency, or as set by occ_grid_topic param)
    - Publishes:
        * /vlmaps/stm/semantic_grid
        * /vlmaps/stm/confidence_grid
        * /vlmaps/stm/top1_grid
        * /vlmaps/ltm/semantic_grid
        * /vlmaps/ltm/confidence_grid
        * /vlmaps/ltm/top1_grid
        * /vlmaps/debug_cloud
        * /vlmaps/semantic_cloud
        * /vlmaps/history_cloud
    """

    def __init__(self):
        super().__init__("crossmaps_3_node")
        self.bridge = CvBridge()
        self.frame_counter = 0

        # -------------------------
        # Params
        # -------------------------
        self.declare_parameter("rgb_topic", "/oak/rgb/image_rect")
        self.declare_parameter("depth_topic", "/oak/stereo/image_raw")
        self.declare_parameter("cam_info_topic", "/oak/rgb/camera_info")

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

        self.declare_parameter("tiles_coarse_x", 8)
        self.declare_parameter("tiles_coarse_y", 4)
        self.declare_parameter("tiles_fine_x", 16)
        self.declare_parameter("tiles_fine_y", 8)
        self.declare_parameter("fuse_coarse_weight", 0.35)

        self.declare_parameter("embed_every_n", 10)
        self.declare_parameter("publish_hz", 2.0)

        # point cloud publishing (keep off for better performance)
        self.declare_parameter("publish_debug_cloud", False)
        self.declare_parameter("publish_semantic_cloud", False)
        self.declare_parameter("publish_history_cloud", False)

        # STM grid publishing
        self.declare_parameter("publish_stm_semantic_grid", True)
        self.declare_parameter("publish_stm_confidence_grid", True)
        self.declare_parameter("publish_stm_top1_grid", True)

        # LTM grid publishing
        self.declare_parameter("publish_ltm_semantic_grid", True)
        self.declare_parameter("publish_ltm_confidence_grid", True)
        self.declare_parameter("publish_ltm_top1_grid", True)

        self.declare_parameter("semantic_cloud_gamma", 0.8)
        self.declare_parameter("semantic_cloud_max_points", 15000)

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
        self.declare_parameter("conf_coarse_boost", 0.6)

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

        # normalization for semantic grid
        self.declare_parameter("norm_low_percentile", 65.0)
        self.declare_parameter("norm_high_percentile", 99.0)

        # history cloud
        self.declare_parameter("history_voxel_m", 0.05)
        self.declare_parameter("history_max_points", 300000)

        # -------------------------
        # Read params (initial)
        # -------------------------
        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.cam_info_topic = self.get_parameter("cam_info_topic").value
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

        # -------------------------
        # STM state (INTERNAL frame grid coords)
        # -------------------------
        self.stm_sum: Dict[Tuple[int, int], np.ndarray] = {}
        self.stm_w: Dict[Tuple[int, int], float] = {}
        self.stm_last: Dict[Tuple[int, int], float] = {}
        self.stm_viewmask: Dict[Tuple[int, int], int] = {}

        # -------------------------
        # LTM state (INTERNAL frame grid coords)
        # -------------------------
        self.ltm_sum: Dict[Tuple[int, int], np.ndarray] = {}
        self.ltm_w: Dict[Tuple[int, int], float] = {}
        self.ltm_last_seen: Dict[Tuple[int, int], float] = {}  # seconds, last evidence accepted
        self.ltm_viewmask: Dict[Tuple[int, int], int] = {}     # optional tracking of view diversity in LTM

        # history cloud stored in INTERNAL frame voxel grid
        self.history_vox: Dict[Tuple[int, int, int], Tuple[float, float, float, float]] = {}
        self.white_rgb = pack_rgb(255, 255, 255)

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
        model_name = "ViT-L-14" # "ViT-B-32" 
        #model_name = "ViT-B-32"
        pretrained = "openai"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.model.to(self.device)

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
        # STM grids
        self.pub_stm_sem = self.create_publisher(OccupancyGrid, "/vlmaps/stm/semantic_grid", 1)
        self.pub_stm_conf = self.create_publisher(OccupancyGrid, "/vlmaps/stm/confidence_grid", 1)
        self.pub_stm_top1 = self.create_publisher(OccupancyGrid, "/vlmaps/stm/top1_grid", 1)

        # LTM grids
        self.pub_ltm_sem = self.create_publisher(OccupancyGrid, "/vlmaps/ltm/semantic_grid", 1)
        self.pub_ltm_conf = self.create_publisher(OccupancyGrid, "/vlmaps/ltm/confidence_grid", 1)
        self.pub_ltm_top1 = self.create_publisher(OccupancyGrid, "/vlmaps/ltm/top1_grid", 1)

        # clouds
        self.pub_debug_cloud = self.create_publisher(PointCloud2, "/vlmaps/debug_cloud", 10)
        self.pub_sem_cloud = self.create_publisher(PointCloud2, "/vlmaps/semantic_cloud", 10)
        self.pub_hist_cloud = self.create_publisher(PointCloud2, "/vlmaps/history_cloud", 1)

        self.timer = self.create_timer(1.0 / float(self.get_parameter("publish_hz").value), self.on_publish)

        self.sub_cam = self.create_subscription(CameraInfo, self.cam_info_topic, self.on_cam_info, 10)

        # occupancy grid subscription (still used by ray-consistency)
        self.sub_occ = self.create_subscription(
            OccupancyGrid, self.get_parameter("occ_grid_topic").value, self.on_occ_grid, 1
        )

        # RGB+Depth sync
        self.rgb_sub = message_filters.Subscriber(self, Image, self.rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=int(self.get_parameter("sync_queue_size").value),
            slop=float(self.get_parameter("sync_slop_s").value),
            allow_headerless=False,
        )
        self.ts.registerCallback(self.on_rgb_depth)

        self.get_logger().info("CrossMaps v3 node started (STM + LTM, no tool map, no save/load).")
        self.get_logger().info(f"internal_frame={self.internal_frame}, publish_frame={self.publish_frame}")
        self.get_logger().info(
            "Publishes: "
            "/vlmaps/stm/{semantic_grid,confidence_grid,top1_grid}, "
            "/vlmaps/ltm/{semantic_grid,confidence_grid,top1_grid}, "
            "/vlmaps/debug_cloud, /vlmaps/semantic_cloud, /vlmaps/history_cloud"
        )

    # -------------------------
    # Basics
    # -------------------------
    def now_sec(self):
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def on_cam_info(self, msg: CameraInfo):
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])
        self.get_logger().info(
            f"Camera intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}"
        )
        self.destroy_subscription(self.sub_cam)

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
    def _prompt(self, text: str) -> str:
        return f"a photo of a {text}"

    def compute_clip_text_embedding(self, text: str) -> np.ndarray:
        prompt = self._prompt(text)
        tokens = self.tokenizer([prompt]).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.squeeze(0).float().cpu().numpy()

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

        with torch.no_grad():
            emb = self.model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.float().cpu().numpy()

        return emb.reshape((tiles_y, tiles_x, -1)).astype(np.float32)

    def ensure_query_embedding(self):
        new_query = str(self.get_parameter("query_text").value)
        if new_query != self.query_text:
            self.query_text = new_query
            self.text_emb = None
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
        if self.stm_w[key] < 1e-3:
            self.stm_w.pop(key, None)
            self.stm_sum.pop(key, None)
            self.stm_last.pop(key, None)
            self.stm_viewmask.pop(key, None)

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
        mask |= (1 << b)
        viewmask_dict[key] = mask

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

    def fuse_stm_cell(self, key, new_emb_unit: np.ndarray, conf: float, now_s: float):
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
            return

        emb_old, _, _ = self.stm_embedding_and_coherence(key)
        if emb_old is None:
            self.stm_sum[key] = (conf * new_emb_unit).astype(np.float32)
            self.stm_w[key] = float(conf)
            self.stm_last[key] = now_s
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
            return

        old_u, _, old_w = self.ltm_embedding_and_coherence(stm_key)
        if old_u is None or old_w is None:
            self.ltm_sum[stm_key] = (ev_w * emb_unit).astype(np.float32)
            self.ltm_w[stm_key] = float(np.clip(ev_w, 0.0, ltm_w_max))
            self.ltm_last_seen[stm_key] = now_s
            self.ltm_viewmask[stm_key] = int(self.stm_viewmask.get(stm_key, 0))
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
            return

        # Consistent: aggregate
        new_sum = (self.ltm_sum[stm_key] + ev_w * emb_unit).astype(np.float32)
        new_w = float(np.clip(self.ltm_w[stm_key] + ev_w, 0.0, ltm_w_max))
        self.ltm_sum[stm_key] = new_sum
        self.ltm_w[stm_key] = new_w
        self.ltm_last_seen[stm_key] = now_s
        self.ltm_viewmask[stm_key] = int(self.ltm_viewmask.get(stm_key, 0) | int(self.stm_viewmask.get(stm_key, 0)))

    # -------------------------
    # Coherence shaping
    # -------------------------
    def _coherence01(self, coh: float) -> float:
        coh_floor = float(self.get_parameter("coherence_floor").value)
        coh_pow = float(self.get_parameter("coherence_power").value)
        coh01 = float(np.clip((coh - coh_floor) / max(1.0 - coh_floor, 1e-6), 0.0, 1.0))
        return float(np.power(coh01, coh_pow))

    # -------------------------
    # Depth->points in INTERNAL frame
    # -------------------------
    def depth_to_points_internal(self, depth_msg: Image):
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth = np.asarray(depth)

        if depth_msg.encoding == "16UC1" or depth.dtype == np.uint16:
            depth_m = depth.astype(np.float32) * 0.001
        else:
            depth_m = depth.astype(np.float32)

        if depth_m.ndim != 2:
            return None

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
            return None

        x = (uu - self.cx) * z / self.fx
        y = (vv - self.cy) * z / self.fy
        pts_cam = np.stack([x, y, z], axis=1)

        source_frame = depth_msg.header.frame_id
        if not source_frame:
            return None

        try:
            tf = self.tf_buffer.lookup_transform(
                self.internal_frame,
                source_frame,
                Time.from_msg(depth_msg.header.stamp),
                timeout=Duration(seconds=0.5),
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed {self.internal_frame} <- {source_frame}: {e}")
            return None

        T = transform_to_matrix(tf)
        pts_h = np.ones((pts_cam.shape[0], 4), dtype=np.float32)
        pts_h[:, :3] = pts_cam.astype(np.float32)
        pts_out = (T @ pts_h.T).T[:, :3]
        cam_pos = T[:3, 3].copy()

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
            if stamp_msg is None:
                tf = self.tf_buffer.lookup_transform(
                    self.publish_frame, self.internal_frame, Time(), timeout=Duration(seconds=0.3)
                )
            else:
                tf = self.tf_buffer.lookup_transform(
                    self.publish_frame,
                    self.internal_frame,
                    Time.from_msg(stamp_msg),
                    timeout=Duration(seconds=0.3),
                )
            return transform_to_matrix(tf)
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
        for p in pts_internal:
            vx = int(np.floor(p[0] * inv))
            vy = int(np.floor(p[1] * inv))
            vz = int(np.floor(p[2] * inv))
            k = (vx, vy, vz)
            if k in self.history_vox:
                continue
            self.history_vox[k] = (float(p[0]), float(p[1]), float(p[2]), self.white_rgb)
            if len(self.history_vox) >= max_pts:
                break

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
    # Sync callback (updates STM)
    # -------------------------
    def on_rgb_depth(self, rgb_msg: Image, depth_msg: Image):
        if self.fx is None:
            return

        self.frame_counter += 1
        if self.frame_counter % int(self.get_parameter("embed_every_n").value) != 0:
            return

        # RGB
        try:
            rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        except Exception as e:
            self.get_logger().warn(f"RGB conversion failed: {e}")
            return

        # tile embeddings (cached)
        if self.tile_stamp != rgb_msg.header.stamp or self.tile_coarse is None or self.tile_fine is None:
            cx = int(self.get_parameter("tiles_coarse_x").value)
            cy = int(self.get_parameter("tiles_coarse_y").value)
            fx = int(self.get_parameter("tiles_fine_x").value)
            fy = int(self.get_parameter("tiles_fine_y").value)
            self.tile_coarse = self.compute_tile_embeddings_batched(rgb_cv, cx, cy)
            self.tile_fine = self.compute_tile_embeddings_batched(rgb_cv, fx, fy)
            self.tile_stamp = rgb_msg.header.stamp

        out = self.depth_to_points_internal(depth_msg)
        if out is None:
            return
        pts_int, cam_int, uu, vv, z_cam, (h, w) = out

        # update history
        self.update_history_cloud(pts_int)

        # debug cloud (publish-frame warped)
        if bool(self.get_parameter("publish_debug_cloud").value):
            T = self.lookup_T_publish_internal(depth_msg.header.stamp)
            if T is not None:
                pts_h = np.ones((pts_int.shape[0], 4), dtype=np.float32)
                pts_h[:, :3] = pts_int.astype(np.float32)
                pts_pub = (T @ pts_h.T).T[:, :3]
                self.pub_debug_cloud.publish(make_cloud_xyz(pts_pub, self.publish_frame, depth_msg.header.stamp))

        # query embedding (for semantic cloud + grids)
        self.ensure_query_embedding()
        if self.text_emb is None:
            return

        # per-point embeddings (multi-scale)
        cx = int(self.get_parameter("tiles_coarse_x").value)
        cy = int(self.get_parameter("tiles_coarse_y").value)
        fx = int(self.get_parameter("tiles_fine_x").value)
        fy = int(self.get_parameter("tiles_fine_y").value)
        w_coarse = float(self.get_parameter("fuse_coarse_weight").value)
        w_fine = 1.0 - w_coarse

        tx_c = np.clip((uu / w * cx).astype(np.int32), 0, cx - 1)
        ty_c = np.clip((vv / h * cy).astype(np.int32), 0, cy - 1)
        tx_f = np.clip((uu / w * fx).astype(np.int32), 0, fx - 1)
        ty_f = np.clip((vv / h * fy).astype(np.int32), 0, fy - 1)

        emb_c = self.tile_coarse[ty_c, tx_c, :]
        emb_f = self.tile_fine[ty_f, tx_f, :]

        emb_c = emb_c / (np.linalg.norm(emb_c, axis=1, keepdims=True) + 1e-6)
        emb_f = emb_f / (np.linalg.norm(emb_f, axis=1, keepdims=True) + 1e-6)

        emb = (w_coarse * emb_c) + (w_fine * emb_f)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-6)

        # similarity (for semantic cloud)
        sim = np.clip(np.sum(emb * self.text_emb[None, :], axis=1), -1.0, 1.0)
        sim_pos = np.maximum(0.0, sim)

        # point confidence
        d0 = float(self.get_parameter("conf_d0_m").value)
        conf_min = float(self.get_parameter("conf_min").value)
        conf_dist = np.exp(-np.clip(z_cam, 0.0, 50.0) / max(d0, 1e-3)).astype(np.float32)
        c_boost = float(self.get_parameter("conf_coarse_boost").value)
        f_boost = float(self.get_parameter("conf_fine_boost").value)
        conf_scale = (w_coarse * c_boost + w_fine * f_boost)
        conf = np.clip(conf_dist * conf_scale, conf_min, 1.0).astype(np.float32)

        # Ray consistency needs map coords
        use_ray = bool(self.get_parameter("use_ray_consistency").value)
        ray_stride = int(self.get_parameter("ray_check_stride").value)
        T_pub_int = self.lookup_T_publish_internal(depth_msg.header.stamp) if use_ray else None

        # fuse into STM (INTERNAL grid)
        now_s = self.now_sec()
        cam_xy_int = (float(cam_int[0]), float(cam_int[1]))

        for idx, ((x, y, z), e, c) in enumerate(zip(pts_int, emb, conf)):
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
                and (idx % ray_stride == 0)
                and (self.occ is not None)
                and (T_pub_int is not None)
            ):
                cam_h = np.array([cam_int[0], cam_int[1], cam_int[2], 1.0], dtype=np.float32)
                pt_h = np.array([x, y, z, 1.0], dtype=np.float32)
                cam_pub = (T_pub_int @ cam_h)[:3]
                pt_pub = (T_pub_int @ pt_h)[:3]
                if self.ray_is_occluded((float(cam_pub[0]), float(cam_pub[1])), (float(pt_pub[0]), float(pt_pub[1]))):
                    continue

            self.fuse_stm_cell(cell, e.astype(np.float32), float(c), now_s)

        # publish semantic cloud (publish frame)
        if bool(self.get_parameter("publish_semantic_cloud").value):
            T = self.lookup_T_publish_internal(depth_msg.header.stamp)
            if T is not None:
                max_pts = int(self.get_parameter("semantic_cloud_max_points").value)
                gamma = float(self.get_parameter("semantic_cloud_gamma").value)
                N = pts_int.shape[0]
                if N > max_pts:
                    idxs = np.random.choice(N, size=max_pts, replace=False)
                    pts_sel = pts_int[idxs]
                    s_sel = sim_pos[idxs]
                else:
                    pts_sel = pts_int
                    s_sel = sim_pos

                pts_h = np.ones((pts_sel.shape[0], 4), dtype=np.float32)
                pts_h[:, :3] = pts_sel.astype(np.float32)
                pts_pub = (T @ pts_h.T).T[:, :3]

                s01 = np.power(np.clip(s_sel, 0.0, 1.0), gamma)
                rgb = np.zeros((pts_pub.shape[0],), dtype=np.float32)
                for i, v01 in enumerate(s01):
                    r, g, b = blue_to_red(float(v01))
                    rgb[i] = pack_rgb(r, g, b)
                self.pub_sem_cloud.publish(make_cloud_xyzrgb(pts_pub, rgb, self.publish_frame, depth_msg.header.stamp))

    # -------------------------
    # Memory -> published grids (shared logic)
    # -------------------------
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

    # -------------------------
    # Publish grids (STM + LTM) + promotions
    # -------------------------
    def on_publish(self):
        self.ensure_query_embedding()
        self.publish_history_cloud()

        # empty defaults
        empty_sem = -np.ones((self.grid_height, self.grid_width), dtype=np.int8)
        empty_conf = -np.ones((self.grid_height, self.grid_width), dtype=np.int8)
        empty_top1 = -np.ones((self.grid_height, self.grid_width), dtype=np.int8)

        if self.text_emb is None:
            if bool(self.get_parameter("publish_stm_semantic_grid").value):
                self.publish_grid(empty_sem, self.pub_stm_sem)
            if bool(self.get_parameter("publish_stm_confidence_grid").value):
                self.publish_grid(empty_conf, self.pub_stm_conf)
            if bool(self.get_parameter("publish_stm_top1_grid").value):
                self.publish_grid(empty_top1, self.pub_stm_top1)

            if bool(self.get_parameter("publish_ltm_semantic_grid").value):
                self.publish_grid(empty_sem, self.pub_ltm_sem)
            if bool(self.get_parameter("publish_ltm_confidence_grid").value):
                self.publish_grid(empty_conf, self.pub_ltm_conf)
            if bool(self.get_parameter("publish_ltm_top1_grid").value):
                self.publish_grid(empty_top1, self.pub_ltm_top1)
            return

        # decay some STM cells
        now_s = self.now_sec()
        stm_keys = list(self.stm_w.keys())
        for k in stm_keys[: min(3000, len(stm_keys))]:
            self.decay_stm_cell(k, now_s)

        # need transform internal->publish for both STM and LTM grids
        T = self.lookup_T_publish_internal()
        if T is None:
            return

        use_view = bool(self.get_parameter("use_visibility_boost").value)
        use_coh_conf = bool(self.get_parameter("use_coherence_in_confidence").value)
        use_coh_sem = bool(self.get_parameter("use_coherence_in_semantic").value)

        # -------------------------
        # STM -> compute and publish
        # -------------------------
        stm_keys = list(self.stm_w.keys())
        if len(stm_keys) < 20:
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

        # -------------------------
        # LTM -> compute and publish
        # -------------------------
        ltm_keys = list(self.ltm_w.keys())
        if len(ltm_keys) < 20:
            if bool(self.get_parameter("publish_ltm_semantic_grid").value):
                self.publish_grid(empty_sem, self.pub_ltm_sem)
            if bool(self.get_parameter("publish_ltm_confidence_grid").value):
                self.publish_grid(empty_conf, self.pub_ltm_conf)
            if bool(self.get_parameter("publish_ltm_top1_grid").value):
                self.publish_grid(empty_top1, self.pub_ltm_top1)
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
            if bool(self.get_parameter("publish_ltm_semantic_grid").value):
                self.publish_grid(empty_sem, self.pub_ltm_sem)
            if bool(self.get_parameter("publish_ltm_confidence_grid").value):
                self.publish_grid(empty_conf, self.pub_ltm_conf)
            if bool(self.get_parameter("publish_ltm_top1_grid").value):
                self.publish_grid(empty_top1, self.pub_ltm_top1)
            return

        _, _, scores_ltm, confs_ltm, _ = res_ltm
        sem_ltm, conf_ltm, top1_ltm = self._build_sem_conf_top1_grids(ltm_cells_pub, scores_ltm, confs_ltm)

        if bool(self.get_parameter("publish_ltm_semantic_grid").value):
            self.publish_grid(sem_ltm, self.pub_ltm_sem)
        if bool(self.get_parameter("publish_ltm_confidence_grid").value):
            self.publish_grid(conf_ltm, self.pub_ltm_conf)
        if bool(self.get_parameter("publish_ltm_top1_grid").value):
            self.publish_grid(top1_ltm, self.pub_ltm_top1)

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


def main():
    rclpy.init()
    node = CrossMaps3Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()