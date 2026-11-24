"""Testable Gazebo protobuf-JSON to compressed ROS payload conversions.

No ROS/Gazebo imports here. Raw images exist only on the local helper pipe;
the ROS publishers expose JPEG RGB and lossless millimetre PNG depth.
"""
import base64
import hashlib
import math
import struct

import cv2
import numpy as np

CAMERA_FRAME = "oak_rgb_camera_optical_frame"
LIDAR_FRAME = "laser_frame"
SIM_DOMAINS = {"rover_a": 80, "rover_b": 81}


class LatestRgbdPair:
    """Bounded exact-stamp join; encode only the newest complete capture.

    Delayed older images cannot overwrite current frames. A simulation clock
    reset requires a new bridge/session, as it does for the mapping adapter.
    """
    def __init__(self, capacity=4):
        if capacity < 1:
            raise ValueError("pair buffer capacity must be positive")
        self.capacity = capacity
        self.pending = {"rgb": {}, "depth": {}}
        self.watermark = -1
        self.dropped = 0

    def add(self, kind, message):
        if kind not in self.pending:
            raise ValueError("pair buffer accepts only RGB or depth")
        tick = stamp_ns(message)
        if tick <= self.watermark:
            self.dropped += 1
            return
        slot = self.pending[kind]
        slot[tick] = message
        while len(slot) > self.capacity:
            del slot[min(slot)]
            self.dropped += 1

    def take_latest(self):
        common = self.pending["rgb"].keys() & self.pending["depth"].keys()
        if not common:
            return None
        tick = max(common)
        pair = (self.pending["rgb"][tick], self.pending["depth"][tick])
        for slot in self.pending.values():
            for old in list(slot):
                if old <= tick:
                    del slot[old]
                    self.dropped += old != tick
        self.watermark = tick
        return pair


def camera_extrinsic(scene, robot):
    """Read generated model metadata; never guess a camera mount transform."""
    try:
        entry = scene["robots"][robot]["camera_optical"]
        translation = tuple(float(x) for x in entry["translation"])
        quaternion = tuple(float(x) for x in entry["quaternion_xyzw"])
        frame = entry["frame_id"]
        status = entry["calibration_status"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or malformed {robot} optical calibration in scene.json") from exc
    if len(translation) != 3 or len(quaternion) != 4:
        raise ValueError("optical extrinsic must contain XYZ translation and XYZW quaternion")
    if not all(math.isfinite(x) for x in (*translation, *quaternion)):
        raise ValueError("optical extrinsic must be finite")
    if max(abs(x) for x in translation) > 5.:
        raise ValueError("optical extrinsic must use metres, within 5m of rover base")
    if abs(sum(x*x for x in quaternion)-1.) > 1e-6:
        raise ValueError("optical quaternion must have unit length")
    if frame != CAMERA_FRAME or not isinstance(status, str) or not status:
        raise ValueError("optical metadata must name the expected frame and calibration status")
    return dict(translation=translation, quaternion_xyzw=quaternion,
                frame_id=frame, calibration_status=status)


def lidar_extrinsic(scene, robot):
    """Require the generated CAD-derived laser mount rather than guessing TF."""
    try:
        entry = scene["robots"][robot]["lidar"]
        translation = tuple(float(x) for x in entry["translation"])
        quaternion = tuple(float(x) for x in entry["quaternion_xyzw"])
        rate = float(entry["update_rate_hz"])
        frame = entry["frame_id"]
        status = entry["calibration_status"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or malformed {robot} lidar metadata; regenerate scene.json") from exc
    if (len(translation) != 3 or len(quaternion) != 4 or
            not all(math.isfinite(x) for x in (*translation, *quaternion, rate)) or
            max(abs(x) for x in translation) > 1. or
            abs(sum(x*x for x in quaternion)-1.) > 1e-6 or not 0.1 <= rate <= 100.):
        raise ValueError("Invalid finite metre-scale lidar extrinsic or update rate")
    if frame != LIDAR_FRAME or not isinstance(status, str) or not status:
        raise ValueError("Lidar metadata must name laser_frame and its calibration status")
    return dict(translation=translation, quaternion_xyzw=quaternion,
                frame_id=frame, update_rate_hz=rate, calibration_status=status)


def laser_scan(message, update_rate_hz=10.):
    """Validate planar Gazebo scan data without inventing free-space returns.

    Protobuf JSON represents nonfinite ranges as strings. Preserve no-return
    +Inf, too-close -Inf and invalid NaN semantics; finite out-of-range readings
    become their corresponding infinities. Gazebo samples all rays at once, so
    time_increment=0 is deliberate (do not fabricate rotating-scan timing).
    """
    stamp_ns(message)
    count = int(message.get("count", 0))
    if not 2 <= count <= 4096 or int(message.get("vertical_count", 1)) != 1:
        raise ValueError("Expected bounded single-plane laser scan")
    ranges = message.get("ranges", [])
    intensities = message.get("intensities", [])
    if len(ranges) != count or len(intensities) not in (0, count):
        raise ValueError("Laser array lengths differ from declared count")
    angle_min, angle_max, step = (float(message.get(key, 0.)) for key in
                                  ("angle_min", "angle_max", "angle_step"))
    range_min, range_max = (float(message.get(key, 0.)) for key in ("range_min", "range_max"))
    rate = float(update_rate_hz)
    if (not all(math.isfinite(x) for x in (angle_min, angle_max, step, range_min, range_max, rate)) or
            not 0 < step <= math.pi or not 0 < angle_max-angle_min <= 2*math.pi+1e-5 or
            not math.isclose(angle_min+(count-1)*step, angle_max, abs_tol=1e-5) or
            not 0 <= range_min < range_max <= 1000. or not .1 <= rate <= 100.):
        raise ValueError("Invalid laser angular/range/timing metadata")
    frame = message.get("frame", LIDAR_FRAME)
    if frame != LIDAR_FRAME:
        raise ValueError("Unexpected laser coordinate frame")
    converted = []
    for raw in ranges:
        value = float(raw)
        converted.append(math.inf if value > range_max else -math.inf if value < range_min else value)
    intensity_values = [float(value) for value in intensities]
    if any(not math.isfinite(value) or value < 0 for value in intensity_values):
        raise ValueError("Invalid laser intensities")
    return dict(angle_min=angle_min, angle_max=angle_max, angle_increment=step,
                range_min=range_min, range_max=range_max, scan_time=1./rate,
                time_increment=0., ranges=converted, intensities=intensity_values)


def sensor_noise(image, kind, profile="clean", *, seed=2026, robot="rover_a", capture_ns=0):
    """Deterministic stress profile, NOT a calibrated OAK-D sensor model.

    clean keeps ideal rendered pixels (JPEG/1mm encoding still applies).
    noisy: RGB sigma=1.2 DN; depth sigma=(.002+.001*z^2)m, dropout
    min(.015+.003*z^2,.20), then 2mm quantization. Seeds depend on capture
    stamp and rover, not callback order or dropped packets.
    """
    if profile not in ("clean", "noisy") or kind not in ("rgb", "depth"):
        raise ValueError("unknown sensor noise profile or image kind")
    if profile == "clean":
        return image
    key = f"{int(seed)}:{robot}:{int(capture_ns)}:{kind}".encode()
    random = np.random.default_rng(int.from_bytes(hashlib.blake2s(key,digest_size=8).digest(),"little"))
    if kind == "rgb":
        noisy = image.astype(np.float32) + random.normal(0.,1.2,image.shape)
        return np.clip(np.rint(noisy),0,255).astype(np.uint8)
    valid = np.isfinite(image) & (image > 0)
    depth = np.where(valid,image,0.).astype(np.float32)
    sigma = .002 + .001*depth*depth
    noisy = depth + random.normal(size=depth.shape)*sigma
    dropout = random.random(depth.shape) < np.minimum(.015+.003*depth*depth,.20)
    noisy = np.rint(noisy/.002)*.002
    noisy[~valid | dropout | (noisy <= 0)] = 0.
    return noisy.astype(np.float32)


def stamp_parts(stamp):
    sec, nsec = int(stamp.get("sec", 0)), int(stamp.get("nsec", 0))
    if sec < 0 or sec > 2_147_483_647 or not 0 <= nsec < 1_000_000_000:
        raise ValueError("invalid simulation stamp")
    return sec, nsec


def stamp_ns(message):
    sec, nsec = stamp_parts(message.get("header", {}).get("stamp", {}))
    return sec * 1_000_000_000 + nsec


def image_array(message, depth=False):
    width, height = int(message.get("width", 0)), int(message.get("height", 0))
    if not 0 < width <= 4096 or not 0 < height <= 2160:
        raise ValueError("invalid image dimensions")
    fmt = message.get("pixel_format_type", "UNKNOWN_PIXEL_FORMAT")
    formats = {3: "RGB_INT8", 4: "RGBA_INT8", 5: "BGRA_INT8", 8: "BGR_INT8", 13: "R_FLOAT32"}
    fmt = formats.get(fmt, fmt)
    if depth:
        if fmt != "R_FLOAT32":
            raise ValueError(f"depth requires R_FLOAT32 metres, got {fmt}")
        channels, dtype = 1, np.dtype("<f4")
    else:
        if fmt not in ("RGB_INT8", "BGR_INT8", "RGBA_INT8", "BGRA_INT8"):
            raise ValueError(f"unsupported RGB pixel format {fmt}")
        channels, dtype = (4 if "A_" in fmt else 3), np.dtype("u1")
    row_bytes = width * channels * dtype.itemsize
    step = int(message.get("step", row_bytes))
    if step < row_bytes or step > row_bytes + 4096:
        raise ValueError("invalid image row stride")
    raw = base64.b64decode(message.get("data", ""), validate=True)
    if len(raw) != step * height:
        raise ValueError("image payload length does not match dimensions")
    rows = np.frombuffer(raw, np.uint8).reshape(height, step)[:, :row_bytes].copy()
    if depth:
        return rows.view(dtype).reshape(height, width)
    data = rows.reshape(height, width, channels)
    codes = {"RGB_INT8": cv2.COLOR_RGB2BGR, "RGBA_INT8": cv2.COLOR_RGBA2BGR,
             "BGRA_INT8": cv2.COLOR_BGRA2BGR}
    return cv2.cvtColor(data, codes[fmt]) if fmt in codes else data


def encode_rgb(message, profile="clean", *, seed=2026, robot="rover_a"):
    image = image_array(message)
    image = sensor_noise(image,"rgb",profile,seed=seed,robot=robot,capture_ns=stamp_ns(message))
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return "bgr8; jpeg compressed bgr8", encoded.tobytes()


def encode_depth(message, profile="clean", *, seed=2026, robot="rover_a"):
    depth = image_array(message, depth=True)
    depth = sensor_noise(depth,"depth",profile,seed=seed,robot=robot,capture_ns=stamp_ns(message))
    valid = np.isfinite(depth) & (depth > 0) & (depth < 65.535)
    millimetres = np.zeros(depth.shape, np.uint16)
    millimetres[valid] = np.clip(np.rint(depth[valid] * 1000.), 1, 65535).astype(np.uint16)
    ok, encoded = cv2.imencode(".png", millimetres, [cv2.IMWRITE_PNG_COMPRESSION, 2])
    if not ok:
        raise ValueError("PNG depth encoding failed")
    # compressed_depth_image_transport ConfigHeader: int format + two float
    # coefficients. For original 16UC1 depths the coefficients are unused.
    return "16UC1; compressedDepth png", struct.pack("<iff", 0, 0., 0.) + encoded.tobytes()


def camera_calibration(message):
    width, height = int(message.get("width", 0)), int(message.get("height", 0))
    k = list(map(float, message.get("intrinsics", {}).get("k", [])))
    p = list(map(float, message.get("projection", {}).get("p", [])))
    if not 0 < width <= 4096 or not 0 < height <= 2160 or len(k) != 9:
        raise ValueError("missing camera calibration")
    if not all(math.isfinite(x) for x in k) or k[0] <= 0 or k[4] <= 0 or k[8] != 1:
        raise ValueError("invalid intrinsics")
    if not p:
        p = [k[0],k[1],k[2],0.,k[3],k[4],k[5],0.,k[6],k[7],k[8],0.]
    if len(p) != 12 or not all(math.isfinite(x) for x in p):
        raise ValueError("invalid projection")
    d = list(map(float, message.get("distortion", {}).get("k", [0.]*5)))
    # The generated room explicitly uses an undistorted RGB-D pinhole camera.
    if any(x != 0 for x in d):
        raise ValueError("simulation bridge expects undistorted RGB-D")
    return dict(width=width, height=height, k=k, p=p, d=[0.]*5,
                r=[1.,0.,0.,0.,1.,0.,0.,0.,1.])


def safe_velocity(forward, yaw):
    forward, yaw = float(forward), float(yaw)
    if not math.isfinite(forward) or not math.isfinite(yaw):
        return 0., 0.
    return max(-.35, min(.35, forward)), max(-.7, min(.7, yaw))
