"""Validated, dependency-light configuration for the experimental fleet server."""
from copy import deepcopy
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = {
    "central_domain": 42,
    "dashboard": {"host": "127.0.0.1", "port": 8766},
    "input": {
        "max_hz": 2.0, "max_age_s": 2.0, "sync_slop_s": 0.08,
        "tf_timeout_s": 0.08, "qos_depth": 5, "max_pixels": 2073600,
        "min_depth_m": 0.2, "max_depth_m": 8.0,
    },
    "perception": {"parameters": {
        "esam_variant": "vitt", "clip_model_name": "ViT-L-14",
        "clip_pretrained": "openai", "dinov2_model_name": "dinov2_vits14",
        "stationary_pose_lock": False,
    }},
    "pose_guard": {
        "enabled": True, "max_translation_error_m": 0.15,
        "max_rotation_error_deg": 5.0, "max_speed_m_s": 2.0,
        "max_yaw_rate_deg_s": 150.0,
    },
    "backend": {"shadow_mode": True},
    "robots": [
        {"id": "rover_a", "domain": 0},
        {"id": "rover_b", "domain": 1},
    ],
}
ROBOT_DEFAULTS = {
    "local_frame": "odom",
    "rgb_topic": "/oak/rgb/image_rect/compressed",
    "depth_topic": "/oak/stereo/image_raw/compressedDepth",
    "camera_info_topic": "/oak/rgb/camera_info",
    "depth_camera_info_topic": "/oak/stereo/camera_info",
    "rgb_compressed": True, "depth_compressed": True,
    "depth_registration": "auto", "depth_scale": 0.001,
    "parameters": {},
}


def merge(base, update):
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _number(value, name, lower, upper):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")


def load_config(path=None):
    custom = {} if path is None else json.loads(Path(path).read_text())
    return validate_config(custom)


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("configuration must be a JSON object")
    unknown = set(config)-set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
    config = merge(DEFAULT_CONFIG, config)
    for section in ('dashboard', 'input', 'perception', 'pose_guard', 'backend'):
        if not isinstance(config[section], dict):
            raise ValueError(f'{section} must be an object')
        if section != 'backend' and set(config[section])-set(DEFAULT_CONFIG[section]):
            raise ValueError(f'unknown fields in {section}')
    if not isinstance(config['perception']['parameters'], dict):
        raise ValueError('perception.parameters must be an object')
    _number(config["central_domain"], "central_domain", 0, 101)
    if not isinstance(config["central_domain"], int):
        raise ValueError("central_domain must be an integer")
    dashboard = config["dashboard"]
    if dashboard["host"] not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("dashboard must bind to loopback; use an authenticated tunnel for remote access")
    _number(dashboard["port"], "dashboard.port", 1024, 65535)
    if not isinstance(dashboard["port"], int):
        raise ValueError("dashboard.port must be an integer")
    limits = {"max_hz": (.1, 30), "max_age_s": (.1, 30),
              "sync_slop_s": (.001, .5), "tf_timeout_s": (0, .5),
              "qos_depth": (1, 30), "max_pixels": (1024, 8294400),
              "min_depth_m": (.01, 5), "max_depth_m": (.1, 100)}
    for name, bounds in limits.items():
        _number(config["input"][name], f"input.{name}", *bounds)
    for name in ('qos_depth', 'max_pixels'):
        if not isinstance(config['input'][name], int):
            raise ValueError(f'input.{name} must be an integer')
    if config["input"]["min_depth_m"] >= config["input"]["max_depth_m"]:
        raise ValueError("min_depth_m must be less than max_depth_m")
    if not isinstance(config["robots"], list) or not 1 <= len(config["robots"]) <= 8:
        raise ValueError("configure between one and eight robots")
    ids, domains = set(), set()
    for i, source in enumerate(config["robots"]):
        if not isinstance(source, dict) or set(source)-set(ROBOT_DEFAULTS)-{'id', 'domain'}:
            raise ValueError('robot configuration must be an object with known fields')
        robot = merge(ROBOT_DEFAULTS, source)
        identifier, domain = robot.get("id"), robot.get("domain")
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", identifier):
            raise ValueError("robot IDs must be lowercase ROS-safe names, not domain numbers")
        _number(domain, f"{identifier}.domain", 0, 101)
        if not isinstance(domain, int):
            raise ValueError("robot domain must be an integer")
        if identifier in ids or domain in domains or domain == config["central_domain"]:
            raise ValueError("robot IDs/domains must be unique and separate from central_domain")
        ids.add(identifier); domains.add(domain)
        if robot["depth_registration"] not in ("auto", "registered", "project"):
            raise ValueError("depth_registration must be auto, registered, or project")
        for field in ("local_frame", "rgb_topic", "depth_topic", "camera_info_topic", "depth_camera_info_topic"):
            if not isinstance(robot[field], str) or not robot[field] or len(robot[field]) > 240:
                raise ValueError(f"{identifier}.{field} must be a nonempty ROS name")
        for field in ("rgb_compressed", "depth_compressed"):
            if not isinstance(robot[field], bool):
                raise ValueError(f"{identifier}.{field} must be boolean")
        _number(robot["depth_scale"], "depth_scale", .000001, 1.)
        if not isinstance(robot["parameters"], dict):
            raise ValueError("robot parameters must be an object")
        config["robots"][i] = robot
    guard = config['pose_guard']
    if not isinstance(guard['enabled'], bool):
        raise ValueError('pose_guard.enabled must be boolean')
    for name, bounds in {'max_translation_error_m': (.01, 10),
                         'max_rotation_error_deg': (.1, 90),
                         'max_speed_m_s': (.01, 20),
                         'max_yaw_rate_deg_s': (1, 720)}.items():
        _number(guard[name], f'pose_guard.{name}', *bounds)
    if config["perception"]["parameters"].get("stationary_pose_lock", False) or any(
            r["parameters"].get("stationary_pose_lock", False) for r in config["robots"]):
        raise ValueError("stationary pose lock is not supported in collaborative mode")
    return config


def owned_run_directory(path=None, resume=False):
    """Keep recordings under the experimental directory, never touch v2 maps."""
    from datetime import datetime, timezone
    import uuid
    parent = ROOT / ".runtime" / "collaboration"
    folder = (Path(path).expanduser() if path else parent / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6])).resolve()
    if not folder.is_relative_to(parent.resolve()) or folder == parent.resolve():
        raise ValueError(f"run directory must be a child of {parent}")
    if folder.exists() and not resume:
        raise ValueError("run directory already exists; use --resume to preserve and resume it")
    if resume and not folder.is_dir():
        raise ValueError("--resume requires an existing run directory")
    return folder
