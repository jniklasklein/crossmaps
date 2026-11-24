#!/usr/bin/env python3
"""Generate a self-contained, deterministic Gazebo RGB-D alignment laboratory.

No network access or Fuel service is required at runtime. Robot geometry and
CAD inertias are imported from the user's Waveshare description. The indoor
room uses locally bundled CC0 photographic materials and detailed geometry.
Run again after moving the generated directory:
mesh resource URIs are absolute for reproducible Gazebo resource resolution.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

WIDTH, HEIGHT = 640, 360
VENDOR = Path(__file__).parent / "vendor/waveshare_ugv"
CALIBRATION_PATH = Path(__file__).parent / "rover_calibration.json"
CAMERA_XYZ = (.06577008779165411, 6.508560202099369e-10, .10195299999999997)
OPTICAL_QUATERNION_XYZW = (-.5, .5, -.5, .5)
SPAWNS = {"rover_a": (0., -.30, .002), "rover_b": (0., .30, .002)}
LIDAR_FRAME = "laser_frame"
LIDAR_HZ, LIDAR_SAMPLES = 10., 720
LIDAR_RANGE = (.12, 12.)
# Sensing plane within the vendored 35mm-high lidar housing. This is an
# engineering approximation, not a calibration of the physical laser module.
LIDAR_PLANE_Z = .028


def node(parent, tag, value=None, **attrs):
    result = ET.SubElement(parent, tag, attrs)
    if value is not None:
        result.text = str(value)
    return result


def values(seq):
    return " ".join(str(v) for v in seq)


def pose(parent, xyz, rpy=(0, 0, 0)):
    return node(parent, "pose", values((*xyz, *rpy)))


def material(parent, color):
    mat = node(parent, "material")
    node(mat, "ambient", values(color))
    node(mat, "diffuse", values(color))
    node(mat, "specular", "0.02 0.02 0.02 1")


def geometry(parent, shape, dimensions):
    geom = node(parent, "geometry")
    item = node(geom, shape)
    if shape == "box":
        node(item, "size", values(dimensions))
    elif shape == "cylinder":
        node(item, "radius", dimensions[0])
        node(item, "length", dimensions[1])
    elif shape == "sphere":
        node(item, "radius", dimensions[0])
    return geom


def solid(link, name, xyz, size, color, shape="box", rpy=(0, 0, 0), collide=True):
    visual = node(link, "visual", name=name)
    pose(visual, xyz, rpy)
    geometry(visual, shape, size)
    material(visual, color)
    if collide:
        collision = node(link, "collision", name=name)
        pose(collision, xyz, rpy)
        geometry(collision, shape, size)
    return visual


def inertia(link, mass, diagonal):
    inertial = node(link, "inertial")
    node(inertial, "mass", mass)
    tensor = node(inertial, "inertia")
    for tag, value in zip(("ixx", "iyy", "izz"), diagonal):
        node(tensor, tag, value)
    for tag in ("ixy", "ixz", "iyz"):
        node(tensor, tag, 0)


def plugin(parent, short, class_name):
    return node(parent, "plugin", filename="gz-sim-" + short + "-system",
                name="gz::sim::systems::" + class_name)


def static_model(world, name, xyz=(0, 0, 0)):
    model = node(world, "model", name=name)
    node(model, "static", "true")
    pose(model, xyz)
    return node(model, "link", name="link")


def load_materials():
    """Resolve bundled, licensed texture maps without runtime downloads."""
    folder = Path(__file__).parent / "material_assets"
    manifest = json.loads((folder / "manifest.json").read_text())
    result = {}
    for name, item in manifest["materials"].items():
        resolved = {"scale_m": item["scale_m"]}
        for kind in ("albedo", "normal", "roughness"):
            path = (folder / item[kind]["path"]).resolve()
            if not path.is_relative_to(folder.resolve()) or not path.is_file():
                raise ValueError(f"Missing or out-of-scope material map: {name}/{kind}")
            resolved[kind] = path
        result[name] = resolved
    return result


def _origin(element):
    return ([float(v) for v in element.get("xyz", "0 0 0").split()],
            [float(v) for v in element.get("rpy", "0 0 0").split()])


def _optical_to_sensor_rpy(q):
    """Convert optical-to-base quaternion to Gazebo's +X-forward sensor pose."""
    x, y, z, w = q
    a, b, c, d = (.5, -.5, .5, .5)  # inverse canonical optical rotation
    x, y, z, w = (w*a + x*d + y*c - z*b, w*b - x*c + y*d + z*a,
                  w*c + x*b - y*a + z*d, w*d - x*a - y*b - z*c)
    return (math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)),
            math.asin(max(-1., min(1., 2 * (w*y - z*x)))),
            math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)))


def validate_calibration(calibration):
    for name in SPAWNS:
        source = calibration["robots"][name]
        K = source["K_640x360"]
        if len(K) != 3 or any(len(row) != 3 for row in K):
            raise ValueError(f"{name}: expected 3x3 K_640x360")
        if not all(math.isfinite(float(v)) for row in K for v in row):
            raise ValueError(f"{name}: nonfinite intrinsics")
        if K[0][0] <= 0 or K[1][1] <= 0 or K[2] != [0, 0, 1]:
            raise ValueError(f"{name}: invalid pinhole intrinsics")
        camera = source["camera_optical"]
        xyz, quaternion = camera["translation"], camera["quaternion_xyzw"]
        if len(xyz) != 3 or not all(math.isfinite(v) and abs(v) < 5 for v in xyz):
            raise ValueError(f"{name}: invalid optical translation")
        if len(quaternion) != 4 or not all(math.isfinite(v) for v in quaternion) or abs(sum(v*v for v in quaternion)-1) > 1e-5:
            raise ValueError(f"{name}: optical rotation must be a unit xyzw quaternion")
        if camera["frame_id"] != "oak_rgb_camera_optical_frame" or not camera["calibration_status"]:
            raise ValueError(f"{name}: expected optical frame and calibration provenance")


def _import_link(model, source_link, parent_joint):
    """Preserve source CAD frames/inertials/visuals; simplify contact geometry."""
    name = source_link.get("name")
    link = node(model, "link", name=name)
    xyz, rpy = _origin(parent_joint.find("origin"))
    parent = parent_joint.find("parent").get("link")
    p = pose(link, xyz, rpy)
    if parent != "base_footprint":
        p.set("relative_to", parent)
    source_inertial = source_link.find("inertial")
    inertial = node(link, "inertial")
    inertial_xyz, inertial_rpy = _origin(source_inertial.find("origin"))
    pose(inertial, inertial_xyz, inertial_rpy)
    node(inertial, "mass", source_inertial.find("mass").get("value"))
    tensor = node(inertial, "inertia")
    for tag, value in source_inertial.find("inertia").attrib.items():
        node(tensor, tag, value)
    for kind in ("visual", "collision"):
        for i, source in enumerate(source_link.findall(kind)):
            # Base mesh includes the two passive center-wheel visuals. Its
            # collision excludes those: they receive free revolute contacts.
            item = node(link, kind, name=f"cad_{kind}_{i}")
            source_xyz, source_rpy = _origin(source.find("origin"))
            if kind == "collision" and name == "base_link":
                pose(item, (0, 0, -.008305))
                geometry(item, "box", (.245198, .23042, .09661))
            elif kind == "collision" and "wheel" in name:
                pose(item, (0, 0, 0))
                geometry(item, "cylinder", (.04, .0425))
                ode = node(node(node(item, "surface"), "friction"), "ode")
                node(ode, "mu", .8); node(ode, "mu2", .15)
                node(ode, "fdir1", "1 0 0")
            else:
                pose(item, source_xyz, source_rpy)
                mesh_name = Path(source.find("geometry/mesh").get("filename")).name
                geom = node(item, "geometry")
                node(node(geom, "mesh"), "uri", (VENDOR / "meshes/ugv_rover" / mesh_name).resolve().as_uri())
            if kind == "visual":
                color = source.find("material/color").get("rgba")
                material(item, [float(v) for v in color.split()])
    return link


def rover(world, name, xyz, hz, calibration, width=WIDTH, height=HEIGHT):
    model = node(world, "model", name=name)
    pose(model, xyz)
    source = ET.parse(VENDOR / "urdf/ugv_rover.urdf").getroot()
    joints = {j.find("child").get("link"): j for j in source.findall("joint")}
    for source_link in source.findall("link"):
        if source_link.find("inertial") is None:
            continue
        link_name = source_link.get("name")
        source_joint = joints[link_name]
        _import_link(model, source_link, source_joint)
        if link_name == "base_link":
            continue
        # Pan/tilt fixed at zero: source servos have zero effort/velocity and no
        # motor model. Do not invent uncontrolled hanging camera articulation.
        joint_type = "revolute" if "wheel" in link_name else "fixed"
        joint = node(model, "joint", name=source_joint.get("name"), type=joint_type)
        node(joint, "parent", source_joint.find("parent").get("link"))
        node(joint, "child", link_name)
        if joint_type == "revolute":
            axis = node(joint, "axis")
            node(axis, "xyz", source_joint.find("axis").get("xyz"))
            limit = node(axis, "limit")
            node(limit, "effort", 2.); node(limit, "velocity", 20.)
    body = model.find("link[@name='base_link']")
    base_xyz, _ = _origin(joints["base_link"].find("origin"))
    footprint = node(model, "frame", name="base_footprint", attached_to="base_link")
    node(footprint, "pose", values((*[-v for v in base_xyz], 0, 0, 0)), relative_to="base_link")
    for side, y in (("left", .08726), ("right", -.08726)):
        wheel_name = side + "_passive_middle"
        wheel = node(model, "link", name=wheel_name)
        node(wheel, "pose", f"0 {y} -.041212 1.5708 0 0", relative_to="base_link")
        # The CAD chassis mass already includes both center wheels. These tiny
        # numerical inertias add 2g total, avoiding double-counting tire masses.
        inertia(wheel, .001, (.0000005, .0000005, .0000008))
        collision = node(wheel, "collision", name="passive_tire")
        geometry(collision, "cylinder", (.04, .0425))
        ode = node(node(node(collision, "surface"), "friction"), "ode")
        node(ode, "mu", .8); node(ode, "mu2", .15)
        node(ode, "fdir1", "1 0 0")
        joint = node(model, "joint", name=wheel_name + "_joint", type="revolute")
        node(joint, "parent", "base_link"); node(joint, "child", wheel_name)
        node(node(joint, "axis"), "xyz", "0 0 -1")
    sensor = node(body, "sensor", name="rgbd", type="rgbd_camera")
    camera_xyz = calibration["camera_optical"]["translation"]
    pose(sensor, [a - b for a, b in zip(camera_xyz, base_xyz)],
         _optical_to_sensor_rpy(calibration["camera_optical"]["quaternion_xyzw"]))
    node(sensor, "always_on", "true")
    node(sensor, "update_rate", hz)
    node(sensor, "topic", f"/{name}/rgbd")
    node(sensor, "gz_frame_id", "oak_rgb_camera_optical_frame")
    camera = node(sensor, "camera")
    K = [[float(value) * (width / WIDTH if row == 0 else height / HEIGHT if row == 1 else 1.)
          for value in values] for row, values in enumerate(calibration["K_640x360"])]
    node(camera, "horizontal_fov", 2 * math.atan(width / (2 * K[0][0])))
    image = node(camera, "image")
    node(image, "anti_aliasing", 4)
    node(image, "width", width); node(image, "height", height)
    node(image, "format", "R8G8B8")
    clip = node(camera, "clip")
    node(clip, "near", .1); node(clip, "far", 8.)
    lens = node(camera, "lens")
    node(lens, "type", "gnomonical")
    node(lens, "scale_to_hfov", "false")
    intrinsics = node(lens, "intrinsics")
    for key, value in {"fx": K[0][0], "fy": K[1][1], "cx": K[0][2], "cy": K[1][2], "s": 0}.items():
        node(intrinsics, key, value)
    projection = node(lens, "projection")
    for key, value in {"p_fx": K[0][0], "p_fy": K[1][1], "p_cx": K[0][2], "p_cy": K[1][2], "tx": 0, "ty": 0}.items():
        node(projection, key, value)
    lidar_link = model.find("link[@name='base_lidar_link']")
    mount_xyz, mount_rpy = _origin(joints["base_lidar_link"].find("origin"))
    if mount_rpy[:2] != [0., 0.]:
        raise ValueError("Lidar mount roll/pitch changed; update the laser extrinsic composition")
    lidar = node(lidar_link, "sensor", name="lidar", type="gpu_lidar")
    pose(lidar, (0., 0., LIDAR_PLANE_Z), (0., 0., -mount_rpy[2]))
    node(lidar, "topic", f"/{name}/scan")
    node(lidar, "gz_frame_id", LIDAR_FRAME)
    node(lidar, "update_rate", LIDAR_HZ)
    node(lidar, "always_on", "true")
    node(lidar, "visualize", "false")
    ray = node(lidar, "lidar")
    scan = node(ray, "scan")
    horizontal = node(scan, "horizontal")
    node(horizontal, "samples", LIDAR_SAMPLES)
    node(horizontal, "resolution", 1)
    node(horizontal, "min_angle", -math.pi)
    # No duplicated endpoint ray: 720 bearings at exact half-degree steps.
    node(horizontal, "max_angle", math.pi - 2 * math.pi / LIDAR_SAMPLES)
    vertical = node(scan, "vertical")
    node(vertical, "samples", 1); node(vertical, "resolution", 1)
    node(vertical, "min_angle", 0); node(vertical, "max_angle", 0)
    ranges = node(ray, "range")
    node(ranges, "min", LIDAR_RANGE[0]); node(ranges, "max", LIDAR_RANGE[1])
    node(ranges, "resolution", .01)
    noise = node(ray, "noise", type="gaussian")
    node(noise, "mean", 0); node(noise, "stddev", .01)
    drive = plugin(model, "diff-drive", "DiffDrive")
    for side in ("left", "right"):
        for position in ("up", "down"):
            node(drive, side + "_joint", f"{side}_{position}_wheel_link_joint")
    for tag, value in {"wheel_separation": .17452, "wheel_radius": .04,
                       "topic": f"/{name}/cmd_vel", "odom_topic": f"/{name}/odom",
                       "tf_topic": f"/{name}/tf", "frame_id": "odom",
                       "child_frame_id": "base_footprint", "odom_publish_frequency": 30,
                       "max_linear_velocity": .4, "min_linear_velocity": -.4,
                       "max_angular_velocity": .8, "min_angular_velocity": -.8,
                       "max_linear_acceleration": .4, "min_linear_acceleration": -.4,
                       "max_angular_acceleration": .8, "min_angular_acceleration": -.8}.items():
        node(drive, tag, value)
    # Ground truth is intentionally separate from wheel odometry and must never
    # be bridged into the matcher as an inter-rover alignment shortcut.
    truth = plugin(model, "pose-publisher", "PosePublisher")
    node(truth, "publish_link_pose", "false")
    node(truth, "publish_sensor_pose", "false")
    node(truth, "publish_collision_pose", "false")
    node(truth, "publish_visual_pose", "false")
    node(truth, "publish_nested_model_pose", "false")
    node(truth, "publish_model_pose", "true")
    node(truth, "use_pose_vector_msg", "true")
    node(truth, "update_frequency", 10)
    return dict(frame_id=LIDAR_FRAME,
                translation=[base_xyz[0] + mount_xyz[0], base_xyz[1] + mount_xyz[1],
                             base_xyz[2] + mount_xyz[2] + LIDAR_PLANE_Z],
                quaternion_xyzw=[0., 0., 0., 1.], update_rate_hz=LIDAR_HZ,
                samples=LIDAR_SAMPLES, range_min=LIDAR_RANGE[0], range_max=LIDAR_RANGE[1],
                gaussian_stddev_m=.01,
                calibration_status="CAD mount; assumed sensing plane 28mm above mount; not hardware calibrated",
                limitations="Simultaneous GPU rays; no spinning-scan distortion, reflectivity or calibrated dropout")


def build(output: Path, render_engine="ogre2", hz=3., calibration_path=None, camera_width=WIDTH):
    if camera_width not in (640, 960, 1280):
        raise ValueError("camera_width must be 640, 960 or 1280 (16:9)")
    width, height = camera_width, camera_width * 9 // 16
    calibration = json.loads(Path(calibration_path or CALIBRATION_PATH).read_text())
    validate_calibration(calibration)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    sdf = ET.Element("sdf", {"version": "1.9"})
    world = node(sdf, "world", name="crossmaps_lab")
    physics = node(world, "physics", name="default", type="ignored")
    node(physics, "max_step_size", .002)
    node(physics, "real_time_factor", 1.)
    plugin(world, "physics", "Physics")
    plugin(world, "user-commands", "UserCommands")
    plugin(world, "scene-broadcaster", "SceneBroadcaster")
    node(plugin(world, "sensors", "Sensors"), "render_engine", render_engine)
    from scene_geometry import build_room
    room_metadata = build_room(world, assets, load_materials())
    robot_metadata = {}
    for name, xyz in SPAWNS.items():
        laser = rover(world, name, xyz, hz, calibration["robots"][name], width, height)
        robot_metadata[name] = dict(calibration["robots"][name], lidar=laser)
    ET.indent(sdf, space="  ")
    world_path = output / "crossmaps_lab.sdf"
    ET.ElementTree(sdf).write(world_path, encoding="utf-8", xml_declaration=True)
    metadata = {"world": "crossmaps_lab", "render_engine": render_engine, "camera_hz": hz,
                "width": width, "height": height,
                "note": "Per-rover intrinsics and extrinsics from recorded ROS input; not independently mechanically measured. Use published camera_info for reconstruction.",
                "spawn_positions": SPAWNS, "odometry": "independent wheel-integrated origins",
                "environment": room_metadata,
                "robots": robot_metadata,
                "model": "Waveshare local CAD, four driven wheels plus two passive center contacts",
                "model_limitations": "CAD inertias unvalidated; passive contacts use 1g numerical mass each; friction/actuation not calibrated; pan/tilt locked at zero; ideal registered RGB-D without stereo failure simulation"}
    (output / "scene.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return world_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "generated")
    parser.add_argument("--render-engine", choices=("ogre", "ogre2"), default="ogre2")
    parser.add_argument("--camera-hz", type=float, default=3.)
    parser.add_argument("--camera-width", type=int, choices=(640, 960, 1280), default=960,
                        help="16:9 render size; intrinsics scale to preserve the real camera field of view")
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_PATH)
    args = parser.parse_args()
    if not .1 <= args.camera_hz <= 15:
        parser.error("--camera-hz must be between .1 and 15")
    print(build(args.output, args.render_engine, args.camera_hz, args.calibration, args.camera_width))


if __name__ == "__main__":
    main()
