"""Dependency-free structural checks; runtime rendering is a separate smoke test."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from generate_world import build, CAMERA_XYZ, OPTICAL_QUATERNION_XYZW, VENDOR, _optical_to_sensor_rpy, load_materials


class WorldTests(unittest.TestCase):
    def test_self_contained_dual_rover(self):
        with tempfile.TemporaryDirectory() as folder:
            path = build(Path(folder))
            world = ET.parse(path).getroot().find("world")
            self.assertEqual(world.attrib["name"], "crossmaps_lab")
            for name in ("rover_a", "rover_b"):
                model = world.find(f"model[@name='{name}']")
                self.assertIsNotNone(model)
                sensor = model.find("link/sensor")
                self.assertEqual(sensor.attrib["type"], "rgbd_camera")
                self.assertEqual(sensor.findtext("topic"), f"/{name}/rgbd")
                self.assertEqual(sensor.findtext("camera/image/width"), "640")
                self.assertEqual(sensor.findtext("camera/image/height"), "360")
                drive = model.find("plugin[@name='gz::sim::systems::DiffDrive']")
                self.assertEqual(len(drive.findall("left_joint")), 2)
                self.assertEqual(len(drive.findall("right_joint")), 2)
                self.assertEqual(float(drive.findtext("wheel_radius")), .04)
                self.assertEqual(float(drive.findtext("wheel_separation")), .17452)
                self.assertIsNotNone(model.find("joint[@name='left_passive_middle_joint']"))
                self.assertIsNotNone(model.find("joint[@name='right_passive_middle_joint']"))
                joints = {j.attrib["name"] for j in model.findall("joint")}
                for joint in drive.findall("left_joint") + drive.findall("right_joint"):
                    self.assertIn(joint.text, joints)
                self.assertEqual(drive.findtext("topic"), f"/{name}/cmd_vel")
                self.assertEqual(drive.findtext("odom_topic"), f"/{name}/odom")
                self.assertEqual(drive.findtext("frame_id"), "odom")
                self.assertEqual(drive.findtext("child_frame_id"), "base_footprint")
            for uri in world.findall(".//mesh/uri"):
                self.assertTrue(uri.text.startswith("file://"))
                self.assertTrue(Path(uri.text.removeprefix("file://")).exists())
            self.assertEqual(world.findall(".//include"), [])
            for png in (Path(folder) / "assets").glob("*.png"):
                self.assertTrue(png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            metadata = json.loads((Path(folder) / "scene.json").read_text())
            self.assertEqual(metadata["robots"]["rover_a"]["camera_optical"]["translation"], list(CAMERA_XYZ))
            self.assertAlmostEqual(sum(q*q for q in OPTICAL_QUATERNION_XYZW), 1.)

    def test_camera_extrinsic_and_independent_origins(self):
        with tempfile.TemporaryDirectory() as folder:
            world = ET.parse(build(Path(folder))).getroot().find("world")
            positions = []
            for name in ("rover_a", "rover_b"):
                model = world.find(f"model[@name='{name}']")
                positions.append([float(x) for x in model.findtext("pose").split()])
                body = model.find("link[@name='base_link']")
                body_pose = [float(x) for x in body.findtext("pose").split()]
                sensor_pose = [float(x) for x in body.findtext("sensor/pose").split()]
                for index, value in enumerate(CAMERA_XYZ):
                    self.assertAlmostEqual(body_pose[index] + sensor_pose[index], value)
                drive = model.find("plugin[@name='gz::sim::systems::DiffDrive']")
                # No absolute/world odometry reference or shared transform.
                self.assertEqual(drive.findtext("frame_id"), "odom")
                self.assertIsNone(drive.find("initial_pose"))
            self.assertAlmostEqual(positions[1][1] - positions[0][1], .6)
            # REP-103 optical +Z forward -> body +X, optical +X -> body -Y.
            x, y, z, w = OPTICAL_QUATERNION_XYZW
            rotation = ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                        (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                        (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))
            self.assertEqual(tuple(row[2] for row in rotation), (1., 0., 0.))
            self.assertEqual(tuple(row[0] for row in rotation), (0., -1., 0.))
            self.assertEqual(_optical_to_sensor_rpy(OPTICAL_QUATERNION_XYZW), (0., 0., 0.))

    def test_preserved_cad_provenance_and_intrinsics(self):
        provenance = json.loads((VENDOR / "provenance.json").read_text())
        for relative, digest in provenance["sha256"].items():
            self.assertEqual(hashlib.sha256((VENDOR / relative).read_bytes()).hexdigest(), digest)
        with tempfile.TemporaryDirectory() as folder:
            world = ET.parse(build(Path(folder))).getroot().find("world")
            self.assertAlmostEqual(float(world.findtext("model[@name='rover_a']/link/sensor/camera/lens/intrinsics/fx")), 452.87445068359375)
            self.assertAlmostEqual(float(world.findtext("model[@name='rover_b']/link/sensor/camera/lens/intrinsics/fx")), 457.4581298828125)
            for name in ("rover_a", "rover_b"):
                base = world.find(f"model[@name='{name}']/link[@name='base_link']")
                self.assertAlmostEqual(float(base.findtext("inertial/mass")), 2.78375064359473)
                self.assertIn("base_link.stl", base.findtext("visual/geometry/mesh/uri"))

    def test_materials_are_local_licensed_and_integrity_checked(self):
        folder = Path(__file__).parent / "material_assets"
        manifest = json.loads((folder / "manifest.json").read_text())
        resolved = load_materials()
        self.assertTrue({"wood", "floor", "plaster", "fabric"} <= resolved.keys())
        for name, item in manifest["materials"].items():
            self.assertEqual(item["license"], "CC0-1.0")
            self.assertTrue(item["source_url"].startswith("https://polyhaven.com/a/"))
            for kind in ("albedo", "normal", "roughness"):
                data = resolved[name][kind].read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), item[kind]["sha256"])
                self.assertEqual(len(data), item[kind]["bytes"])

    def test_no_posters_or_printed_test_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            world = ET.parse(build(Path(folder))).getroot().find("world")
            names = [element.get("name", "").lower() for element in world.iter()]
            for forbidden in ("wall_art", "other_art", "printed_front", "field_kit_carton", "storage_carton"):
                self.assertNotIn(forbidden, names)
            self.assertFalse(any("poster" in name for name in names))
            self.assertGreater(len(world.findall(".//material/pbr/metal/albedo_map")), 5)
            self.assertEqual(world.findtext("scene/shadows"), "true")
            self.assertTrue(any(light.findtext("cast_shadows") == "true" for light in world.findall("light")))

    def test_higher_resolution_preserves_calibrated_field_of_view(self):
        with tempfile.TemporaryDirectory() as folder:
            world = ET.parse(build(Path(folder), camera_width=960)).getroot().find("world")
            camera = world.find("model[@name='rover_a']/link/sensor/camera")
            self.assertEqual(camera.findtext("image/width"), "960")
            self.assertEqual(camera.findtext("image/height"), "540")
            self.assertAlmostEqual(float(camera.findtext("lens/intrinsics/fx")), 452.87445068359375 * 1.5)
            self.assertAlmostEqual(float(camera.findtext("lens/intrinsics/cx")), 323.2257385253906 * 1.5)
            self.assertEqual(camera.findtext("image/anti_aliasing"), "4")
            info = json.loads((Path(folder) / "scene.json").read_text())
            self.assertEqual((info["width"], info["height"]), (960, 540))
        with self.assertRaises(ValueError):
            build(Path("/not-written"), camera_width=641)

    def test_engine_and_camera_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            world = ET.parse(build(Path(folder), "ogre2", 1.5)).getroot().find("world")
            self.assertEqual(world.findtext("plugin[@name='gz::sim::systems::Sensors']/render_engine"), "ogre2")
            self.assertEqual(world.findtext("model[@name='rover_a']/link/sensor/update_rate"), "1.5")


if __name__ == "__main__":
    unittest.main()
