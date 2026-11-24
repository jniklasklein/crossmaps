"""Offline validation of the vendored, data-only plant model and provenance."""
import hashlib
import json
import math
from pathlib import Path
import struct
import unittest
from urllib.parse import unquote, urlsplit


ASSETS = Path(__file__).resolve().parent / "object_assets"
MODEL = ASSETS / "potted_plant_02/potted_plant_02_1k.gltf"


def _uris(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "uri":
                yield child
            else:
                yield from _uris(child)
    elif isinstance(value, list):
        for child in value:
            yield from _uris(child)


class ObjectAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ASSETS / "manifest.json").read_text())
        cls.model = json.loads(MODEL.read_text())

    def test_cc0_provenance_size_and_every_hash(self):
        self.assertEqual(self.manifest["license"], "CC0-1.0")
        self.assertEqual(self.manifest["source"], "https://polyhaven.com/a/potted_plant_02")
        self.assertEqual(self.manifest["author"], "Rico Cilliers")
        recorded, total = set(), 0
        for entry in self.manifest["files"]:
            path = (ASSETS / entry["path"]).resolve()
            self.assertTrue(path.is_relative_to(ASSETS))
            self.assertEqual(urlsplit(entry["source_url"]).netloc, "dl.polyhaven.org")
            payload = path.read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), entry["sha256"], str(path))
            total += len(payload)
            recorded.add(path)
        actual = {p.resolve() for p in (ASSETS / "potted_plant_02").rglob("*") if p.is_file()}
        self.assertEqual(actual, recorded)
        self.assertLess(total, 20_000_000)

    def test_every_gltf_resource_is_local_and_contained(self):
        self.assertEqual(self.model["asset"]["version"], "2.0")
        self.assertFalse(self.model.get("extensionsRequired"))
        resources = list(_uris(self.model))
        self.assertEqual(len(resources), 7)
        for uri in resources:
            parsed = urlsplit(uri)
            self.assertFalse(parsed.scheme or parsed.netloc or parsed.query or parsed.fragment)
            decoded = unquote(parsed.path)
            self.assertNotIn("\\", decoded)
            self.assertFalse(Path(decoded).is_absolute())
            path = (MODEL.parent / decoded).resolve()
            self.assertTrue(path.is_relative_to(MODEL.parent))
            self.assertTrue(path.is_file(), uri)
        for buffer in self.model["buffers"]:
            self.assertEqual((MODEL.parent / buffer["uri"]).stat().st_size, buffer["byteLength"])

    def test_original_geometry_dimensions_and_upright_conversion(self):
        buffers = [(MODEL.parent / entry["uri"]).read_bytes() for entry in self.model["buffers"]]
        minimum, maximum, triangles = [math.inf] * 3, [-math.inf] * 3, 0
        for node in self.model["nodes"]:
            # The original model has only translations, not hidden scale/rotation.
            self.assertFalse({"matrix", "rotation", "scale"} & node.keys())
            translation = node.get("translation", (0, 0, 0))
            for primitive in self.model["meshes"][node["mesh"]]["primitives"]:
                self.assertEqual(primitive.get("mode", 4), 4)  # TRIANGLES
                triangles += self.model["accessors"][primitive["indices"]]["count"] // 3
                accessor = self.model["accessors"][primitive["attributes"]["POSITION"]]
                self.assertEqual((accessor["componentType"], accessor["type"]), (5126, "VEC3"))
                view = self.model["bufferViews"][accessor["bufferView"]]
                payload = buffers[view["buffer"]]
                offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
                stride = view.get("byteStride", 12)
                self.assertLessEqual(offset + stride * (accessor["count"] - 1) + 12, len(payload))
                for index in range(accessor["count"]):
                    point = struct.unpack_from("<fff", payload, offset + stride * index)
                    for axis in range(3):
                        value = point[axis] + translation[axis]
                        minimum[axis] = min(minimum[axis], value)
                        maximum[axis] = max(maximum[axis], value)
        self.assertEqual(triangles, 69_806)
        coordinates = self.manifest["coordinate_system"]
        self.assertEqual(coordinates["units"], "meters")
        self.assertEqual(coordinates["up"], "Y")
        for actual, expected in zip(minimum, coordinates["native_bounds_min"]):
            self.assertAlmostEqual(actual, expected, places=6)
        for actual, expected in zip(maximum, coordinates["native_bounds_max"]):
            self.assertAlmostEqual(actual, expected, places=6)
        roll, pitch, yaw = coordinates["gazebo_visual_rpy"]
        self.assertAlmostEqual(roll, math.pi / 2)
        self.assertEqual((pitch, yaw), (0, 0))
        # Rotation +pi/2 about X maps native Y into positive Gazebo Z.
        self.assertGreaterEqual(minimum[1], -0.0001)
        self.assertAlmostEqual(maximum[1] - minimum[1], 0.8411787392, places=6)
        self.assertLess(maximum[0] - minimum[0], 0.75)
        self.assertLess(maximum[2] - minimum[2], 0.70)

    def test_leaf_edges_have_true_alpha_not_jpeg_rectangles(self):
        leaves = next(m for m in self.model["materials"] if m["name"].endswith("_leaves"))
        self.assertEqual(leaves["alphaMode"], "MASK")
        self.assertEqual(leaves["alphaCutoff"], 0.5)
        texture = self.model["textures"][leaves["pbrMetallicRoughness"]["baseColorTexture"]["index"]]
        source = self.model["images"][texture["source"]]
        self.assertEqual(source["mimeType"], "image/png")
        payload = (MODEL.parent / source["uri"]).read_bytes()
        self.assertEqual(payload[:8], b"\x89PNG\r\n\x1a\n")
        width, height, bits, color_type = struct.unpack_from(">IIBB", payload, 16)
        self.assertEqual((width, height, bits, color_type), (1024, 1024, 16, 6))  # RGBA


if __name__ == "__main__":
    unittest.main()
