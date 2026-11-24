"""Structural and mesh invariants for the furnished scene (no renderer needed)."""
import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from scene_geometry import build_room, rounded_box_mesh, lathe_mesh

NS = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}


class SceneGeometryTests(unittest.TestCase):
    def test_bevel_mesh_has_valid_unit_normals_triangles_and_metre_uvs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'rounded.dae'
            count = rounded_box_mesh(path, (2., 1., .1), .015, scale=.5)
            root = ET.parse(path).getroot()
            arrays = {a.attrib['id']: [float(x) for x in a.text.split()]
                      for a in root.findall('.//c:float_array', NS)}
            normals = arrays['normals-array']
            positions = arrays['positions-array']
            self.assertEqual(len(normals), len(positions))
            for index in range(0, len(normals), 3):
                self.assertAlmostEqual(sum(x*x for x in normals[index:index+3]), 1., places=6)
            triangles = root.find('.//c:triangles', NS)
            indices = [int(x) for x in triangles.findtext('c:p', namespaces=NS).split()]
            self.assertEqual(len(indices), count*9)
            self.assertLess(max(indices), len(positions)//3)
            self.assertEqual(int(triangles.attrib['count']), count)
            self.assertEqual(max(arrays['uv-array']), 4.)
            self.assertTrue(all(math.isfinite(x) for values in arrays.values() for x in values))

    def test_lathe_preserves_open_rim_and_has_finite_normals(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'cup.dae'
            profile = ((.14,0.),(.19,.5),(.17,.5),(.12,.03),(0.,.03))
            count = lathe_mesh(path, profile, segments=24)
            root = ET.parse(path).getroot()
            normals = [float(x) for x in root.find('.//c:float_array[@id="normals-array"]', NS).text.split()]
            self.assertEqual(count, 2*24*(len(profile)-1))
            self.assertTrue(all(math.isfinite(x) for x in normals))

    def test_room_keeps_floor_clear_and_has_detailed_furniture_without_posters(self):
        with tempfile.TemporaryDirectory() as folder:
            world = ET.Element('world')
            metadata = build_room(world, Path(folder), {})
            names = {m.attrib['name'] for m in world.findall('model')}
            self.assertTrue({'workbench', 'office_chair', 'oak_bookshelf', 'potted_rubber_plant', 'oak_door', 'side_window'} <= names)
            self.assertEqual(world.findtext("model[@name='room_architecture']/link/collision[@name='oak_floor']/pose"), '1.5 0 -0.05 0 0 0')
            self.assertFalse(any('poster' in item.attrib.get('name', '') or 'printed' in item.attrib.get('name','')
                                 for item in world.iter()))
            self.assertEqual(metadata['visuals'], len(world.findall('.//visual')))
            self.assertLess(metadata['estimated_visual_triangles'], 130000)
            for link in world.findall('model/link'):
                names = [v.attrib['name'] for v in link.findall('visual')]
                self.assertEqual(len(names), len(set(names)))
                for visual in link.findall('visual'):
                    self.assertIsNotNone(visual.find('geometry'))
            for uri in world.findall('.//mesh/uri'):
                self.assertTrue(Path(uri.text.removeprefix('file://')).is_file())
            plant = world.find("model[@name='potted_rubber_plant']/link/visual")
            self.assertIsNone(plant.find('material'))  # retain imported glTF PBR/alpha
            self.assertEqual(world.findtext('scene/shadows'), 'true')
            self.assertEqual(len(world.findall('light')), 3)

    def test_material_maps_are_explicit_and_open_gl_tangent_normal_is_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            maps = {name: folder/(name+'.jpg') for name in ('albedo','normal','roughness')}
            world = ET.Element('world')
            build_room(world, folder, {'floor': dict(maps, scale_m=1.7)})
            pbr = world.find("model[@name='room_architecture']/link/visual[@name='oak_floor']/material/pbr/metal")
            for field, tag in (('albedo','albedo_map'),('normal','normal_map'),('roughness','roughness_map')):
                self.assertEqual(pbr.findtext(tag), maps[field].resolve().as_uri())
            self.assertEqual(pbr.find('normal_map').attrib['type'], 'tangent')


if __name__ == '__main__':
    unittest.main()
