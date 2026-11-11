import unittest
import numpy as np
from world_model import new_world_id, validate_world_id, node_uid, interior_descriptor, surface_role, dominant_surface_core


class WorldModelTests(unittest.TestCase):
    def test_connected_core_excludes_outliers_without_mutating_map(self):
        panel = {(x,y,0) for x in range(6) for y in range(4)}
        support = panel | {(100,100,100), (-50,0,0)}
        original = support.copy()
        core, reliable = dominant_surface_core(support)
        self.assertTrue(reliable)
        self.assertEqual(core, panel)
        self.assertEqual(support, original)

    def test_fragmented_geometry_does_not_arbitrarily_select_a_part(self):
        support = {(x,0,0) for x in range(5)} | {(x,0,0) for x in range(20,25)}
        core, reliable = dominant_surface_core(support)
        self.assertFalse(reliable)
        self.assertEqual(core, support)
        self.assertEqual(dominant_surface_core(set()), (set(),False))

    def test_namespaces_survive_serialization_and_do_not_collide(self):
        first, second = new_world_id(), new_world_id()
        self.assertEqual(first, validate_world_id(first))
        self.assertNotEqual(node_uid(first, 'entity', 1), node_uid(second, 'entity', 1))
        self.assertNotEqual(node_uid(first, 'entity', 1), node_uid(first, 'segment', 1))
        with self.assertRaises(ValueError):
            validate_world_id('not-a-world-id')

    def test_grouping_interior_is_independent_of_context_contrast(self):
        a = np.array([1.,0., 0., .35])
        b = np.array([1.,0., 0., -.35])
        np.testing.assert_allclose(interior_descriptor(a, 2), interior_descriptor(b, 2))
        self.assertIsNone(interior_descriptor(a, 0))
        self.assertIsNone(interior_descriptor(a, 3))
        self.assertIsNone(interior_descriptor(np.zeros(4), 2))

    def test_large_floor_and_wall_are_surface_candidates_not_sofa_panels(self):
        x,y=np.meshgrid(np.arange(0,3,.1),np.arange(0,2,.1))
        floor=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
        self.assertEqual(surface_role(floor,.1),'horizontal_support_surface')
        self.assertEqual(surface_role(floor[:,[0,2,1]],.1),'vertical_structural_surface')
        self.assertEqual(surface_role(floor*.3,.1),'unknown')
        self.assertEqual(surface_role(floor[:5],.1),'unknown')


if __name__ == '__main__':
    unittest.main()
