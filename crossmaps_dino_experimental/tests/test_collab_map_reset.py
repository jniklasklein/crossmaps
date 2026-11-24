"""Explicit rebuild drops local map history, not the expensive model cache."""
import unittest

from collab_perception import PerceptionFleet
from test_collab_perception import Factory, config, frame


class MapResetTests(unittest.TestCase):
    def test_reset_preserves_models_not_geometry_identity_or_guards(self):
        factory = Factory()
        fleet = PerceptionFleet(config(), node_factory=factory)
        self.addCleanup(fleet.close)
        fleet.process(frame())
        old_node = fleet.nodes[0]
        old_models = fleet._models
        old_guard = fleet._guards['A']
        self.assertTrue(fleet.health())
        fleet.reset_maps()
        self.assertTrue(old_node.destroyed)
        self.assertEqual(fleet.nodes, ())
        self.assertEqual(fleet.health(), {})
        self.assertEqual(fleet.image_metadata('A'), {})
        self.assertEqual(fleet._guards, {})
        self.assertEqual(fleet._last_integrated, {})
        self.assertIs(fleet._models, old_models)
        fleet.process(frame(seq=2))
        self.assertIsNot(fleet.nodes[0], old_node)
        self.assertIsNot(fleet._guards['A'], old_guard)
        self.assertIs(fleet.nodes[0].model, old_models['model'])


if __name__ == '__main__':
    unittest.main()
