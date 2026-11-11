"""Read-only dashboard API checks, no ROS node or HTTP server startup."""
from types import SimpleNamespace
import unittest

import numpy as np

from crossmaps_dashboard import CLOUD_LAYERS, PARAM_GROUPS, CloudSnapshot, CrossMapsDashboard, DashboardState, now_sec


class CloudPayloadTests(unittest.TestCase):
    def test_grouping_and_overlay_controls_have_node_defaults(self):
        from test_node_identity_integration import DEFAULTS
        names = ('use_object_assembly', 'entity_grouping_sensitivity',
                 'entity_assembly_confirm_observations', 'entity_assembly_max_extent_m',
                 'segment_mask_min_depth_points', 'segment_mask_min_support_voxels',
                 'segment_overlay_fill_masks', 'segment_overlay_alpha')
        controls = {p['name']: p for g in PARAM_GROUPS for p in g['params']}
        for name in names:
            self.assertIn(name, controls)
            self.assertIn(name, DEFAULTS)
        self.assertEqual(controls['entity_grouping_sensitivity']['min'], 0.)
        self.assertEqual(controls['entity_grouping_sensitivity']['max'], 1.)

    def setUp(self):
        self.node = SimpleNamespace(state=DashboardState(), robot_pose_source_topic=None, robot_odom_topic="")

    def layer(self, id="entity_voxels", max_points=0):
        payload = CrossMapsDashboard.clouds_payload(self.node, max_points)
        return next(layer for layer in payload["layers"] if layer["id"] == id)

    def snapshot(self, count=4, age=0):
        return CloudSnapshot(np.zeros((count, 3)), np.full((count, 3), 120), "map", 0,
                             now_sec() - age, count, count)

    def test_independent_layer_contract(self):
        layers = {layer["id"]: layer for layer in CLOUD_LAYERS}
        self.assertNotIn("object_instances", layers)
        for name in ("segment", "entity"):
            layer = layers[name + "_voxels"]
            self.assertEqual(layer["publishParam"], "publish_" + name + "_voxels")
            self.assertEqual(layer["topic"], f"/crossmaps/{'segments' if name == 'segment' else 'entities'}/voxels")

    def test_waiting_empty_and_ready_are_distinct(self):
        self.assertEqual(self.layer()["state"], "waiting")
        self.node.state.clouds["entity_voxels"] = self.snapshot(0)
        self.assertEqual(self.layer()["state"], "empty")
        self.node.state.clouds["entity_voxels"] = self.snapshot()
        layer = self.layer(max_points=2)
        self.assertEqual(layer["state"], "ready")
        self.assertEqual(layer["count"], 2)
        self.assertEqual(len(layer["positions"]), 6)

    def test_disabled_publisher_does_not_display_cached_cloud(self):
        self.node.state.clouds["entity_voxels"] = self.snapshot()
        self.node.state.params["publish_entity_voxels"] = False
        layer = self.layer()
        self.assertEqual(layer["state"], "disabled")
        self.assertEqual(layer["disabled_by"], ["publish_entity_voxels"])
        self.assertFalse(layer["available"])
        self.assertEqual(layer["positions"], [])

    def test_disabled_memory_and_stale_cloud_are_explained(self):
        self.node.state.params["use_segment_memory"] = False
        self.assertEqual(self.layer()["disabled_by"], ["use_segment_memory"])
        self.node.state.params["use_segment_memory"] = True
        self.node.state.clouds["entity_voxels"] = self.snapshot(age=12)
        self.assertEqual(self.layer()["state"], "stale")
        self.assertTrue(self.layer()["available"])


if __name__ == "__main__":
    unittest.main()
