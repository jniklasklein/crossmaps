import copy
import unittest
import numpy as np

from test_node_identity_integration import make_node, observation


class DisplayIntegrationTests(unittest.TestCase):
    def test_segment_and_entity_display_update_without_subscribers(self):
        node = make_node()
        for seq in range(1, 5):
            node.now = float(seq)
            node.apply_semantic_result(observation(seq), 'test')
        self.assertEqual(node.projection['segment_display_decisions'][0]['display_owner'], 101)
        self.assertEqual(node.projection['entity_display_decisions'][0]['display_owner'],
                         node.entity_memory.segment_entity[101])
        self.assertEqual(node.segment_display_memory.last_seq, 4)
        self.assertEqual(node.entity_display_memory.last_seq, 4)

    def test_rendering_and_overlay_toggle_do_not_change_world_evidence(self):
        node = make_node()
        node.stride = 8
        for seq in range(1, 5):
            node.now = float(seq)
            node.apply_semantic_result(observation(seq), 'test')
        before = copy.deepcopy(node.entity_memory.to_dict())
        ownership = node.voxel_segment_id.copy()
        for dominant in (True, False, True):
            node.params['segment_overlay_dominant_owner'] = dominant
            for _ in range(5):
                for color in (node._segment_instance_rgb, node._segment_entity_rgb):
                    view = node._render_id_projection(node.projection, color)
                    self.assertEqual(view.shape, (4, 10, 3))
                    self.assertTrue(np.any(view))
        self.assertEqual(before, node.entity_memory.to_dict())
        self.assertEqual(ownership, node.voxel_segment_id)
        self.assertEqual(node.entity_display_memory.last_seq, 4)


if __name__ == '__main__':
    unittest.main()
