"""Bounded, reversible grouping; no model downloads or ROS required."""
import copy
import json
import math
import unittest

from object_assembly import AssemblyConfig, ObjectAssemblies


def part(x=0, width=0.6, height=0.2, dino=(1., 0.)):
    return {'min': (x, 0., 0.), 'max': (x+width, height, 0.2), 'dino': dino,
            'support': {(i, j, 0) for i in range(round(x/.1), round((x+width)/.1))
                        for j in range(max(1, round(height/.1)))}}


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.memory = ObjectAssemblies()
        self.nodes = {1: part(), 2: part(.6, width=.2, height=.6)}
        # Different shapes (seat vs back), not identical adjacent boxes.
        self.nodes[2]['max'] = (.8, .6, .6)

    def update(self, seq, **kwargs):
        return self.memory.update(seq, float(seq), self.nodes, kwargs.pop('observed', self.nodes),
                                  voxel_size=.1, **kwargs)

    def confirm(self):
        for seq in (1, 2, 3):
            self.update(seq)
        self.assertEqual(self.memory.canonical(2), 1)

    def test_complementary_parts_group_without_changing_geometry(self):
        before = copy.deepcopy(self.nodes)
        for seq in (1, 2):
            self.update(seq)
            self.assertEqual(self.memory.canonical(2), 2)
        self.update(3)
        self.assertEqual(self.memory.canonical(2), 1)
        self.assertEqual(self.nodes, before)

    def test_duplicate_observation_and_diagnostics_never_vote(self):
        self.update(1)
        for _ in range(10):
            self.assertFalse(self.update(1))
            self.memory.diagnostics()
        self.assertEqual(self.memory.pairs[(1, 2)]['hits'], 1)
        self.assertEqual(self.memory.aliases, {})

    def test_identical_boxes_need_shared_mask_and_two_views(self):
        self.nodes[2] = part(.6)
        for seq in (1, 2, 3, 4):
            self.update(seq)
        self.assertEqual(self.memory.aliases, {})
        self.assertEqual(self.memory.pairs[(1, 2)]['reason'], 'similar_shape_needs_shared_mask')
        for seq in (5, 6, 7, 8):
            self.update(seq, shared_masks=[(1, 2)], camera=(0, 0, 0))
        self.assertEqual(self.memory.aliases, {})
        self.update(9, shared_masks=[(1, 2)], camera=(.6, 0, 0))
        self.assertEqual(self.memory.canonical(2), 1)

    def test_high_similarity_far_objects_do_not_group(self):
        self.nodes[2] = part(5.)
        for seq in range(1, 7):
            self.update(seq)
        self.assertEqual(self.memory.aliases, {})

    def test_fragmented_geometry_is_uncertain_not_a_grouping_shortcut(self):
        self.nodes[2]['geometry_reliable'] = False
        for seq in range(1, 7):
            self.update(seq)
        self.assertEqual(self.memory.aliases, {})
        self.assertEqual(self.memory.pairs[(1, 2)]['reason'], 'fragmented_geometry')

    def test_single_point_contact_does_not_group_objects(self):
        self.nodes[2]['support']={(6,0,0)}
        for seq in (1,2,3,4):
            self.update(seq)
        self.assertEqual(self.memory.aliases,{})
        self.assertEqual(self.memory.pairs[(1,2)]['reason'],'insufficient_surface_contact')

    def test_floor_cannot_become_part_of_an_object_despite_high_similarity(self):
        self.nodes[1]['surface_role']='horizontal_support_surface'
        for seq in (1,2,3,4):
            self.update(seq)
        self.assertEqual(self.memory.aliases,{})
        self.assertEqual(self.memory.pairs[(1,2)]['reason'],'structural_surface_not_object_part')

    def test_missing_fresh_dino_holds_but_does_not_confirm_a_join(self):
        self.nodes[2]['fresh_dino']=False
        for seq in (1,2,3,4):
            self.update(seq)
        self.assertEqual(self.memory.aliases,{})
        self.nodes[2]['fresh_dino']=True
        for seq in (5,6,7):
            self.update(seq)
        self.assertEqual(self.memory.canonical(2),1)
        self.nodes[2]['fresh_dino']=False
        for seq in (8,9,10):
            self.update(seq)
            self.assertEqual(self.memory.canonical(2),1)

    def test_old_assembly_confirmations_reset_but_corrections_survive(self):
        self.confirm()
        saved=self.memory.to_dict()
        saved['version']=1
        saved['blocked']=[(3,4)]
        restored=ObjectAssemblies.from_dict(saved)
        self.assertEqual(restored.aliases,{})
        self.assertEqual(restored.pairs,{})
        self.assertEqual(restored.blocked,{(3,4)})

    def test_bbox_overlap_without_surface_contact_is_insufficient(self):
        self.nodes[2]['support'] = {(40, 40, 40)}
        for seq in range(1, 5):
            self.update(seq)
        self.assertEqual(self.memory.pairs[(1, 2)]['reason'], 'insufficient_surface_contact')
        self.assertEqual(self.memory.aliases, {})

    def test_hysteresis_and_occlusion_preserve_groups(self):
        self.confirm()
        self.nodes[2]['dino'] = ()
        self.update(4)
        self.assertEqual(self.memory.canonical(2), 1)
        self.nodes[2]['dino'] = (0., 1.)
        for seq in (5, 6):
            self.update(seq)
            self.assertEqual(self.memory.canonical(2), 1)
        self.update(7, observed=[1])
        self.assertEqual(self.memory.canonical(2), 1)
        self.update(8)
        self.assertEqual(self.memory.canonical(2), 2)

    def test_similarity_chain_cannot_join_incompatible_endpoints(self):
        self.nodes = {1: part(0, dino=(1., 0.)),
                      2: part(.6, height=.7, dino=(.85, math.sqrt(1-.85**2))),
                      3: part(1.2, height=1.5, dino=(.445, math.sqrt(1-.445**2)))}
        for seq in (1, 2, 3):
            self.update(seq)
        self.assertEqual(self.memory.canonical(2), 1)
        self.assertEqual(self.memory.canonical(3), 3)
        self.assertEqual(self.memory.pairs[(2, 3)]['group_blocker'], 'incompatible_group_appearance')

    def test_established_group_rechecks_distant_endpoint_appearance(self):
        self.nodes = {1: part(0), 2: part(.6, height=.7), 3: part(1.2, height=1.5)}
        for seq in (1, 2, 3):
            self.update(seq)
        self.assertEqual(self.memory.canonical(3), 1)
        self.nodes[1]['dino'] = (.85, math.sqrt(1-.85**2))
        self.nodes[3]['dino'] = (.85, -math.sqrt(1-.85**2))
        for seq in (4, 5):
            self.update(seq)
            self.assertEqual(self.memory.canonical(3), 1)
        self.update(6)
        self.assertNotEqual(self.memory.canonical(1), self.memory.canonical(3))

    def test_explicit_separation_and_allow_require_fresh_evidence(self):
        self.confirm()
        self.memory.separate(1, 2)
        for seq in (4, 5, 6):
            self.update(seq)
        self.assertEqual(self.memory.aliases, {})
        self.memory.allow(1, 2)
        for seq in (7, 8):
            self.update(seq)
            self.assertEqual(self.memory.aliases, {})
        self.update(9)
        self.assertEqual(self.memory.canonical(2), 1)

    def test_forbidden_pair_blocks_indirect_group(self):
        self.nodes[3] = part(.8, height=1.4)
        self.memory.separate(1, 3)
        for seq in (1, 2, 3):
            self.update(seq)
        self.assertNotEqual(self.memory.canonical(1), self.memory.canonical(3))

    def test_persistence_preserves_groups_and_pair_corrections(self):
        self.confirm()
        self.memory.config.sensitivity = .7
        saved = json.loads(json.dumps(self.memory.to_dict()))
        restored = ObjectAssemblies.from_dict(saved)
        self.assertEqual(restored.to_dict(), self.memory.to_dict())
        self.assertFalse(restored.update(3, 3., self.nodes, self.nodes))
        self.assertEqual(restored.canonical(2), 1)

    def test_snapshot_is_independent_and_bad_aliases_rejected(self):
        self.confirm()
        saved = self.memory.to_dict()
        self.update(4)
        self.assertEqual(saved['pairs'][0][2]['hits'], 3)
        saved['aliases'] = {1: 2, 2: 1}
        with self.assertRaises(ValueError):
            ObjectAssemblies.from_dict(saved)

    def test_spatial_gap_slider_works_at_fine_voxel_resolution(self):
        self.nodes[1] = {'min': (0.,0.,0.), 'max': (.05,.1,.1), 'support': {(0,0,0), (0,1,0)}, 'dino': (1.,0.)}
        self.nodes[2] = {'min': (.25,0.,0.), 'max': (.35,.5,.5), 'support': {(5,0,0), (5,1,0)}, 'dino': (1.,0.)}
        for seq in (1, 2, 3):
            self.memory.update(seq, float(seq), self.nodes, self.nodes, voxel_size=.05)
        self.assertEqual(self.memory.aliases, {})
        self.memory.config.sensitivity = 1.
        for seq in (4, 5, 6):
            self.memory.update(seq, float(seq), self.nodes, self.nodes, voxel_size=.05)
        self.assertEqual(self.memory.canonical(2), 1)

    def test_sensitivity_monotonic_and_changes_acceptance(self):
        configs = [AssemblyConfig(sensitivity=s) for s in (0, .5, 1)]
        self.assertGreater(configs[0].min_dino, configs[1].min_dino)
        self.assertGreater(configs[1].min_dino, configs[2].min_dino)
        self.assertLess(configs[0].max_gap, configs[1].max_gap)
        self.assertLess(configs[1].max_gap, configs[2].max_gap)
        self.nodes[2]['dino'] = (.85, math.sqrt(1-.85**2))
        self.memory.config.sensitivity = 0
        for seq in (1, 2, 3):
            self.update(seq)
        self.assertEqual(self.memory.aliases, {})
        self.memory.config.sensitivity = .5
        for seq in (4, 5, 6):
            self.update(seq)
        self.assertEqual(self.memory.canonical(2), 1)

    def test_missing_dino_cannot_create_a_group(self):
        self.nodes[2]['dino'] = ()
        for seq in range(1, 5):
            self.update(seq)
        self.assertEqual(self.memory.aliases, {})

    def test_disable_restores_base_ids_and_clears_evidence(self):
        self.confirm()
        self.memory.config.enabled = False
        self.update(4)
        self.assertEqual(self.memory.aliases, {})
        self.assertEqual(self.memory.pairs, {})

    def test_extent_member_and_pair_limits(self):
        self.memory.config.max_extent_m = .4
        for seq in (1, 2, 3):
            self.update(seq)
        self.assertEqual(self.memory.aliases, {})
        self.memory.config.max_extent_m = 3.5
        self.memory.config.max_members = 1
        for seq in (4, 5, 6):
            self.update(seq)
        self.assertEqual(self.memory.aliases, {})
        self.memory.config.max_pairs = 0
        self.update(7)
        self.assertTrue(self.memory.budget_limited)


if __name__ == '__main__':
    unittest.main()
