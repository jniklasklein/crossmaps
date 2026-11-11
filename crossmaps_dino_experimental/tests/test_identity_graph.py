import copy
import json
import unittest

from identity_graph import IdentityConfig, ObservationEdges, PersistentEntities


def surface(start=0, count=20, dino=(1.0, 0.0)):
    return {"support": {(i, 0, 0) for i in range(start, start + count)}, "dino": dino}


class IdentityReplayTests(unittest.TestCase):
    def setUp(self):
        self.memory = PersistentEntities(IdentityConfig(birth_observations=2))

    def observe(self, seq, segments, observed=None, camera=(0, 0, 0)):
        return self.memory.update(seq, float(seq), segments,
                                  set(segments) if observed is None else observed, camera)

    def birth(self, segments):
        self.observe(1, segments)
        self.observe(2, segments)

    def test_stationary_birth_hysteresis_and_deduplication(self):
        scene = {101: surface()}
        self.observe(1, scene)
        self.assertFalse(self.memory.records)
        self.assertFalse(self.observe(1, scene))
        self.observe(2, scene)
        eid = self.memory.segment_entity[101]
        self.assertNotEqual(eid, 101)
        for seq in range(3, 10):
            self.observe(seq, scene)
        record = self.memory.records[eid]
        self.assertEqual(record.members, {101})
        self.assertEqual(len(record.views), 1)
        self.assertEqual(record.observations, 9)
        before = self.memory.to_dict()
        self.observe(3, scene)
        self.assertEqual(before, self.memory.to_dict())

    def test_touching_identical_boxes_never_group(self):
        scene = {101: surface(0), 102: surface(20), 103: surface(40)}
        self.birth(scene)
        for seq in range(3, 9):
            self.observe(seq, scene, camera=(seq * 0.3, 0, 0))
        self.assertEqual(len(set(self.memory.segment_entity.values())), 3)
        self.assertFalse(self.memory.can_merge_segments(101, 102))

    def test_id_survives_complete_segment_replacement(self):
        self.birth({101: surface()})
        eid = self.memory.segment_entity[101]
        self.observe(3, {202: surface()})
        self.assertNotIn(202, self.memory.segment_entity)
        self.observe(4, {202: surface()})
        self.assertEqual(self.memory.segment_entity[202], eid)
        self.assertEqual(self.memory.records[eid].members, {202})
        self.assertEqual(len(self.memory.records), 1)

    def test_fragment_probation_preserves_anchor_when_parent_shrinks(self):
        self.birth({101: surface(0, 40)})
        eid = self.memory.segment_entity[101]
        split_masks = {101: surface(0, 20), 202: surface(20, 20)}
        self.observe(3, split_masks)
        self.observe(4, split_masks)
        self.assertEqual(self.memory.segment_entity[202], eid)
        self.assertEqual(self.memory.records[eid].members, {101, 202})
        self.assertEqual(self.memory.records[eid].observations, 4)

    def test_disconnected_maintenance_split_is_not_part_of_evidence(self):
        self.birth({101: surface(0, 40)})
        eid = self.memory.segment_entity[101]
        self.memory.segment_split(101, 202)
        scene = {101: surface(0, 20), 202: surface(20, 20)}
        self.observe(3, scene, {101})  # A split itself is not an observation.
        self.assertNotIn(202, self.memory.pending)
        self.observe(4, scene)
        self.observe(5, scene)
        self.assertNotEqual(self.memory.segment_entity[202], eid)
        self.assertEqual(self.memory.lineage[1]["event"], "split_from")

    def test_two_objects_under_one_mask_remain_ambiguous(self):
        self.birth({101: surface(0), 102: surface(20)})
        ids = set(self.memory.records)
        for seq in range(3, 9):
            self.observe(seq, {303: surface(0, 40)})
        self.assertNotIn(303, self.memory.segment_entity)
        self.assertEqual(set(self.memory.records), ids)
        self.assertEqual(self.memory.pending[303]["reason"], "ambiguous_support")

    def test_dino_conflict_vetoes_reassociation_but_similarity_cannot_create_it(self):
        self.birth({101: surface()})
        old = self.memory.segment_entity[101]
        scene = {202: surface(dino=(-1.0, 0.0)), 303: surface(50)}
        self.observe(3, scene)
        self.observe(4, scene)
        self.assertNotEqual(self.memory.segment_entity[202], old)
        self.assertNotEqual(self.memory.segment_entity[303], old)
        self.assertEqual(len(set(self.memory.segment_entity.values())), 2)

    def test_missing_dino_allows_strong_geometry(self):
        self.birth({101: surface()})
        old = self.memory.segment_entity[101]
        self.observe(3, {202: surface(dino=())})
        self.observe(4, {202: surface(dino=())})
        self.assertEqual(self.memory.segment_entity[202], old)

    def test_occlusion_does_not_add_observations_or_erase_identity(self):
        scene = {101: surface()}
        self.birth(scene)
        eid = self.memory.segment_entity[101]
        for seq in range(3, 20):
            self.observe(seq, scene, set())
        self.assertEqual(self.memory.records[eid].observations, 2)
        self.observe(20, scene)
        self.assertEqual(self.memory.segment_entity[101], eid)

    def test_dormant_reappearance_and_expiry_do_not_reuse_ids(self):
        self.birth({101: surface()})
        old = self.memory.segment_entity[101]
        self.observe(3, {})
        self.observe(4, {202: surface()})
        self.observe(5, {202: surface()})
        self.assertEqual(self.memory.segment_entity[202], old)
        self.observe(200, {})
        self.assertNotIn(old, self.memory.records)
        self.observe(201, {303: surface()})
        self.observe(202, {303: surface()})
        self.assertGreater(self.memory.segment_entity[303], old)

    def test_explicit_membership_removal_is_non_destructive_and_reversible(self):
        self.birth({101: surface(0, 40)})
        scene = {101: surface(), 202: surface(20)}
        self.observe(3, scene)
        self.observe(4, scene)
        old = self.memory.segment_entity[101]
        self.assertTrue(self.memory.detach_segment(202))
        self.observe(5, scene)
        self.observe(6, scene)
        self.assertEqual(self.memory.segment_entity[101], old)
        self.assertNotEqual(self.memory.segment_entity[202], old)
        self.assertEqual(len(scene[202]["support"]), 20)

    def test_save_restore_keeps_ids_pending_and_lineage(self):
        self.birth({101: surface()})
        self.observe(3, {202: surface()})
        snapshot = json.loads(json.dumps(self.memory.to_dict()))
        self.memory = PersistentEntities.from_dict(snapshot, self.memory.config)
        self.observe(4, {202: surface()})
        self.assertEqual(self.memory.segment_entity[202], 1)
        self.assertEqual(self.memory.next_id, 2)
        self.assertEqual(self.memory.records[1].observations, 3)

    def test_memory_support_is_bounded(self):
        self.memory.config.support_limit = 10
        self.birth({101: surface(0, 200)})
        self.assertEqual(len(self.memory.records[1].support), 10)
        self.assertEqual(len(self.memory.support_index), 10)

    def test_descriptor_bank_keeps_multiple_appearances_without_extra_inference(self):
        self.birth({101: surface()})
        eid = self.memory.segment_entity[101]
        self.observe(3, {101: surface(dino=(0.0, 1.0))})
        self.assertEqual(len(self.memory.records[eid].dino_prototypes), 2)
        self.observe(4, {202: surface()})
        self.observe(5, {202: surface()})
        self.assertEqual(self.memory.segment_entity[202], eid)
        restored = PersistentEntities.from_dict(json.loads(json.dumps(self.memory.to_dict())))
        self.assertEqual(restored.records[eid].dino_prototypes, self.memory.records[eid].dino_prototypes)


class RelationEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.memory = ObservationEdges()
        self.near = {"from": 1, "to": 2, "relation": "near", "strength": 0.8}

    def test_publication_is_read_only_and_duplicates_do_not_confirm(self):
        self.memory.update(1, [self.near, self.near], {1, 2}, {1, 2})
        before = copy.deepcopy(self.memory.states)
        for _ in range(20):
            self.assertEqual(self.memory.snapshot({1, 2}), [])
        self.memory.update(1, [self.near], {1, 2}, {1, 2})
        self.assertEqual(before, self.memory.states)
        self.memory.update(2, [self.near], {1, 2}, {1, 2})
        self.assertEqual(self.memory.snapshot({1, 2})[0]["hits"], 2)

    def test_occluded_endpoints_and_display_budgets_are_not_misses(self):
        for seq in (1, 2):
            self.memory.update(seq, [self.near], {1, 2}, {1, 2})
        for seq in range(3, 10):
            self.memory.snapshot({1})
            self.memory.update(seq, [], {1}, {1, 2})
        self.assertEqual(self.memory.snapshot({1, 2})[0]["misses"], 0)

    def test_relation_disappears_only_after_evaluated_negative_observations(self):
        for seq in (1, 2):
            self.memory.update(seq, [self.near], {1, 2}, {1, 2})
        self.memory.update(3, [], {1, 2}, {1, 2}, max_misses=1)
        self.assertEqual(self.memory.snapshot({1, 2})[0]["misses"], 1)
        self.memory.update(4, [], {1, 2}, {1, 2}, max_misses=1)
        self.assertFalse(self.memory.states)

    def test_unobserved_static_map_cannot_confirm_edges(self):
        for seq in range(10):
            self.memory.update(seq, [self.near], {1}, {1, 2})
        self.assertFalse(self.memory.states)

    def test_direction_and_retirement(self):
        self.assertEqual(self.memory.key(2, 1, "near"), (1, 2, "near"))
        self.assertEqual(self.memory.key(2, 1, "above"), (2, 1, "above"))
        self.memory.update(1, [self.near], {1, 2}, {1, 2})
        self.memory.update(2, [], set(), {1})
        self.assertFalse(self.memory.states)


if __name__ == "__main__":
    unittest.main()
