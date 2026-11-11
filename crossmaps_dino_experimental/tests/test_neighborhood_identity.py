import copy
import json
import unittest

from identity_graph import PersistentEntities


def node(support, **extra):
    return dict(support=set(support), dino=(1., 0.), group_dino=(1., 0.),
                clip=(1., 0.), fresh_dino=True, **extra)


def fixture():
    memory = PersistentEntities()
    query = {(x, 0, 0) for x in range(20)}
    far = {(x, 0, 0) for x in range(100, 120)}
    adjacent = {(x, y, 0) for x in range(20) for y in range(1, 5)}
    scene = {101: node(far), 202: node(adjacent), 303: node(query)}
    for sid, overlap in ((101, 16), (202, 15)):
        record = memory._new(1.)
        memory._attach(sid, record.id, 'fixture')
        record.observations = 5
        record.dino = (1., 0.)
        memory._support(record, scene[sid]['support'] | {(x, 0, 0) for x in range(overlap)})
    memory.last_seq = 1
    return memory, scene


class NeighborhoodTests(unittest.TestCase):
    def test_lower_overlap_candidate_wins_only_after_repeated_contact_evidence(self):
        memory, scene = fixture()
        before = copy.deepcopy(scene)
        for seq in (2, 3):
            memory.update(seq, float(seq), scene, set(scene))
            self.assertNotIn(303, memory.segment_entity)
            self.assertEqual(memory.pending[303]['candidate'], 2)
            self.assertEqual(memory.pending[303]['reason'], 'neighborhood_tiebreak')
            memory.update(seq, float(seq), scene, set(scene))
        memory.update(4, 4., scene, set(scene))
        self.assertEqual(memory.segment_entity[303], 2)
        self.assertEqual(memory.segment_entity[101], 1)
        self.assertEqual(memory.segment_entity[202], 2)
        self.assertEqual(memory.lineage[-1]['reason'], 'neighborhood_tiebreak')
        self.assertEqual(before, scene)

    def test_weak_absent_conflicting_structural_or_identical_neighbor_cannot_break_tie(self):
        for mode in ('off', 'stale', 'missing', 'low_dino', 'clip_conflict', 'floor', 'same_shape'):
            memory, scene = fixture()
            if mode == 'off': memory.config.neighborhood_tiebreak = False
            if mode == 'stale': scene[202]['fresh_dino'] = False
            if mode == 'missing': scene[202]['group_dino'] = ()
            if mode == 'low_dino': scene[202]['group_dino'] = (0., 1.)
            if mode == 'clip_conflict': scene[202]['clip'] = (-1., 0.)
            if mode == 'floor': scene[202]['surface_role'] = 'horizontal_support_surface'
            if mode == 'same_shape': scene[202]['support'] = {(x,1,0) for x in range(20)}
            for seq in range(2, 7): memory.update(seq, float(seq), scene, set(scene))
            self.assertNotIn(303, memory.segment_entity, mode)

    def test_outside_window_or_below_overlap_floor_cannot_receive_bonus(self):
        memory, scene = fixture()
        for candidates in ([(.9, 1), (.75, 2)], [(.70, 1), (.64, 2)]):
            selected, _ = memory._resolve_near_tie(303, scene[303], candidates, scene, set(scene))
            self.assertIsNone(selected)

    def test_occlusion_interrupts_confirmation_and_does_not_supply_context(self):
        memory, scene = fixture()
        memory.update(2, 2., scene, set(scene))
        memory.update(3, 3., scene, {101, 303})
        self.assertIsNone(memory.pending[303]['candidate'])
        memory.update(4, 4., scene, set(scene))
        self.assertEqual(memory.pending[303]['hits'], 1)
        self.assertNotIn(303, memory.segment_entity)

    def test_roundtrip_preserves_pending_confirmation(self):
        memory, scene = fixture()
        memory.update(2, 2., scene, set(scene))
        memory = PersistentEntities.from_dict(json.loads(json.dumps(memory.to_dict())))
        for seq in (3, 4): memory.update(seq, float(seq), scene, set(scene))
        self.assertEqual(memory.segment_entity[303], 2)

    def test_explicit_exclusion_and_established_membership_are_not_overridden(self):
        memory, scene = fixture()
        memory.split_exclusions[303] = 2
        self.assertNotIn(2, [eid for _, eid in memory._candidates(303, scene[303], 2.)])
        memory._attach(303, 1, 'fixture')
        for seq in range(2, 8): memory.update(seq, float(seq), scene, set(scene))
        self.assertEqual(memory.segment_entity[303], 1)


if __name__ == '__main__':
    unittest.main()
