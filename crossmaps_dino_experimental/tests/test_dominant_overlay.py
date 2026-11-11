import copy
import unittest
import numpy as np

from dominant_overlay import DominantOverlay
from segment_overlay import mask_consensus_ids, render_mask_consensus, id_color


def decision(votes, winner=None, center=0., reason='multiple_supported_owners'):
    return dict(mask=0, votes=votes, winner=winner, segments=list(votes), reason=reason,
                samples=sum(votes.values()), centroid=np.array([center, 0., 1.]),
                thumbnail=np.ones((64, 96), dtype=bool))


class DominantOverlayTests(unittest.TestCase):
    def test_near_tied_votes_do_not_flicker_or_change_mapping_evidence(self):
        memory = DominantOverlay()
        mask = np.ones((40, 60), dtype=bool)
        for seq in range(1, 61):
            raw = decision({1: 52 if seq % 2 else 48, 2: 48 if seq % 2 else 52})
            before = copy.deepcopy(raw)
            result = memory.update(seq, float(seq), [raw], lambda s: s, .1)
            self.assertEqual(result[0]['display_owner'], 1)
            self.assertTrue(result[0]['display_provisional'])
            self.assertIsNone(raw['winner'])
            self.assertEqual(raw['votes'], before['votes'])
            np.testing.assert_array_equal(raw['thumbnail'], before['thumbnail'])
            labels, uncertain = mask_consensus_ids(mask.shape, [mask], result, dominant=True)
            self.assertTrue(np.all(labels == 1))
            self.assertTrue(np.all(uncertain))

    def test_switch_requires_three_distinct_observations(self):
        memory = DominantOverlay()
        memory.update(1, 1., [decision({1: 80, 2: 20})], lambda s: s, .1)
        for seq in (2, 3):
            result = memory.update(seq, float(seq), [decision({1: 20, 2: 80})], lambda s: s, .1)
            self.assertEqual(result[0]['display_owner'], 1)
            memory.update(seq, float(seq), [decision({1: 20, 2: 80})], lambda s: s, .1)
        self.assertEqual(memory.update(4, 4., [decision({1: 20, 2: 80})], lambda s: s, .1)[0]['display_owner'], 2)

    def test_entity_votes_aggregate_before_ranking(self):
        memory = DominantOverlay()
        result = memory.update(1, 1., [decision({1: 6, 2: 6, 3: 8})],
                               lambda s: 10 if s in (1, 2) else 20, .1)
        self.assertEqual(result[0]['display_owner'], 10)
        self.assertAlmostEqual(result[0]['display_confidence'], .6)

    def test_missing_depth_conflict_retirement_and_motion_do_not_paint_old_owner(self):
        for mode in ('no_depth', 'conflict', 'retired', 'motion'):
            memory = DominantOverlay()
            memory.update(1, 1., [decision({1: 80, 2: 20})], lambda s: s, .1)
            item = decision({1: 45, 2: 55}, center=2. if mode == 'motion' else 0.)
            if mode == 'no_depth': item = dict(mask=0, votes={}, winner=None)
            if mode == 'conflict': item['reason'] = 'appearance_conflict'
            result = memory.update(2, 2., [item], lambda s: None if mode == 'retired' and s == 1 else s, .1)
            self.assertEqual(result[0]['display_owner'], 2 if mode in ('motion', 'retired') else None)

    def test_render_has_one_interior_color_despite_mixed_seeds(self):
        mask = np.ones((40, 60), dtype=bool)
        result = DominantOverlay().update(1, 1., [decision({1: 60, 2: 40})], lambda s: s, .1)
        seeds = np.zeros(mask.shape, dtype=np.int32)
        seeds[10, 10], seeds[10, 45] = 1, 2
        view = render_mask_consensus(mask.shape, [mask], result, None, lambda s: s,
                                     id_color, seeds=seeds, dominant=True)
        np.testing.assert_array_equal(view[10, 10], view[10, 45])
        self.assertEqual(seeds[10, 45], 2)

    def test_missing_observation_breaks_challenger_run(self):
        memory = DominantOverlay()
        memory.update(1, 1., [decision({1: 80, 2: 20})], lambda s: s, .1)
        memory.update(2, 2., [decision({1: 20, 2: 80})], lambda s: s, .1)
        result = memory.update(4, 4., [decision({1: 20, 2: 80})], lambda s: s, .1)
        self.assertEqual(result[0]['display_owner'], 1)
        self.assertEqual(memory.records[0]['hits'], 1)

    def test_mask_dropout_retains_history_but_does_not_paint_missing_mask(self):
        memory = DominantOverlay()
        memory.update(1, 1., [decision({1: 80, 2: 20})], lambda s: s, .1)
        self.assertEqual(memory.update(2, 2., [], lambda s: s, .1), [])
        result = memory.update(3, 3., [decision({1: 49, 2: 51})], lambda s: s, .1)
        self.assertEqual(result[0]['display_owner'], 1)

    def test_unassigned_entity_support_is_not_confirmed_consensus(self):
        result = DominantOverlay().update(1, 1., [decision({1: 30, 2: 70})],
                                          lambda s: 10 if s == 1 else None, .1)
        self.assertEqual(result[0]['display_owner'], 10)
        self.assertTrue(result[0]['display_provisional'])


if __name__ == '__main__':
    unittest.main()
