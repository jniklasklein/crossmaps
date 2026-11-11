import copy
import unittest
import numpy as np

from surface_tracking import SurfaceConfig, SurfaceConsensus, BoundaryOwnershipVotes
from segment_overlay import mask_consensus_ids


class SurfaceTests(unittest.TestCase):
    def setUp(self):
        self.memory = SurfaceConsensus()
        self.mask = np.zeros((60, 100), dtype=bool)
        self.mask[5:55, 5:95] = True
        uu, vv = np.meshgrid(np.arange(10, 95, 5), np.arange(10, 55, 5))
        self.uu, self.vv = uu.ravel(), vv.ravel()
        self.points = np.column_stack((self.uu*.01, self.vv*.01, np.ones(self.uu.size)))
        self.owners = np.ones(self.uu.size, dtype=np.int32)

    def propose(self, seq, owners=None, points=None, compatible=None):
        return self.memory.propose([self.mask], self.owners if owners is None else owners,
            self.points if points is None else points, self.uu, self.vv, float(seq), .1,
            {1,2}, compatible or (lambda *_: True))

    def commit(self, seq, decisions):
        return self.memory.commit(seq, float(seq), decisions, [self.mask], {1,2})

    def test_stationary_noise_replay_preserves_identity_and_sam_outline(self):
        rng = np.random.default_rng(7)
        before = self.points.copy()
        for seq in range(1, 81):
            owners = self.owners.copy()
            owners[rng.choice(len(owners), size=8, replace=False)] = 2
            jittered = self.points + rng.normal(0, .005, self.points.shape)
            decisions = self.propose(seq, owners, jittered)
            self.assertEqual(decisions[0]['winner'], 1)
            self.commit(seq, decisions)
            labels, ambiguous = mask_consensus_ids(self.mask.shape, [self.mask], decisions)
            np.testing.assert_array_equal(labels > 0, self.mask)
            self.assertTrue(np.all(labels[self.mask] == 1))
            self.assertFalse(np.any(ambiguous))
        np.testing.assert_array_equal(before, self.points)
        self.assertEqual(self.memory.switches, 0)

    def test_mixed_mask_is_not_a_majority_merge_or_voronoi_partition(self):
        owners = self.owners.copy()
        owners[self.uu > 50] = 2
        decisions = self.propose(1, owners)
        self.assertIsNone(decisions[0]['winner'])
        labels, ambiguous = mask_consensus_ids(self.mask.shape, [self.mask], decisions)
        self.assertFalse(np.any(labels))
        np.testing.assert_array_equal(ambiguous, self.mask)
        entity_labels, ambiguous = mask_consensus_ids(self.mask.shape, [self.mask], decisions, lambda _: 7)
        self.assertTrue(np.all(entity_labels[self.mask] == 7))
        self.assertFalse(np.any(ambiguous))

    def test_temporal_hold_needs_current_depth_and_world_agreement(self):
        self.commit(1, self.propose(1))
        unknown = np.zeros_like(self.owners)
        held = self.propose(2, unknown)
        self.assertEqual(held[0]['winner'], 1)
        self.assertEqual(held[0]['reason'], 'temporal_depth_hold')
        moved = self.propose(2, unknown, self.points + [1,0,0])
        self.assertIsNone(moved[0]['winner'])
        no_depth = self.memory.propose([self.mask], unknown[:0], self.points[:0], self.uu[:0], self.vv[:0],
                                      2., .1, {1,2}, lambda *_: True)
        self.assertIsNone(no_depth[0]['winner'])
        self.assertIsNone(self.propose(30, unknown)[0]['winner'])

    def test_owner_switch_requires_consecutive_observations_not_refreshes(self):
        self.commit(1, self.propose(1))
        challenger = np.full_like(self.owners, 2)
        for seq in (2,3):
            decision = self.propose(seq, challenger)
            self.assertEqual(decision[0]['winner'], 1)
            self.commit(seq, decision)
            self.assertFalse(self.commit(seq, decision))
        decision = self.propose(4, challenger)
        self.assertEqual(decision[0]['winner'], 2)
        self.commit(4, decision)
        self.commit(5, self.propose(5, challenger))
        self.assertEqual(self.memory.switches, 1)

    def test_appearance_conflict_blocks_temporal_reuse(self):
        self.commit(1, self.propose(1))
        decision = self.propose(2, compatible=lambda *_: False)
        self.assertIsNone(decision[0]['winner'])
        self.assertEqual(decision[0]['reason'], 'appearance_conflict')

    def test_missing_observation_interrupts_owner_switch_confirmation(self):
        self.commit(1, self.propose(1))
        challenger = np.full_like(self.owners, 2)
        self.commit(2, self.propose(2, challenger))
        self.memory.commit(3, 3., [], [], {1,2})
        decision = self.propose(4, challenger)
        self.assertEqual(decision[0]['switch_hits'], 1)
        self.assertEqual(decision[0]['winner'], 1)

    def test_split_masks_keep_surface_id_without_creating_tracks(self):
        self.commit(1, self.propose(1))
        left, right = self.mask.copy(), self.mask.copy()
        left[:, 50:] = False
        right[:, :50] = False
        decisions = self.memory.propose([left, right], self.owners, self.points, self.uu, self.vv,
                                        2., .1, {1}, lambda *_: True)
        self.assertEqual([d['winner'] for d in decisions], [1,1])
        self.memory.commit(2, 2., decisions, [left,right], {1})
        self.assertEqual(set(self.memory.records), {1})

    def test_retired_ids_never_reappear_from_mask_history(self):
        self.commit(1, self.propose(1))
        decision = self.memory.propose([self.mask], self.owners, self.points, self.uu, self.vv,
                                       2., .1, {2}, lambda *_: True)
        self.assertIsNone(decision[0]['winner'])


class CorrectionTests(unittest.TestCase):
    def test_repeated_votes_and_duplicate_protection(self):
        votes = BoundaryOwnershipVotes()
        proposal = [((0,0,0), 2, 1)]
        for seq in (1,2,3):
            self.assertEqual(votes.update(seq, proposal, {2:100}), [])
            self.assertEqual(votes.update(seq, proposal, {2:100}), [])
        self.assertEqual(votes.update(4, proposal, {2:100}), [((0,0,0),2,1)])

    def test_missing_evidence_resets_votes_and_budget_prevents_track_erasure(self):
        votes = BoundaryOwnershipVotes()
        proposal = [((0,0,0), 2, 1)]
        votes.update(1, proposal, {2:20})
        votes.update(2, [], {2:20})
        self.assertEqual(votes.update(3, proposal, {2:20}), [])
        accepted = []
        for seq in range(4, 80):
            accepted += votes.update(seq, [((seq//4,0,0), 2, 1)], {2:20})
        self.assertLessEqual(len(accepted), 2)
        restored = BoundaryOwnershipVotes.from_dict(copy.deepcopy(votes.to_dict()))
        self.assertEqual(restored.losses, votes.losses)
        self.assertFalse(restored.pending)


if __name__ == '__main__':
    unittest.main()
