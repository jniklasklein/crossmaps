"""Coverage/budget regressions for two rovers pointed at repetitive scenes."""
import tempfile
import unittest
from unittest.mock import patch

from collab_backend import Backend
from collab_retrieval import VerificationSchedule, pair_key
from test_collab_backend import observation


def rejected(*_, **__):
    return dict(ok=False, reason='insufficient_geometric_inliers', inliers=0)


class RetrievalCoverageTests(unittest.TestCase):
    def test_stale_counterpart_does_not_pin_the_budget_to_two_frames(self):
        # This fixture exercises scheduling across ten retained frames, not
        # stationary admission (covered independently by test_collab_retention).
        backend = Backend(dict(candidate_top_k=2, submap_keyframes=1, stationary_bootstrap_keyframes=16))
        def scores(_a, b):
            return {'clip': .99-.001*b['seq'], 'dino': .93-.001*b['seq']}
        pairs = []
        def check(a, b, **kwargs):
            self.assertEqual(kwargs, dict(min_inliers=12, min_overlap=.25, max_rmse=.065))
            pairs.append(pair_key(a['id'], b['id']))
            return rejected()
        with patch.object(backend, '_scores', side_effect=scores), patch('collab_backend.verify_rgbd', side_effect=check):
            for seq in range(1, 9):
                backend.ingest(dict(observation('A', seq), received_monotonic=float(seq)))
            for seq in range(1, 11):
                n = len(pairs)
                backend.ingest(dict(observation('B', seq), received_monotonic=1000.+seq))
                self.assertLessEqual(len(pairs)-n, 2)
        self.assertEqual(len(set(pairs)), len(pairs))
        targets = {f for p in pairs for f in p if backend.keyframes[f]['robot_id'] == 'A'}
        self.assertGreaterEqual(len(targets), 6)
        self.assertGreater(backend.verification_schedule.backfilled, 0)
        self.assertTrue(all(not c.get('active') for c in backend.constraints.values() if c['kind']=='loop'))

    def test_idle_backfill_budget_and_persistence(self):
        with tempfile.TemporaryDirectory() as folder:
            backend = Backend(dict(verifications_per_keyframe=1), folder)
            with patch('collab_backend.verify_rgbd', side_effect=rejected) as verifier:
                for seq in (1, 2, 3):
                    backend.ingest(observation('A', seq))
                backend.ingest(observation('B'))
                self.assertEqual(verifier.call_count, 1)
                before = backend.revision
                self.assertTrue(backend.backfill_one())
                self.assertEqual((verifier.call_count, backend.revision), (2, before+1))
                backend = Backend(store_dir=folder)
                self.assertEqual(backend.verification_schedule.report()['checked_unique_pairs'], 2)
                self.assertTrue(backend.backfill_one())
                self.assertEqual(verifier.call_count, 3)
                self.assertFalse(backend.backfill_one())
                for frame in list(backend.keyframes.values())[::-1]:
                    backend._retrieve(frame)   # Reversed source/target can't vote/check twice.
                self.assertEqual(verifier.call_count, 3)
                calls = [pair_key(a.args[0]['id'], a.args[1]['id']) for a in verifier.call_args_list]
                self.assertEqual(len(set(calls)), 3)
                cards = [c for c in backend.candidates.values() if c.get('verification_basis')=='historical_backfill']
                self.assertEqual(len(cards), 2)

    def test_retired_session_has_no_backfill_votes_and_zero_budget_stays_off(self):
        backend = Backend(dict(verifications_per_keyframe=1))
        with patch('collab_backend.verify_rgbd', side_effect=rejected) as verify:
            for i in (1,2,3):
                backend.ingest(observation('A',i))
            backend.ingest(observation('B'))
            self.assertTrue(backend.verification_schedule.pending)
            backend.ingest(observation('A',session='restarted'))
            count = verify.call_count
            self.assertFalse(backend.backfill_one())
            self.assertEqual(verify.call_count, count)
            backend.config['verifications_per_keyframe'] = 0
            backend.ingest(observation('A',seq=2,session='restarted'))
            self.assertFalse(backend.backfill_one())
            self.assertEqual(verify.call_count, count)

    def test_operator_rejection_survives_reverse_retrieval_and_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            backend = Backend(store_dir=folder)
            backend.ingest(observation('A')); backend.ingest(observation('B'))
            edge = next(e for e in backend.constraints.values() if e['kind']=='loop')
            backend.command(dict(action='reject_constraint', id=edge['id']))
            reopened = Backend(store_dir=folder)
            with patch('collab_backend.verify_rgbd', side_effect=AssertionError('duplicate verification')):
                for frame in reopened.keyframes.values():
                    reopened._retrieve(frame)
                self.assertFalse(reopened.backfill_one())
            self.assertEqual(reopened.constraints[edge['id']]['state'], 'rejected')
            self.assertEqual(len(reopened.verification_schedule.checked), 1)

    def test_pending_queue_bounded_per_cell_and_globally(self):
        plan = VerificationSchedule()
        for i in range(plan.MAX_PENDING*2):
            row = dict(source='a'+str(i), target='b', source_submap=str(i%90), target_submap='other')
            plan.add(row)
        self.assertLessEqual(len(plan.pending), plan.MAX_PENDING)
        for i in range(90):
            self.assertLessEqual(sum(c['source_submap']==str(i) for c in plan.pending.values()), plan.PER_SUBMAP_PAIR)

    def test_idle_coverage_eventually_checks_even_non_retrieved_pairs(self):
        backend = Backend(dict(verifications_per_keyframe=1, candidate_top_k=1, stationary_bootstrap_keyframes=8))
        with patch.object(backend, '_scores', return_value={'clip': .97}), \
                patch('collab_backend.verify_rgbd', side_effect=rejected) as check:
            for robot, n in [('A', 7), ('B', 5)]:
                for i in range(1, n+1):
                    backend.ingest(observation(robot,i))
            for _ in range(100):
                backend.backfill_one()
            self.assertEqual(len(backend.verification_schedule.checked), 35)
            self.assertEqual(check.call_count, 35)
            self.assertTrue(any(c.get('proposal_basis')=='retained_trajectory_exploration'
                                for c in backend.candidates.values()))
