"""Storage UI actions exercise real persistence, never user's recordings."""
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from collab_config import validate_config
from collab_demo import observations
from collab_storage import StorageManager
from crossmaps_collaborative import FleetApplication


class StorageApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'recordings'
        self.run = self.root / 'current'
        self.storage = StorageManager(self.root, self.run, min_free_bytes=0)
        self.config = validate_config({'backend': {'verifications_per_keyframe': 0}})
        self.app = FleetApplication(self.config, self.run, demo=True, storage=self.storage)
        self.addCleanup(self.app.close)

    def ingest(self, step=0):
        for obs in observations(step, self.config['robots']):
            self.assertTrue(self.app.offer(obs))
            self.assertTrue(self.app.process_one())

    def prepare_reset(self):
        return self.app.command({'action': 'prepare_storage', 'operation': 'rebuild_current'})['plan']

    def test_preparation_is_read_only_and_confirmation_clears_all_map_state(self):
        self.ingest()
        original_files = sorted((self.run / 'evidence').glob('frame-*.npz'))
        generation = self.app.map_generation
        fleet = Mock()
        self.app.fleet = fleet
        plan = self.prepare_reset()
        self.assertEqual(plan['run_ids'], ['current'])
        self.assertTrue(all(p.exists() for p in original_files))
        self.assertEqual(len(self.app.backend.keyframes), 2)
        stale = observations(1, self.config['robots'])[0]
        self.app.offer(stale)
        self.app.command({'action': 'confirm_storage', 'token': plan['token']})
        self.assertFalse(any(p.exists() for p in original_files))
        self.assertEqual(len(self.app.backend.keyframes), 0)
        self.assertEqual(self.app.backend.constraints, {})
        self.assertEqual(self.app.backend.entity_links, {})
        self.assertEqual(self.app.scheduler.pending, {})
        self.assertNotEqual(self.app.map_generation, generation)
        self.assertFalse(self.app.offer(dict(stale, seq=99)))
        fleet.reset_maps.assert_called_once()
        self.app.fleet = None
        for row in self.app.snapshot()['robots']:
            self.assertIsNone(row['shared_evidence_age_s'])
            self.assertIsNone(row['local_map_update_age_s'])
        self.ingest(2)
        self.assertEqual(len(self.app.backend.keyframes), 2)
        with self.assertRaises(ValueError):
            self.app.command({'action': 'confirm_storage', 'token': plan['token']})

    def test_capacity_does_not_refresh_shared_evidence_timer(self):
        self.app.backend.config['max_keyframes'] = 1
        self.ingest()
        rows = {r['robot_id']: r for r in self.app.snapshot()['robots']}
        self.assertIsNotNone(rows['rover_a']['shared_evidence_age_s'])
        self.assertIsNone(rows['rover_b']['shared_evidence_age_s'])
        self.assertIsNotNone(rows['rover_b']['local_map_update_age_s'])
        self.assertEqual(rows['rover_b']['mapping_status'], 'capacity_reached')

    def test_only_explicit_selected_inactive_recording_is_deleted(self):
        for name in ('old', 'keep'):
            with StorageManager(self.root, self.root / name, min_free_bytes=0):
                pass
        plan = self.app.command({'action': 'prepare_storage', 'operation': 'delete_runs',
                                 'run_ids': ['old']})['plan']
        self.assertTrue((self.root / 'old').exists())
        self.app.command({'action': 'confirm_storage', 'token': plan['token']})
        self.assertFalse((self.root / 'old').exists())
        self.assertTrue((self.root / 'keep').exists())
        self.assertTrue(self.run.exists())
        with self.assertRaises(ValueError):
            self.app.command({'action': 'prepare_storage', 'operation': 'delete_runs', 'run_ids': ['current']})

    def test_read_only_replay_refuses_prepare_and_confirm(self):
        self.app.replay = True
        for command in ({'action': 'prepare_storage', 'operation': 'rebuild_current'},
                        {'action': 'confirm_storage', 'token': 'invalid'}):
            with self.assertRaises(ValueError):
                self.app.command(command)
        self.app.replay = False

    def test_rebuild_waits_for_inflight_processing_and_does_not_leak_it(self):
        self.ingest()
        plan = self.prepare_reset()
        entered, release = threading.Event(), threading.Event()
        original = self.app.backend.ingest
        def slow_ingest(obs):
            entered.set()
            self.assertTrue(release.wait(5))
            return original(obs)
        self.app.backend.ingest = slow_ingest
        self.app.offer(observations(1, self.config['robots'])[0])
        process = threading.Thread(target=self.app.process_one)
        process.start()
        self.assertTrue(entered.wait(5))
        outcomes = []
        def reset():
            try:
                outcomes.append(self.app.command({'action': 'confirm_storage', 'token': plan['token']}))
            except Exception as exc:
                outcomes.append(exc)
        rebuild = threading.Thread(target=reset)
        rebuild.start()
        time.sleep(.02)
        self.assertEqual(outcomes, [])
        release.set()
        process.join(5); rebuild.join(5)
        self.assertFalse(process.is_alive() or rebuild.is_alive())
        self.assertEqual(len(outcomes), 1)
        self.assertIsInstance(outcomes[0], dict, outcomes)
        self.assertEqual(len(self.app.backend.keyframes), 0)

    def test_invalid_storage_configuration_is_rejected(self):
        for storage in ({'max_total_bytes': -1}, {'max_runs': True}, {'min_free_bytes': float('nan')},
                        {'unrestricted_delete': True}):
            with self.assertRaises(ValueError):
                validate_config({'storage': storage})

    def test_backend_construction_failure_does_not_delete_evidence(self):
        self.ingest()
        original = self.app.backend
        files = sorted((self.run / 'evidence').glob('frame-*.npz'))
        plan = self.prepare_reset()
        with patch('crossmaps_collaborative.Backend', side_effect=RuntimeError('construction failed')):
            with self.assertRaisesRegex(RuntimeError, 'construction failed'):
                self.app.command({'action': 'confirm_storage', 'token': plan['token']})
        self.assertIs(self.app.backend, original)
        self.assertTrue(all(p.exists() for p in files))

    def test_local_reset_failure_keeps_empty_graph_and_blocks_processing(self):
        self.ingest()
        old_backend = self.app.backend
        fleet = Mock()
        fleet.reset_maps.side_effect = RuntimeError('local cleanup failed')
        self.app.fleet = fleet
        plan = self.prepare_reset()
        result = self.app.command({'action': 'confirm_storage', 'token': plan['token']})
        self.assertTrue(result['result']['reset_applied'])
        self.assertIn('local cleanup failed', result['result']['warning'])
        self.assertIsNot(self.app.backend, old_backend)
        self.assertEqual(self.app.backend.keyframes, {})
        self.app.offer(observations(2, self.config['robots'])[0])
        self.assertFalse(self.app.process_one())
        for row in self.app.snapshot()['robots']:
            self.assertEqual(row['status'], 'processing_error')
            self.assertEqual(row['mapping_status'], 'reset_failed')
            self.assertIn('local cleanup failed', row['last_error'])
        self.assertIsNone(self.app.image('rover_a', 'segmentation'))
        fleet.reset_maps.side_effect = None
        plan = self.prepare_reset()
        self.app.command({'action': 'confirm_storage', 'token': plan['token']})
        self.app.fleet = None
        self.ingest(3)
        self.assertEqual(len(self.app.backend.keyframes), 2)

    def test_checkpoint_failure_after_reset_does_not_resurrect_old_state(self):
        self.ingest()
        plan = self.prepare_reset()
        with patch('collab_backend.Backend.command', side_effect=OSError('disk unavailable')):
            result = self.app.command({'action': 'confirm_storage', 'token': plan['token']})
        self.assertTrue(result['result']['reset_applied'])
        self.assertIn('empty checkpoint', result['result']['warning'])
        self.assertEqual(self.app.backend.keyframes, {})
        self.assertEqual(list((self.run / 'evidence').glob('frame-*.npz')), [])

    def test_busy_shutdown_keeps_recording_ownership_lock(self):
        self.app.worker = Mock()
        self.app.worker.is_alive.return_value = True
        with patch.object(self.storage, 'close', wraps=self.storage.close) as close:
            self.app.close()
            close.assert_not_called()
        with self.assertRaises((ValueError, RuntimeError)):
            StorageManager(self.root, self.run, min_free_bytes=0)
        self.storage.close()  # Test worker is a mock, not an actual writer.


if __name__ == '__main__':
    unittest.main()
