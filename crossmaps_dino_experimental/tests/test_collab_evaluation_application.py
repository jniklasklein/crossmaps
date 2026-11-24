"""Evaluation stays downstream of mapping and scoped to the current recording."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from collab_config import validate_config
from collab_demo import observations
from collab_storage import StorageManager
from crossmaps_collaborative import FleetApplication, main


class EvaluationApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.storage = StorageManager(self.root, self.root/'current', min_free_bytes=0)
        self.config = validate_config({'backend': {'verifications_per_keyframe': 0}})
        self.evaluator = Mock()
        self.evaluator.snapshot.return_value = dict(enabled=True, schema=1, status='ready', marker=17)
        self.app = FleetApplication(self.config, self.root/'current', demo=True,
                                    storage=self.storage, evaluation=self.evaluator)
        self.addCleanup(self.app.close)

    def test_truth_evaluation_cannot_change_observation_or_backend(self):
        packet = observations(0, self.config['robots'])[0]
        keys = set(packet)
        self.app.offer(packet)
        self.app.process_one()
        before = json.dumps(self.app.backend.snapshot(), sort_keys=True)
        self.app._refresh_evaluation()
        self.assertEqual(before, json.dumps(self.app.backend.snapshot(), sort_keys=True))
        self.assertEqual(set(packet), keys)
        self.evaluator.snapshot.assert_called_once_with(self.app.backend)
        self.assertEqual(self.app.snapshot()['simulation_evaluation']['marker'], 17)
        path = self.root/'current/evidence/evaluation.json'
        self.assertEqual(json.loads(path.read_text())['marker'], 17)
        self.app._refresh_evaluation()
        self.assertEqual(self.evaluator.snapshot.call_count, 1)

    def test_failure_is_visible_and_does_not_stop_mapping(self):
        self.evaluator.snapshot.side_effect = ValueError('bad truth')
        self.app._refresh_evaluation()
        self.assertEqual(self.app.snapshot()['simulation_evaluation']['status'], 'invalid')
        self.app.offer(observations(0, self.config['robots'])[0])
        self.assertTrue(self.app.process_one())

    def test_rebuild_clears_report_and_evaluation_gauge(self):
        self.app._refresh_evaluation()
        plan = self.app.command(dict(action='prepare_storage', operation='rebuild_current'))['plan']
        self.app.command(dict(action='confirm_storage', token=plan['token']))
        self.evaluator.reset.assert_called_once()
        self.assertIsNone(self.app.snapshot()['simulation_evaluation'])
        self.assertFalse((self.root/'current/evidence/evaluation.json').exists())

    def test_physical_profile_cannot_enable_truth(self):
        with patch('crossmaps_collaborative.check_environment') as check, patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                main(['--simulation-truth', '/tmp/truth.json', '--simulation-scene', '/tmp/scene.json'])
            check.assert_not_called()

    def test_both_sidecar_arguments_are_required(self):
        with patch('crossmaps_collaborative.check_environment') as check, patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                main(['--simulation-truth', '/tmp/truth.json'])
            check.assert_not_called()


if __name__ == '__main__':
    unittest.main()
