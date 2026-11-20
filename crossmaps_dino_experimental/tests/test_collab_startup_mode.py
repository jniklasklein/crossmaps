"""Check live startup policy without connecting to hardware."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from collab_config import ROOT, load_config, validate_config
from crossmaps_collaborative import FleetApplication, main


class StartupModeTests(unittest.TestCase):
    def check(self, *args):
        output = StringIO()
        with patch('crossmaps_collaborative.check_environment'), redirect_stdout(output):
            self.assertEqual(main([*args, '--check']), 0)
        return json.loads(output.getvalue())['startup_alignment']

    def test_new_live_auto_but_shadow_mode_is_available(self):
        self.assertFalse(validate_config({})['backend']['shadow_mode'])
        self.assertFalse(load_config(ROOT / 'config' / 'collaborative_two_rovers.json')['backend']['shadow_mode'])
        self.assertEqual(self.check(), 'automatic')
        self.assertEqual(self.check('--shadow-mode'), 'shadow')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'shadow.json'
            path.write_text(json.dumps({'backend': {'shadow_mode': True}}))
            self.assertEqual(self.check('--config', str(path)), 'shadow')

    def test_default_demo_remains_review_first(self):
        self.assertEqual(self.check('--demo'), 'shadow')
        explicit = ROOT / 'config' / 'collaborative_two_rovers.json'
        self.assertEqual(self.check('--demo', '--config', str(explicit)), 'automatic')

    def test_resume_and_replay_honor_checkpoint_not_startup_mode(self):
        for initial in (False, True):
            with self.subTest(initial=initial), tempfile.TemporaryDirectory() as tmp:
                config = validate_config({'backend': {'shadow_mode': initial}})
                app = FleetApplication(config, tmp, demo=True)
                try:
                    self.assertIs(app.snapshot()['shadow_mode'], initial)
                    app.command({'action': 'set_shadow_mode', 'enabled': not initial})
                finally:
                    app.close()
                for replay in (True, False):
                    app = FleetApplication(config, tmp, demo=not replay, replay=replay)
                    try:
                        self.assertIs(app.snapshot()['shadow_mode'], not initial)
                    finally:
                        app.close()

    def test_startup_flag_cannot_override_saved_session(self):
        for args in (('--resume', '--run-dir', 'existing'), ('--replay', 'existing')):
            with self.subTest(args=args), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                main(['--shadow-mode', *args])
