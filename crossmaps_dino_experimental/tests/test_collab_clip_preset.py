"""Fresh encoder trials must never reinterpret stored descriptors."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from collab_config import configure_shared_clip, validate_config
from crossmaps_collaborative import main


class ClipPresetTests(unittest.TestCase):
    mobile = ("MobileCLIP2-S2", "dfndr2b")
    baseline = ("ViT-L-14", "openai")

    def check(self, *args):
        output = StringIO()
        with patch('crossmaps_collaborative.check_environment') as preflight, redirect_stdout(output):
            self.assertEqual(main([*args, '--check']), 0)
        clip = json.loads(output.getvalue())['clip']
        self.assertFalse(clip['weights_checked'])
        shared = preflight.call_args.args[0]['perception']['parameters']
        return ((clip['model_name'], clip['pretrained']),
                (shared['clip_model_name'], shared['clip_pretrained']))

    def test_packaged_trial_and_explicit_baseline_reach_shared_settings(self):
        self.assertEqual(self.check(), (self.mobile, self.mobile))
        self.assertEqual(self.check('--clip-preset', 'vit-l-14'), (self.baseline, self.baseline))

    def test_partial_custom_profile_keeps_old_defaults_unless_overridden(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'custom.json'
            path.write_text('{}')
            self.assertEqual(self.check('--config', str(path)), (self.baseline, self.baseline))
            self.assertEqual(self.check('--config', str(path), '--clip-preset', 'mobileclip2-s2'),
                             (self.mobile, self.mobile))
            self.assertEqual(path.read_text(), '{}')

    def test_presets_cannot_change_recorded_runs_even_during_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = Path(tmp) / 'config.json'
            saved.write_text('{}')
            for mode in (('--resume', '--run-dir', tmp), ('--replay', tmp)):
                with self.subTest(mode=mode), patch('crossmaps_collaborative.owned_run_directory',
                                                    return_value=Path(tmp)):
                    self.assertEqual(self.check(*mode), (self.baseline, self.baseline))
                    error = StringIO()
                    with redirect_stderr(error), self.assertRaises(SystemExit) as refusal:
                        main([*mode, '--clip-preset', 'mobileclip2-s2', '--check'])
                    self.assertEqual(refusal.exception.code, 2)
                    self.assertIn('saved runs retain their recorded encoder', error.getvalue())
            self.assertEqual(saved.read_text(), '{}')

    def test_per_rover_pins_cannot_silently_override_explicit_preset(self):
        config = validate_config({})
        config['robots'][1]['parameters']['clip_pretrained'] = 'openai'
        with self.assertRaisesRegex(ValueError, 'conflicts with rover_b'):
            configure_shared_clip(config, 'mobileclip2-s2')
        self.assertEqual(config['perception']['parameters']['clip_model_name'], self.baseline[0])
        config['robots'][1]['parameters']['clip_pretrained'] = 'dfndr2b'
        self.assertEqual(configure_shared_clip(config, 'mobileclip2-s2')['clip_pretrained'], 'dfndr2b')

    def test_check_reports_effective_shared_rover_override(self):
        # A custom profile can legitimately pin *both* rovers to the same encoder.
        robots = [{'id': f'rover_{n}', 'domain': n, 'parameters': {
            'clip_model_name': 'ViT-B-32', 'clip_pretrained': 'openai'}} for n in (0, 1)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'custom.json'
            path.write_text(json.dumps({'robots': robots}))
            reported, _ = self.check('--config', str(path))
            self.assertEqual(reported, ('ViT-B-32', 'openai'))


if __name__ == '__main__':
    unittest.main()
