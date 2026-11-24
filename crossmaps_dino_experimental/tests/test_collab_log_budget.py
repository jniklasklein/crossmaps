import os
from pathlib import Path
import tempfile
import unittest

from collab_log_budget import ScopedLogBudget


class ScopedLogBudgetTests(unittest.TestCase):
    def test_append_writer_continues_after_tail_compaction_without_holes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'console.log'
            with path.open('ab', buffering=0) as writer:
                writer.write(b'old line\n'*400 + b'RECENT-MARKER\n')
                budget = ScopedLogBudget(root, max_file_bytes=1024, max_total_bytes=4096)
                report = budget.tick(force=True)
                self.assertLessEqual(path.stat().st_size, 1024)
                self.assertIn(b'RECENT-MARKER', path.read_bytes())
                writer.write(b'NEXT-LINE\n')
                self.assertTrue(path.read_bytes().endswith(b'NEXT-LINE\n'))
                self.assertNotIn(b'\0', path.read_bytes())
                self.assertEqual(report['compactions'], 1)
                self.assertGreater(report['dropped_bytes'], 0)

    def test_total_budget_covers_ros_subfolders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('one.log', 'ros/two.log', 'ros_logs/node/three.log'):
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'x'*900)
            report = ScopedLogBudget(root, max_file_bytes=2048, max_total_bytes=1600).tick(force=True)
            self.assertEqual(report['files'], 3)
            self.assertLessEqual(report['bytes'], 1600)

    def test_links_world_evidence_and_nonlogs_untouched(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as external:
            root, other = Path(directory), Path(external)
            secret = other/'outside.log'; secret.write_bytes(b'outside'*1000)
            (root/'linked.log').symlink_to(secret)
            (root/'ros').symlink_to(other, target_is_directory=True)
            os.link(secret, root/'hardlinked.log')
            protected = []
            for name in ('world/model.log', 'evidence/frame.npz', 'ros_logs/data.npz', 'asset.txt'):
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'protected'*1000); protected.append(path)
            report = ScopedLogBudget(root, max_file_bytes=1024, max_total_bytes=2048).tick(force=True)
            self.assertEqual(report['files'], 0)
            self.assertEqual(secret.read_bytes(), b'outside'*1000)
            for path in protected:
                self.assertEqual(path.read_bytes(), b'protected'*1000)

    def test_root_replacement_and_root_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory); root = outer/'owned'; root.mkdir()
            budget = ScopedLogBudget(root)
            root.rename(outer/'old'); root.mkdir()
            with self.assertRaises(RuntimeError):
                budget.tick(force=True)
            link = outer/'alias'; link.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                ScopedLogBudget(link)

    def test_tick_is_rate_limited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); log = root/'one.log'
            clock = [1.]
            budget = ScopedLogBudget(root, max_file_bytes=1024, max_total_bytes=4096, clock=lambda: clock[0])
            budget.tick(); log.write_bytes(b'x'*2048)
            budget.tick(); self.assertEqual(log.stat().st_size, 2048)
            clock[0] = 2.
            budget.tick(); self.assertLessEqual(log.stat().st_size, 1024)


if __name__ == '__main__':
    unittest.main()
