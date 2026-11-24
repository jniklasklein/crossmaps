"""Temporary-directory-only retention, quota and destructive-boundary tests."""
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collab_storage as storage


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'collaboration'
        self.root.mkdir()

    def manager(self, name='current', **kwargs):
        result = storage.StorageManager(self.root, self.root/name, min_free_bytes=0, **kwargs)
        self.addCleanup(result.close)
        return result

    def evidence(self, name='current', value=b'frame'):
        folder = self.root/name/'evidence'
        folder.mkdir(parents=True, exist_ok=True)
        (folder/('frame-'+'a'*20+'.npz')).write_bytes(value)
        (folder/'state.json').write_text(json.dumps({'schema': 'crossmaps_collaborative_store_v1', 'sessions': {}}))
        return folder

    def legacy(self, name='old', domain=42):
        folder = self.root/name
        folder.mkdir()
        (folder/'config.json').write_text(json.dumps({'central_domain': domain, 'robots': []}))
        self.evidence(name)
        return folder

    def test_inventory_and_explicit_cleanup_never_delete_current(self):
        old = self.manager('old')
        self.evidence('old')
        old.close()
        current = self.manager()
        report = current.inventory()
        rows = {r['id']: r for r in report['runs']}
        self.assertTrue(rows['old']['deletable'])
        self.assertFalse(rows['current']['deletable'])
        self.assertFalse(report['auto_delete'])
        with self.assertRaises(storage.StorageError):
            current.prepare_cleanup(['current'])
        preview = current.prepare_cleanup(['old'])
        self.assertTrue((self.root/'old').exists())
        result = current.cleanup(preview['token'])
        self.assertEqual(result['removed'], ['old'])
        self.assertGreater(result['freed_bytes'], 0)
        self.assertFalse(result['recoverable'])
        self.assertTrue((self.root/'current').exists())
        with self.assertRaises(storage.StorageError):
            current.cleanup(preview['token'])

    def test_live_and_replay_locks_protect_recordings(self):
        old = self.manager('old')
        self.evidence('old')
        current = self.manager()
        with self.assertRaises(storage.StorageError):
            current.prepare_cleanup(['old'])
        with self.assertRaises(storage.StorageError):
            self.manager('old', readonly=True)
        old.close()
        replay = self.manager('old', readonly=True)
        with self.assertRaises(storage.StorageError):
            current.prepare_cleanup(['old'])
        replay.close()
        preview = current.prepare_cleanup(['old'])
        reopened = self.manager('old')
        with self.assertRaises(storage.StorageError):
            current.cleanup(preview['token'])
        self.assertTrue((self.root/'old').exists())
        reopened.close()

    def test_readonly_replay_does_not_change_recording_files(self):
        old = self.legacy()
        before = {str(p.relative_to(old)): p.read_bytes() for p in old.rglob('*') if p.is_file()}
        replay = self.manager('old', readonly=True)
        self.assertFalse(replay.check_write(1)[0])
        for method, args in ((replay.prepare_reset, ()), (replay.prepare_cleanup, (['old'],))):
            with self.assertRaises(storage.StorageError):
                method(*args)
        replay.close()
        after = {str(p.relative_to(old)): p.read_bytes() for p in old.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_cleanup_refuses_changed_files_and_validates_all_before_delete(self):
        for name in ('old-a', 'old-b'):
            item = self.manager(name)
            self.evidence(name)
            item.close()
        current = self.manager()
        preview = current.prepare_cleanup(['old-a', 'old-b'])
        (self.root/'old-b'/'new-user-file').write_text('keep me')
        with self.assertRaises(storage.StorageError):
            current.cleanup(preview['token'])
        self.assertTrue((self.root/'old-a').exists())
        self.assertTrue((self.root/'old-b'/'new-user-file').exists())

    def test_invalid_paths_symlinks_and_special_files_are_protected(self):
        current = self.manager()
        for name in ('../outside', '/tmp/outside', '.', '..', 'a/b'):
            with self.subTest(name=name), self.assertRaises(storage.StorageError):
                current.prepare_cleanup([name])
        outside = Path(self.temp.name)/'outside'
        outside.mkdir()
        (outside/'precious').write_text('safe')
        (self.root/'symlink').symlink_to(outside, target_is_directory=True)
        old = self.manager('old')
        old.close()
        (self.root/'old'/'nested').symlink_to(outside, target_is_directory=True)
        for name in ('symlink', 'old'):
            with self.assertRaises(storage.StorageError):
                current.prepare_cleanup([name])
        self.assertFalse(current.check_write(1)[0])
        self.assertEqual((outside/'precious').read_text(), 'safe')

    def test_constructor_rejects_nested_and_symlink_runs(self):
        (self.root/'nested').mkdir()
        with self.assertRaises(storage.StorageError):
            storage.StorageManager(self.root, self.root/'nested'/'child')
        target = Path(self.temp.name)/'outside'
        target.mkdir()
        (self.root/'linked').symlink_to(target)
        with self.assertRaises(storage.StorageError):
            self.manager('linked')

    def test_reset_accepts_new_frames_but_not_new_generation(self):
        current = self.manager()
        evidence = self.evidence()
        (self.root/'current'/'config.json').write_text('{}')
        preview = current.prepare_reset()
        (evidence/('frame-'+'b'*20+'.npz')).write_bytes(b'new frame since preview')
        result = current.reset_current(preview['token'])
        self.assertTrue(result['reset_applied'])
        self.assertFalse(list(evidence.iterdir()))
        self.assertTrue((self.root/'current'/'config.json').exists())
        self.assertGreater(result['freed_bytes'], 0)
        stale = current.prepare_reset()
        evidence.rename(self.root/'current'/'unexpected-old-evidence')
        evidence.mkdir()
        with self.assertRaises(storage.StorageError):
            current.reset_current(stale['token'])

    def test_reset_cleanup_failure_keeps_new_empty_generation_and_retries(self):
        current = self.manager()
        evidence = self.evidence()
        preview = current.prepare_reset()
        with patch.object(storage, '_remove_checked_tree', side_effect=PermissionError('test permission failure')):
            result = current.reset_current(preview['token'])
        self.assertTrue(result['ok'])
        self.assertTrue(result['reset_applied'])
        self.assertIn('pending', result['warning'])
        self.assertFalse(list(evidence.iterdir()))
        pending = list((self.root/'current').glob('.discarded-evidence-*'))
        self.assertEqual(len(pending), 1)
        self.assertGreater(current.inventory(force=True)['current_bytes'], 0)
        preview = current.prepare_reset()
        result = current.reset_current(preview['token'])
        self.assertIsNone(result['warning'])
        self.assertFalse(list((self.root/'current').glob('.discarded-evidence-*')))

    def test_expired_or_wrong_operation_token_cannot_delete(self):
        now = [100.]
        current = self.manager(clock=lambda: now[0])
        self.evidence()
        preview = current.prepare_reset()
        with self.assertRaises(storage.StorageError):
            current.cleanup(preview['token'])
        preview = current.prepare_reset()
        now[0] += 301.
        with self.assertRaises(storage.StorageError):
            current.reset_current(preview['token'])
        self.assertTrue(list((self.root/'current'/'evidence').iterdir()))

    def test_reservations_prevent_concurrent_quota_overspend(self):
        mib = 1024**2
        a = self.manager('a', max_total_bytes=7*mib)
        b = self.manager('b', max_total_bytes=7*mib)
        self.assertEqual(a.check_write(mib), (True, None))
        self.assertFalse(b.check_write(mib)[0])
        a.finish_write()
        self.assertEqual(b.check_write(mib), (True, None))
        b.finish_write()
        self.assertEqual(a.inventory(force=True)['reserved_bytes'], 0)

    def test_disk_reserve_and_managed_run_count_stop_admission_not_delete(self):
        old = self.manager('old')
        old.close()
        current = self.manager(max_runs=1)
        allowed, reason = current.check_write(10)
        self.assertFalse(allowed)
        self.assertIn('count', reason)
        self.assertTrue((self.root/'old').exists())
        current.max_runs = 10
        disk = type('Disk', (), {'free': 100})()
        with patch.object(storage.shutil, 'disk_usage', return_value=disk):
            allowed, reason = current.check_write(10)
        self.assertFalse(allowed)
        self.assertIn('disk', reason)

    def test_legacy_cleanup_requires_known_schema_and_inactive_domain(self):
        self.legacy('old-a', domain=42)
        self.legacy('old-b', domain=42)
        current = self.manager()
        fd = os.open(self.root.parent/'collaboration_domain_42.lock', os.O_RDWR | os.O_CREAT, 0o600)
        self.addCleanup(os.close, fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with self.assertRaises(storage.StorageError):
            current.prepare_cleanup(['old-a'])
        fcntl.flock(fd, fcntl.LOCK_UN)
        plan = current.prepare_cleanup(['old-a', 'old-b'])
        result = current.cleanup(plan['token'])
        self.assertEqual(result['removed'], ['old-a', 'old-b'])

    def test_legacy_current_owned_domain_and_replay_protection(self):
        self.legacy('old', domain=82)
        fd = os.open(self.root.parent/'collaboration_domain_82.lock', os.O_RDWR | os.O_CREAT, 0o600)
        self.addCleanup(os.close, fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = self.manager(owned_domain=82, domain_lock_held=True)
        self.assertTrue(current.prepare_cleanup(['old'])['permanent'])
        review = os.open(self.root.parent/'collaboration_review_8768.lock', os.O_RDWR | os.O_CREAT, 0o600)
        self.addCleanup(os.close, review)
        fcntl.flock(review, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with self.assertRaises(storage.StorageError):
            current.prepare_cleanup(['old'])

    def test_managed_run_opened_by_old_build_still_has_domain_protection(self):
        old = self.manager('old')
        (old.run_dir/'config.json').write_text(json.dumps({'central_domain': 42, 'robots': []}))
        self.evidence('old')
        old.close()
        current = self.manager()
        legacy_lock = os.open(self.root.parent/'collaboration_domain_42.lock', os.O_RDWR | os.O_CREAT, 0o600)
        self.addCleanup(os.close, legacy_lock)
        fcntl.flock(legacy_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with self.assertRaises(storage.StorageError):
            current.prepare_cleanup(['old'])
        fcntl.flock(legacy_lock, fcntl.LOCK_UN)
        self.assertTrue(current.prepare_cleanup(['old'])['permanent'])

    def test_unknown_legacy_user_file_is_never_deleted(self):
        folder = self.legacy()
        (folder/'important.txt').write_text('not application evidence')
        current = self.manager()
        row = next(r for r in current.inventory()['runs'] if r['id'] == 'old')
        self.assertFalse(row['deletable'])
        self.assertIn('Unrecognized files', row['protected_reason'])
        self.assertGreater(row['bytes'], 0)

    def test_partial_cleanup_reports_what_was_removed(self):
        for name in ('old-a', 'old-b'):
            old = self.manager(name)
            self.evidence(name)
            old.close()
        current = self.manager()
        preview = current.prepare_cleanup(['old-a', 'old-b'])
        real_remove = storage._remove_checked_tree
        def remove(parent, name, inode):
            if name == 'old-b':
                raise PermissionError('test deletion failure')
            real_remove(parent, name, inode)
        with patch.object(storage, '_remove_checked_tree', side_effect=remove):
            result = current.cleanup(preview['token'])
        self.assertFalse(result['ok'])
        self.assertEqual(result['removed'], ['old-a'])
        self.assertGreater(result['freed_bytes'], 0)
        self.assertIn('old-b', result['error'])
        self.assertTrue((self.root/'old-b').exists())


if __name__ == '__main__':
    unittest.main()
