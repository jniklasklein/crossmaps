"""Long-run evidence admission, measured quality improvements and exact quotas."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from collab_backend import Backend


def observation(seq=1, robot='A', *, seconds=None, seed=14, blur=0., coverage=1., x=0.):
    rng = np.random.default_rng(seed)
    rgb = rng.integers(30, 220, (240, 320, 3), dtype=np.uint8)
    if blur:
        rgb = cv2.GaussianBlur(rgb, (0, 0), blur)
    depth = np.full((240, 320), 2., np.float32)
    depth[int(240*coverage):] = 0
    pose = np.eye(4); pose[0, 3] = x
    stamp = int((seq*10 if seconds is None else seconds)*1e9)
    return dict(robot_id=robot, session_id='session', seq=seq, stamp_ns=stamp,
                rgb_stamp_ns=stamp, rgb=rgb, depth=depth, K=np.array([[250., 0, 160], [0, 250, 120], [0, 0, 1]]),
                T_local_camera=pose, calibration_id='camera', feature_space_id='features',
                points_camera=np.array([[0, 0, 2], [.1, 0, 2], [0, .1, 2]], np.float32),
                colors=np.ones((3, 3), np.uint8), surfaces=[])


class RetentionTests(unittest.TestCase):
    def backend(self, **config):
        return Backend(dict(verifications_per_keyframe=0, **config))

    def fill_bootstrap(self, backend, **kwargs):
        for seq in range(1, 7):
            backend.ingest(observation(seq, **kwargs))

    def test_two_parked_rovers_remain_bounded_over_hours(self):
        backend = self.backend()
        for seq in range(1, 127):
            for robot in ('A', 'B'):
                backend.ingest(observation(seq, robot, seconds=seq*60.))
        state = backend.snapshot()
        self.assertEqual(len(backend.keyframes), 12)
        self.assertEqual(len(backend.submaps), 6)  # sparse test input crosses90s per2captures
        self.assertIsNone(state['stats']['capacity_reason'])
        self.assertEqual(state['stats']['evidence_capacity']['counters']['redundant'], 240)
        for robot in state['robots']:
            self.assertEqual(robot['evidence']['outcome'], 'redundant')
            self.assertFalse(robot['trajectory']['tracking_lost'])
            self.assertEqual(robot['trajectory']['latest']['stamp_ns'], int(126*60e9))

    def test_bootstrap_is_time_spaced_and_six_separate_captures(self):
        backend = self.backend()
        for seq in range(1, 51):
            backend.ingest(observation(seq, seconds=seq))
        self.assertEqual(len(backend.keyframes), 6)
        self.assertEqual([f['stamp_ns']/1e9 for f in backend.keyframes.values()], [1, 9, 17, 25, 33, 41])

    def test_motion_calibration_and_feature_space_changes_still_admit(self):
        backend = self.backend()
        self.fill_bootstrap(backend)
        backend.ingest(observation(7, x=.2))
        changed = observation(8, x=.2); changed['calibration_id'] = 'new_camera'
        backend.ingest(changed)
        changed = observation(9, x=.2); changed['feature_space_id'] = 'new_features'
        backend.ingest(changed)
        self.assertEqual(len(backend.keyframes), 9)
        self.assertGreaterEqual(len(backend.submaps), 3)

    def test_changed_image_or_depth_admits_but_old_view_does_not(self):
        backend = self.backend()
        self.fill_bootstrap(backend)
        backend.ingest(observation(7, seed=29))
        depth_change = observation(8); depth_change['depth'][:120] = 2.4
        backend.ingest(depth_change)
        self.assertEqual(len(backend.keyframes), 8)
        backend.ingest(observation(9))
        self.assertEqual(len(backend.keyframes), 8)

    def test_better_depth_before_bootstrap_interval_is_retained(self):
        backend = self.backend()
        backend.ingest(observation(1, seconds=1, coverage=.5))
        backend.ingest(observation(2, seconds=2, coverage=.8))
        self.assertEqual(len(backend.keyframes), 2)
        self.assertEqual(backend.snapshot()['robots'][0]['evidence']['reason'], 'depth_coverage_improved')

    def test_improved_depth_frontier_not_quality_oscillation(self):
        backend = self.backend()
        self.fill_bootstrap(backend, coverage=.6)
        backend.ingest(observation(7, coverage=.9))
        self.assertEqual(len(backend.keyframes), 7)
        for seq in range(8, 20):
            backend.ingest(observation(seq, coverage=.6 if seq%2 else .9))
        self.assertEqual(len(backend.keyframes), 7)

    def test_new_depth_support_with_equal_coverage_is_retained_once(self):
        backend = self.backend()
        self.fill_bootstrap(backend, coverage=.5)
        changed = observation(7, coverage=.5)
        changed['depth'] = np.flipud(changed['depth']).copy()
        backend.ingest(changed)
        self.assertEqual(len(backend.keyframes), 7)
        self.assertEqual(backend.snapshot()['robots'][0]['evidence']['reason'], 'depth_support_improved')
        changed.update(seq=8, stamp_ns=int(80e9), rgb_stamp_ns=int(80e9))
        backend.ingest(changed)
        self.assertEqual(len(backend.keyframes), 7)
        backend.ingest(observation(9, coverage=.5))
        self.assertEqual(len(backend.keyframes), 7)

    def test_sharper_same_view_is_retained_once(self):
        backend = self.backend()
        self.fill_bootstrap(backend, blur=3.)
        backend.ingest(observation(7, blur=1.))
        self.assertEqual(len(backend.keyframes), 7)
        self.assertEqual(backend.snapshot()['robots'][0]['evidence']['reason'], 'image_detail_improved')
        for seq in range(8, 20):
            backend.ingest(observation(seq, blur=3. if seq%2 else 1.))
        self.assertEqual(len(backend.keyframes), 7)

    def test_exposure_and_small_pixel_noise_do_not_retain(self):
        backend = self.backend()
        self.fill_bootstrap(backend)
        changed = observation(7)
        changed['rgb'] = np.clip(changed['rgb'].astype(float)*1.05+3, 0, 255).astype(np.uint8)
        backend.ingest(changed)
        noisy = observation(8)
        noise = np.random.default_rng(22).normal(0, .5, noisy['rgb'].shape)
        noisy['rgb'] = np.clip(noisy['rgb'].astype(float)+noise, 0, 255).astype(np.uint8)
        backend.ingest(noisy)
        self.assertEqual(len(backend.keyframes), 6)

    def test_capacity_preserves_graph_but_redundant_view_keeps_trajectory(self):
        backend = self.backend(max_keyframes=6)
        self.fill_bootstrap(backend)
        retained = list(backend.keyframes)
        backend.ingest(observation(7, x=.3))
        self.assertEqual(backend.snapshot()['robots'][0]['evidence']['reason'], 'max_keyframes')
        backend.ingest(observation(8))
        state = backend.snapshot()
        self.assertEqual(list(backend.keyframes), retained)
        self.assertEqual(state['robots'][0]['status'], 'observing')
        self.assertFalse(state['robots'][0]['trajectory']['tracking_lost'])

    def test_exact_byte_quota_never_commits_oversized_frame(self):
        sample = self.backend()
        sample.ingest(observation(1)); one = sample.store_bytes
        sample.ingest(observation(2)); both = sample.store_bytes
        with tempfile.TemporaryDirectory() as directory:
            backend = Backend(dict(verifications_per_keyframe=0, max_store_bytes=both-1), directory)
            backend.ingest(observation(1)); backend.ingest(observation(2))
            self.assertEqual(backend.store_bytes, one)
            self.assertEqual(len(backend.keyframes), 1)
            self.assertEqual(len(list(Path(directory).glob('frame-*.npz'))), 1)
            self.assertEqual(backend.snapshot()['robots'][0]['evidence']['reason'], 'max_store_bytes')
        exact = self.backend(max_store_bytes=both)
        exact.ingest(observation(1)); exact.ingest(observation(2))
        self.assertEqual(exact.store_bytes, both)
        self.assertEqual(len(exact.keyframes), 2)

    def test_external_storage_budget_and_write_failure_precede_graph_mutation(self):
        sizes = []
        def deny(size):
            sizes.append(size)
            return False, 'Archive budget exhausted'
        backend = Backend(dict(verifications_per_keyframe=0), evidence_budget=deny)
        backend.ingest(observation())
        self.assertGreater(sizes[0], 1024)
        self.assertFalse(backend.keyframes)
        self.assertFalse(backend.submaps)
        self.assertEqual(backend.store_bytes, 0)
        backend.evidence_budget = None
        with patch.object(backend, '_persist_frame', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                backend.ingest(observation(2))
        self.assertFalse(backend.keyframes)
        self.assertFalse(backend.submaps)

    def test_restore_recovers_quality_and_old_sources_without_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = Backend(dict(verifications_per_keyframe=0), directory)
            self.fill_bootstrap(backend)
            for source in backend.sessions.values():
                source.pop('evidence')
            backend.command(dict(action='save'))
            restored = Backend(store_dir=directory)
            restored.ingest(observation(7))
            self.assertEqual(len(restored.keyframes), 6)
            self.assertEqual(restored.snapshot()['robots'][0]['evidence']['outcome'], 'redundant')

    def test_unreferenced_crash_payloads_are_preserved_and_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = Backend(dict(verifications_per_keyframe=0), directory)
            backend.ingest(observation())
            original_bytes = backend.store_bytes
            orphan = Path(directory)/('frame-'+'f'*20+'.npz')
            orphan.write_bytes(b'orphan-evidence'*100)
            resumed = Backend(dict(max_store_bytes=original_bytes+orphan.stat().st_size), directory)
            self.assertEqual(resumed.store_bytes, original_bytes+orphan.stat().st_size)
            resumed.ingest(observation(2))
            self.assertEqual(len(resumed.keyframes), 1)
            self.assertEqual(orphan.read_bytes(), b'orphan-evidence'*100)

    def test_failed_payload_write_removes_only_its_unfinished_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = Backend(dict(verifications_per_keyframe=0), directory)
            backend.ingest(observation())
            previous = {p.name:p.read_bytes() for p in Path(directory).glob('frame-*.npz')}
            with patch('collab_backend.os.fsync', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    backend.ingest(observation(2))
            self.assertFalse(list(Path(directory).glob('*.tmp')))
            self.assertEqual({p.name:p.read_bytes() for p in Path(directory).glob('frame-*.npz')}, previous)


if __name__ == '__main__':
    unittest.main()
