"""End-to-end orchestration with real backend, no ROS or model downloads."""
from copy import deepcopy
import json
from pathlib import Path
import queue
import tempfile
import time
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from collab_config import validate_config
from collab_demo import observations
from crossmaps_collaborative import FleetApplication


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.config = validate_config({'backend': {'verifications_per_keyframe': 0}})
        self.app = FleetApplication(self.config, self.directory.name, demo=True)
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(self.app.close)

    def pair(self, step=0):
        packets = observations(step, self.config['robots'])
        for packet in packets:
            self.assertTrue(self.app.offer(packet))
        for _ in packets:
            self.assertTrue(self.app.process_one())
        return packets

    def test_two_sources_own_local_maps_until_supplied_alignment(self):
        self.pair()
        state = self.app.snapshot()
        self.assertEqual(state['mode'], 'synthetic_demo')
        self.assertEqual([r['aligned'] for r in state['robots']], [True, False])
        T = np.eye(4); T[:3, 3] = [1.5, .5, 0]
        self.app.command(dict(action='anchor', robot_id='rover_b', reference_robot_id='rover_a', T_reference_local=T.tolist()))
        self.assertTrue(all(r['aligned'] for r in self.app.snapshot()['robots']))
        self.app.command(dict(action='clear_anchor', robot_id='rover_b'))
        self.assertFalse(self.app.snapshot()['robots'][1]['aligned'])

    def test_preview_advances_before_perception_and_duplicate_does_not_refresh(self):
        packets = self.pair()
        before = self.app.snapshot()['robots'][0]
        packet = observations(1, self.config['robots'])[0]
        self.app.offer(packet)
        after = self.app.snapshot()['robots'][0]
        self.assertGreater(after['last_input_stamp_ns'], before['last_processed_stamp_ns'])
        self.assertEqual(after['last_processed_stamp_ns'], before['last_processed_stamp_ns'])
        jpeg, mime = self.app.image('rover_a', 'rgb')
        self.assertEqual(mime, 'image/jpeg')
        self.assertIsNotNone(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR))
        self.assertFalse(self.app.offer(packets[0]))
        self.assertEqual(self.app.snapshot()['robots'][0]['last_input_stamp_ns'], after['last_input_stamp_ns'])

    def test_new_session_cannot_inherit_old_alignment_while_waiting(self):
        self.pair()
        new = observations(1, self.config['robots'])[0]
        new['session_id'] = 'restarted'
        self.app.offer(new)
        state = self.app.snapshot()['robots'][0]
        self.assertEqual(state['mapping_status'], 'new_session_pending')
        self.assertFalse(state['aligned'])
        self.assertIsNone(state['T_world_local'])
        self.assertIsNone(state['map_observation_age_s'])

    def test_tf_missing_preview_does_not_advance_evidence_or_mapped_time(self):
        self.pair()
        packet = observations(1, self.config['robots'])[0]
        old = self.app.snapshot()['robots'][0]
        mapped = self.app.telemetry['rover_a']['last_mapped_monotonic']
        self.app.offer(dict(packet, preview_only=True, T_local_camera=None,
                            integration_allowed=False))
        self.assertTrue(self.app.process_one())
        new = self.app.snapshot()['robots'][0]
        self.assertEqual(new['status'], 'waiting_for_tf')
        self.assertEqual(old['frames'], new['frames'])
        self.assertEqual(self.app.telemetry['rover_a']['last_mapped_monotonic'], mapped)
        self.app.offer(dict(packet, seq=packet['seq']+1))
        self.assertTrue(self.app.process_one())
        self.assertEqual(self.app.snapshot()['robots'][0]['frames'], old['frames']+1)

    def test_acquisition_feeds_and_recovery_are_independent_of_worker(self):
        reports = queue.Queue()
        self.app.adapters['rover_a'] = dict(process=Mock(), health=reports, output=queue.Queue())
        self.addCleanup(self.app.adapters.clear)
        now = time.monotonic()
        reports.put(dict(status='waiting_for_tf', last_error='late TF',
                         depth_preview_jpeg=b'depth bytes', depth_preview_monotonic=now-1))
        self.app.poll_inputs()
        self.assertEqual(self.app.image('rover_a', 'depth')[0], b'depth bytes')
        state = self.app.snapshot()['robots'][0]
        self.assertGreaterEqual(state['image_age_s']['depth'], 1.)
        self.assertNotIn('depth_preview_jpeg', state['health'])
        self.app.telemetry['rover_a'].update(status='processing_error', last_error='model unavailable')
        reports.put(dict(status='receiving', last_error=None))
        self.app.poll_inputs()
        recovered = self.app.snapshot()['robots'][0]
        self.assertIsNone(recovered['health']['last_error'])
        self.assertEqual(recovered['last_error'], 'model unavailable')

    def test_repeated_camera_stamp_is_reported_stale_despite_fresh_callbacks(self):
        self.pair()
        self.app.telemetry['rover_a']['health'] = dict(rgb_advanced_monotonic=time.monotonic()-20,
                                                     last_rgb_monotonic=time.monotonic())
        state = self.app.snapshot()['robots'][0]
        self.assertGreater(state['frame_age_s'], 19)
        self.assertEqual(state['status'], 'input_stale')

    def test_snapshot_is_detached_and_strict_json(self):
        self.pair()
        state = self.app.snapshot()
        json.dumps(state, allow_nan=False)
        state['robots'][0]['status'] = 'tampered'
        state['submaps'].clear()
        fresh = self.app.snapshot()
        self.assertNotEqual(fresh['robots'][0]['status'], 'tampered')
        self.assertEqual(len(fresh['submaps']), 2)

    def test_processing_error_preserves_preview_and_other_source_can_continue(self):
        packet = observations(0, self.config['robots'])[0]
        packet['K'] = np.zeros((3, 3))
        self.app.offer(packet)
        self.assertFalse(self.app.process_one())
        self.assertEqual(self.app.snapshot()['robots'][0]['status'], 'processing_error')
        self.assertIsNotNone(self.app.image('rover_a', 'rgb'))
        self.app.offer(observations(0, self.config['robots'])[1])
        self.assertTrue(self.app.process_one())
        self.assertIsNone(self.app.snapshot()['active_robot'])

    def test_recorded_replay_no_workers_no_source_writes_and_explicit_ages(self):
        self.pair()
        self.app.close()
        folder = Path(self.directory.name)
        before = {p.name: p.read_bytes() for p in (folder/'evidence').iterdir()}
        replay = FleetApplication(self.config, folder, replay=True)
        self.addCleanup(replay.close)
        self.assertIsNone(replay.worker)
        self.assertIsNone(replay.fleet)
        self.assertEqual(replay.snapshot()['mode'], 'recorded_replay')
        self.assertTrue(all(r['status'] == 'recorded' for r in replay.snapshot()['robots']))
        self.assertIsNone(replay.snapshot()['robots'][0]['image_age_s']['rgb'])
        self.assertIsNotNone(replay.image('rover_a', 'rgb'))
        self.assertIsNotNone(replay.image('rover_a', 'depth'))
        self.assertIsNone(replay.image('rover_a', 'sam'))
        with self.assertRaises(ValueError):
            replay.command({'action': 'save'})
        with self.assertRaises(ValueError):
            replay.offer(observations(1, self.config['robots'])[0])
        replay.command({'action': 'set_shadow_mode', 'enabled': False})
        replay.close()
        self.assertEqual(before, {p.name: p.read_bytes() for p in (folder/'evidence').iterdir()})


if __name__ == '__main__':
    unittest.main()
