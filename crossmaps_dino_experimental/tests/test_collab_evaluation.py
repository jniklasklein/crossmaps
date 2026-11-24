"""Ground-truth scoring cannot change estimated maps or invent associations."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from collab_evaluation import SimulationEvaluator, associate, error, rigid


def pose(x=0., y=0., angle=0.):
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    result = np.eye(4)
    result[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    result[:3, 3] = [x, y, 0]
    return result


def backend(times=(1., 9.), scale=1., b_y=2., b_angle=0.):
    frames, sessions, submaps = {}, {}, {}
    robots = {'rover_a': 'a', 'rover_b': 'b'}
    for robot, source in robots.items():
        keys = []
        for i, tick in enumerate(times):
            key = f'{source}{i}'
            local = pose(x=(tick-times[0])*scale)
            frames[key] = dict(id=key, source_id=source, robot_id=robot, session_id='test',
                               submap_id=source, stamp_ns=int(tick*1e9),
                               T_local_camera=local, T_submap_camera=local.copy())
            keys.append(key)
        sessions[source] = dict(keyframe_ids=keys)
        submaps[source] = dict(aligned=True, T_world_submap=pose(y=b_y, angle=b_angle) if source == 'b' else pose())
    return SimpleNamespace(robots=robots, keyframes=frames, submaps=submaps, sessions=sessions,
                           reference_session='a', constraints={}, _preview_cache=([], {}))


class SimulationEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.truth = self.root/'truth.json'
        self.scene = self.root/'scene.json'
        self.scene.write_text(json.dumps(dict(world='crossmaps_lab', robots={robot: dict(camera_optical=dict(
            translation=[0, 0, 0], quaternion_xyzw=[0, 0, 0, 1])) for robot in ('rover_a', 'rover_b')})))
        self.now = 100.
        self.write_truth()
        self.evaluator = SimulationEvaluator(self.truth, self.scene, clock=lambda: self.now)

    def write_truth(self, *, run='test-run', start=0, end=100, updated=100., valid=True, receipt=100.):
        self.payload = dict(schema=1, run_id=run, updated_unix=updated, valid=valid, robots={})
        if receipt is not None:
            self.payload['last_received_unix'] = {'rover_a': receipt, 'rover_b': receipt}
        for robot in ('rover_a', 'rover_b'):
            self.payload['robots'][robot] = [dict(stamp_ns=i*100_000_000,
                T_world_base=pose(x=i/10+10, y=2 if robot == 'rover_b' else 0).tolist()) for i in range(start, end+1)]
        self.save_truth()

    def save_truth(self):
        self.truth.write_text(json.dumps(self.payload))

    def test_exact_and_interpolated_truth_no_extrapolation(self):
        samples = [(0, pose()), (100_000_000, pose(x=1, angle=90))]
        middle = associate(samples, 50_000_000)
        self.assertAlmostEqual(middle[0, 3], .5)
        self.assertAlmostEqual(error(middle, pose(x=.5, angle=45))[1], 0.)
        self.assertIsNone(associate(samples, -1))
        self.assertIsNone(associate(samples, 100_000_001))
        self.assertIsNone(associate([(0, pose()), (1_000_000_000, pose())], 500_000_000))

    def test_perfect_graph_has_zero_error_despite_unknown_global_origin(self):
        result = self.evaluator.snapshot(backend())
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['coverage']['associated'], 4)
        self.assertAlmostEqual(result['shared']['confirmed']['translation_rmse_m'], 0.)
        self.assertAlmostEqual(result['optimization']['optimized_position_rmse_m'], 0.)
        for row in result['local']:
            self.assertAlmostEqual(row['position_rmse_m'], 0.)
            self.assertEqual(row['rpe_interval_s'], dict(min=8., max=8.))
            self.assertEqual(row['rpe_pairs'], 1)

    def test_wrong_interrover_alignment_is_not_hidden_by_perrobot_fit(self):
        result = self.evaluator.snapshot(backend(b_y=3.))
        self.assertAlmostEqual(result['shared']['confirmed']['translation_rmse_m'], 1.)
        self.assertAlmostEqual(result['optimization']['optimized_position_rmse_m'], math.sqrt(.5))
        self.assertAlmostEqual(result['optimization']['raw_position_rmse_m'], math.sqrt(.5))
        self.assertEqual(result['local'][1]['position_rmse_m'], 0.)  # explicitly LOCAL only

    def test_single_global_rotation_and_translation_are_only_gauge_changes(self):
        global_change = pose(x=20., y=-4., angle=70.)
        for rows in self.payload['robots'].values():
            for row in rows:
                row['T_world_base'] = (global_change @ row['T_world_base']).tolist()
        self.save_truth()
        result = self.evaluator.snapshot(backend())
        self.assertAlmostEqual(result['shared']['confirmed']['translation_rmse_m'], 0., places=7)
        self.assertAlmostEqual(result['optimization']['optimized_position_rmse_m'], 0., places=7)

    def test_local_rotational_drift_is_reported_in_degrees(self):
        graph = backend()
        graph.keyframes['a1']['T_local_camera'] = pose(x=8, angle=10.)
        row = self.evaluator.snapshot(graph)['local'][0]
        self.assertAlmostEqual(row['rotation_rmse_deg'], 10/math.sqrt(2), places=6)
        self.assertAlmostEqual(row['rpe_rotation_rmse_deg'], 10., places=6)

    def test_local_drift_and_rotation_errors_have_correct_units(self):
        result = self.evaluator.snapshot(backend(scale=1.2, b_angle=10.))
        self.assertAlmostEqual(result['local'][0]['position_rmse_m'], 1.6/math.sqrt(2))
        self.assertAlmostEqual(result['local'][0]['rpe_translation_rmse_m'], 1.6)
        self.assertAlmostEqual(result['shared']['confirmed']['rotation_rmse_deg'], 10.)

    def test_optimized_internal_poses_compared_with_same_samples_and_gauge(self):
        graph = backend(scale=1.2)
        graph.keyframes['a1']['T_submap_camera'] = pose(x=8.)
        graph.keyframes['b1']['T_submap_camera'] = pose(x=8.)
        result = self.evaluator.snapshot(graph)['optimization']
        self.assertEqual(result['samples'], 4)
        self.assertGreater(result['raw_position_rmse_m'], 1.)
        self.assertAlmostEqual(result['optimized_position_rmse_m'], 0.)

    def test_missing_stale_and_stalled_stream_never_show_zero_accuracy(self):
        self.truth.unlink()
        self.assertEqual(self.evaluator.snapshot(backend())['status'], 'waiting')
        self.write_truth(updated=90.)
        stale = self.evaluator.snapshot(backend())
        self.assertEqual(stale['status'], 'stale')
        self.assertIsNone(stale['shared']['confirmed']['translation_rmse_m'])
        self.write_truth(updated=100., receipt=90.)
        self.assertEqual(self.evaluator.snapshot(backend())['status'], 'stale')
        self.write_truth(valid=False)
        self.assertEqual(self.evaluator.snapshot(backend())['status'], 'invalid')
        self.write_truth(receipt=None)
        self.assertEqual(self.evaluator.snapshot(backend())['status'], 'invalid')
        self.write_truth()
        self.payload.pop('valid'); self.save_truth()
        self.assertEqual(self.evaluator.snapshot(backend())['status'], 'invalid')

    def test_cached_truth_survives_ring_rollover_but_not_new_run_or_reset(self):
        graph = backend()
        self.assertEqual(self.evaluator.snapshot(graph)['coverage']['associated'], 4)
        self.write_truth(start=200, end=300)
        self.assertEqual(self.evaluator.snapshot(graph)['coverage']['associated'], 4)
        self.write_truth(run='next')
        self.assertEqual(self.evaluator.snapshot(graph)['status'], 'invalid')
        self.assertEqual(len(self.evaluator.cache), 0)
        self.evaluator.reset()
        self.assertEqual(self.evaluator.snapshot(graph)['status'], 'ready')

    def test_clock_regression_requires_explicit_reset(self):
        self.evaluator.snapshot(backend())
        self.write_truth(start=0, end=50)
        self.assertEqual(self.evaluator.snapshot(backend())['status'], 'invalid')
        self.write_truth()
        self.assertEqual(self.evaluator.snapshot(backend())['status'], 'invalid')
        self.evaluator.reset()
        self.assertEqual(self.evaluator.snapshot(backend())['status'], 'ready')

    def test_invalid_matrices_order_and_nan_are_rejected(self):
        for modify in (lambda p: p['robots']['rover_a'][0]['T_world_base'][0].__setitem__(0, 2),
                       lambda p: p['robots']['rover_a'][1].__setitem__('stamp_ns', 0),
                       lambda p: p.__setitem__('updated_unix', float('nan'))):
            self.write_truth()
            modify(self.payload); self.save_truth()
            self.assertEqual(self.evaluator.snapshot(backend())['status'], 'invalid')
        with self.assertRaises(ValueError):
            rigid(np.diag([-1., 1, 1, 1]))

    def test_provisional_and_unaligned_are_not_confirmed(self):
        graph = backend()
        graph.submaps['b']['aligned'] = False
        result = self.evaluator.snapshot(graph)
        self.assertEqual(result['shared']['confirmed']['pairs'], 0)
        self.assertEqual(result['shared']['provisional']['pairs'], 0)
        graph._preview_cache = ([], {'b': pose(y=2)})
        result = self.evaluator.snapshot(graph)
        self.assertEqual(result['shared']['confirmed']['pairs'], 0)
        self.assertEqual(result['shared']['provisional']['pairs'], 2)
        self.assertEqual(result['optimization']['samples'], 2)

    def test_loop_measurements_use_separate_capture_times_and_explicit_thresholds(self):
        graph = backend()
        graph.constraints = {'good': dict(kind='loop', state='accepted', active=True,
            source_frame='a1', target_frame='b0', T_targetcamera_sourcecamera=pose(x=8, y=-2)),
            'bad': dict(kind='loop', state='verified', active=False,
            source_frame='a0', target_frame='b1', T_targetcamera_sourcecamera=pose(x=-7, y=-2))}
        result = self.evaluator.snapshot(graph)['loops']
        self.assertEqual(result['evaluated'], 2)
        self.assertEqual(result['correct'], 1)
        self.assertEqual(result['precision'], .5)
        self.assertEqual(result['accepted_precision'], 1.)

    def test_evaluation_never_changes_estimator_or_frames(self):
        graph = backend()
        before = deepcopy(graph)
        self.evaluator.snapshot(graph)
        self.assertEqual(graph.robots, before.robots)
        self.assertEqual(graph.constraints, before.constraints)
        for key in graph.keyframes:
            np.testing.assert_array_equal(graph.keyframes[key]['T_local_camera'], before.keyframes[key]['T_local_camera'])
            np.testing.assert_array_equal(graph.keyframes[key]['T_submap_camera'], before.keyframes[key]['T_submap_camera'])
        self.assertFalse(any('truth' in key for frame in graph.keyframes.values() for key in frame))

    def test_optical_mount_is_applied_to_ground_truth(self):
        scene = json.loads(self.scene.read_text())
        for value in scene['robots'].values():
            value['camera_optical']['translation'] = [0, 0, .5]
        self.scene.write_text(json.dumps(scene))
        evaluator = SimulationEvaluator(self.truth, self.scene, clock=lambda: self.now)
        evaluator.snapshot(backend())
        self.assertEqual(evaluator.cache['a0'][1][2, 3], .5)


if __name__ == '__main__':
    unittest.main()
