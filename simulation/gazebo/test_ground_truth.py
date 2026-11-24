import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ground_truth import TruthBuffer, matrix


def packet(robot='rover_a', sec=1, x=2):
    return dict(kind=robot, message=dict(header=dict(stamp=dict(sec=str(sec), nsec=0)),
        pose=[dict(name=robot, position=dict(x=x, y=3, z=.02), orientation=dict(w=1))]))


class GroundTruthTests(unittest.TestCase):
    def test_actual_pose_publisher_per_pose_header_and_world_labels(self):
        value = packet(sec=53)
        header = value['message'].pop('header')
        header['stamp']['nsec'] = 800000000
        header['data'] = [dict(key='frame_id', value=['crossmaps_lab']),
                          dict(key='child_frame_id', value=['rover_a'])]
        value['message']['pose'][0]['header'] = header
        truth = TruthBuffer('run')
        self.assertTrue(truth.consume(value))
        self.assertEqual(truth.robots['rover_a'][0]['stamp_ns'], 53_800_000_000)
        broken = copy.deepcopy(value)
        broken['message']['header'] = dict(stamp=dict(sec=54))
        self.assertFalse(truth.consume(broken))
        for field, wrong in (('frame_id', 'odom'), ('child_frame_id', 'rover_b')):
            broken = copy.deepcopy(value)
            for label in broken['message']['pose'][0]['header']['data']:
                if label['key'] == field:
                    label['value'] = [wrong]
            self.assertFalse(TruthBuffer('run').consume(broken))

    def test_named_pose_preserves_world_transform(self):
        truth = TruthBuffer('run')
        self.assertTrue(truth.consume(packet()))
        value = truth.snapshot()['robots']['rover_a'][0]
        self.assertEqual(value['stamp_ns'], 10**9)
        self.assertEqual(value['T_world_base'], [[1.,0.,0.,2.], [0.,1.,0.,3.], [0.,0.,1.,.02], [0.,0.,0.,1.]])
        self.assertFalse(truth.consume(packet()))
        self.assertEqual(len(truth.robots['rover_a']), 1)

    def test_capacity_and_atomic_snapshot(self):
        truth = TruthBuffer('run')
        for i in range(350):
            truth.consume(packet(sec=i))
        self.assertEqual(len(truth.robots['rover_a']), 300)
        self.assertEqual(truth.robots['rover_a'][0]['stamp_ns'], 50*10**9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'truth.json'
            truth.write(path)
            saved = json.loads(path.read_text())
            self.assertEqual(saved['run_id'], 'run')
            self.assertTrue(saved['valid'])
            self.assertEqual(list(Path(directory).iterdir()), [path])
            self.assertLess(path.stat().st_size, 200000)

    def test_invalid_names_and_numbers_never_become_truth(self):
        original = packet()
        broken = []
        for name in ('base_link', 'odom', 'world::unexpected'):
            value = copy.deepcopy(original); value['message']['pose'][0]['name'] = name; broken.append(value)
        value = copy.deepcopy(original); value['message']['pose'] *= 2; broken.append(value)
        for orientation in ({}, {'w':0}, {'w':float('nan')}, {'w':2}):
            value = copy.deepcopy(original); value['message']['pose'][0]['orientation'] = orientation; broken.append(value)
        value = packet(x=float('inf')); broken.append(value)
        value = packet(); value['message']['header']['stamp']['nsec'] = 10**9; broken.append(value)
        for value in broken:
            truth = TruthBuffer('run')
            self.assertFalse(truth.consume(value), value)
            self.assertEqual(len(truth.robots['rover_a']), 0)

    def test_pose_or_clock_regression_invalidates_until_restart(self):
        for clock in (False, True):
            truth = TruthBuffer('run')
            truth.consume(packet(sec=5))
            if clock:
                truth.consume(dict(kind='clock', message=dict(sim=dict(sec=5))))
                truth.consume(dict(kind='clock', message=dict(sim=dict(sec=4))))
            else:
                truth.consume(packet(sec=4))
            self.assertFalse(truth.valid)
            self.assertFalse(truth.consume(packet(sec=6)))
            self.assertEqual(list(truth.robots['rover_a']), [])
            self.assertIn('restart', truth.error)

    def test_quaternion_rotation(self):
        result = matrix(dict(orientation=dict(z=2**-.5, w=2**-.5)))
        self.assertAlmostEqual(result[0][1], -1.)
        self.assertAlmostEqual(result[1][0], 1.)

    def test_receipt_freshness_requires_advancing_pose_not_writer_or_duplicate(self):
        truth = TruthBuffer('run')
        with patch('ground_truth.time.time', return_value=100.):
            truth.consume(packet())
        with patch('ground_truth.time.time', return_value=106.):
            truth.consume(packet())
            value = truth.snapshot()
        self.assertEqual(value['updated_unix'], 106.)
        self.assertEqual(value['last_received_unix']['rover_a'], 100.)
        self.assertEqual(value['sample_age_s']['rover_a'], 6.)
        self.assertIsNone(value['sample_age_s']['rover_b'])

    def test_native_evaluation_branch_has_no_motion_publisher(self):
        source = Path(__file__).with_name('gz_capture.cpp').read_text()
        branch = source.split('if (std::string(argv[1]) == "evaluation") {',1)[1].split('const std::string root',1)[0]
        self.assertIn('Pose_V', branch)
        self.assertNotIn('Advertise', branch)
        self.assertNotIn('cmd_vel', branch)


if __name__ == '__main__':
    unittest.main()
