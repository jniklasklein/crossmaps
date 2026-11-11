import unittest
import json
from pathlib import Path
import tempfile

from replay_identity import demo_frames, replay, replay_assembly
from world_model import EvidenceRecorder, new_world_id
from object_assembly import ObjectAssemblies


class ReplayMetricsTests(unittest.TestCase):
    def test_demo_has_no_switches_or_false_groups(self):
        report = replay(demo_frames())
        self.assertEqual(report["accepted_frames"], 14)
        self.assertEqual(report["identity_switches"], 0)
        self.assertEqual(report["duplicate_entity_frame_count"], 0)
        self.assertEqual(report["false_group_entity_frame_count"], 0)
        self.assertEqual(report["final"]["entities"], 2)
        self.assertIsNone(report["provided_full_cycle_timing"])

    def test_unlabeled_replay_does_not_claim_perfect_accuracy(self):
        frames = list(demo_frames())
        for frame in frames:
            for segment in frame["segments"]:
                segment.pop("truth_id")
        report = replay(frames)
        self.assertIsNone(report["identity_switches"])
        self.assertIsNone(report["duplicate_entity_frame_count"])
        self.assertIsNone(report["false_group_entity_frame_count"])

    def test_duplicate_sequence_not_counted_as_frame(self):
        frames = list(demo_frames())
        report = replay([frames[0], *frames])
        self.assertEqual(report["accepted_frames"], len(frames))

    def test_bounded_recording_replays_actual_decisions_without_claiming_accuracy(self):
        with tempfile.TemporaryDirectory(prefix='.replay-test-', dir=Path(__file__).parent) as directory:
            recorder = EvidenceRecorder(new_world_id(), max_frames=5, directory=directory)
            memory = ObjectAssemblies()
            nodes = {1: {'min':(0,0,0), 'max':(.5,.2,.2), 'dino':(1.,0.),
                         'support':{(i,j,0) for i in range(5) for j in range(2)}},
                     2: {'min':(.5,0,0), 'max':(.7,.6,.6), 'dino':(1.,0.),
                         'support':{(i,j,0) for i in range(5,7) for j in range(6)}}}
            for seq in range(1,8):
                initial = memory.to_dict()
                memory.update(seq,float(seq),nodes,nodes,voxel_size=.1)
                frame = dict(seq=seq,time_s=float(seq),voxel_size=.1,camera_position=None,
                    nodes={str(e): {**n,'support':sorted(n['support'])} for e,n in nodes.items()},
                    observed=[1,2],shared_masks=[],exclusions=[],expected_aliases=memory.aliases,
                    assembly_config=memory.to_dict()['config'])
                recorder.record(frame,initial)
                self.assertFalse(recorder.record(frame,initial))
            self.assertEqual(recorder.frames,5)
            with recorder.path.open() as stream:
                frames = [json.loads(line) for line in stream]
            report = replay_assembly(frames)
            self.assertEqual(report['accepted_frames'],5)
            self.assertEqual(report['decision_mismatch_frames'],0)
            self.assertEqual(report['final']['joined_hypotheses'],1)
            self.assertIsNone(report['grouping_accuracy'])

    def test_recorder_rejects_out_of_scope_output(self):
        with self.assertRaises(ValueError):
            EvidenceRecorder(new_world_id(), directory=Path(__file__).resolve().parents[2])


if __name__ == "__main__":
    unittest.main()
