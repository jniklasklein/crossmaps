#!/usr/bin/env python3
"""Replay post-segmentation observations without ROS or neural inference.

Input JSONL schema (one accepted observation per line):
  {"seq": 1, "time_s": 1.0, "camera_position": [0,0,0],
   "segments": [{"id": 10, "support": [[1,2,3], ...], "dino": [...],
                 "truth_id": "box_a"}], "observed": [10], "cycle_ms": 123.0}

segments must include retained hidden tracks; observed lists only fresh ones.
truth_id and cycle_ms are optional. Unknown metrics are null, never zero.
This measures the identity layer, NOT segmentation quality or model latency.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import time

from identity_graph import PersistentEntities
from object_assembly import ObjectAssemblies, AssemblyConfig


def timing_summary(samples):
    if not samples:
        return None
    ordered = sorted(samples)
    return {"samples": len(samples), "mean_ms": round(statistics.mean(samples), 3),
            "p95_ms": round(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)], 3)}


def replay(frames):
    memory = PersistentEntities()
    timings, cycles = [], []
    last_identity = {}
    switches = duplicates = false_groups = labeled_observations = 0
    assigned_count = observed_count = frame_count = 0
    for frame in frames:
        segments = {int(s["id"]): {"support": {tuple(v) for v in s["support"]},
                                    "dino": tuple(s.get("dino", ())), "truth_id": s.get("truth_id")}
                    for s in frame["segments"]}
        observed = set(map(int, frame["observed"])) & set(segments)
        started = time.perf_counter()
        accepted = memory.update(int(frame["seq"]), float(frame["time_s"]), segments,
                                 observed, frame.get("camera_position"))
        elapsed = (time.perf_counter() - started) * 1000.0
        if not accepted:
            continue
        frame_count += 1
        timings.append(elapsed)
        if frame.get("cycle_ms") is not None:
            value = float(frame["cycle_ms"])
            if math.isfinite(value) and value >= 0:
                cycles.append(value)
        truth_entities, entity_truth = defaultdict(set), defaultdict(set)
        for sid in observed:
            observed_count += 1
            eid = memory.segment_entity.get(sid)
            truth = segments[sid]["truth_id"]
            if truth is not None:
                labeled_observations += 1
            if eid is not None:
                assigned_count += 1
                if truth is not None:
                    truth_entities[str(truth)].add(eid)
                    entity_truth[eid].add(str(truth))
        for truth, eids in truth_entities.items():
            duplicates += max(0, len(eids) - 1)
            # Ambiguous duplicate tracks are counted separately, not arbitrarily
            # selected as the identity against which switches are measured.
            if len(eids) == 1:
                eid = next(iter(eids))
                if truth in last_identity and last_identity[truth] != eid:
                    switches += 1
                last_identity[truth] = eid
        false_groups += sum(len(truths) > 1 for truths in entity_truth.values())
    labeled = labeled_observations > 0
    return {"accepted_frames": frame_count, "observed_surface_samples": observed_count,
            "assigned_fraction": assigned_count / observed_count if observed_count else None,
            "labeled_surface_samples": labeled_observations,
            "identity_switches": switches if labeled else None,
            "duplicate_entity_frame_count": duplicates if labeled else None,
            "false_group_entity_frame_count": false_groups if labeled else None,
            "identity_update_timing": timing_summary(timings),
            "provided_full_cycle_timing": timing_summary(cycles),
            "final": memory.diagnostics()}


def demo_frames():
    # Two touching identical boxes; one changes segment ID, then both disappear
    # from view and return. Ground truth is deliberately separate from features.
    for seq in range(1, 15):
        segments = []
        for sid, start, truth in ((101 if seq < 6 else 201, 0, "box_a"), (102, 20, "box_b")):
            segments.append({"id": sid, "support": [[i, 0, 0] for i in range(start, start + 20)],
                             "dino": [1, 0], "truth_id": truth})
        yield {"seq": seq, "time_s": float(seq), "camera_position": [0.3 * (seq // 4), 0, 0],
               "segments": segments, "observed": [] if 9 <= seq <= 11 else [s["id"] for s in segments]}


def replay_assembly(frames):
    """Reproduce recorded grouping exactly; no invented accuracy labels."""
    memory = None
    accepted = mismatches = 0
    timings = []
    for frame in frames:
        if frame.get('schema') != 'crossmaps_identity_trace_v1':
            raise ValueError('Unsupported identity trace schema')
        if memory is None:
            memory = ObjectAssemblies.from_dict(frame.get('initial_assembly'))
        memory.config = AssemblyConfig(**frame['assembly_config'])
        nodes = {int(e): {**n, 'support': {tuple(k) for k in n['support']}, 'dino': tuple(n['dino'])}
                 for e,n in frame['nodes'].items()}
        started = time.perf_counter()
        if not memory.update(int(frame['seq']), float(frame['time_s']), nodes, frame['observed'],
                             frame['shared_masks'], float(frame['voxel_size']), frame['camera_position'], frame['exclusions']):
            continue
        timings.append(1000*(time.perf_counter()-started))
        accepted += 1
        expected = {int(e): int(root) for e,root in frame['expected_aliases'].items()}
        mismatches += int(memory.aliases != expected)
    return dict(accepted_frames=accepted, decision_mismatch_frames=mismatches,
                grouping_accuracy=None, accuracy_note='No ground-truth physical object labels in this recording',
                grouping_timing=timing_summary(timings), final=memory.diagnostics() if memory else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="JSONL file inside the experimental directory")
    source.add_argument("--demo", action="store_true", help="Run the deterministic synthetic replay")
    source.add_argument("--assembly-input", type=Path, help="Replay an opt-in recorded identity trace inside this directory")
    args = parser.parse_args()
    if args.demo:
        result = replay(demo_frames())
    else:
        path = (args.assembly_input or args.input).resolve()
        root = Path(__file__).resolve().parent
        if not path.is_relative_to(root):
            parser.error("Replay input must be inside crossmaps_dino_experimental")
        with path.open() as stream:
            runner = replay_assembly if args.assembly_input else replay
            result = runner(json.loads(line) for line in stream if line.strip())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
