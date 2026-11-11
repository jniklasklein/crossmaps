"""Execute the real node methods without ROS, model loads, or external files.

Only the AST of the experimental node is loaded. Sensor/model/transport edges
are stubbed; association, fusion, identity, serialization and graph code run.
"""
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import queue
import struct
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import cv2
from segment_overlay import dense_display_ids, id_color, render_overlay, render_mask_consensus
from surface_tracking import SurfaceConfig, SurfaceConsensus, BoundaryOwnershipVotes
from dominant_overlay import DominantOverlay
from world_model import new_world_id, validate_world_id, node_uid, interior_descriptor, surface_role, dominant_surface_core, EvidenceRecorder
from object_assembly import AssemblyConfig
from pipeline_runtime import LatestFrame, PipelineHealth
from voxel_scoring import score_voxels

from identity_graph import IdentityConfig, ObservationEdges, PersistentEntities


NODE_PATH = Path(__file__).resolve().parents[1] / "crossmaps_3_node_3d_revised-exp.py"
TREE = ast.parse(NODE_PATH.read_text())
CLASS = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "CrossMaps3Node")
DEFAULTS = {}
for statement in ast.walk(next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")):
    if isinstance(statement, ast.Call) and isinstance(statement.func, ast.Attribute) and statement.func.attr == "declare_parameter":
        try:
            DEFAULTS[ast.literal_eval(statement.args[0])] = ast.literal_eval(statement.args[1])
        except (ValueError, TypeError):
            pass
TEST_CLASS = copy.deepcopy(CLASS)
TEST_CLASS.bases = []
TEST_CLASS.body = [n for n in TEST_CLASS.body if isinstance(n, ast.FunctionDef) and n.name != "__init__"]
MODULE = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] +
                  [n for n in TREE.body if isinstance(n, ast.FunctionDef)] + [TEST_CLASS], type_ignores=[])
NAMESPACE = dict(np=np, cv2=cv2, math=math, hashlib=hashlib, time=time, json=json,
                 os=os, pickle=pickle, tempfile=tempfile, struct=struct,
                 dense_display_ids=dense_display_ids, id_color=id_color, render_overlay=render_overlay,
                 render_mask_consensus=render_mask_consensus, SurfaceConfig=SurfaceConfig,
                 SurfaceConsensus=SurfaceConsensus, BoundaryOwnershipVotes=BoundaryOwnershipVotes,
                 DominantOverlay=DominantOverlay,
                 new_world_id=new_world_id, validate_world_id=validate_world_id, node_uid=node_uid,
                 interior_descriptor=interior_descriptor, surface_role=surface_role,
                 dominant_surface_core=dominant_surface_core,
                 EvidenceRecorder=EvidenceRecorder,
                 LatestFrame=LatestFrame, PipelineHealth=PipelineHealth, score_voxels=score_voxels,
                 queue=queue, threading=threading,
                 AssemblyConfig=AssemblyConfig,
                 IdentityConfig=IdentityConfig, ObservationEdges=ObservationEdges, PersistentEntities=PersistentEntities)
exec(compile(ast.fix_missing_locations(MODULE), str(NODE_PATH), "exec"), NAMESPACE)


def make_node():
    node = NAMESPACE["CrossMaps3Node"]()
    node.params = dict(DEFAULTS)
    node.params.update(publish_semantic_cloud=False, use_ray_consistency=False,
                       segment_track_debug_log=False, segment_maintenance_debug_log=False,
                       segment_frame_assignment_max_depth_frac=0.0, esam_min_depth_points=3,
                       entity_birth_observations=2, object_graph_min_observations=1,
                       object_graph_min_support_voxels=3)
    node.get_parameter = lambda name: SimpleNamespace(value=node.params[name])
    node.get_logger = lambda: SimpleNamespace(warn=lambda *_: None, info=lambda *_: None, error=lambda *_: None)
    node.pipeline_health = PipelineHealth()
    node.latest_frame = LatestFrame()
    node.async_worker_busy = threading.Event()
    node.semantic_job_queue = queue.Queue(maxsize=1)
    node.semantic_result_queue = queue.Queue(maxsize=2)
    node.async_results_dropped = 0
    for name in ("sum", "w", "first_seen", "last_seen", "obs_count", "last_obs_seq", "viewmask",
                 "centroid", "bbox_min", "bbox_max", "support_cells", "support_voxels", "dino_sum",
                 "dino_w", "merge_block_until"):
        setattr(node, "seg_" + name, {})
    for name in ("voxel_segment_id", "segment_voxel_miss_hits", "segment_semantic_conflict_hits",
                 "segment_semantic_conflict_last_seq", "segment_query_support_cache", "stm_viewmask",
                 "vox_stm_viewmask", "vox_stm_sum", "vox_stm_w", "vox_ltm_sum", "vox_ltm_w"):
        setattr(node, name, {})
    node.seg_next_id = 101
    node.entity_memory = PersistentEntities()
    node.world_model_id = new_world_id()
    node.dino_base_dim = 2  # Synthetic descriptors have no concatenated context.
    node.seg_last_dino_seq = {}
    node.entity_edges = ObservationEdges()
    node.segment_edges = ObservationEdges()
    node.last_applied_observation_seq = -1
    node.last_accepted_depth_stamp_ns = 0
    node.last_segment_projection_context = None
    node.last_identity_update_ms = 0.0
    node.text_emb = None
    node.query_text = "box"
    node.publish_frame = "map"
    node.internal_frame = "odom"
    node.now = 1.0
    node.now_sec = lambda: node.now
    node.voxel_size = node.resolution = 0.05
    node.voxel_z_min = 0.0
    node.voxel_z_max = 5.0
    node.origin_x = node.origin_y = 0.0
    node.voxel_width = node.voxel_height = node.grid_width = node.grid_height = 200
    node.voxel_depth = 100
    node.last_segment_maintenance_s = 0.0
    node.last_segment_maintenance_debug_log_s = 0.0
    node._mark_state_dirty = lambda: None
    node._profile_add_ms = node._profile_add_value = node._profile_add_elapsed = lambda *_: None
    node.publish_esam_mask_overlay = node.publish_feature_diagnostics = lambda *_: None
    node.publish_entity_id_projection = lambda: None
    def capture_projection(context, header):
        node.projection = context
        node.last_segment_projection_context = (context, header)
    node.publish_segment_id_projection = capture_projection
    node.fuse_stm_cell = lambda *_, **__: None
    node.fuse_vox_stm_cell = lambda *_, **__: None
    return node


def observation(seq, width=10, height=4, masks=None):
    uu, vv = np.meshgrid(np.arange(width), np.arange(height))
    points = np.stack([(uu.ravel() + 2.5) * 0.05, (vv.ravel() + 2.5) * 0.05,
                       np.full(width * height, 0.525)], axis=1).astype(np.float32)
    masks = [np.ones((height, width), dtype=bool)] if masks is None else masks
    return dict(seq=seq, obs_time_s=float(seq), rgb_cv=np.zeros((height, width, 3), dtype=np.uint8),
                rgb_header=SimpleNamespace(stamp=None), depth_header=SimpleNamespace(stamp=None),
                pts_int=points, cam_int=np.array([0, 0, 0], dtype=np.float32),
                uu=uu.ravel(), vv=vv.ravel(), z_cam=points[:, 2], image_shape=(height, width),
                masks=masks, mask_scores=np.ones(len(masks), dtype=np.float32),
                mask_emb=np.tile(np.array([[1, 0]], dtype=np.float32), (len(masks), 1)),
                dino_mask_emb=np.tile(np.array([[1, 0]], dtype=np.float32), (len(masks), 1)),
                dino_mask_quality=np.ones(len(masks), dtype=np.float32))


class NodeIntegrationTests(unittest.TestCase):
    def apply(self, node, seq, masks=None):
        node.now = float(seq)
        return node.apply_semantic_result(observation(seq, masks=masks), "test")

    def test_real_fusion_and_entity_birth(self):
        node = make_node()
        for seq in range(1, 5):
            self.assertTrue(self.apply(node, seq))
        self.assertEqual(len(node.seg_w), 1)
        self.assertEqual(len(node.entity_memory.records), 1)
        self.assertEqual(node.seg_obs_count[101], 4)
        self.assertEqual(node.entity_memory.segment_entity[101], 1)
        self.assertEqual(node.last_applied_observation_seq, 4)

    def test_small_mask_with_six_depth_samples_gets_stable_segment(self):
        node = make_node()
        node.params['esam_min_depth_points'] = 25  # Semantic fusion is separate.
        small = np.zeros((4, 10), dtype=bool)
        small[:2, :3] = True
        for seq in range(1, 6):
            self.apply(node, seq, [small])
        self.assertEqual(set(node.seg_w), {101})
        self.assertEqual(node.seg_obs_count[101], 5)
        self.assertEqual(node.last_mask_attribution[0]['segments'], [101])
        self.assertEqual(node.last_mask_attribution[0]['depth_points'], 6)

    def test_stationary_depth_quantization_replay_keeps_one_surface(self):
        node = make_node()
        node.params['segment_assoc_min_bbox_iou'] = .99
        for seq in range(1, 41):
            frame = observation(seq, width=3, height=1)
            frame['pts_int'][:] = (.049 if seq % 2 else .051, .025, .525)
            node.now = float(seq)
            self.assertTrue(node.apply_semantic_result(frame, 'test'))
            self.assertEqual(set(node.seg_w), {101})
            self.assertEqual(node.projection['decisions'][0]['winner'], 101)
        self.assertEqual(node.seg_obs_count[101], 40)

    def test_broad_mask_does_not_homogenize_component_descriptors(self):
        node = make_node()
        left = np.zeros((4,10), dtype=bool)
        left[:, :5] = True
        for seq in (1,2):
            frame = observation(seq, masks=[left, ~left])
            frame['mask_emb'] = np.eye(2, dtype=np.float32)
            frame['dino_mask_emb'] = np.eye(2, dtype=np.float32)
            node.now = float(seq)
            node.apply_semantic_result(frame, 'test')
        before = {s: node.segment_dino_embedding(s).copy() for s in node.seg_w}
        for seq in (3,4,5):
            self.apply(node, seq)
        for sid in before:
            np.testing.assert_array_equal(node.segment_dino_embedding(sid), before[sid])
        self.assertEqual(node.entity_memory.assemblies.aliases, {})

    def test_grouping_uses_interior_not_context_adjusted_cosine(self):
        node = make_node()
        left = np.zeros((4,10), dtype=bool)
        left[:, :2] = True
        a=np.array([1.,0.,0.,.35], dtype=np.float32)
        b=np.array([.85,math.sqrt(1-.85**2),-.35,0.], dtype=np.float32)
        a/=np.linalg.norm(a)
        b/=np.linalg.norm(b)
        self.assertLess(float(a@b), .8)
        for seq in range(1,7):
            frame=observation(seq,masks=[left,~left])
            frame['dino_mask_emb']=np.stack((a,b))
            node.now=float(seq)
            node.apply_semantic_result(frame,'test')
        self.assertEqual(node._effective_entity_id(2), 1)
        self.assertEqual(len(node.seg_w), 2)

    def test_mixed_components_pool_separate_existing_tokens_without_inference(self):
        node = make_node()
        node.params.update(dinov2_context_contrast_weight=0., dinov2_min_effective_tokens=1.,
                           segment_mask_interior_px=0, stride=1)
        left = np.zeros((4,10), dtype=bool)
        left[:,:5] = True
        for seq in (1,2):
            frame = observation(seq, masks=[left, ~left])
            frame['dino_mask_emb'] = np.eye(2, dtype=np.float32)
            node.now = float(seq)
            node.apply_semantic_result(frame, 'test')
        grid = np.zeros((10,10,2), dtype=np.float32)
        grid[:,:5,0], grid[:,5:,1] = 1., 1.
        frame = observation(3)
        frame['dense_dino'] = dict(features=grid, meta=dict(input_size=10, resized_w=10,
                                                          resized_h=4, offset_x=0, offset_y=0))
        # Fail if refreshing mixed components invokes another neural forward.
        node.compute_dinov2_dense_features = lambda *_: self.fail('Unexpected DINO inference')
        node.compute_mask_embeddings_batched = lambda *_: self.fail('Unexpected CLIP inference')
        node.now = 3.
        node.apply_semantic_result(frame, 'test')
        self.assertEqual(node.seg_last_dino_seq, {101:3, 102:3})
        self.assertLess(float(node.segment_dino_embedding(101) @ node.segment_dino_embedding(102)), .1)

    def test_boundary_repair_requires_four_frames_and_protects_independent_masks(self):
        node = make_node()
        node.params['segment_mask_interior_px'] = 0
        full = np.ones((4,10), dtype=bool)
        small = np.zeros_like(full)
        small[:,:5] = True
        for seq in (1,2):
            self.apply(node, seq, [small, ~small])
        frame = observation(3)
        points, uu, vv = frame['pts_int'], frame['uu'], frame['vv']
        stray = node.xyz_to_voxel(*map(float, points[0]))
        node._claim_segment_voxels(102, {stray})
        node.seg_support_voxels[102].add(stray)
        # A mask covering only S101 plus one stray S102 sample, not all of S102.
        target_mask = small.copy()
        for seq in (3,4,5):
            owners = np.array([node.voxel_segment_id[node.xyz_to_voxel(*map(float,p))] for p in points])
            node._prepare_surface_consensus([target_mask], owners, points, uu, vv, full.shape,
                np.array([[1,0]], np.float32), np.array([[1,0]], np.float32), np.ones(1), seq, float(seq))
            self.assertEqual(node.voxel_segment_id[stray], 102)
        owners = np.array([node.voxel_segment_id[node.xyz_to_voxel(*map(float,p))] for p in points])
        node._prepare_surface_consensus([target_mask], owners, points, uu, vv, full.shape,
            np.array([[1,0]], np.float32), np.array([[1,0]], np.float32), np.ones(1), 6, 6.)
        self.assertEqual(node.voxel_segment_id[stray], 101)
        self.assertEqual(len(node.seg_support_voxels[102]), 20)
        self.assertEqual(node.entity_memory.lineage[-1]['event'], 'boundary_owner_corrected')
        # Independent evidence for the loser prevents further corrections.
        stray2 = node.xyz_to_voxel(*map(float, points[1]))
        node._claim_segment_voxels(102, {stray2})
        node.seg_support_voxels[102].add(stray2)
        for seq in range(7,12):
            owners = np.array([node.voxel_segment_id[node.xyz_to_voxel(*map(float,p))] for p in points])
            node._prepare_surface_consensus([target_mask, ~target_mask], owners, points, uu, vv, full.shape,
                np.array([[1,0],[1,0]], np.float32), np.array([[1,0],[1,0]], np.float32), np.ones(2), seq, float(seq))
        self.assertEqual(node.voxel_segment_id[stray2], 102)

    def test_recording_failure_does_not_interrupt_perception(self):
        node = make_node()
        node.params['record_identity_replay'] = True
        warnings = []
        node.get_logger = lambda: SimpleNamespace(warn=warnings.append, info=lambda *_: None)
        with patch.dict(NAMESPACE, EvidenceRecorder=lambda *_: (_ for _ in ()).throw(OSError('disk unavailable'))):
            for seq in (1,2,3):
                self.assertTrue(self.apply(node, seq))
        self.assertEqual(node.seg_obs_count[101], 3)
        self.assertEqual(len(warnings), 1)
        self.assertIn('recording stopped', warnings[0])

    def test_node_recording_replays_and_toggle_starts_new_trace(self):
        from replay_identity import replay_assembly
        node = make_node()
        with tempfile.TemporaryDirectory(dir=NODE_PATH.parent / 'tests') as directory:
            with patch.dict(NAMESPACE, EvidenceRecorder=lambda world, count: EvidenceRecorder(world, count, directory)):
                node.params['record_identity_replay'] = True
                for seq in range(1,6):
                    self.apply(node, seq)
                old = node.evidence_recorder.path
                node.params['record_identity_replay'] = False
                self.apply(node, 6)
                self.assertEqual(node.evidence_recorder.frames, 5)
                self.assertEqual(node.evidence_recorder.status(False)['state'], 'stopped')
                node.params['record_identity_replay'] = True
                self.apply(node, 7)
                self.assertNotEqual(old, node.evidence_recorder.path)
            for path in (old, node.evidence_recorder.path):
                frames = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(replay_assembly(frames)['decision_mismatch_frames'], 0)

    def test_no_depth_resets_boundary_votes_without_losing_lifetime_budget(self):
        node = make_node()
        self.apply(node, 1)
        node.boundary_votes.budgets[101] = 4
        node.boundary_votes.losses[101] = 1
        node.boundary_votes.pending[((2,2,10),101,102)] = 3
        node.assign_frame_segments_fine_image(
            masks=[np.ones((4,10), dtype=bool)], mask_emb=np.array([[1,0]], np.float32),
            mask_scores=np.ones(1), dino_mask_emb=np.array([[1,0]], np.float32), dino_mask_quality=np.ones(1),
            pts_int=np.zeros((0,3)), uu_i=np.zeros(0,dtype=int), vv_i=np.zeros(0,dtype=int),
            image_shape=(4,10), conf_dist=np.zeros(0), conf_scale=1., conf_min=.1,
            now_s=2., cam_xy_int=(0,0), obs_seq=2, projected_ctx=None, header_msg=None)
        self.assertFalse(node.boundary_votes.pending)
        self.assertEqual(node.boundary_votes.losses[101], 1)
        self.assertEqual(node.boundary_votes.budgets[101], 4)
        self.assertFalse(node.projection['decisions'])

    def test_invalid_mask_shape_does_not_crash_fine_assignment(self):
        node = make_node()
        frame = observation(1)
        node.assign_frame_segments_fine_image(
            masks=[np.ones(3, dtype=bool)], mask_emb=frame['mask_emb'], mask_scores=frame['mask_scores'],
            dino_mask_emb=frame['dino_mask_emb'], dino_mask_quality=frame['dino_mask_quality'],
            pts_int=frame['pts_int'], uu_i=frame['uu'], vv_i=frame['vv'], image_shape=(4,10),
            conf_dist=np.ones(40), conf_scale=1., conf_min=.1, now_s=1., cam_xy_int=(0,0),
            obs_seq=1, projected_ctx=None, header_msg=None)
        self.assertEqual(node.last_mask_attribution[0]['reason'], 'bad_shape')
        self.assertFalse(node.seg_w)

    def test_world_snapshot_has_unambiguous_ids_and_uncertainty(self):
        node = make_node()
        for seq in (1,2,3):
            self.apply(node,seq)
        nodes = node._segment_graph_candidates(entity_view=True,for_evidence=True)
        snapshot = node._object_graph_descriptor_payload(nodes,[],np.eye(4))
        self.assertFalse(snapshot['world_model']['absence_is_negative_evidence'])
        self.assertEqual(snapshot['world_model']['observation_seq'],3)
        self.assertEqual(snapshot['nodes'][0]['uid'],f'{node.world_model_id}/entity/1')
        self.assertEqual(snapshot['nodes'][0]['semantic_labels'],[])
        self.assertIn('not_object_probability',snapshot['nodes'][0]['confidence_semantics'])
        self.assertEqual(snapshot['memberships'][0]['from_uid'],f'{node.world_model_id}/segment/101')

    def test_rejected_sensor_duplicate_does_not_clear_camera_or_evidence(self):
        node = make_node()
        first = observation(1)
        first['depth_header'].stamp = SimpleNamespace(sec=10, nanosec=25)
        self.assertTrue(node.apply_semantic_result(first, 'test'))
        previous_rgb = node.last_assignment_rgb
        previous_attribution = node.last_mask_attribution
        previous_counts = node.last_visible_segment_counts
        node.last_segment_projection_context = node.projection
        previous_context = node.last_segment_projection_context
        duplicate = observation(2)
        duplicate['depth_header'].stamp = SimpleNamespace(sec=10, nanosec=25)
        self.assertFalse(node.apply_semantic_result(duplicate, 'test'))
        self.assertIs(node.last_assignment_rgb, previous_rgb)
        self.assertIs(node.last_mask_attribution, previous_attribution)
        self.assertIs(node.last_visible_segment_counts, previous_counts)
        self.assertIs(node.last_segment_projection_context, previous_context)

    def test_trace_sensor_stamp_matches_current_not_previous_observation(self):
        node = make_node()
        node.params['record_identity_replay'] = True
        with tempfile.TemporaryDirectory(dir=NODE_PATH.parent / 'tests') as directory:
            with patch.dict(NAMESPACE, EvidenceRecorder=lambda world, count: EvidenceRecorder(world, count, directory)):
                for seq in (1,2):
                    frame = observation(seq)
                    frame['depth_header'].stamp = SimpleNamespace(sec=seq+10, nanosec=5)
                    self.assertTrue(node.apply_semantic_result(frame, 'test'))
            frames = [json.loads(line) for line in node.evidence_recorder.path.read_text().splitlines()]
            self.assertEqual([f['sensor_depth_stamp_ns'] for f in frames], [11000000005,12000000005])

    def test_single_voxel_small_mask_can_remain_a_surface_not_an_entity(self):
        node = make_node()
        for seq in range(1, 6):
            frame = observation(seq, width=3, height=1)
            frame['pts_int'][:] = (.125, .125, .525)
            node.now = float(seq)
            node.apply_semantic_result(frame, 'test')
        self.assertEqual(set(node.seg_w), {101})
        self.assertEqual(len(node.seg_support_voxels[101]), 1)
        self.assertEqual(node.seg_obs_count[101], 5)
        self.assertEqual(node.entity_memory.segment_entity, {})

    def test_missing_sam_keeps_current_owners_without_fabricating_observations(self):
        node = make_node()
        for seq in (1, 2):
            self.apply(node, seq)
        node.params['publish_segment_id_projection'] = False
        obs_before = node.seg_obs_count.copy()
        entity_observations = node.entity_memory.records[1].observations
        for seq in (3, 4, 5):
            self.apply(node, seq, [])
        self.assertEqual(set(np.unique(node.projection['id_image'])), {101})
        self.assertEqual(node.seg_obs_count, obs_before)
        self.assertEqual(node.entity_memory.records[1].observations, entity_observations)
        self.assertEqual(node.seg_last_seen[101], 5.)

    def test_appearance_veto_retains_display_ownership_not_descriptor_update(self):
        node = make_node()
        self.apply(node, 1)
        before = node.seg_sum[101].copy()
        frame = observation(2)
        frame['mask_emb'][:] = [-1, 0]
        frame['dino_mask_emb'][:] = [-1, 0]
        node.now = 2.
        node.apply_semantic_result(frame, 'test')
        self.assertEqual(set(np.unique(node.projection['id_image'])), {101})
        np.testing.assert_array_equal(node.seg_sum[101], before)
        self.assertEqual(node.seg_obs_count[101], 1)

    def test_sofa_parts_group_non_destructively_and_share_display_id(self):
        node = make_node()
        left = np.zeros((4, 10), dtype=bool)
        left[:, :2] = True  # Complementary sizes; not identical boxes.
        for seq in range(1, 7):
            self.apply(node, seq, [left, ~left])
        self.assertEqual(len(node.seg_w), 2)
        self.assertEqual(len(node.entity_memory.records), 2)
        self.assertEqual(len(set(node.voxel_segment_id.values())), 2)
        self.assertFalse(node.entity_memory.can_merge_segments(101, 102))
        self.assertEqual(node._effective_entity_id(2), 1)
        self.assertEqual(node._segment_entity_rgb(101), node._segment_entity_rgb(102))
        graph = node._segment_graph_candidates(entity_view=True, for_evidence=True)
        self.assertEqual(len(graph), 1)
        self.assertEqual(graph[0]['component_entity_ids'], [1, 2])
        self.assertEqual(graph[0]['segment_ids'], [101, 102])
        self.assertEqual(graph[0]['membership_basis'], 'dino_geometry_assembly')
        self.assertEqual(graph[0]['observations'], 6)

    def test_same_shape_box_segments_do_not_group_when_stationary(self):
        node = make_node()
        left = np.zeros((4, 10), dtype=bool)
        left[:, :5] = True
        for seq in range(1, 8):
            self.apply(node, seq, [left, ~left])
        self.assertEqual(node.entity_memory.assemblies.aliases, {})
        for seq in range(8, 14):
            self.apply(node, seq)  # Broad mask alone still isn't proof.
        self.assertEqual(node.entity_memory.assemblies.aliases, {})
        self.assertEqual(node.last_mask_attribution[0]['segments'], [101, 102])

    def test_unsupported_mask_reported_instead_of_inventing_geometry(self):
        node = make_node()
        small = np.zeros((4, 10), dtype=bool)
        small[0, :2] = True
        self.apply(node, 1, [small])
        self.assertEqual(node.seg_w, {})
        self.assertEqual(node.last_mask_attribution[0]['reason'], 'insufficient_depth')

    def test_masks_with_no_depth_are_visible_as_unassigned(self):
        node = make_node()
        frame = observation(1)
        for name in ('uu', 'vv', 'z_cam', 'pts_int'):
            frame[name] = frame[name][:0]
        node.apply_semantic_result(frame, 'test')
        self.assertEqual(node.seg_w, {})
        self.assertEqual(node.last_mask_attribution[0]['reason'], 'no_valid_depth')
        self.assertEqual(len(node.projection['masks']), 1)

    def test_split_is_not_explicit_grouping_veto_but_detach_is(self):
        node = make_node()
        for seq in (1, 2):
            self.apply(node, seq)
        node.entity_memory.segment_split(101, 102)
        self.assertEqual(node.entity_memory.assembly_exclusions, {})
        node.entity_memory.detach_segment(101)
        self.assertEqual(node.entity_memory.assembly_exclusions, {101: 1})
        restored = PersistentEntities.from_dict(node.entity_memory.to_dict())
        self.assertEqual(restored.assembly_exclusions, {101: 1})

    def test_duplicate_and_out_of_order_results_do_not_fuse(self):
        node = make_node()
        self.apply(node, 2)
        before = node.seg_w.copy()
        self.assertFalse(self.apply(node, 2))
        self.assertFalse(self.apply(node, 1))
        self.assertEqual(node.seg_w, before)

    def test_small_masks_keep_large_surface_owner(self):
        node = make_node()
        self.apply(node, 1)
        self.apply(node, 2)
        small = np.zeros((4, 10), dtype=bool)
        small[:, :2] = True  # 20% IoU, below the old 25% threshold.
        for seq in (3, 4, 5):
            self.apply(node, seq, [small])
        self.assertEqual(set(node.seg_w), {101})
        self.assertEqual(node.seg_obs_count[101], 5)
        self.assertEqual(node.entity_memory.segment_entity[101], 1)

    def test_same_sensor_frame_with_new_job_sequence_is_not_new_evidence(self):
        node = make_node()
        first = observation(1)
        first["depth_header"].stamp = SimpleNamespace(sec=10, nanosec=3)
        self.assertTrue(node.apply_semantic_result(first, "test"))
        first["seq"] = 2
        self.assertFalse(node.apply_semantic_result(first, "test"))
        self.assertEqual(node.seg_obs_count[101], 1)

    def test_live_membership_correction_preserves_geometry(self):
        node = make_node()
        self.apply(node, 1)
        self.apply(node, 2)
        before = node.voxel_segment_id.copy()
        node.on_entity_correction(SimpleNamespace(data=json.dumps({"action": "detach_segment", "segment_id": 101})))
        self.assertNotIn(101, node.entity_memory.segment_entity)
        self.assertEqual(before, node.voxel_segment_id)
        # Wrong types and unsupported actions are rejected without changes.
        node.on_entity_correction(SimpleNamespace(data='{"action":"detach_segment","segment_id":true}'))
        node.on_entity_correction(SimpleNamespace(data='[]'))
        self.assertEqual(before, node.voxel_segment_id)

    def test_maintenance_split_owners_survive_the_next_broad_mask(self):
        node = make_node()
        node.params["use_segment_maintenance"] = False
        for seq in (1, 2):
            frame = observation(seq)
            frame["pts_int"][frame["uu"] >= 5, 0] += 1.0
            node.now = float(seq)
            node.apply_semantic_result(frame, "test")
        self.assertEqual(len(node.seg_w), 1)
        stats = node.split_disconnected_segments()
        self.assertEqual(stats["new_segments"], 1)
        child = max(node.seg_w)
        self.assertEqual(node.seg_last_obs_seq[child], -1)
        self.assertTrue(node._valid_projected_segment_owner(child))
        owners = node.voxel_segment_id.copy()
        for seq in (3, 4, 5):
            frame = observation(seq)
            frame["pts_int"][frame["uu"] >= 5, 0] += 1.0
            node.now = float(seq)
            node.apply_semantic_result(frame, "test")
        self.assertEqual(owners, node.voxel_segment_id)
        self.assertEqual(len(node.seg_w), 2)
        self.assertEqual(len(set(node.entity_memory.segment_entity.values())), 2)

    def test_fusion_guards_ownership_even_without_a_projected_context(self):
        node = make_node()
        self.apply(node, 1)
        before = node.voxel_segment_id.copy()
        frame = observation(2)
        obs = node.build_segment_observation(frame["pts_int"], frame["mask_emb"][0], np.ones(40))
        rejected = node.fuse_segment_observation(obs, 2.0, (0, 0), obs_seq=2,
                                                forced_sid=999, force_new=True)
        self.assertIsNone(rejected)
        self.assertEqual(before, node.voxel_segment_id)
        self.assertNotIn(999, node.seg_w)

    def test_one_mask_spanning_two_boxes_does_not_steal_owners(self):
        node = make_node()
        left = np.zeros((4, 10), dtype=bool)
        left[:, :5] = True
        for seq in (1, 2):
            self.apply(node, seq, [left, ~left])
        before = node.voxel_segment_id.copy()
        self.assertEqual(len(node.entity_memory.records), 2)
        for seq in (3, 4, 5):
            self.apply(node, seq)
        self.assertEqual(before, node.voxel_segment_id)
        self.assertEqual(len(node.entity_memory.records), 2)

    def test_graph_filters_do_not_change_entity_membership_or_evidence(self):
        node = make_node()
        left = np.zeros((4, 10), dtype=bool)
        left[:, :5] = True
        for seq in range(1, 5):
            self.apply(node, seq, [left, ~left])
        before = copy.deepcopy(node.entity_memory.to_dict())
        edges_before = copy.deepcopy(node.entity_edges.states)
        node.params["object_graph_max_nodes"] = 1
        for _ in range(10):
            nodes = node._segment_graph_candidates()
            node._object_graph_edges(nodes)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(before, node.entity_memory.to_dict())
        self.assertEqual(edges_before, node.entity_edges.states)
        node.params["object_graph_max_nodes"] = 18
        nodes = node._segment_graph_candidates()
        self.assertEqual({n["id"] for n in nodes}, {1, 2})
        edges = node._object_graph_edges(nodes)
        self.assertTrue(edges)
        payload = node._object_graph_descriptor_payload(nodes, edges, np.eye(4, dtype=np.float32))
        self.assertEqual(payload["schema"], "crossmaps_hierarchical_scene_graph_v3")
        self.assertEqual(len(payload["memberships"]), 2)
        json.dumps(payload)

    def test_missing_mask_observation_does_not_create_negative_edges(self):
        node = make_node()
        left = np.zeros((4, 10), dtype=bool)
        left[:, :5] = True
        for seq in range(1, 5):
            self.apply(node, seq, [left, ~left])
        before = copy.deepcopy(node.entity_edges.states)
        for seq in range(5, 10):
            self.apply(node, seq, [])
        self.assertEqual(before, node.entity_edges.states)

    def test_segment_serialization_roundtrip(self):
        node = make_node()
        self.apply(node, 1)
        self.apply(node, 2)
        restored = make_node()
        restored._deserialize_segments(node._serialize_segments())
        restored.entity_memory = PersistentEntities.from_dict(node.entity_memory.to_dict())
        self.assertEqual(node.voxel_segment_id, restored.voxel_segment_id)
        self.assertEqual(node.entity_memory.segment_entity, restored.entity_memory.segment_entity)

    def test_mixed_descriptor_dimensions_roundtrip_without_dropping_tracks(self):
        node = make_node()
        left = np.zeros((4,10), dtype=bool)
        left[:,:5] = True
        self.apply(node, 1, [left, ~left])
        node.seg_dino_sum[102] = np.array([0.,1.,.5,0.], dtype=np.float32)
        restored = make_node()
        restored._deserialize_segments(node._serialize_segments())
        for sid in node.seg_w:
            np.testing.assert_array_equal(restored.seg_dino_sum[sid], node.seg_dino_sum[sid])

    def test_invalid_dino_updates_never_poison_memory_or_count_as_fresh(self):
        node = make_node()
        self.apply(node, 1)
        old = node.segment_dino_embedding(101).copy()
        for bad in ([float('nan'), 0.], [float('inf'), 0.], [0.,0.]):
            node._update_segment_dino_memory(101, np.array(bad), 1., 2)
        node._update_segment_dino_memory(101, np.array([0.,1.]), float('nan'), 2)
        node.params['dinov2_track_ema_alpha'] = .5
        node._update_segment_dino_memory(101, -old, 1., 2)  # Undefined cancellation.
        np.testing.assert_array_equal(node.segment_dino_embedding(101), old)
        self.assertEqual(node.seg_last_dino_seq[101], 1)
        node._drop_segment_record(101)
        self.assertNotIn(101, node.seg_last_dino_seq)

    def test_full_save_reload_preserves_ids_and_resumes_sequence(self):
        node = make_node()
        self.apply(node, 1)
        self.apply(node, 2)
        node.grid_size_m = 10.0
        node.last_T_publish_internal = np.eye(4, dtype=np.float32)
        node.history_vox = {}
        node.latest_grids = node._empty_grid_bundle()
        node.state_dirty = True
        for prefix in ("stm", "ltm", "vox_stm", "vox_ltm"):
            for suffix in ("sum", "w", "last", "last_seen", "viewmask", "z_min", "z_max", "z_sum", "z_count"):
                if not hasattr(node, prefix + "_" + suffix):
                    setattr(node, prefix + "_" + suffix, {})
        # Runtime test artifacts are created and removed only inside this
        # experimental directory, never in the user's shared map-state path.
        with tempfile.TemporaryDirectory(prefix=".identity-test-", dir=NODE_PATH.parent / "tests") as directory:
            node.map_state_path = str(Path(directory) / "state.pkl")
            node.save_map_state()
            self.assertFalse(node.state_dirty)
            self.assertTrue(Path(node.map_state_path).is_file())
            restored = make_node()
            restored.grid_size_m = 10.0
            restored.async_job_seq = 0
            restored.map_state_path = node.map_state_path
            restored.load_map_state()
            self.assertEqual(restored.async_job_seq, 2)
            self.assertEqual(restored.last_applied_observation_seq, 2)
            self.assertEqual(restored.entity_memory.segment_entity, node.entity_memory.segment_entity)
            self.assertEqual(restored.world_model_id, node.world_model_id)
            self.assertEqual(restored.dino_base_dim, node.dino_base_dim)
            self.assertFalse(restored.entity_edges.states)
            self.apply(restored, 3)
            self.assertEqual(restored.entity_memory.segment_entity[101], 1)
            self.assertEqual(restored.entity_memory.records[1].observations, 3)

            incompatible = make_node()
            incompatible.grid_size_m = 10.0
            incompatible.async_job_seq = 0
            incompatible.voxel_size = 0.10
            incompatible.map_state_path = node.map_state_path
            warnings = []
            incompatible.get_logger = lambda: SimpleNamespace(warn=warnings.append, info=lambda *_: None)
            incompatible.load_map_state()
            self.assertEqual(incompatible.seg_w, {})
            self.assertIn("voxel_size", warnings[0])

            node._stationary_pose_anchor = np.eye(4, dtype=np.float32)
            node._stationary_pose_anchor[0, 3] = 1.25
            node.save_map_state()
            anchored = make_node()
            anchored.grid_size_m = 10.0
            anchored.async_job_seq = 0
            anchored.map_state_path = node.map_state_path
            warnings.clear()
            anchored.get_logger = lambda: SimpleNamespace(warn=warnings.append, info=lambda *_: None)
            anchored.load_map_state()
            self.assertFalse(anchored.seg_w)
            self.assertIn('stationary_pose_lock', warnings[0])
            anchored.params['stationary_pose_lock'] = True
            anchored.load_map_state()
            np.testing.assert_array_equal(anchored._stationary_pose_anchor, node._stationary_pose_anchor)
            self.assertEqual(set(anchored.seg_w), {101})
            node._stationary_pose_anchor = None
            node.save_map_state()

            # Legacy segment maps load without inventing prior entity evidence.
            with open(node.map_state_path, "rb") as stream:
                legacy = pickle.load(stream)
            legacy["version"] = 2
            legacy.pop("entities")
            legacy.pop("identity_metadata")
            legacy.pop("world_model_id")
            with open(node.map_state_path, "wb") as stream:
                pickle.dump(legacy, stream)
            restored = make_node()
            restored.grid_size_m = 10.0
            restored.async_job_seq = 0
            restored.map_state_path = node.map_state_path
            restored.load_map_state()
            self.assertEqual(set(restored.seg_w), {101})
            self.assertEqual(restored.entity_memory.records, {})
            self.assertEqual(restored.async_job_seq, 2)
            self.assertTrue(restored.state_dirty)  # Persist the newly allocated namespace.

    def test_actual_fallback_rejects_adjacent_identical_no_overlap(self):
        node = make_node()
        self.apply(node, 1)
        new = observation(2)["pts_int"].copy()
        new[:, 0] += 0.65
        obs = node.build_segment_observation(new, np.array([1, 0], dtype=np.float32), np.ones(len(new)))
        self.assertIsNone(node._segment_association_metrics(obs, 101))

    def test_stacked_surfaces_do_not_get_overlap_from_xy_cells(self):
        node = make_node()
        node.seg_support_voxels[101] = {(1, 1, 1)}
        node.seg_support_cells[101] = {(1, 1)}
        obs = {"support_voxels": {(1, 1, 5)}, "support_cells": {(1, 1)}}
        self.assertEqual(node._segment_support_overlap_with_obs(101, obs), 0.0)

    def test_touching_padded_boxes_are_not_enough_for_fallback(self):
        node = make_node()
        self.apply(node, 1)
        new = observation(2)["pts_int"].copy()
        new[:, 0] += 0.50
        obs = node.build_segment_observation(new, np.array([1, 0], dtype=np.float32), np.ones(len(new)))
        self.assertIsNone(node._segment_association_metrics(obs, 101))

    def test_ambiguous_fallback_remains_unmatched(self):
        node = make_node()
        node.seg_w = {1: 1, 2: 1}
        node._segment_association_metrics = lambda obs, sid: {"score": 0.80 if sid == 1 else 0.79}
        matches, _ = node.match_segment_observations({0: {}})
        self.assertEqual(matches, {})

    def test_legacy_merge_cannot_collapse_independent_entities(self):
        node = make_node()
        left = np.zeros((4, 10), dtype=bool)
        left[:, :5] = True
        self.apply(node, 1, [left, ~left])
        self.apply(node, 2, [left, ~left])
        self.assertFalse(node._merge_segment_records(101, 102))
        self.assertEqual(len(node.seg_w), 2)

    def test_dashboard_parameters_exist_in_node(self):
        dashboard = ast.parse((NODE_PATH.parent / "crossmaps_dashboard.py").read_text())
        assignments = {n.targets[0].id: n.value for n in dashboard.body
                       if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
        groups = ast.literal_eval(assignments["PARAM_GROUPS"])
        for group in groups:
            if group["id"] in ("identity", "segments", "graph"):
                for parameter in group["params"]:
                    self.assertIn(parameter["name"], DEFAULTS)
        images = ast.literal_eval(assignments["IMAGE_LAYERS"])
        self.assertIn("entity_projection", {image["id"] for image in images})

    def test_entity_camera_colors_and_unknowns(self):
        node = make_node()
        self.apply(node, 1)
        self.apply(node, 2)
        node.entity_memory.pending[303] = {"reason": "ambiguous_support"}
        node.last_segment_projection_context = ({"id_image": np.array([[0, 101, 202, 303]])}, object())
        node.pub_entity_id_projection = object()
        node.pub_entity_id_projection_compressed = object()
        node.debug_rgb_image_has_subscribers = lambda *_: True
        captured = []
        node.publish_debug_rgb_image = lambda raw, compressed, rgb, header, **kwargs: captured.append(rgb)
        NAMESPACE["CrossMaps3Node"].publish_entity_id_projection(node)
        pixels = captured[0][0]
        self.assertEqual(tuple(pixels[0]), (0, 0, 0))
        self.assertEqual(tuple(pixels[1]), node._segment_instance_rgb(1))
        self.assertEqual(tuple(pixels[2]), (175, 175, 175))
        self.assertEqual(tuple(pixels[3]), (255, 180, 50))

    def cloud_node(self):
        node = make_node()
        for seq in (1, 2):
            left = np.zeros((4, 10), dtype=bool)
            left[:, :5] = True
            self.apply(node, seq, [left, ~left])
        for name in ("pub_segment_voxels", "pub_entity_voxels", "pub_obj_instance_voxels"):
            publisher = SimpleNamespace(messages=[], subscribers=1)
            publisher.publish = publisher.messages.append
            publisher.get_subscription_count = lambda p=publisher: p.subscribers
            setattr(node, name, publisher)
        node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: 123))
        return node

    def capture_clouds(self, node, transform=None):
        def cloud(points, rgb, frame, stamp):
            return SimpleNamespace(points=points.copy(), rgb=rgb.copy(), frame=frame, stamp=stamp)
        with patch.dict(NAMESPACE, make_cloud_xyzrgb=cloud):
            node.publish_segment_instance_voxels(np.eye(4) if transform is None else transform)

    def test_separate_clouds_share_geometry_not_colors(self):
        node = self.cloud_node()
        # Simulate a confirmed common membership without changing surface owners.
        node.entity_memory.segment_entity = {101: 7, 102: 7}
        transform = np.eye(4)
        transform[0, 3] = 2
        self.capture_clouds(node, transform)
        segment = node.pub_segment_voxels.messages[0]
        entity = node.pub_entity_voxels.messages[0]
        np.testing.assert_array_equal(segment.points, entity.points)
        self.assertTrue(np.all(segment.points[:, 0] >= 2))
        self.assertEqual(segment.frame, "map")
        self.assertEqual(segment.stamp, entity.stamp)
        self.assertEqual(len(np.unique(segment.rgb)), 2)
        np.testing.assert_array_equal(entity.rgb, NAMESPACE["pack_rgb"](*node._segment_instance_rgb(7)))
        self.assertFalse(node.pub_obj_instance_voxels.messages)
        self.assertEqual(set(node.voxel_segment_id.values()), {101, 102})

    def test_entity_cloud_unknown_and_ambiguous_match_camera_palette(self):
        node = self.cloud_node()
        node.entity_memory.segment_entity.clear()
        node.entity_memory.pending[101] = {"reason": "ambiguous_support"}
        self.capture_clouds(node)
        expected = {NAMESPACE["pack_rgb"](175, 175, 175), NAMESPACE["pack_rgb"](255, 180, 50)}
        self.assertEqual(set(node.pub_entity_voxels.messages[0].rgb), expected)

    def test_cloud_publication_switches_subscribers_and_budgets(self):
        node = self.cloud_node()
        node.params.update(publish_segment_voxels=False, object_instance_cloud_max_points=7,
                           publish_object_instance_voxels=True, instance_color_by_entity=False)
        node.pub_entity_voxels.subscribers = 0
        self.capture_clouds(node)
        self.assertFalse(node.pub_segment_voxels.messages)
        self.assertFalse(node.pub_entity_voxels.messages)
        self.assertEqual(len(node.pub_obj_instance_voxels.messages[0].points), 7)
        node.pub_obj_instance_voxels.subscribers = 0
        node._segment_instance_candidate_ids = lambda: self.fail("No subscribers: must not build geometry")
        self.capture_clouds(node)

    def test_empty_clouds_clear_old_geometry_even_with_graph_off(self):
        node = self.cloud_node()
        self.capture_clouds(node)
        node.params.update(use_segment_memory=False, use_object_abstraction=False)
        with patch.dict(NAMESPACE, make_cloud_xyzrgb=lambda points, *args: points):
            node.publish_entity_views(np.eye(4))
        self.assertEqual(node.pub_segment_voxels.messages[-1].shape, (0, 3))
        self.assertEqual(node.pub_entity_voxels.messages[-1].shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
