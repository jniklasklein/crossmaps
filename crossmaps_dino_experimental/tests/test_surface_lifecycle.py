import unittest
import numpy as np
from test_node_identity_integration import make_node, observation
from segment_overlay import render_mask_consensus, id_color
from test_object_assembly import part
from object_assembly import ObjectAssemblies


def populated(count=6):
    node = make_node()
    for seq in range(1, count+1):
        node.now = float(seq)
        node.apply_semantic_result(observation(seq), 'test')
    return node


class LifecycleTests(unittest.TestCase):
    def test_semantic_eviction_does_not_erase_geometric_identity(self):
        node = populated()
        support = node.seg_support_voxels[101].copy()
        for key in support:
            node._drop_segment_voxel_owner_if_empty(key)
        self.assertEqual(node.seg_support_voxels[101], support)
        self.assertTrue(all(node.voxel_segment_id[k] == 101 for k in support))

    def test_mature_track_survives_looking_away_but_tentative_expires(self):
        node = populated()
        self.assertEqual(node.retire_stale_segments(1000.), 0)
        self.assertIn(101, node.seg_w)
        tentative = populated(1)
        self.assertEqual(tentative.retire_stale_segments(1000.), 1)

    def test_misses_must_be_consecutive_and_mature_tracks_require_more(self):
        node = populated()
        key = next(iter(node.seg_support_voxels[101]))
        for _ in range(40):
            node.prune_segment_support_from_free_space({key})
            node.prune_segment_support_from_free_space(set())
        self.assertEqual(node.voxel_segment_id[key], 101)
        for _ in range(7):
            node.prune_segment_support_from_free_space({key})
        self.assertIn(key, node.seg_support_voxels[101])
        node.prune_segment_support_from_free_space({key})
        self.assertNotIn(key, node.seg_support_voxels[101])

    def test_historical_support_repack_is_not_positive_evidence(self):
        node = populated()
        key = next(iter(node.seg_support_voxels[101]))
        node.segment_voxel_miss_hits[key] = 3
        node._set_segment_support(101, node.seg_support_voxels[101].copy(), set())
        self.assertEqual(node.segment_voxel_miss_hits[key], 3)
        node._claim_segment_voxels(101, {key})
        self.assertNotIn(key, node.segment_voxel_miss_hits)

    def test_current_depth_endpoints_override_crossing_free_space_rays(self):
        node = populated()
        points = observation(7)['pts_int']
        key = node.xyz_to_voxel(*map(float, points[0]))
        neighbour = (key[0]-1, key[1], key[2])
        far = (150,150,50)
        node.vox_clear_hits = {key:7, neighbour:7}
        node.segment_voxel_miss_hits = {key:7, neighbour:7}
        node._clear_voxel_candidate_keys = lambda *_: {key, neighbour, far}
        candidates = []
        node.prune_segment_support_from_free_space = lambda keys: (candidates.append(set(keys)) or (0,0))
        node.clear_voxels_from_depth_rays(points, np.zeros(3))
        self.assertEqual(candidates, [{far}])
        self.assertNotIn(key, node.segment_voxel_miss_hits)
        self.assertNotIn(neighbour, node.vox_clear_hits)

    def test_bounded_descriptor_evidence_does_not_grow_beyond_weight(self):
        node = populated()
        node.params['segment_weight_max'] = 2.
        for seq in range(7, 107):
            node.now=float(seq);node.apply_semantic_result(observation(seq), 'test')
        self.assertLessEqual(float(np.linalg.norm(node.seg_sum[101])), node.seg_w[101]+1e-5)
        self.assertEqual(set(node.seg_w), {101})

    def test_temporal_guidance_cannot_expand_owner_across_a_large_mask(self):
        node = populated()
        frame = observation(7)
        points = frame['pts_int'].copy()
        points[12:] += [3.,0.,0.]
        owners = np.zeros(len(points), dtype=np.int32); owners[:12] = 101
        guided, _, _, _ = node._prepare_surface_consensus(frame['masks'], owners, points,
            frame['uu'], frame['vv'], frame['image_shape'], frame['mask_emb'], frame['dino_mask_emb'],
            frame['dino_mask_quality'], 7, 7.)
        self.assertTrue(np.all(guided[12:] == 0))

    def test_temporal_match_reinforces_incumbent_instead_of_birth_for_remainder(self):
        node = populated()
        # Current world depth moved by a voxel: still within the validated
        # temporal geometry gate, but projected anchors and fallback are absent.
        node.match_segment_observations = lambda *a,**k: ({},{})
        node.build_projected_segment_id_context = lambda pts,*a: dict(
            point_sids=np.zeros(len(pts),dtype=np.int32),visible_counts={})
        before=node.seg_obs_count[101]
        for seq in range(7,47):
            frame=observation(seq)
            frame['pts_int'][:,2]+=.05+.004*np.sin(seq)
            frame['mask_emb'][:]=[np.cos(.02*seq),np.sin(.02*seq)]
            frame['dino_mask_emb'][:]=[np.cos(.01*seq),np.sin(.01*seq)]
            node.now=float(seq)
            node.apply_semantic_result(frame,'test')
            self.assertEqual(set(node.seg_w),{101})
        self.assertGreater(node.seg_obs_count[101],before)
        self.assertIn(101,node.projection['id_image'])

    def test_stationary_anchor_cancels_pose_drift_only_when_explicitly_enabled(self):
        node = make_node(); node.params['stationary_pose_lock'] = True
        local = np.array([[1.,0.,1.], [0.,1.,1.]], dtype=np.float32)
        for angle in np.linspace(0., 1., 20):
            c,s=np.cos(angle),np.sin(angle)
            transform = np.eye(4);transform[:2,:2]=[[c,-s],[s,c]]
            node.last_depth_transform = transform
            raw = local @ transform[:3,:3].T
            stable,_ = node._apply_stationary_pose_anchor(raw, np.zeros(3))
            np.testing.assert_allclose(stable,local,atol=1e-6)
        node.params['stationary_pose_lock']=False
        raw,_=node._apply_stationary_pose_anchor(raw,np.zeros(3))
        self.assertIsNone(node._stationary_pose_anchor)
        self.assertFalse(np.allclose(raw,local))

    def test_confirmed_assembly_does_not_flicker_on_one_noisy_extent(self):
        nodes={1:part(),2:part(.6,height=.7)}
        memory=ObjectAssemblies()
        for seq in range(1,4):memory.update(seq,float(seq),nodes,nodes,voxel_size=.1)
        self.assertEqual(memory.canonical(2),1)
        nodes[2]['max']=(6.,1.,1.)
        memory.update(4,4.,nodes,nodes,voxel_size=.1)
        self.assertEqual(memory.canonical(2),1)
        for seq in (5,6):memory.update(seq,float(seq),nodes,nodes,voxel_size=.1)
        self.assertEqual(memory.canonical(2),2)

    def test_mixed_overlay_preserves_local_support_without_claiming_whole_mask(self):
        shape=(40,60);mask=np.ones(shape,dtype=bool)
        seeds=np.zeros(shape,dtype=np.int32);seeds[20,10]=1;seeds[20,50]=2
        decisions=[dict(mask=0,winner=None,segments=[1,2],reason='multiple_supported_owners')]
        rgb=np.zeros((*shape,3),dtype=np.uint8)
        view=render_mask_consensus(shape,[mask],decisions,rgb,lambda s:s,id_color,seeds=seeds,radius=2)
        np.testing.assert_array_equal(view[20,10],(np.array(id_color(1))*.78).astype(np.uint8))
        np.testing.assert_array_equal(view[20,50],(np.array(id_color(2))*.78).astype(np.uint8))
        self.assertFalse(np.array_equal(view[20,30],view[20,10]))
        self.assertEqual(np.count_nonzero(seeds),2)
