"""No recorded rover images needed. Test the held-out contract and transform convention."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from collab_geometry import verify_rgbd
from collab_pnp import _balanced_holdout, audit_reprojection, prepare_reprojection, reprojection_partition


class OrderInvariantVerification(unittest.TestCase):
    def test_partition_is_invariant_to_camera_order_and_row_order(self):
        rng = np.random.default_rng(901)
        dims = [(160, 240), (320, 320)]
        points = [rng.uniform([1,1], [w-2,h-2], size=(120,2)) for h,w in dims]
        valid = rng.random(120) > .1
        h1 = _balanced_holdout(*points, dims, valid)
        order = rng.permutation(len(valid))
        reverse = _balanced_holdout(points[1][order], points[0][order], dims[::-1], valid[order])
        np.testing.assert_array_equal(reverse, h1[order])
        self.assertEqual(h1.sum(), round(.24 * valid.sum()))
        self.assertFalse(np.any(h1 & ~valid))
        for x, (h,w) in zip(points, dims):
            self.assertGreaterEqual(len(np.unique(np.floor(x[h1] / [w/4,h/3]), axis=0)), 4)

    @staticmethod
    def pair():
        h,w = 240,320
        depth = np.full((h,w), 2, dtype=np.float32)
        K = np.array([[220.,0,w/2],[0,220.,h/2],[0,0,1.]])
        rgb = np.random.default_rng(5).integers(0,256,size=(h,w,3),dtype=np.uint8)
        other = cv2.warpAffine(rgb, np.array([[1.,0.,14.],[0.,1.,-4.]]), (w,h))
        # Correspondences on a frontoparallel plane shift by f * t / z.
        expected = np.eye(4); expected[:2,3] = np.array([14.,-4.]) * 2 / 220
        return (dict(id='a',rgb=rgb,depth=depth,K=K),
                dict(id='z',rgb=other,depth=depth.copy(),K=K.copy()), expected)

    def test_precommitted_holdout_is_excluded_from_the_entire_visual_fit(self):
        import collab_geometry as g
        a,b,_ = self.pair()
        px,py,_ = g._orb_matches(a,b)
        train, held = reprojection_partition(a,b,px,py)
        self.assertFalse(np.intersect1d(train, held).size)
        with patch.object(g,'_seed_from_matches',wraps=g._seed_from_matches) as seed:
            result = verify_rgbd(a,b,matcher_mode='orb')
        self.assertTrue(result['ok'], result)
        np.testing.assert_array_equal(seed.call_args.args[2], px[train])
        self.assertEqual(result['holdout_matches'], len(held))
        self.assertEqual(result['reprojection_reserved_count'], len(held))
        self.assertEqual(result['reprojection_holdout_inliers'], result['reprojection_holdout_nonseed_inliers'])
        self.assertGreaterEqual(min(result['reprojection_holdout_forward_inliers'],
                                    result['reprojection_holdout_reverse_inliers']), 2)

    def test_one_fit_per_request_with_inverse_pose_and_directional_diagnostics(self):
        import collab_geometry as g
        a,b,expected = self.pair()
        with patch.object(g,'_verify_ordered',wraps=g._verify_ordered) as solver:
            f = verify_rgbd(a,b,matcher_mode='orb')
            r = verify_rgbd(b,a,matcher_mode='orb')
        self.assertEqual(solver.call_count, 2)  # not best-of-two in either direction
        self.assertTrue(all(call.args[0] is a and call.args[1] is b for call in solver.call_args_list))
        self.assertTrue(f['ok'] and r['ok'], (f,r))
        # ORB's bounded ICP refinement on a smooth plane can move ~1.5 cm;
        # metric correctness is approximate, but reversal must be exact.
        np.testing.assert_allclose(f['T_target_source'][:2,3], expected[:2,3], atol=.02)
        for key in ('T_target_source','T_ransac_target_source'):
            np.testing.assert_allclose(r[key] @ f[key], np.eye(4), atol=1.e-8)
        for k in ('rmse','inliers','translation_sigma','rotation_sigma','reprojection_holdout_count'):
            self.assertAlmostEqual(f[k], r[k])
        self.assertEqual(f['reprojection_holdout_cells'], r['reprojection_holdout_cells'][::-1])
        self.assertEqual(f['icp_inliers'], r['icp_reverse_inliers'])
        self.assertEqual(f['image_cells'], r['target_image_cells'])
        self.assertAlmostEqual(f['overlap_source'], r['overlap_target'])

    def test_cannot_accept_holdout_with_depth_from_only_one_camera(self):
        a,b,_ = self.pair()
        py,px = np.mgrid[20:220:20,20:300:20]
        grid = np.column_stack((px.ravel(), py.ravel())).astype(float)
        data = prepare_reprojection(a,b,grid,grid,np.eye(4))
        data['gb'][data['hold']] = False
        evidence = audit_reprojection(data,np.eye(4))
        self.assertGreater(evidence['reprojection_holdout_inliers'], 8)
        self.assertEqual(evidence['reprojection_holdout_reverse_inliers'], 0)
        self.assertFalse(evidence['ok'])

    def test_unrelated_rgb_is_rejected_in_both_orders_with_tied_ids(self):
        a,b,_ = self.pair()
        a.pop('id'); b.pop('id')
        b['rgb'] = np.random.default_rng(9).integers(0,256,size=b['rgb'].shape,dtype=np.uint8)
        f,r = verify_rgbd(a,b), verify_rgbd(b,a)
        self.assertFalse(f['ok'] or r['ok'])
        self.assertEqual(f['reason'], r['reason'])
        self.assertEqual(f['attempts'], r['attempts'])
