import unittest

import cv2
import numpy as np

from collab_geometry import pose_delta, transform
from collab_pnp import audit_reprojection, bidirectional_pnp_seed, prepare_reprojection, refine_reprojection


def scene(seed=7, localized=False, competing=False):
    rng = np.random.default_rng(seed)
    K = np.array([[540.0, 0, 320.0], [0, 530.0, 240.0], [0, 0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.array([0.015, 0.09, -0.025]))[0]
    T[:3, 3] = [-0.18, 0.04, 0.02]
    uv = rng.uniform([245, 165] if localized else [60, 55], [286, 216] if localized else [575, 415], (120, 2))
    z = rng.uniform(1.5, 3.8, len(uv))
    p = np.column_stack(((uv[:, 0] - 320) * z / 540, (uv[:, 1] - 240) * z / 530, z))
    q = transform(p, T)
    if competing:
        q[60:, 0] += 0.32
    u = q @ K.T
    uv2 = u[:, :2] / u[:, 2:]
    frames = []
    for pixels, points in ((uv, p), (uv2, q)):
        d = np.zeros((480, 640), np.float32)
        for (x, y), v in zip(np.rint(pixels).astype(int), points):
            if 2 <= x < 638 and 2 <= y < 478:
                d[y - 1 : y + 2, x - 1 : x + 2] = v[2]
        frames.append(dict(depth=d, K=K))
    return frames, (uv, uv2), T


class OneSidedDepthTests(unittest.TestCase):
    def test_joint_plane_and_texture_recovers_small_true_offset(self):
        # An independent depth observation of a slanted plane at each camera,
        # not a point cloud transformed using the starting (noisy) estimate.
        K = np.array([[420.0, 0, 200.0], [0, 410.0, 140.0], [0, 0, 1.0]])
        actual = np.eye(4)
        actual[:3, :3] = cv2.Rodrigues(np.array([-0.02, 0.035, 0.015]))[0]
        actual[:3, 3] = [-0.08, 0.012, 0.02]
        ys, xs = np.mgrid[:280, :400]
        rays = np.stack(((xs - 200) / 420, (ys - 140) / 410, np.ones_like(xs)), axis=-1)
        n = np.array([0.13, -0.24, 1.0])
        n2 = actual[:3, :3] @ n
        z1 = 2.0 / (rays @ n)
        z2 = (2.0 + n2 @ actual[:3, 3]) / (rays @ n2)
        frames = ({"K": K, "depth": z1.astype("f")}, {"K": K, "depth": z2.astype("f")})
        a = np.column_stack((xs[50:231:30, 55:348:34].ravel(), ys[50:231:30, 55:348:34].ravel()))
        point = rays[a[:, 1], a[:, 0]] * z1[a[:, 1], a[:, 0], None]
        pixel = transform(point, actual) @ K.T
        b = pixel[:, :2] / pixel[:, 2:]
        old = actual.copy()
        old[:3, 3] += [0.008, -0.012, 0.025]
        data = prepare_reprojection(*frames, a, b, old)
        self.assertGreater(min(len(s) for s in data["surfaces"]), 100)
        fit = refine_reprojection(data, old)
        self.assertTrue(fit["ok"], fit)
        self.assertLess(pose_delta(actual, fit["T_target_source"])[0], 0.002)

    def test_refinement_repairs_small_metric_seed_error_and_audits_unseen_pixels(self):
        frames, pairs, T = scene()
        seed = T.copy()
        seed[:3, 3] += [0.012, -0.016, 0.03]
        data = prepare_reprojection(*frames, *pairs, seed, seed_indices=np.arange(80))
        result = refine_reprojection(data, seed)
        self.assertTrue(result["ok"], result)
        self.assertGreaterEqual(result["reprojection_holdout_nonseed_inliers"], 3)
        self.assertLess(pose_delta(T, result["T_target_source"])[0], 0.002)
        # A small ICP step below the usual metric ICP motion guard can still
        # move every visual feature several tens of pixels.
        icp = T.copy()
        icp[0, 3] += 0.12
        self.assertFalse(audit_reprojection(data, icp)["ok"])

    def test_holdout_corruption_fails_even_with_a_good_training_fit(self):
        frames, pairs, T = scene()
        data = prepare_reprojection(*frames, *pairs, T)
        data["b"][data["hold"]] += [100.0, 60.0]
        result = refine_reprojection(data, T)
        self.assertFalse(result["ok"])
        self.assertGreaterEqual(result["reprojection_inliers"], 30)
        self.assertEqual(result["reason"], "insufficient_held_out_pixel_support")

    def test_complementary_holes_preserve_signed_pose(self):
        frames, pairs, T = scene()
        for i, (d, uv) in enumerate(zip(frames, pairs)):
            for x, y in np.rint(uv[i::2]).astype(int):
                d["depth"][y - 1 : y + 2, x - 1 : x + 2] = 0
        fit = bidirectional_pnp_seed(*frames, *pairs)
        self.assertTrue(fit["ok"], fit)
        self.assertLess(fit["pnp_paired_depth"], 12)
        np.testing.assert_allclose(fit["T_target_source"], T, atol=0.002)
        rev = bidirectional_pnp_seed(*frames[::-1], *pairs[::-1])
        self.assertTrue(rev["ok"], rev)
        self.assertLess(pose_delta(rev["T_target_source"], np.linalg.inv(T))[0], 0.002)

    def test_missing_reverse_depth_must_not_accept(self):
        frames, pairs, _ = scene()
        frames[1]["depth"].fill(0)
        fit = bidirectional_pnp_seed(*frames, *pairs)
        self.assertFalse(fit["ok"])
        self.assertEqual(fit["reason"], "pnp_insufficient_depth_in_either_direction")

    def test_wrong_or_inconsistent_depth_is_not_geometry(self):
        frames, pairs, _ = scene()
        frames[1]["depth"] *= 3.0
        fit = bidirectional_pnp_seed(*frames, *pairs)
        self.assertFalse(fit["ok"])

    def test_repeated_or_localized_features_are_not_independent(self):
        frames, pairs, _ = scene()
        fit = bidirectional_pnp_seed(*frames, *(np.repeat(u[:6], 10, axis=0) for u in pairs))
        self.assertFalse(fit["ok"])
        frames, pairs, _ = scene(localized=True)
        fit = bidirectional_pnp_seed(*frames, *pairs)
        self.assertFalse(fit["ok"])
        self.assertEqual(fit["reason"], "pnp_localized_or_degenerate_support")

    def test_conflicting_motion_and_mismatched_pixels_do_not_seed(self):
        frames, pairs, _ = scene(competing=True)
        fit = bidirectional_pnp_seed(*frames, *pairs)
        self.assertFalse(fit["ok"], fit)
        frames, (a, b), _ = scene()
        fit = bidirectional_pnp_seed(*frames, a, np.roll(b, 2, axis=0))
        self.assertFalse(fit["ok"], fit)
