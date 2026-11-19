import gc
import unittest
import weakref

import cv2
import numpy as np

from collab_visual_matches import _CACHE, _CACHE_LIMIT, visual_matches


class VisualProposalTests(unittest.TestCase):
    def test_uniform_and_ephemeral_cache(self):
        for _ in range(_CACHE_LIMIT + 2):
            img = np.full((64, 96, 3), 118, np.uint8)
            a, b, diag = visual_matches({'rgb': img}, {'rgb': img})
            self.assertEqual(a.shape, (0, 2))
            self.assertEqual(b.shape, a.shape)
            self.assertEqual(diag['visual_matches'], 0)
        ref = weakref.ref(img)
        del img
        gc.collect()
        self.assertIsNone(ref())
        self.assertLessEqual(len(_CACHE), _CACHE_LIMIT)

    def test_contrast_rotation_and_scale_keep_pixel_coordinates(self):
        rng = np.random.default_rng(14)
        gray = cv2.GaussianBlur(rng.integers(0, 255, size=(360, 502), dtype=np.uint8), (3, 3), 0)
        M = cv2.getRotationMatrix2D((251, 180), 13., .85)
        M[:, 2] += (17., -8.)
        im2 = cv2.warpAffine(gray, M, (502, 360))
        im2 = (np.sqrt(im2 / 255.) * 175 + 30).astype(np.uint8)
        a, b, d = visual_matches({'rgb': gray}, {'rgb': im2})
        self.assertGreater(len(a), 50)
        self.assertEqual(d['visual_matches'], len(a))
        right = np.linalg.norm(a @ M[:, :2].T + M[:, 2] - b, axis=1) < 2.
        self.assertGreater(right.mean(), .9)
        for pixels in (a, b):
            self.assertEqual(len(np.unique(np.rint(pixels/4), axis=0)), len(pixels))
