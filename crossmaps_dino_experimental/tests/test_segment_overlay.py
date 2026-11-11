import unittest
import numpy as np
from segment_overlay import dense_display_ids, id_color, render_overlay


class OverlayTests(unittest.TestCase):
    def test_mask_fill_preserves_seeds_and_multiple_owners(self):
        seeds = np.zeros((12, 16), dtype=np.int32)
        seeds[6, 3], seeds[6, 12] = 101, 202
        before = seeds.copy()
        mask = np.ones_like(seeds, dtype=bool)
        dense = dense_display_ids(seeds, [mask], radius=0)
        np.testing.assert_array_equal(seeds, before)
        self.assertEqual(set(np.unique(dense)), {101, 202})
        self.assertEqual(dense[6, 3], 101)
        self.assertEqual(dense[6, 12], 202)

    def test_small_mask_boundaries_win_without_leaking(self):
        seeds = np.zeros((12, 16), dtype=np.int32)
        seeds[6, 3], seeds[6, 12] = 1, 2
        small = np.zeros_like(seeds, dtype=bool)
        small[:, :5] = True
        large = np.ones_like(seeds, dtype=bool)
        dense = dense_display_ids(seeds, [large, small], radius=0)
        self.assertTrue(np.all(dense[small] == 1))
        self.assertTrue(np.all(small[:, :5]))

    def test_no_masks_only_expand_current_seeds_locally(self):
        seeds = np.zeros((30, 30), dtype=np.int32)
        seeds[15, 15] = 1
        dense = dense_display_ids(seeds, [], radius=3)
        self.assertEqual(dense[15, 18], 1)
        self.assertEqual(dense[15, 20], 0)
        self.assertEqual(dense[0, 0], 0)

    def test_unseeded_small_mask_is_not_colored_by_parent_or_dilation(self):
        seeds = np.zeros((12, 16), dtype=np.int32)
        seeds[6, 5] = 1
        small = np.zeros_like(seeds, dtype=bool)
        small[:, :5] = True
        large = np.ones_like(seeds, dtype=bool)
        dense = dense_display_ids(seeds, [large, small], radius=4)
        self.assertTrue(np.all(dense[small] == 0))
        self.assertTrue(np.all(dense[~small] == 1))

    def test_unseeded_mask_is_gray_not_invented_id(self):
        seeds = np.zeros((10, 10), dtype=np.int32)
        mask = np.ones_like(seeds, dtype=bool)
        dense = dense_display_ids(seeds, [mask])
        self.assertFalse(np.any(dense))
        out = render_overlay(dense, [mask], None, id_color)
        self.assertTrue(np.all(out == out[:, :, :1]))
        self.assertGreater(int(out[0, 0, 0]), 100)

    def test_palette_and_default_blending_are_bright_on_black(self):
        ids = np.arange(1, 101).reshape(10, 10)
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        out = render_overlay(ids, [], rgb, id_color)
        self.assertTrue(np.all(out.max(axis=2) >= 198))
        self.assertTrue(np.all(rgb == 0))


if __name__ == '__main__':
    unittest.main()
