import unittest
import numpy as np
from fixed_scan_probe import fixed_ranges


class FixedScanTests(unittest.TestCase):
    def test_variable_width_is_mapped_by_angle_not_truncated(self):
        for size in (500,501,502,503):
            source=np.full(size,2.)
            result=fixed_ranges(source,0.,2*np.pi/(size-1),0.,2*np.pi,502,.02,25.)
            self.assertEqual(len(result),502)
            self.assertEqual(result[0],2.)
            self.assertEqual(result[-1],2.)
            self.assertTrue(np.all(result[np.isfinite(result)]==2.))

    def test_missing_returns_are_not_interpolated(self):
        result=fixed_ranges([1.,np.nan,np.inf,0.,30.,2.],0.,1.,0.,5.,6,.02,25.)
        np.testing.assert_allclose(result,[1.,np.nan,np.nan,np.nan,np.nan,2.],equal_nan=True)

    def test_collision_keeps_foreground(self):
        result=fixed_ranges([1.,3.,2.,4.],0.,1.,0.,3.,2,.02,25.)
        np.testing.assert_allclose(result,[1.,2.])

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaises(ValueError):fixed_ranges([1.,2.],0.,0.,0.,1.,2,.02,25.)
