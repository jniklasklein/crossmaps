from pathlib import Path
from types import SimpleNamespace as NS
import math
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from ros_node import parked, planar_pose
from install import patch_vendor


def seed(yaw=1., frame='odom', z=0.):
    p = NS(position=NS(x=2., y=-1., z=z),
           orientation=NS(x=0., y=0., z=math.sin(yaw/2), w=math.cos(yaw/2)))
    return NS(header=NS(frame_id=frame), child_frame_id='base_footprint', pose=NS(pose=p))


class GuardTests(unittest.TestCase):
    def test_handoff_preserves_signed_pose(self):
        self.assertEqual(planar_pose(seed(yaw=-2.)),(2.,-1.,-2.))

    def test_forbid_seed_from_unrelated_or_corrupt_frame(self):
        for bad in (seed(frame='map'),seed(z=.8),seed(yaw=float('nan'))):
            with self.assertRaises(ValueError):
                planar_pose(bad)
        bad = seed(); bad.pose.pose.orientation.x=.6
        with self.assertRaises(ValueError): planar_pose(bad)

    def test_parked_requires_fresh_unchanging_encoders(self):
        quiet=[(i*.1,123.,456.) for i in range(14)]
        self.assertTrue(parked(quiet,1.35))
        self.assertFalse(parked(quiet,2.))
        self.assertFalse(parked(quiet[-2:],1.35))
        self.assertFalse(parked(quiet+[(1.34,123.01,456.)],1.35))

    def test_never_overwrite_unrecognized_launch(self):
        with self.assertRaises(ValueError):
            patch_vendor(b'# user has a different hardware configuration\n')
