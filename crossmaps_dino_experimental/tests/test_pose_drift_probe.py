import json
import math
from types import SimpleNamespace
import unittest
from pose_drift_probe import yaw, summarize


class PoseProbeTests(unittest.TestCase):
    def test_yaw_normalizes_quaternion_and_rejects_invalid(self):
        q=SimpleNamespace(x=0.,y=0.,z=2*math.sin(.3),w=2*math.cos(.3))
        self.assertAlmostEqual(yaw(q),.6)
        self.assertIsNone(yaw(SimpleNamespace(x=0.,y=0.,z=0.,w=0.)))

    def test_unwrap_does_not_invent_a_full_turn_at_pi(self):
        rows=[dict(t=i,yaw=math.radians(v),wz=.01,orientation_available=True)
              for i,v in enumerate((178,179,-180,-179,-178))]
        report=summarize(rows)
        self.assertAlmostEqual(report['yaw']['delta'],4.)
        self.assertAlmostEqual(report['yaw']['slope_per_s'],1.)
        self.assertAlmostEqual(report['wz']['mean'],.01)
        json.dumps(report)

    def test_missing_orientation_not_reported_as_zero_yaw(self):
        report=summarize([dict(t=0,yaw=None,wz=0.),dict(t=2,yaw=None,wz=0.)])
        self.assertNotIn('yaw',report)
        self.assertEqual(summarize([]),{'count':0})
