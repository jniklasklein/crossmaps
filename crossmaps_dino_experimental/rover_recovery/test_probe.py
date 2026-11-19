"""Pure timestamp checks; no ROS or rover needed."""

import unittest
from types import SimpleNamespace as NS

from probe import Stream


def stamped(ns):
    return NS(header=NS(stamp=NS(sec=ns // 10**9, nanosec=ns % 10**9)))


class StreamTest(unittest.TestCase):
    def test_replayed_and_expired_stamps_arent_fresh(self):
        s = Stream()
        for t, ns in [(1, 0), (2, 100), (3, 100), (4, 55)]:
            s.observe(stamped(ns), t)
        report = s.report(4, 1)
        self.assertFalse(report["fresh"])
        self.assertEqual(report["distinct_stamps"], 2)
        self.assertEqual(report["backwards_steps"], 1)
        self.assertEqual(report["advancing_stamps"], 1)
        s.observe(stamped(101), 5)
        self.assertTrue(s.report(5.1, 1)["fresh"])
        self.assertFalse(s.report(6.1, 1)["fresh"])

    def test_velocity_does_not_require_stamp(self):
        s = Stream()
        for val in [0., 0.21]:
            s.observe(NS(linear=NS(x=val, y=0., z=0.), angular=NS(x=0., y=0., z=0.)), 1, cmd=True)
        self.assertEqual(s.nonzero, 1)
        self.assertEqual(s.received, 2)
        self.assertFalse(s.report(2, 1)["fresh"])


if __name__ == "__main__":
    unittest.main()
