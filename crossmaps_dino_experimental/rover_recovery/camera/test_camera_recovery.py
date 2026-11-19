import os
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossmaps_oak_guard import FrameProbe, proc_identity, stop_private_group


class CameraRecoveryTests(unittest.TestCase):
    def test_duplicate_stamps_are_not_live(self):
        p = FrameProbe()
        for _ in range(20):
            p.observe('depth', 1)
            p.observe('rgb', 2)
        self.assertFalse(p.complete())
        p.observe('rgb', 3)
        self.assertFalse(p.complete())
        p.observe('depth', 2)
        self.assertTrue(p.complete())

    def test_zero_and_regression_are_not_progress(self):
        p = FrameProbe()
        for stamp in (0, 0, 20, 19):
            p.observe('rgb', stamp)
            p.observe('depth', stamp)
        self.assertFalse(p.complete())
        # A source-clock reset does not leave the supervisor permanently blind.
        for kind in ('rgb', 'depth'):
            p.observe(kind, 20)
        self.assertTrue(p.complete())

    @unittest.skipUnless(Path('/proc/self/stat').exists(), 'Linux process identity')
    def test_owner_identity(self):
        self.assertIsNotNone(proc_identity(os.getpid()))
        self.assertIsNone(proc_identity(99999999))

    @unittest.skipUnless(Path('/proc/self/stat').exists(), 'target Linux process groups')
    def test_shutdown_private_group_leaves_unrelated_service_alive(self):
        # A test double for RTAB-Map, in a different group, must never be signalled.
        unrelated = subprocess.Popen(['sleep', '20'], start_new_session=True)
        child = subprocess.Popen(['sleep', '20'], start_new_session=True)
        try:
            stop_private_group(child)
            self.assertIsNotNone(child.poll())
            self.assertIsNone(unrelated.poll())
        finally:
            for proc in (child, unrelated):
                if proc.poll() is None:
                    proc.terminate()
                proc.wait(timeout=4)


if __name__ == '__main__':
    unittest.main()
