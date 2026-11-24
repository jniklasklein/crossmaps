"""Bundle tests use tiny fake binaries and never open a GUI."""
import hashlib
import os
from pathlib import Path
import plistlib
import tempfile
import unittest
from unittest.mock import patch

import macos_rviz


class RvizBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.prefix = self.root / "ros"
        (self.prefix / "bin").mkdir(parents=True)
        (self.prefix / "lib").mkdir()
        (self.prefix / "opt/rviz_ogre_vendor/lib").mkdir(parents=True)
        self.python = self.prefix / "bin/python3"
        self.python.write_bytes(b"fake python")
        self.source = self.prefix / "bin/rviz2"
        self.source.write_bytes(b"original signed RViz bytes")
        self.source.chmod(0o755)
        self.apps = self.root / ".runtime/rviz-apps"
        self.patches = [patch.object(macos_rviz, "APP_ROOT", self.apps),
                        patch.object(macos_rviz.platform, "system", return_value="Darwin")]
        for mocked in self.patches:
            mocked.start()

    def tearDown(self):
        for mocked in reversed(self.patches):
            mocked.stop()
        self.temporary.cleanup()

    def test_identifiable_per_rover_bundles_preserve_binary_bytes(self):
        for robot, letter in (("rover_a", "A"), ("rover_b", "B")):
            executable = Path(macos_rviz.prepare_rviz_app(self.python, robot))
            self.assertEqual(executable.read_bytes(), self.source.read_bytes())
            self.assertTrue(os.access(executable, os.X_OK))
            contents = executable.parent.parent
            self.assertEqual(contents.parent.name, f"CrossMaps Rover {letter} Navigation.app")
            metadata = plistlib.loads((contents / "Info.plist").read_bytes())
            self.assertEqual(metadata["CFBundleExecutable"], "RViz")
            self.assertEqual(metadata["CFBundleIdentifier"], "org.crossmaps.simulation.rviz." + robot.replace("_", "-"))
            self.assertEqual((contents / "lib").resolve(), (self.prefix / "lib").resolve())
            self.assertEqual((contents / "opt").resolve(), (self.prefix / "opt").resolve())

    def test_idempotent_and_refreshes_only_when_source_hash_changes(self):
        destination = Path(macos_rviz.prepare_rviz_app(self.python, "rover_a"))
        before = destination.stat().st_mtime_ns
        macos_rviz.prepare_rviz_app(self.python, "rover_a")
        self.assertEqual(destination.stat().st_mtime_ns, before)
        self.source.write_bytes(b"new unmodified official RViz")
        macos_rviz.prepare_rviz_app(self.python, "rover_a")
        self.assertEqual(hashlib.sha256(destination.read_bytes()).digest(), hashlib.sha256(self.source.read_bytes()).digest())

    def test_non_mac_returns_original_and_does_not_make_bundle(self):
        with patch.object(macos_rviz.platform, "system", return_value="Linux"):
            self.assertEqual(macos_rviz.prepare_rviz_app(self.python, "rover_a"), str(self.source.resolve()))
        self.assertFalse(self.apps.exists())

    def test_invalid_robot_cannot_choose_a_destination(self):
        with self.assertRaises(ValueError):
            macos_rviz.prepare_rviz_app(self.python, "../../outside")
        self.assertFalse(self.apps.exists())

    def test_preexisting_symlink_cannot_redirect_bundle_writes(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.apps.parent.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "unexpected path"):
            macos_rviz.prepare_rviz_app(self.python, "rover_a")
        self.assertEqual(list(outside.iterdir()), [])

    def test_library_links_do_not_replace_existing_real_directories(self):
        destination = Path(macos_rviz.prepare_rviz_app(self.python, "rover_a"))
        link = destination.parent.parent / "lib"
        link.unlink()
        link.mkdir()
        with self.assertRaisesRegex(RuntimeError, "real directory"):
            macos_rviz.prepare_rviz_app(self.python, "rover_a")
        self.assertTrue(link.is_dir())
        self.assertFalse(link.is_symlink())


if __name__ == "__main__":
    unittest.main()
