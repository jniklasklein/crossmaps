"""The native dependency installer must not migrate the perception environment."""
import unittest

from setup_nav2 import validate_plan


class DependencyPlanTests(unittest.TestCase):
    def setUp(self):
        self.installed = {name: {"version": version} for name, version in
                          (("python", "3.11.14"), ("numpy", "1.26.4"), ("ros2-distro-mutex", "0.7.0"))}

    def test_add_only_plan(self):
        plan = {"success": True, "actions": {"LINK": [{"name": "ros-humble-nav2-bringup", "version": "1.1.18"}]}}
        self.assertIn("ros-humble-nav2-bringup", validate_plan(plan, self.installed))

    def test_same_version_boost_build_adjustment(self):
        plan = {"success": True, "actions": {
            "UNLINK": [{"name": "libboost-python", "version": "1.86.0"}],
            "LINK": [{"name": "libboost-python", "version": "1.86.0", "build_string": "py311h9fe4b7f_4"}],
        }}
        self.assertIn("libboost-python", validate_plan(plan, self.installed))

    def test_reject_environment_upgrade(self):
        plan = {"success": True, "actions": {
            "UNLINK": [{"name": "python", "version": "3.11.14"}],
            "LINK": [{"name": "python", "version": "3.12.0"}],
        }}
        with self.assertRaisesRegex(RuntimeError, "Refusing to replace"):
            validate_plan(plan, self.installed)

    def test_reject_boost_version_change(self):
        plan = {"success": True, "actions": {
            "UNLINK": [{"name": "libboost-python", "version": "1.86.0"}],
            "LINK": [{"name": "libboost-python", "version": "1.88.0", "build_string": "new"}],
        }}
        with self.assertRaisesRegex(RuntimeError, "Refusing to replace"):
            validate_plan(plan, self.installed)

    def test_reject_different_ros_generation(self):
        self.installed["ros2-distro-mutex"]["version"] = "0.9.0"
        with self.assertRaisesRegex(RuntimeError, "expects existing"):
            validate_plan({"success": True}, self.installed)

    def test_reject_solver_failure(self):
        with self.assertRaisesRegex(RuntimeError, "Dependency solve failed"):
            validate_plan({"success": False}, self.installed)


if __name__ == "__main__":
    unittest.main()
