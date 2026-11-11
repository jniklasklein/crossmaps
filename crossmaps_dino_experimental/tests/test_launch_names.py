"""Keep the public entry point, ROS name and dashboard default in agreement."""
import ast
from pathlib import Path
import unittest


class LaunchNamingTests(unittest.TestCase):
    def test_node_name_matches_dashboard_target(self):
        root = Path(__file__).resolve().parents[1]
        node = ast.parse((root / 'crossmaps_v2_node.py').read_text())
        node_class = next(n for n in node.body if isinstance(n, ast.ClassDef) and n.name == 'CrossMapsNode')
        init = next(n for n in node_class.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        base_init = next(n for n in ast.walk(init) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute) and n.func.attr == '__init__'
                         and isinstance(n.func.value, ast.Call)
                         and isinstance(n.func.value.func, ast.Name) and n.func.value.func.id == 'super')
        name = ast.literal_eval(base_init.args[0])
        self.assertEqual(name, 'crossmaps_v2_node')
        dashboard = ast.parse((root / 'crossmaps_dashboard.py').read_text())
        target = next(n for n in ast.walk(dashboard) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument'
                      and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == '--target-node')
        default = next(k.value for k in target.keywords if k.arg == 'default')
        self.assertEqual(ast.literal_eval(default), '/' + name)


if __name__ == '__main__':
    unittest.main()
