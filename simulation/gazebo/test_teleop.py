from types import SimpleNamespace
from unittest.mock import Mock
import unittest

from teleop import check_control_graph


class ControlGraphTests(unittest.TestCase):
    def test_only_expected_simulation_receiver_and_no_competing_controller(self):
        node = Mock()
        node.get_subscriptions_info_by_topic.return_value = [SimpleNamespace(node_name='crossmaps_gazebo_rover_a')]
        node.count_publishers.return_value = 0
        check_control_graph(node, 'rover_a')
        node.count_publishers.return_value = 1
        check_control_graph(node, 'rover_a', own_publisher=True)
        with self.assertRaisesRegex(RuntimeError, 'Another controller'):
            check_control_graph(node, 'rover_a')
        with self.assertRaisesRegex(RuntimeError, 'only command receiver'):
            check_control_graph(node, 'rover_b')
        node.get_subscriptions_info_by_topic.return_value = []
        with self.assertRaises(RuntimeError):
            check_control_graph(node, 'rover_a')


if __name__ == '__main__':
    unittest.main()
