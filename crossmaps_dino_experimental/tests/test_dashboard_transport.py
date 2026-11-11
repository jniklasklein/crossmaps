"""Transport failure checks without starting ROS or an HTTP server."""
import unittest
from http.server import BaseHTTPRequestHandler
from types import SimpleNamespace
from unittest.mock import patch

from crossmaps_dashboard import CrossMapsDashboard, DashboardState, make_handler


class TransportTests(unittest.TestCase):
    def test_expected_disconnects_close_only_that_connection(self):
        handler_type = make_handler(None)
        for error in (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            handler = handler_type.__new__(handler_type)
            handler.close_connection = False
            with patch.object(BaseHTTPRequestHandler, 'handle', side_effect=error):
                handler.handle()
            self.assertTrue(handler.close_connection)

    def test_unexpected_server_errors_are_not_hidden(self):
        handler_type = make_handler(None)
        handler = handler_type.__new__(handler_type)
        with patch.object(BaseHTTPRequestHandler, 'handle', side_effect=ValueError('bug')):
            with self.assertRaises(ValueError):
                handler.handle()

    def test_health_receipt_and_malformed_payload(self):
        node = SimpleNamespace(state=DashboardState())
        with patch('crossmaps_dashboard.now_sec', return_value=42.):
            CrossMapsDashboard.on_pipeline_health(node, SimpleNamespace(data='{"state":"healthy"}'))
        self.assertEqual(node.state.pipeline_health, {'state':'healthy', '_received_s':42.})
        for data in ('broken', '[]', 'null'):
            CrossMapsDashboard.on_pipeline_health(node, SimpleNamespace(data=data))
        self.assertEqual(node.state.pipeline_health['state'], 'healthy')

    def test_status_exposes_heartbeat_age_without_private_receipt_time(self):
        node = SimpleNamespace(state=DashboardState(), robot_pose_source_topic=None,
            robot_odom_topic='', target_node='/test', get_name=lambda:'dashboard',
            url='http://localhost', camera_streams_enabled=False)
        node.state.pipeline_health = {'state':'processing', 'stage':'DINO', '_received_s':40.}
        with patch('crossmaps_dashboard.now_sec', return_value=42.):
            payload = CrossMapsDashboard.status_payload(node, light=True)
        self.assertEqual(payload['pipeline_health'], {'state':'processing', 'stage':'DINO', 'age_s':2.})
