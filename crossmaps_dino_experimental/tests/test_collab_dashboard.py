"""Live loopback HTTP tests; no ROS, model weights, or robot required."""
import http.client
import json
import unittest

from collab_dashboard import CollaborativeDashboard


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.commands = []
        self.state = {"schema": "test", "robots": [{"robot_id": "A"}], "revision": 1}

        def command(value):
            self.commands.append(value)
            return {"ok": True, "message": "applied"}

        self.dashboard = CollaborativeDashboard(lambda: self.state, command,
            lambda robot, view: (b"sample-image", "image/jpeg") if (robot, view) == ("A", "rgb") else None,
            port=0).start()
        self.addCleanup(self.dashboard.close)

    def request(self, method="GET", path="/api/state", payload=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.dashboard.port, timeout=3)
        connection.request(method, path, payload, headers or {})
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def command_headers(self):
        _, _, body = self.request()
        return {"Content-Type": "application/json", "X-CrossMaps-Token": json.loads(body)["_dashboard"]["command_token"]}

    def test_state_without_mutating_provider(self):
        status, headers, body = self.request()
        self.assertEqual(status, 200)
        state = json.loads(body)
        self.assertTrue(state["_dashboard"]["commands_enabled"])
        self.assertEqual(state["robots"][0]["robot_id"], "A")
        self.assertNotIn("_dashboard", self.state)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])

    def test_static_allowlist_and_traversal(self):
        for path, content in (("/", b"CrossMaps"), ("/app.js", b"buildCloud"), ("/styles.css", b".topbar")):
            status, _, body = self.request(path=path)
            self.assertEqual(status, 200)
            self.assertIn(content, body)
        for path in ("/../collab_dashboard.py", "/%2e%2e/collab_dashboard.py", "/api/secrets"):
            self.assertEqual(self.request(path=path)[0], 404)

    def test_image_routing_and_parameters(self):
        status, headers, body = self.request(path="/api/image?robot=A&view=rgb")
        self.assertEqual((status, headers["Content-Type"], body), (200, "image/jpeg", b"sample-image"))
        self.assertEqual(self.request(path="/api/image?robot=B&view=rgb")[0], 404)
        self.assertEqual(self.request(path="/api/image?robot=../../file&view=rgb")[0], 400)
        self.assertEqual(self.request(path="/api/image?robot=A&view=../../file")[0], 400)

    def test_token_required_and_valid_command(self):
        payload = json.dumps({"action": "set_shadow_mode", "enabled": True})
        self.assertEqual(self.request("POST", "/api/command", payload, {"Content-Type": "application/json"})[0], 403)
        self.assertEqual(self.commands, [])
        status, _, body = self.request("POST", "/api/command", payload, self.command_headers())
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["ok"])
        self.assertEqual(self.commands, [{"action": "set_shadow_mode", "enabled": True}])

    def test_origin_and_dns_rebinding_rejected(self):
        headers = self.command_headers()
        headers["Origin"] = "https://attacker.example"
        self.assertEqual(self.request("POST", "/api/command", '{"action":"save"}', headers)[0], 403)
        self.assertEqual(self.request(headers={"Host": "attacker.example"})[0], 403)
        self.assertEqual(self.commands, [])

    def test_clear_anchor_and_identity_review_actions_are_routed(self):
        headers = self.command_headers()
        for payload in ({'action': 'clear_anchor', 'robot_id': 'A'},
                        {'action': 'accept_entity_link', 'id': 'link-1'},
                        {'action': 'unlink_entities', 'id': 'link-1'}):
            self.assertEqual(self.request('POST', '/api/command', json.dumps(payload), headers)[0], 200)
            self.assertEqual(self.commands[-1], payload)

    def test_unknown_nonfinite_and_oversized_commands(self):
        headers = self.command_headers()
        for body in ('{"action":"move_robot"}', '[]', '{"action":"anchor","x":NaN}', 'broken'):
            self.assertEqual(self.request("POST", "/api/command", body, headers)[0], 400)
        self.assertEqual(self.request("POST", "/api/command", " " * 65537, headers)[0], 413)
        self.assertEqual(self.commands, [])

    def test_provider_failure_is_contained(self):
        self.dashboard.state_provider = lambda: (_ for _ in ()).throw(RuntimeError("private-path"))
        status, _, body = self.request()
        self.assertEqual(status, 500)
        self.assertNotIn(b"private-path", body)
        self.dashboard.state_provider = lambda: self.state
        self.assertEqual(self.request()[0], 200)

    def test_json_content_type_required(self):
        headers = self.command_headers()
        headers["Content-Type"] = "text/plain"
        self.assertEqual(self.request("POST", "/api/command", '{"action":"save"}', headers)[0], 415)

    def test_lifecycle_idempotent(self):
        port = self.dashboard.port
        self.dashboard.start()
        self.assertEqual(self.dashboard.port, port)
        self.dashboard.close()
        self.dashboard.close()


if __name__ == "__main__":
    unittest.main()
