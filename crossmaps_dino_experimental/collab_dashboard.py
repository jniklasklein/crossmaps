"""Dependency-free, read-mostly collaborative mapping dashboard.

ROS stays in the runner: providers return snapshots and encoded camera images.
Commands are token protected, same-origin, and accepted only from loopback.
"""
from __future__ import annotations

import ipaddress
import json
from pathlib import Path
import re
import secrets
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit


_STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
           "/index.html": ("index.html", "text/html; charset=utf-8"),
           "/app.js": ("app.js", "text/javascript; charset=utf-8"),
           "/styles.css": ("styles.css", "text/css; charset=utf-8")}
_VIEWS = frozenset(("rgb", "sam", "segments", "entities", "dino", "dino_similarity", "clip", "depth"))
_COMMANDS = frozenset(("anchor", "clear_anchor", "accept_constraint", "reject_constraint", "set_shadow_mode", "accept_entity_link", "unlink_entities", "save"))
_ROBOT = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _loopback(host):
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def _json_default(value):
    # NumPy data can occur in provider snapshots; avoid a mandatory dependency.
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


class CollaborativeDashboard:
    """HTTP view over a backend; providers must be thread-safe and bounded.

    start() is nonblocking. port=0 selects an available port (useful in tests).
    Bind non-loopback explicitly only when remote read-only viewing is wanted.
    """

    def __init__(self, state_provider, command_handler, image_provider=None,
                 host="127.0.0.1", port=8766):
        self.state_provider = state_provider
        self.command_handler = command_handler
        self.image_provider = image_provider
        self.host, self.port = str(host), int(port)
        self._server = None
        self._thread = None
        self._token = secrets.token_urlsafe(32)
        self._static_root = Path(__file__).with_name("collab_dashboard")

    @property
    def url(self):
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"

    def start(self):
        if self._server is not None:
            return self
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def setup(self):
                super().setup()
                self.connection.settimeout(10)

            def handle(self):
                try:
                    super().handle()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
                    self.close_connection = True

            def log_message(self, *_args):
                pass

            def _host_valid(self):
                try:
                    hostname = urlsplit("http://" + self.headers.get("Host", "")).hostname
                except ValueError:
                    return False
                return hostname in {"localhost", "127.0.0.1", "::1", owner.host}

            def _reply(self, status, payload, content_type="application/json; charset=utf-8"):
                if not isinstance(payload, bytes):
                    payload = json.dumps(payload, default=_json_default, allow_nan=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                if not self._host_valid():
                    self._reply(403, {"ok": False, "error": "Unrecognized dashboard host"})
                    return
                parsed = urlsplit(self.path)
                try:
                    if parsed.path in _STATIC:
                        name, mime = _STATIC[parsed.path]
                        self._reply(200, (owner._static_root / name).read_bytes(), mime)
                    elif parsed.path == "/api/state":
                        state = dict(owner.state_provider())
                        local = _loopback(self.client_address[0])
                        state["_dashboard"] = {"commands_enabled": local,
                                               "command_token": owner._token if local else None}
                        self._reply(200, state)
                    elif parsed.path == "/api/image":
                        query = parse_qs(parsed.query, max_num_fields=8)
                        robot = query.get("robot", [""])[0]
                        view = query.get("view", ["rgb"])[0]
                        if not _ROBOT.fullmatch(robot) or view not in _VIEWS:
                            self._reply(400, {"ok": False, "error": "Invalid robot or camera view"})
                            return
                        result = owner.image_provider(robot, view) if owner.image_provider else None
                        if result is None:
                            self._reply(404, {"ok": False, "error": "No image received for this view"})
                            return
                        data, mime = result
                        if not isinstance(data, bytes) or len(data) > 16 * 1024 * 1024 or mime not in ("image/jpeg", "image/png", "image/webp"):
                            raise ValueError("Invalid image provider result")
                        self._reply(200, data, mime)
                    else:
                        self._reply(404, {"ok": False, "error": "Not found"})
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    self.close_connection = True
                except Exception:
                    self._reply(500, {"ok": False, "error": "Dashboard provider failed; inspect runner logs"})

            def do_POST(self):
                if urlsplit(self.path).path != "/api/command":
                    self._reply(404, {"ok": False, "error": "Not found"})
                    return
                origin = self.headers.get("Origin")
                expected_origin = "http://" + self.headers.get("Host", "")
                if (not self._host_valid() or not _loopback(self.client_address[0])
                        or (origin is not None and origin != expected_origin)
                        or not secrets.compare_digest(self.headers.get("X-CrossMaps-Token", ""), owner._token)):
                    self._reply(403, {"ok": False, "error": "Commands require a local, same-origin dashboard session"})
                    return
                if self.headers.get_content_type() != "application/json" or self.headers.get("Transfer-Encoding"):
                    self._reply(415, {"ok": False, "error": "Use application/json with Content-Length"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 65536:
                        self._reply(413, {"ok": False, "error": "Command exceeds 64 KiB or is empty"})
                        return
                    command = json.loads(self.rfile.read(length), parse_constant=lambda _s: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
                    if not isinstance(command, dict) or command.get("action") not in _COMMANDS:
                        raise ValueError("Unknown dashboard action")
                    result = owner.command_handler(command)
                    self._reply(200 if result.get("ok", True) else 400, result)
                except (ValueError, TypeError, KeyError) as exc:
                    self._reply(400, {"ok": False, "error": str(exc)[:250]})
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    self.close_connection = True
                except Exception:
                    self._reply(500, {"ok": False, "error": "Command failed; inspect runner logs"})

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True
            request_queue_size = 32

            def __init__(self, *args, **kwargs):
                self._slots = threading.BoundedSemaphore(16)
                super().__init__(*args, **kwargs)

            def process_request(self, request, client_address):
                if not self._slots.acquire(blocking=False):
                    self.shutdown_request(request)
                    return
                try:
                    super().process_request(request, client_address)
                except Exception:
                    self._slots.release()
                    raise

            def process_request_thread(self, request, client_address):
                try:
                    super().process_request_thread(request, client_address)
                finally:
                    self._slots.release()

        if ":" in self.host:
            Server.address_family = socket.AF_INET6
        self._server = Server((self.host, self.port), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="collaborative-dashboard", daemon=True)
        self._thread.start()
        return self

    def close(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            self._thread.join(timeout=3)
            self._thread = None
