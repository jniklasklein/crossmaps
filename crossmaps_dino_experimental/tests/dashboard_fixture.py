"""Offline browser fixture; serves only experimental UI and synthetic data.

Run: python3 -B tests/dashboard_fixture.py
All parameter writes stay in memory. No ROS node, models, presets or map loads.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crossmaps_dashboard import CLOUD_LAYERS, GRID_LAYERS, IMAGE_LAYERS, PARAM_GROUPS

CONFIG = dict(cloudLayers=CLOUD_LAYERS, imageLayers=IMAGE_LAYERS, gridLayers=GRID_LAYERS,
              paramGroups=PARAM_GROUPS, targetNode="offline UI fixture", cameraStreamsEnabled=False)
PARAMS = {p["name"]: (True if p["type"] == "bool" else p.get("min", 1) if p["type"] != "str"
                       else p.get("choices", [""])[0]) for g in PARAM_GROUPS for p in g["params"]}
PARAMS.update(query_text="sofa", diagnostic_similarity_x=0.5, diagnostic_similarity_y=0.5)
IMAGE = b'''<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540">
<rect width="960" height="540" fill="#162b30"/><rect x="4" y="4" width="952" height="532" fill="none" stroke="#36b68f" stroke-width="8"/>
<path d="M0 0L960 540M960 0L0 540" stroke="#55a7e8" stroke-width="3"/>
<g fill="#f4f1e8" font-family="sans-serif" font-size="28"><text x="20" y="42">TOP LEFT</text>
<text x="755" y="42">TOP RIGHT</text><text x="20" y="516">BOTTOM LEFT</text>
<text x="690" y="516">BOTTOM RIGHT</text><text x="280" y="280">Camera aspect-ratio fixture</text></g></svg>'''


class Handler(BaseHTTPRequestHandler):
    def send(self, body, mime="application/json", status=200):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        files = {"/": ("index.html", "text/html"), "/static/styles.css": ("styles.css", "text/css"),
                 "/static/app.js": ("app.js", "application/javascript")}
        if path in files:
            filename, mime = files[path]
            return self.send((ROOT / "crossmaps_dashboard" / filename).read_bytes(), mime)
        if path.startswith(("/api/image/", "/api/grid/")):
            return self.send(IMAGE, "image/svg+xml")
        if path == "/api/config":
            return self.send(CONFIG)
        if path == "/api/params":
            return self.send({"params": PARAMS})
        if path == "/api/clouds":
            return self.send({"layers": [dict(id=layer["id"], count=2, available=True, state="ready", age_s=0.1,
                                             frame_id="map", positions=[0, 0, 0, 0.3, 0.3, 0.3],
                                             colors=[255, 180, 50, 100, 100, 100]) for layer in CLOUD_LAYERS]})
        if path == "/api/graph":
            return self.send({"descriptor": {"identity": {"entities": 2, "whole_object_grouping": True,
                "assembly": {"enabled": True, "sensitivity": .5, "min_dino_cos": .8, "max_gap_m": .15,
                    "joined_hypotheses": 1, "pairs": [{"a": 7, "b": 8, "hits": 3, "views": [[0,0,0]],
                        "reason": "dino_contact", "dino_cos": .93, "gap_m": .03, "confirmed": True}]}},
                "mask_attribution": [{"segments": [101]}, {"segments": [102]}, {"segments": [], "reason": "no_valid_depth"}],
                "entity_type": "entity", "nodes": [{"id": 7, "segment_ids": [101, 102], "observations": 10,
                "view_count": 2, "confidence": 0.8, "support_count": 40}]}})
        if path == "/api/status":
            return self.send({"topics": {}, "graph": {}, "events": [], "camera_streams_enabled": False})
        if path == "/api/presets":
            return self.send({"presets": []})
        if path == "/api/events":
            return self.send(b"data: {\"topics\": {}, \"graph\": {}, \"events\": []}\n\n", "text/event-stream")
        self.send({}, status=404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/params":
            PARAMS.update(body["updates"])
            return self.send({"ok": True, "results": {key: {"ok": True} for key in body["updates"]}})
        self.send({"ok": False, "error": "not supported by fixture"}, status=400)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    print(f"Offline fixture: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
