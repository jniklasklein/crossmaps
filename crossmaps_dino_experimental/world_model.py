"""Portable evidence metadata; no cross-map identity/pose claims are inferred."""
import uuid
import json
from pathlib import Path
import numpy as np


def new_world_id():
    return str(uuid.uuid4())


def validate_world_id(value):
    return str(uuid.UUID(str(value)))


def node_uid(world_id, kind, identifier):
    return f'{world_id}/{kind}/{int(identifier)}'


def interior_descriptor(descriptor, base_dim):
    if descriptor is None or base_dim <= 0:
        return None
    vector = np.asarray(descriptor, dtype=np.float32).reshape(-1)
    if vector.size not in (base_dim, 2*base_dim):
        return None
    interior = vector[:base_dim]
    norm = float(np.linalg.norm(interior))
    return interior/norm if np.isfinite(norm) and norm > 1e-6 else None


def dominant_surface_core(support, min_fraction=.6):
    """Separate isolated depth outliers without deleting stored observations.

    A core must explain most measured voxels. Multiple substantial disconnected
    components are reported as uncertain, never arbitrarily selected to group.
    """
    support = set(support)
    remaining = support.copy()
    components = []
    offsets = [(x,y,z) for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1) if (x,y,z)!=(0,0,0)]
    while remaining:
        seed = remaining.pop(); component = {seed}; stack = [seed]
        while stack:
            x,y,z = stack.pop()
            for dx,dy,dz in offsets:
                key = (x+dx,y+dy,z+dz)
                if key in remaining:
                    remaining.remove(key);component.add(key);stack.append(key)
        components.append(component)
    if not components:
        return set(), False
    core = max(components, key=lambda c: (len(c), min(c)))
    reliable = len(core) >= min_fraction*len(support)
    return (core if reliable else support), reliable


def surface_role(points, voxel_size):
    """Conservative geometric candidates, not semantic floor/wall labels.

    Large planar surfaces must not become parts of contacting objects. Small
    sofa panels/cushions do not meet the breadth/area requirements.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 12:
        return 'unknown'
    sample = points[::max(1, int(np.ceil(len(points)/256)))]
    centered = sample - np.mean(sample, axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered / max(1, len(sample)))
    if values[1] <= voxel_size*voxel_size or values[0]/max(values.sum(), 1e-9) > .025:
        return 'unknown'
    local = centered @ vectors[:, 1:]
    spans = np.percentile(local, 95, axis=0) - np.percentile(local, 5, axis=0)
    if min(spans) < .75 or max(spans) < 1.5 or np.prod(spans) < 1.5:
        return 'unknown'
    vertical = abs(float(vectors[2, 0]))
    if vertical > .85:
        return 'horizontal_support_surface'
    if vertical < .25:
        return 'vertical_structural_surface'
    return 'unknown'


class EvidenceRecorder:
    """Opt-in bounded local JSONL; records geometry/features, never camera RGB.

    Every output path must resolve inside this experimental package. Recording
    failure is reported by the caller and must never stop perception.
    """
    def __init__(self, world_id, max_frames=120, directory=None):
        root = Path(__file__).resolve().parent
        folder = (Path(directory) if directory is not None else root/'.runtime'/'identity_replays').resolve()
        if not folder.is_relative_to(root):
            raise ValueError('Evidence recordings must remain inside the experimental directory')
        self.path = folder/f'{validate_world_id(world_id)}-{new_world_id()}.jsonl'
        self.max_frames = max(1, min(1000, int(max_frames)))
        self.frames = self.bytes = 0
        self.last_seq = -1
        self.error = None

    def record(self, frame, initial=None):
        if self.error or self.frames >= self.max_frames or int(frame['seq']) <= self.last_seq:
            return False
        payload = dict(schema='crossmaps_identity_trace_v1', **frame)
        if self.frames == 0:
            payload['initial_assembly'] = initial
        encoded = json.dumps(payload, separators=(',', ':'), allow_nan=False) + '\n'
        size = len(encoded.encode('utf-8'))
        if self.bytes+size > 32*1024*1024:
            self.error = '32 MiB recording limit reached'
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('x' if self.frames == 0 else 'a', encoding='utf-8') as stream:
            stream.write(encoded)
        self.frames += 1
        self.bytes += size
        self.last_seq = int(frame['seq'])
        return True

    def status(self, enabled=True):
        return dict(frames=self.frames, max_frames=self.max_frames, bytes=self.bytes,
                    path=str(self.path), error=self.error,
                    state='error' if self.error else 'complete' if self.frames >= self.max_frames
                    else 'recording' if enabled else 'stopped')
