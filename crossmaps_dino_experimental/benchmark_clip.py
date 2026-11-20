"""Offline crop-encoder A/B; run in the perception environment on a completed run.

python benchmark_clip.py .runtime/collaboration/RUN_ID --device mps

No ROS, network, evidence writes, or saved descriptors. The benchmark uses boxes
around measured surface support, not the original (unrecorded) SAM masks.
"""
import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from collab_identity_inspector import _pixels


MODELS = {"vit": ("ViT-L-14", "openai"), "mobile": ("MobileCLIP2-S2", "dfndr2b")}


def spread(items, limit):
    indices = np.linspace(0, len(items) - 1, min(limit, len(items)), dtype=int)
    return [items[i] for i in indices]


def recorded_crops(run, frames=16):
    paths = sorted((run / "evidence").glob("frame-*.npz"))
    if not paths:
        raise ValueError(f"No retained frame-*.npz in {run / 'evidence'}")
    crops, keys = [], []
    digest = hashlib.sha256()
    for path in spread(paths, frames):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(data["metadata"].item())
            # Collaborative storage calls its OpenCV BGR array 'rgb'.
            bgr = data["rgb"]
            frame = {"K": data["K"], "rgb": bgr}
            boxes = []
            for index, surf in enumerate(meta["surfaces"]):
                pixels = _pixels(frame, [{"points_camera": data[f"s{index}_points_camera"]}])
                if len(pixels) < 3:
                    continue
                x0, y0 = pixels.min(axis=0)
                x1, y1 = pixels.max(axis=0) + 1
                if min(x1 - x0, y1 - y0) >= 32:
                    boxes.append((surf["local_id"], x0, y0, x1, y1))
            for sid, x0, y0, x1, y1 in spread(boxes, 4):
                rgb = np.ascontiguousarray(bgr[y0:y1, x0:x1, ::-1])
                key = (meta["robot_id"], meta["session_id"], meta["id"], str(sid))
                digest.update(json.dumps([*key, *rgb.shape]).encode())
                digest.update(rgb)
                keys.append(key)
                crops.append(Image.fromarray(rgb))
    if len(crops) < 8:
        raise ValueError(f"Need at least 8 eligible recorded surface boxes; found {len(crops)}")
    return crops, keys, digest.hexdigest()


def agreement(keys, features):
    """Nearest neighbor in *another* frame, limited to the same source/session."""
    scores = features @ features.T  # normalized features
    correct = total = 0
    for i, (robot, session, frame, sid) in enumerate(keys):
        pool = [j for j, (r, s, f, _) in enumerate(keys) if (r, s) == (robot, session) and f != frame]
        labels = {keys[j][3] for j in pool}
        if sid not in labels or len(labels) < 2:
            continue  # Must have both a matching track and a distractor in the gallery.
        best = pool[int(np.argmax(scores[i, pool]))]
        correct += int(keys[best][3] == sid)
        total += 1
    return {"top1_matches": correct, "eligible_queries": total,
            "fraction": correct / total if total else None}


def run_one(args):
    # HF reads this on import; both named OpenCLIP weights use HF. Reject any
    # accidental fallback to a direct remote URL before opening a connection.
    os.environ["HF_HUB_OFFLINE"] = "1"
    def no_network(event, _args):
        if event in ("socket.getaddrinfo", "socket.connect"):
            raise RuntimeError("Offline CLIP benchmark: cache both checkpoints before running")
    sys.addaudithook(no_network)
    import torch
    from clip_inference import load_clip_for_inference

    crops, keys, digest = recorded_crops(args.run, args.frames)
    device = torch.device(args.device)
    name, tag = MODELS[args.model]
    model, preprocess, _ = load_clip_for_inference(name, tag, device)
    model.eval()
    sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(device.type, lambda: None)
    def prepare(images):
        return torch.stack([preprocess(im) for im in images]).to(device)

    times = {}
    with torch.inference_mode():
        for size in (1, 8):
            encoder, total = [], []
            for i in range(args.warmup + args.repeats):
                subset = [crops[(i * size + j) % len(crops)] for j in range(size)]
                sync()
                start = time.perf_counter()
                batch = prepare(subset)
                sync()
                begin_encoder = time.perf_counter()
                emb = model.encode_image(batch)
                sync()
                end_encoder = time.perf_counter()
                (emb / emb.norm(dim=-1, keepdim=True)).float().cpu().numpy()
                sync()
                end = time.perf_counter()
                if i >= args.warmup:
                    encoder.append(1000 * (end_encoder - begin_encoder))
                    total.append(1000 * (end - start))
            times[size] = {"encode_ms": round(statistics.median(encoder), 2),
                           "pil_to_cpu_embedding_ms": round(statistics.median(total), 2)}

        vectors = []
        for offset in range(0, len(crops), 8):
            e = model.encode_image(prepare(crops[offset:offset + 8])).float()
            vectors.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
        features = np.concatenate(vectors)
    if not np.isfinite(features).all() or not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-3):
        raise ValueError("Encoder produced non-finite or non-unit embeddings")

    def footprint(module):
        return {"parameters": sum(p.numel() for p in module.parameters()),
                "parameter_MiB": round(sum(p.numel() * p.element_size() for p in module.parameters()) / 2**20, 2)}
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(model=name, pretrained=tag, device=str(device), torch=str(torch.__version__),
                platform=platform.platform(), input_tensor=list(batch.shape[1:]), input_dtype=str(batch.dtype),
                embedding_dim=features.shape[1], full_model=footprint(model), image_tower=footprint(model.visual),
                process_peak_rss_MiB=round(rss / (2**20 if sys.platform == "darwin" else 1024), 1),
                median_batch_latency=times, crops=len(crops), source_frames=len({k[2] for k in keys}),
                crops_sha256=digest, recorded_track_agreement=agreement(keys, features))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path, help="archived run directory containing evidence/frame-*.npz")
    p.add_argument("--device", choices=("mps", "cpu", "cuda"), default="cpu")
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--model", choices=MODELS, help="measure only this model (default: both, separate processes)")
    args = p.parse_args()
    if min(args.frames, args.repeats, args.warmup) < 1:
        p.error("--frames, --warmup, and --repeats must all be positive")
    if args.model:
        print(json.dumps(run_one(args), allow_nan=False))
        return
    results = {}
    for key in MODELS:
        print(f"Benchmarking {key} in a fresh process…", file=sys.stderr, flush=True)
        out = subprocess.check_output([sys.executable, __file__, *sys.argv[1:], "--model", key], text=True)
        results[key] = json.loads(out)
    if results["vit"]["crops_sha256"] != results["mobile"]["crops_sha256"]:
        raise RuntimeError("The inputs changed between models. Use a completed recording.")
    print(json.dumps(dict(run=str(args.run), warmup=args.warmup, repeats=args.repeats, results=results,
                          notes=["RGB boxes around retained 3D support; not original SAM masks.",
                                 "Batch times exclude model load, SAM, DINO and map fusion.",
                                 "Peak RSS is per process including load; it is not peak GPU memory.",
                                 "Recorded track agreement is a consistency proxy, not ground-truth or text retrieval accuracy."]),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
