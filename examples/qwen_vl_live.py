"""T6-11 live pass — Qwen3-VL grounding on the local vLLM host, end to end.

Draws a synthetic tabletop scene (numpy only), sends it to the host VLM for
2D grounding, back-projects through a flat synthetic depth into camera-frame
boxes, lifts to world via `BoxDetectorAdapter`, ingests into `TempoMem`, and
queries it back — the full eyes→memory path with a REAL model in the loop.

Run:  REAL_LLM_HOST=172.16.1.12 uv run python worldsense/examples/qwen_vl_live.py
"""

from __future__ import annotations

import os

import numpy as np
from tempomem import TempoMem

from worldsense import BoxDetectorAdapter, VLMDetector

HOST = os.environ.get("REAL_LLM_HOST", "172.16.1.12")


def _scene() -> tuple[np.ndarray, np.ndarray]:
    """A simple synthetic tabletop: brown table band, red cup, blue ball."""
    rgb = np.full((480, 640, 3), 235, dtype=np.uint8)  # light wall
    rgb[300:480, :] = (139, 90, 43)  # table
    rgb[180:300, 200:280] = (200, 30, 30)  # red cup-ish block
    yy, xx = np.mgrid[0:480, 0:640]
    ball = (xx - 450) ** 2 + (yy - 250) ** 2 < 45**2
    rgb[ball] = (30, 60, 200)  # blue ball
    depth = np.full((480, 640), 2.0, dtype=np.float32)
    return rgb, depth


class _MeanColorEncoder:
    """Deterministic 4-d demo encoder: normalized mean RGB + bias."""

    dim = 4

    def encode_image(self, images):
        out = []
        for im in images:
            m = np.asarray(im, dtype=np.float32).reshape(-1, 3).mean(axis=0) / 255.0
            v = np.array([m[0], m[1], m[2], 1.0], dtype=np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


def main() -> int:
    rgb, depth = _scene()
    print(f"1) grounding a synthetic 640x480 scene on http://{HOST}:8001 …")
    detector = VLMDetector(base_url=f"http://{HOST}:8001/v1", timeout_s=120.0)
    adapter = BoxDetectorAdapter(detector, _MeanColorEncoder(), source="qwen3-vl")
    pose = np.eye(4)  # camera at world origin, optical frame
    dets = adapter.process_frame(rgb, depth, pose)
    for d in dets:
        c = tuple(round(v, 2) for v in d.center_xyz)
        print(f"   {d.label:<16} center={c}  conf={d.confidence:.2f}")
    if not dets:
        print("no boxes grounded — NO-GO")
        return 1

    print("2) ingest into TempoMem + query back …")
    mem = TempoMem.open(":memory:", embedding_dim=4)
    mem.add_detections(dets)
    mem.commit()
    hits = mem.query(dets[0].label, k=3).nodes
    print(f"   query({dets[0].label!r}) -> {[(n.id, n.label) for n in hits]}")
    mem.close()

    ok = bool(hits)
    print(f"\nqwen3-vl eyes→memory {'GO' if ok else 'NO-GO'} — {len(dets)} object(s) grounded")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
