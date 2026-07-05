"""T6-11 — `VLMDetector`: an open-vocab `Detector3D` over a VLM endpoint.

Qwen3-VL grounds objects as 2D pixel boxes (JSON); the detector back-projects
them through the depth map into camera-frame `Box3D`s, so it slots straight
into the existing `BoxDetectorAdapter` lift+encode pipeline. The endpoint call
is behind an injected ``transport`` — everything here is network-free.
"""

from __future__ import annotations

import json

import numpy as np
from tempomem import TempoMem

from worldsense.adapters import Box3D, BoxDetectorAdapter, Detector3D
from worldsense.vlm import VLMDetector, _png_bytes

_K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]])


def _frame(depth_m: float = 2.0):
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    depth = np.full((64, 64), depth_m, dtype=np.float32)
    return rgb, depth


def _transport(reply: str):
    calls: list[dict] = []

    def t(payload: dict) -> str:
        calls.append(payload)
        return reply

    t.calls = calls  # type: ignore[attr-defined]
    return t


_CUP = json.dumps([{"label": "cup", "bbox_2d": [22, 22, 42, 42]}])


def test_detect_backprojects_box_through_depth() -> None:
    rgb, depth = _frame(2.0)
    det = VLMDetector(transport=_transport(_CUP))
    boxes = det.detect(rgb, depth, _K)
    assert len(boxes) == 1
    b = boxes[0]
    assert b.label == "cup"
    # box center = principal point -> camera X/Y ~ 0, Z = median depth
    assert abs(b.center[0]) < 1e-6 and abs(b.center[1]) < 1e-6
    assert abs(b.center[2] - 2.0) < 1e-6
    # 20 px at fx=100, z=2 -> 0.4 m extent; flat depth -> default thickness
    assert abs(b.size[0] - 0.4) < 1e-6 and abs(b.size[1] - 0.4) < 1e-6
    assert b.size[2] > 0
    assert b.box2d == (22, 22, 42, 42)


def test_box_without_valid_depth_is_skipped() -> None:
    rgb, depth = _frame(0.0)  # no depth anywhere
    det = VLMDetector(transport=_transport(_CUP))
    assert det.detect(rgb, depth, _K) == []


def test_parse_tolerates_markdown_fences_and_garbage() -> None:
    rgb, depth = _frame()
    fenced = f"```json\n{_CUP}\n```"
    assert len(VLMDetector(transport=_transport(fenced)).detect(rgb, depth, _K)) == 1
    assert VLMDetector(transport=_transport("no objects, sorry")).detect(rgb, depth, _K) == []


def test_default_intrinsics_fallback_still_lifts() -> None:
    rgb, depth = _frame()
    boxes = VLMDetector(transport=_transport(_CUP)).detect(rgb, depth, None)
    assert boxes and boxes[0].center[2] > 0


def test_payload_carries_png_data_url() -> None:
    rgb, depth = _frame()
    t = _transport(_CUP)
    VLMDetector(transport=t).detect(rgb, depth, _K)
    payload = t.calls[0]  # type: ignore[attr-defined]
    assert "data:image/png;base64," in json.dumps(payload)


def test_png_bytes_has_magic_header() -> None:
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    assert _png_bytes(rgb).startswith(b"\x89PNG\r\n\x1a\n")


class _StubEncoder:
    dim = 4

    def encode_image(self, images):
        return np.ones((len(images), 4), dtype=np.float32) / 2.0


def test_end_to_end_through_box_adapter_into_memory() -> None:
    # the VLM fill is a drop-in Detector3D: same lift+encode pipeline, one store
    rgb, depth = _frame(2.0)
    det = VLMDetector(transport=_transport(_CUP))
    assert isinstance(det, Detector3D)
    adapter = BoxDetectorAdapter(det, _StubEncoder(), source="qwen3-vl")
    pose = np.eye(4)
    dets = adapter.process_frame(rgb, depth, pose, _K)
    assert len(dets) == 1 and isinstance(dets[0].label, str)
    assert abs(dets[0].center_xyz[2] - 2.0) < 1e-6
    mem = TempoMem.open(":memory:", embedding_dim=4)
    mem.add_detections(dets)
    mem.commit()
    assert mem.query("cup", k=1).nodes
    mem.close()


def test_box3d_type_is_reused() -> None:
    rgb, depth = _frame()
    boxes = VLMDetector(transport=_transport(_CUP)).detect(rgb, depth, _K)
    assert isinstance(boxes[0], Box3D)


def test_bare_object_reply_parses() -> None:
    # Qwen3-VL sometimes emits a single JSON object, no array brackets
    rgb, depth = _frame()
    bare = '{"label": "cup", "bbox_2d": [22, 22, 42, 42]}'
    assert len(VLMDetector(transport=_transport(bare)).detect(rgb, depth, _K)) == 1


def test_normalized_thousand_coords_are_scaled() -> None:
    # live finding: the model grounds in 0-1000 normalized space when it likes;
    # any coordinate beyond the image extent flags the box as normalized
    rgb, depth = _frame(2.0)  # 64x64 image
    norm = json.dumps([{"label": "cup", "bbox_2d": [344, 344, 656, 656]}])
    boxes = VLMDetector(transport=_transport(norm)).detect(rgb, depth, _K)
    assert len(boxes) == 1
    b = boxes[0]
    # 344..656 of 1000 -> 22..42 px on a 64px image -> same lift as _CUP
    assert b.box2d == (22, 22, 42, 42)
    assert abs(b.center[2] - 2.0) < 1e-6
