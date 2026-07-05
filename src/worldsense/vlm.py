"""Qwen3-VL detector — an open-vocab `Detector3D` over a VLM endpoint (T6-11).

The VLM (any OpenAI-compatible ``/v1/chat/completions`` serving a grounding-
capable model — Qwen3-VL on the local vLLM host) answers with 2D pixel boxes
as JSON; this module back-projects them through the depth map into
camera-frame `Box3D`s. That makes the VLM a drop-in `Detector3D` fill: the
existing `BoxDetectorAdapter` does the world-frame lift + feature encode, so
memory/query see the exact same `Detection` pipeline as any other detector.

Seams and units:

* ``transport`` — inject a ``Callable[[payload dict], str]`` returning the
  assistant text; the default posts to ``base_url`` over stdlib urllib. Tests
  script it: fully network-free.
* Camera frame is the standard optical pinhole frame: x right, y down,
  z forward, **meters**; ``intrinsics`` is a (3, 3) camera matrix (``None``
  falls back to a nominal pinhole from the image size — fine for demos, pass
  the real K for metric accuracy).
* Images are encoded as PNG via a small stdlib encoder — worldsense's core
  stays numpy-only, no pillow.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import json
import logging
import re
import struct
import urllib.request
import zlib
from collections.abc import Callable
from typing import Any

import numpy as np

from .adapters import Box3D

_log = logging.getLogger("worldsense.vlm")

__all__ = ["VLMDetector", "qwen_vl_transport"]

Transport = Callable[[dict], str]

# Qwen-VL grounding format: pixel boxes, origin top-left, x=column / y=row.
_GROUND_PROMPT = (
    "Detect every distinct physical object in the image. Reply with ONLY a JSON "
    'array, one element per object: {"label": "<short lowercase name>", '
    '"bbox_2d": [x1, y1, x2, y2]} using pixel coordinates (x = column, y = row, '
    "origin top-left, x2 > x1, y2 > y1). No prose, no markdown, no other text."
)


# --------------------------------------------------------------------------- #
# stdlib PNG (truecolor, filter 0) — keeps worldsense pillow-free
# --------------------------------------------------------------------------- #
def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
    )


def _png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an (H, W, 3) uint8 array as a PNG (stdlib only)."""
    arr = np.asarray(rgb, dtype=np.uint8)
    h, w = arr.shape[:2]
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))  # filter 0 per row
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit truecolor
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )


# --------------------------------------------------------------------------- #
# default HTTP transport (vLLM / any OpenAI-compatible server)
# --------------------------------------------------------------------------- #
def qwen_vl_transport(
    base_url: str,
    model: str,
    *,
    api_key: str = "local",
    timeout_s: float = 60.0,
) -> Transport:
    """The default transport: POST the payload to ``/v1/chat/completions``.

    Coalesces ``content`` / ``reasoning_content`` / ``reasoning`` (vLLM 0.23
    puts text in ``reasoning`` when a parser is configured — same resilience
    as the mindloop verbalizers).
    """

    def transport(payload: dict) -> str:
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps({"model": model, **payload}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        msg = body["choices"][0]["message"]
        return msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""

    return transport


# --------------------------------------------------------------------------- #
# reply parsing
# --------------------------------------------------------------------------- #
def _candidate_items(text: str) -> list[list]:
    """Ordered parse attempts: whole reply, bracket-sliced array, bare objects.

    Live finding: Qwen3-VL sometimes emits ONE bare JSON object (no array), so
    a naive bracket slice grabs the inner bbox ints — each attempt is filtered
    and the first that yields valid boxes wins.
    """
    attempts: list[list] = []
    with contextlib.suppress(json.JSONDecodeError, AttributeError):
        loaded = json.loads(text.strip().strip("`").removeprefix("json"))
        attempts.append([loaded] if isinstance(loaded, dict) else loaded)
    start, end = text.find("["), text.rfind("]")
    if start >= 0 and end > start:
        with contextlib.suppress(json.JSONDecodeError):
            attempts.append(json.loads(text[start : end + 1]))
    objs = []
    for m in re.finditer(r"\{[^{}]*\}", text):
        with contextlib.suppress(json.JSONDecodeError):
            objs.append(json.loads(m.group(0)))
    if objs:
        attempts.append(objs)
    return attempts


def _parse_boxes(text: str) -> list[tuple[str, tuple[int, int, int, int], float]]:
    """The VLM reply -> [(label, (x1, y1, x2, y2), confidence)]; garbage -> []."""
    for items in _candidate_items(text):
        out: list[tuple[str, tuple[int, int, int, int], float]] = []
        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict):
                continue
            label = str(it.get("label", "")).strip()
            box = it.get("bbox_2d") or it.get("bbox")
            if not label or not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            x1, y1, x2, y2 = (int(v) for v in box)
            if x2 <= x1 or y2 <= y1:
                continue
            out.append((label, (x1, y1, x2, y2), float(it.get("confidence", 1.0))))
        if out:
            return out
    _log.warning("VLM reply yielded no parseable boxes: %r", text[:120])
    return []


class VLMDetector:
    """An open-vocab `Detector3D` backed by a grounding VLM (Qwen3-VL).

    ``detect`` sends the RGB frame (PNG data URL) with the grounding prompt,
    parses the 2D pixel boxes, and back-projects each through the depth map:
    Z = median valid depth inside the box; X/Y from the pinhole model; width/
    height from the box extent at that depth; thickness from the in-box depth
    spread (floored at ``default_thickness_m``). A box with no valid depth is
    skipped — an unliftable grounding is expected absence, not an error.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001/v1",
        model: str = "Qwen/Qwen3-VL-32B-Instruct-FP8",
        *,
        api_key: str = "local",
        transport: Transport | None = None,
        prompt: str = _GROUND_PROMPT,
        timeout_s: float = 60.0,
        default_thickness_m: float = 0.15,
        max_tokens: int = 1024,
    ) -> None:
        self._transport = transport or qwen_vl_transport(
            base_url, model, api_key=api_key, timeout_s=timeout_s
        )
        self._prompt = prompt
        self._thickness = float(default_thickness_m)
        self._max_tokens = int(max_tokens)

    def detect(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray | None = None,
    ) -> list[Box3D]:
        """Posed-frame grounding: RGB -> VLM 2D boxes -> camera-frame `Box3D`s.

        ``rgb`` (H, W, 3) uint8; ``depth`` (H, W) float32 **meters**; boxes are
        returned in the camera optical frame (x right, y down, z forward),
        meters — `BoxDetectorAdapter` lifts them to world.
        """
        data_url = "data:image/png;base64," + base64.b64encode(_png_bytes(rgb)).decode("ascii")
        payload: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": self._max_tokens,
        }
        reply = self._transport(payload)
        k = self._intrinsics(intrinsics, rgb)
        h, w = np.asarray(rgb).shape[:2]
        boxes: list[Box3D] = []
        for label, box2d, conf in _parse_boxes(reply):
            box2d = self._to_pixels(box2d, h, w)
            lifted = self._lift(box2d, depth, k)
            if lifted is None:
                _log.info("dropping %r: no valid depth inside %s", label, box2d)
                continue
            center, size = lifted
            boxes.append(
                Box3D(label=label, center=center, size=size, box2d=box2d, confidence=conf)
            )
        _log.debug("VLM grounded %d box(es)", len(boxes))
        return boxes

    @staticmethod
    def _to_pixels(
        box2d: tuple[int, int, int, int], h: int, w: int
    ) -> tuple[int, int, int, int]:
        """Absolute pixels, or Qwen-style 0-1000 normalized when out of range.

        Live finding: Qwen3-VL grounds in 0-1000 normalized space at will; any
        coordinate beyond the image extent flags the box as normalized.
        """
        if max(box2d) <= max(h, w):
            return box2d
        x1, y1, x2, y2 = box2d
        return (
            round(x1 / 1000.0 * w),
            round(y1 / 1000.0 * h),
            round(x2 / 1000.0 * w),
            round(y2 / 1000.0 * h),
        )

    @staticmethod
    def _intrinsics(k: np.ndarray | None, rgb: np.ndarray) -> np.ndarray:
        if k is not None:
            return np.asarray(k, dtype=np.float64).reshape(3, 3)
        h, w = np.asarray(rgb).shape[:2]
        f = float(max(h, w))  # nominal pinhole: fine for demos, not metric
        return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])

    def _lift(
        self,
        box2d: tuple[int, int, int, int],
        depth: np.ndarray,
        k: np.ndarray,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        """2D pixel box + depth -> camera-frame (center, size) in meters, or None."""
        d = np.asarray(depth, dtype=np.float64)
        h, w = d.shape[:2]
        x1, y1, x2, y2 = box2d
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        patch = d[y1:y2, x1:x2]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size == 0:
            return None
        z = float(np.median(valid))
        fx, fy = float(k[0, 0]), float(k[1, 1])
        cx, cy = float(k[0, 2]), float(k[1, 2])
        u0, v0 = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        center = ((u0 - cx) / fx * z, (v0 - cy) / fy * z, z)
        thickness = max(
            float(np.percentile(valid, 90) - np.percentile(valid, 10)), self._thickness
        )
        size = ((x2 - x1) / fx * z, (y2 - y1) / fy * z, thickness)
        return center, size
