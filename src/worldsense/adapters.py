"""Perception adapters: a posed RGB-D frame -> world-frame ``tempomem.Detection``.

The concrete open-vocab detector (NVIDIA Cosmos 3 Reasoner, or the ConceptGraphs
stack — SAM + Grounding DINO + OpenCLIP) is heavy and GPU-bound, so it is
*injected* as a ``Detector3D``. The adapter itself is pure-numpy and
CPU-testable with a stub detector. This is the generalization of the
Cosmos3 / ConceptGraphs designs in ``docs/cosmos3-perception-adapter.md``:
both are simply different ``Detector3D`` implementations behind the same seam.

Conventions: right-handed, meters. A detector emits **camera-frame** oriented
boxes; ``pose`` is a 4x4 camera->world transform; the produced ``Detection``s
are in **world** coordinates (meters).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from tempomem import Detection

from .geometry import transform_point, world_aabb_from_obb

if TYPE_CHECKING:
    from .encoders import ImageEncoder

_log = logging.getLogger("worldsense.adapters")

Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Box3D:
    """One camera-frame oriented 3D box from a detector.

    - ``center`` / ``size``: meters, camera frame (``size`` is full extent).
    - ``rpy``: (roll, pitch, yaw) radians orienting the box's local axes.
    - ``box2d``: (x1, y1, x2, y2) pixel crop for the image encoder, or None
      (whole frame). Pixels are (x=column, y=row), origin top-left, matching
      row-major ``rgb[y1:y2, x1:x2]`` indexing.
    - ``confidence``: in [0, 1].
    """

    label: str
    center: Vec3
    size: Vec3
    rpy: Vec3 = (0.0, 0.0, 0.0)
    box2d: tuple[int, int, int, int] | None = None
    confidence: float = 1.0


@runtime_checkable
class Detector3D(Protocol):
    """Open-vocab 3D box detector over a posed RGB-D frame, in the camera frame.

    Implemented by GPU backends (Cosmos 3, SAM + Grounding DINO); a stub
    satisfies it for tests. ``rgb`` is (H, W, 3) uint8, ``depth`` is (H, W)
    float32 meters, ``intrinsics`` is a (3, 3) camera matrix or None.
    """

    def detect(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray | None = None,
    ) -> list[Box3D]: ...


class BoxDetectorAdapter:
    """A ``tempomem.PerceptionAdapter`` from an injected detector + encoder.

    ``process_frame`` runs the detector to get camera-frame boxes, lifts each to
    a world-frame AABB + center via :mod:`worldsense.geometry`, and
    encodes its image crop into the per-object feature via an ``ImageEncoder``.
    Pure-numpy; the only heavy parts are the injected ``detector`` / ``encoder``,
    so the adapter is fully CPU-testable with stubs.
    """

    def __init__(
        self,
        detector: Detector3D,
        encoder: ImageEncoder,
        *,
        source: str = "box-detector",
    ) -> None:
        self._detector = detector
        self._encoder = encoder
        self._source = source

    def process_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        pose: np.ndarray,
        intrinsics: np.ndarray | None = None,
    ) -> list[Detection]:
        """Posed RGB-D frame -> world-frame detections (meters). See module docstring."""
        boxes = self._detector.detect(rgb, depth, intrinsics)
        if not boxes:
            return []
        crops = [self._crop(rgb, b.box2d) for b in boxes]
        feats = np.asarray(self._encoder.encode_image(crops))
        if feats.shape[0] != len(boxes):
            raise ValueError(f"encoder returned {feats.shape[0]} features for {len(boxes)} boxes")
        dets: list[Detection] = []
        for b, feat in zip(boxes, feats, strict=True):
            cw = transform_point(pose, b.center)
            bbox_min, bbox_max = world_aabb_from_obb(pose, b.center, b.size, b.rpy)
            aux: dict[str, Any] = {
                "source": self._source,
                "rpy": tuple(float(v) for v in b.rpy),
            }
            if b.box2d is not None:
                aux["box2d"] = tuple(int(v) for v in b.box2d)
            dets.append(
                Detection(
                    label=b.label,
                    feature=feat,
                    center_xyz=(float(cw[0]), float(cw[1]), float(cw[2])),
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    confidence=float(b.confidence),
                    aux=aux,
                )
            )
        _log.debug("frame -> %d detection(s) (source=%s)", len(dets), self._source)
        return dets

    @staticmethod
    def _crop(rgb: np.ndarray, box2d: tuple[int, int, int, int] | None) -> np.ndarray:
        """Pixel crop, clamped to image bounds; whole frame if ``box2d`` is None."""
        arr = np.asarray(rgb)
        if box2d is None:
            return arr
        h, w = arr.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in box2d)
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
        return arr[y1:y2, x1:x2]
