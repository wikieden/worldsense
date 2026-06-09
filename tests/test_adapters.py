"""Tests for BoxDetectorAdapter — network-free, no GPU (stub detector + hash encoder)."""

from __future__ import annotations

import numpy as np
import pytest
import spatialmem

from spatialmem_perception.adapters import Box3D, BoxDetectorAdapter, Detector3D


class _StubDetector:
    """Returns canned camera-frame boxes; satisfies Detector3D."""

    def __init__(self, boxes: list[Box3D]) -> None:
        self._boxes = boxes

    def detect(self, rgb, depth, intrinsics=None):
        return list(self._boxes)


class _HashImageEncoder:
    """Deterministic, network-free ImageEncoder: feature from crop content."""

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode_image(self, images):
        out = np.empty((len(images), self._dim), dtype=np.float32)
        for i, im in enumerate(images):
            a = np.asarray(im, dtype=np.float32).ravel()
            seed = (int(abs(a.sum())) + a.size) % (2**32)
            rng = np.random.default_rng(seed)
            out[i] = rng.standard_normal(self._dim).astype(np.float32)
        n = np.linalg.norm(out, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return out / n


def _frame(h: int = 8, w: int = 8):
    return np.zeros((h, w, 3), dtype=np.uint8), np.ones((h, w), dtype=np.float32)


def test_stub_satisfies_detector_protocol():
    assert isinstance(_StubDetector([]), Detector3D)


def test_adapter_satisfies_perception_adapter_protocol():
    ad = BoxDetectorAdapter(_StubDetector([]), _HashImageEncoder())
    assert isinstance(ad, spatialmem.PerceptionAdapter)


def test_empty_detector_returns_no_detections():
    ad = BoxDetectorAdapter(_StubDetector([]), _HashImageEncoder())
    rgb, depth = _frame()
    assert ad.process_frame(rgb, depth, np.eye(4)) == []


def test_identity_pose_keeps_camera_center_and_builds_aabb():
    box = Box3D(label="mug", center=(1.0, 2.0, 3.0), size=(0.2, 0.2, 0.2), box2d=(0, 0, 4, 4))
    ad = BoxDetectorAdapter(_StubDetector([box]), _HashImageEncoder(dim=16))
    rgb, depth = _frame()
    dets = ad.process_frame(rgb, depth, np.eye(4))
    assert len(dets) == 1
    d = dets[0]
    assert d.label == "mug"
    assert d.center_xyz == pytest.approx((1.0, 2.0, 3.0))
    # axis-aligned (rpy=0), identity pose: AABB = center +/- size/2
    assert d.bbox_min == pytest.approx((0.9, 1.9, 2.9))
    assert d.bbox_max == pytest.approx((1.1, 2.1, 3.1))
    assert d.dim == 16
    assert float(np.linalg.norm(d.feature)) == pytest.approx(1.0, abs=1e-5)
    assert d.aux["source"] == "box-detector"
    assert d.aux["box2d"] == (0, 0, 4, 4)


def test_translation_pose_lifts_to_world():
    pose = np.eye(4)
    pose[:3, 3] = (10.0, 0.0, -5.0)
    box = Box3D(label="chair", center=(1.0, 0.0, 0.0), size=(0.5, 0.5, 0.5))
    ad = BoxDetectorAdapter(_StubDetector([box]), _HashImageEncoder())
    rgb, depth = _frame()
    d = ad.process_frame(rgb, depth, pose)[0]
    assert d.center_xyz == pytest.approx((11.0, 0.0, -5.0))


def test_crop_clamps_out_of_bounds_box2d():
    box = Box3D(label="x", center=(0.0, 0.0, 1.0), size=(0.1, 0.1, 0.1), box2d=(-5, -5, 999, 999))
    ad = BoxDetectorAdapter(_StubDetector([box]), _HashImageEncoder())
    rgb, depth = _frame(h=8, w=8)
    assert len(ad.process_frame(rgb, depth, np.eye(4))) == 1  # clamped, no crash


def test_multiple_boxes_preserve_order_and_count():
    boxes = [Box3D(f"o{i}", (float(i), 0.0, 1.0), (0.1, 0.1, 0.1)) for i in range(3)]
    ad = BoxDetectorAdapter(_StubDetector(boxes), _HashImageEncoder())
    rgb, depth = _frame()
    dets = ad.process_frame(rgb, depth, np.eye(4))
    assert [d.label for d in dets] == ["o0", "o1", "o2"]


def test_encoder_box_count_mismatch_raises():
    """A custom encoder returning the wrong number of features is rejected."""

    class _ShortEncoder:
        dim = 16

        def encode_image(self, images):
            return np.zeros((len(images) - 1, 16), dtype=np.float32)

    boxes = [
        Box3D("a", (1.0, 0.0, 1.0), (0.1, 0.1, 0.1)),
        Box3D("b", (2.0, 0.0, 1.0), (0.1, 0.1, 0.1)),
    ]
    ad = BoxDetectorAdapter(_StubDetector(boxes), _ShortEncoder())
    rgb, depth = _frame()
    with pytest.raises(ValueError, match="features"):
        ad.process_frame(rgb, depth, np.eye(4))


def test_non_square_frame_crop_row_col_mapping():
    """A non-square frame with an asymmetric box2d crops without transposing."""
    box = Box3D("x", (0.0, 0.0, 1.0), (0.1, 0.1, 0.1), box2d=(1, 2, 5, 4))
    ad = BoxDetectorAdapter(_StubDetector([box]), _HashImageEncoder())
    rgb, depth = _frame(h=6, w=10)
    dets = ad.process_frame(rgb, depth, np.eye(4))
    assert len(dets) == 1  # arr[2:4, 1:5] valid on a 6x10 frame, no IndexError
