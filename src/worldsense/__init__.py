"""worldsense — perception adapters for Chronotope.

Companion to the numpy-only `tempomem` memory core. Turns RGB(-D) frames into
world-frame `tempomem.Detection`s via the `tempomem.PerceptionAdapter` seam.
Concrete adapters (Cosmos 3, ConceptGraphs) + the camera->world geometry lift +
image-crop encoders live here so the core stays perception-free.
"""

from __future__ import annotations

from .adapters import Box3D, BoxDetectorAdapter, Detector3D
from .encoders import ImageEncoder, OpenClipImageEncoder, l2_normalize
from .geometry import (
    oriented_box_corners,
    rpy_to_matrix,
    transform_point,
    transform_points,
    world_aabb_from_obb,
)

__version__ = "0.1.0a0"

__all__ = [
    "Box3D",
    "BoxDetectorAdapter",
    "Detector3D",
    "ImageEncoder",
    "OpenClipImageEncoder",
    "l2_normalize",
    "oriented_box_corners",
    "rpy_to_matrix",
    "transform_point",
    "transform_points",
    "world_aabb_from_obb",
]
