> 🌐 **English** · [中文](../zh/cosmos3-perception-adapter.md)

# Design — `Cosmos3PerceptionAdapter`

**Status:** draft / GPU-blocked. **Repo:** `worldsense` (companion to
the `tempomem` memory core). **Depends on:** a CUDA box (Cosmos 3 Nano 16B /
Super 64B) or a Cosmos NIM endpoint.

## Goal

Let `tempomem.TempoMem.add_frame(rgb, depth, pose)` use **NVIDIA Cosmos 3**
as the perception backend. The adapter turns one posed RGB frame into
`list[tempomem.Detection]` in world coordinates, so the core's fusion → memory
→ query pipeline runs unchanged. The core ships only the `PerceptionAdapter`
protocol; this adapter (and the geometry lift + image encoder it uses) lives
here.

## Why this is realistic (2026-06 finding)

The Cosmos 3 technical report: the Reasoner emits **structured camera-frame 3D
boxes** — each object as `{label, center, dimensions, orientation}` in a unified
**JSON** format — and `Cosmos3-Nano` can **estimate metric-scale ego-pose**. So
Cosmos supplies ~70% of a `Detection` (label + metric 3D box, camera frame). The
adapter only has to (1) lift camera-frame → world-frame, (2) add a per-object
feature vector (Cosmos emits none), (3) parse the JSON robustly.

## Contract

```python
# tempomem (core) — the seam
class PerceptionAdapter(Protocol):
    def process_frame(self, rgb, depth, pose, intrinsics=None) -> list[Detection]: ...

# this repo provides the pieces:
from worldsense.geometry import world_aabb_from_obb, transform_point
from worldsense.encoders import OpenClipImageEncoder  # ImageEncoder
```

## Pipeline (`process_frame`)

```
rgb ──► Cosmos 3 Reasoner (NIM or local Cosmos3OmniPipeline)
          prompt: "detect objects; per object JSON
                   {label, center[x,y,z] m (camera), size[l,w,h] m,
                    orient[roll,pitch,yaw] rad, box2d[x1,y1,x2,y2]}"
          ──► parse JSON (robust)
   pose (arg, preferred) or Cosmos ego-pose ──► cam→world 4×4
   per object:
     center_world = transform_point(cam2world, center_cam)
     bbox_min/max = world_aabb_from_obb(cam2world, center, size, rpy)
     feature      = encoder.encode_image(crop(rgb, box2d))   # OpenClipImageEncoder
     Detection(label, feature, center_world, bbox_min, bbox_max, confidence, ts,
               aux={"source":"cosmos3", "orient": rpy, "box2d": (...)})
```

Pure-numpy geometry (`worldsense.geometry`) is CPU-testable with a
fake reasoner; only the Cosmos call needs a GPU.

## Interface sketch

```python
class Cosmos3PerceptionAdapter:  # implements tempomem.PerceptionAdapter
    def __init__(self, *, reasoner, encoder, pose_source="arg", confidence=0.8,
                 grounding_prompt=_DEFAULT, transport=None): ...
    def process_frame(self, rgb, depth, pose, intrinsics=None) -> list[Detection]:
        boxes = self._detect(rgb)                       # Cosmos -> [_Box3D]
        c2w = pose if self.pose_source == "arg" else self._estimate_pose(rgb)
        return [self._to_detection(b, c2w, rgb) for b in boxes]
```

Packaging: behind a `[cosmos]` extra; default to a hosted NIM so no local weights
are required.

## Open questions (probe on real weights before implementing)

1. **JSON box schema** — exact keys, 2D-vs-3D, metric-vs-normalized — undocumented.
2. **Per-box confidence** — undocumented; use a config prior.
3. **Latency** — System-2 (seconds/clip): sample keyframes, do not call per frame.
4. **Camera-frame metric scale** — confirm meters, not normalized.
5. **Pose source** — trust caller SLAM/ARKit pose over Cosmos estimate unless
   benchmarked; `axis_remap` config for the (undocumented) camera convention.
6. **`aux` persistence** — core's store has no `aux` column yet, so orient/box2d
   are dropped on persist; needs a core schema bump to survive sessions.

## Offline test plan (no GPU)

- Inject a fake `reasoner`/`transport` returning canned JSON boxes.
- Assert parse → world-frame transform (known pose) → `Detection` with correct
  `center_xyz` / AABB; feature L2-norm; bad JSON → `tempomem.AdapterError`.
- Geometry helpers already unit-tested (`tests/test_geometry.py`).

## Alternatives in this repo

- **`ConceptGraphsAdapter`** (SAM + Grounding DINO + OpenCLIP) — open-vocab
  RGB-D perception, same seam.
- **Isaac nvblox + cuVSLAM** feeder — metric 3D + 6-DoF pose (CUDA); add labels
  + features.
