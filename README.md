> 🌐 **English** · [中文](README.zh.md)

# spatialmem-perception

Perception adapters for **[SpatialMem](https://github.com/wikieden/spatialmem)** —
turn RGB(-D) frames into world-frame `Detection`s the memory core can ingest.

> **Why a separate repo?** SpatialMem core is a *memory* layer (like Mem0 is for
> text) and stays numpy-only. Recognition is not its job. All concrete
> perception — open-vocab detectors, the camera→world geometry lift, image-crop
> feature encoding — lives here, behind heavy GPU deps, so the core stays light.

## What's here

| Piece | Status |
|---|---|
| `geometry` — camera-frame oriented 3D box → world AABB lift (pure numpy) | ✅ |
| `ImageEncoder` protocol + `OpenClipImageEncoder` (text+image towers) | ✅ (`[clip]`) |
| `Cosmos3PerceptionAdapter` — Cosmos 3 Reasoner 3D boxes + ego-pose → `Detection` | ⬜ design (needs GPU) — see [docs/cosmos3-perception-adapter.md](docs/en/cosmos3-perception-adapter.md) |
| `ConceptGraphsAdapter` (SAM + Grounding DINO + OpenCLIP) | ⬜ planned |

All adapters implement the `spatialmem.PerceptionAdapter` protocol, so:

```python
mem.add_frame(rgb, depth, pose, adapter=my_adapter)  # adapter -> Detections -> fusion
```

## Install

```bash
pip install spatialmem-perception          # core (numpy)
pip install spatialmem-perception[clip]    # + open-clip / torch / pillow
```

## Relationship to the core

`spatialmem-perception` **depends on** `spatialmem`. The core owns the seam
(`PerceptionAdapter`), the memory, and query; this repo fills in the eyes. See
the full system design — Cosmos 3 + SpatialMem + LLM "brain" — in the core repo
at `docs/design/cosmos3-spatialmem-llm-brain.md`.

## License

Apache-2.0 — see [LICENSE](LICENSE).
