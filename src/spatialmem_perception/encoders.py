"""Image-crop encoders for perception adapters.

A box-only detector (e.g. Cosmos 3) emits no per-object embedding. An
`ImageEncoder` turns each object's image crop into the feature a
`spatialmem.Detection` needs, in the SAME embedding space as the query encoder
(`spatialmem.Encoder.encode_text`) so semantic query still aligns.

`OpenClipImageEncoder` implements BOTH towers — pass one instance to
`SpatialMemory.open(encoder=...)` (text, for query) and to a perception adapter
(image, for per-object features). Requires the `clip` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


@runtime_checkable
class ImageEncoder(Protocol):
    """Maps image crops into the store's embedding space.

    `encode_image` returns an (N, dim) float32 array, L2-normalized per row, in
    the SAME space as the query encoder's `encode_text`. Images are (H, W, 3)
    uint8 RGB arrays.
    """

    @property
    def dim(self) -> int: ...

    def encode_image(self, images: Sequence[np.ndarray]) -> np.ndarray: ...


class OpenClipImageEncoder:
    """CLIP encoder with text + image towers. Requires `pip install
    spatialmem-perception[clip]`. Lazy-imports open_clip/torch/PIL.

    Satisfies `spatialmem.Encoder` (encode_text) AND `ImageEncoder`
    (encode_image), so a single instance serves both query and perception.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
    ) -> None:
        try:
            import open_clip
            import torch
        except ImportError as e:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "OpenClipImageEncoder needs the 'clip' extra: "
                "pip install spatialmem-perception[clip]"
            ) from e
        self._torch = torch
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._device = device
        self._dim = int(self._model.text_projection.shape[1])

    @property
    def dim(self) -> int:
        return self._dim

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - needs torch
        torch = self._torch
        toks = self._tokenizer(list(texts)).to(self._device)
        with torch.no_grad():
            feats = self._model.encode_text(toks)
        return l2_normalize(feats.cpu().numpy())

    def encode_image(self, images: Sequence[np.ndarray]) -> np.ndarray:  # pragma: no cover
        from PIL import Image

        torch = self._torch
        batch = torch.stack(
            [self._preprocess(Image.fromarray(np.asarray(im, dtype=np.uint8))) for im in images]
        ).to(self._device)
        with torch.no_grad():
            feats = self._model.encode_image(batch)
        return l2_normalize(feats.cpu().numpy())
