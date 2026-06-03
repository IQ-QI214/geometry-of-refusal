from __future__ import annotations
import numpy as np
from PIL import Image


def blank_image(size: tuple[int, int] = (336, 336)) -> Image.Image:
    return Image.fromarray(np.zeros((*size, 3), dtype=np.uint8))


def noise_image(size: tuple[int, int] = (336, 336), seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr)
