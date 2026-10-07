"""Low-level drawing helpers for scalable text and translucent boxes."""

from typing import Tuple

import cv2
import numpy as np

from .theme import FONT


def thickness(scale: float) -> int:
    """Stroke width that looks right for a given font scale."""
    return max(1, int(scale * 1.6 + 0.35))


def text_size(text: str, scale: float, stroke: int) -> Tuple[int, int, int]:
    """(width, height, baseline) in pixels."""
    (width, height), baseline = cv2.getTextSize(text, FONT, scale, stroke)
    return width, height, baseline


def fit_scale(text: str, max_width: float, scale: float, min_scale: float = 0.3) -> float:
    """Shrink a font scale until the text fits in max_width pixels."""
    while scale > min_scale and text_size(text, scale, thickness(scale))[0] > max_width:
        scale *= 0.92
    return scale


def darken(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int,
           keep: float = 0.3) -> None:
    """Semi-transparent dark box (keeps `keep` of the underlying brightness)."""
    h, w = canvas.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 > x0 and y1 > y0:
        roi = canvas[y0:y1, x0:x1]
        roi[:] = (roi.astype(np.float32) * keep).astype(np.uint8)
