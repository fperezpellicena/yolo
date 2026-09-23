"""Writes the rendered view to a video file at a fixed size."""

import sys
from typing import Optional, Tuple

import cv2
import numpy as np

from .ui.theme import BG


def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Fit an image into a fixed size without distortion."""
    ih, iw = image.shape[:2]
    scale = min(width / iw, height / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    out = np.full((height, width, 3), BG, np.uint8)
    x, y = (width - nw) // 2, (height - nh) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


class Recorder:
    """The window can be any size; the recording always has `size`."""

    def __init__(self, path: str, size: Tuple[int, int], fps: float = 20.0):
        self.path = path
        self.size = size
        self.fps = fps
        self._writer: Optional[cv2.VideoWriter] = None
        self._failed = False

    def write(self, canvas: np.ndarray) -> None:
        if self._failed:
            return
        if self._writer is None:
            self._writer = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
            if not self._writer.isOpened():
                print(f"Could not open '{self.path}' for writing.", file=sys.stderr)
                self._writer, self._failed = None, True
                return
        self._writer.write(letterbox(canvas, *self.size))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            print(f"Recording written to {self.path}")
