"""Reading recorded video with trustworthy timestamps."""

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


class VideoReader:
    """Iterates (frame_index, seconds, frame) over a file, optionally trimmed.

    Timestamps come from the container (CAP_PROP_POS_MSEC), so variable frame
    rate phone footage still gives correct tempo. OpenCV applies the phone's
    rotation metadata automatically; `rotate` is a manual override on top.
    """

    def __init__(self, path: str, start: float = 0.0, end: Optional[float] = None,
                 rotate: int = 0):
        if rotate not in (0, *_ROTATIONS):
            raise ValueError("rotate must be 0, 90, 180 or 270")
        self.path, self.start, self.end, self.rotate = path, start, end, rotate
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video '{path}'.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"'{path}' contains no readable frames.")
        h, w = self._rotated(frame).shape[:2]
        self.info = VideoInfo(path, float(fps), count, w, h)

    def _rotated(self, frame: np.ndarray) -> np.ndarray:
        return cv2.rotate(frame, _ROTATIONS[self.rotate]) if self.rotate else frame

    def frames(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        cap = cv2.VideoCapture(self.path)
        index, last_t = -1, -1.0
        try:
            while True:
                if not cap.grab():
                    break
                index += 1
                msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                t = msec / 1000.0 if msec > 0 else index / self.info.fps
                t = max(t, last_t + 1e-6)          # keep strictly increasing
                last_t = t
                if t < self.start:
                    continue                        # grab() skips cheaply
                if self.end is not None and t > self.end:
                    break
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield index, t, self._rotated(frame)
        finally:
            cap.release()

    def read_frame(self, index: int) -> Optional[np.ndarray]:
        """Random access to one frame (used for report snapshots)."""
        cap = cv2.VideoCapture(self.path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            return self._rotated(frame) if ok else None
        finally:
            cap.release()
