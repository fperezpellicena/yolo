"""How the camera moved during the clip, from the still background.

A phone that drifts moves the whole picture: the machine, the floor and the
athlete's feet. The background (everything outside the athlete's box) stands
still, so matching its features between frames gives the camera's motion:
for each sampled frame, the similarity transform (shift, turn and zoom, a 2x3
matrix) that takes first-frame pixels to that frame's pixels. The force model
uses it to work in the first frame's camera, where the calibration was
clicked, and to draw back onto each frame.

Sampled SAMPLE_HZ times a second (a camera drifts slowly), at half resolution:
a few milliseconds per sample. Matched against the first frame; when too few
features still match it (a long drift, the light changing), against the last
frame that matched well, chaining the transforms.
"""

from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

SAMPLE_HZ = 4.0
DOWN = 0.5                 # analyse at this fraction of the frame's size
FEATURES = 1500
MIN_MATCHES = 25           # a transform needs this many consistent background matches
REKEY_BELOW = 60           # fewer matches with the reference than this: move the reference on
BOX_PAD = 0.10             # the athlete's box grows by this share of its height (arms, handles)
RANSAC_PX = 3.0


def _mat(a: np.ndarray) -> np.ndarray:
    return np.vstack([a, [0.0, 0.0, 1.0]])


class CameraTracker:
    """Feed it every frame in order (`update`); `result` gives (N, 2, 3), NaN where a frame
    was not sampled or could not be matched. Frame 0 (the first fed) is the identity."""

    def __init__(self, fps: float):
        self.every = max(1, int(round(fps / SAMPLE_HZ))) if fps > 0 else 1
        self.orb = cv2.ORB_create(FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ref = None            # (keypoints, descriptors, transform first -> reference frame)
        self.out = {}
        self.n = 0

    def _features(self, frame: np.ndarray, box: Optional[np.ndarray]):
        g = cv2.cvtColor(cv2.resize(frame, None, fx=DOWN, fy=DOWN), cv2.COLOR_BGR2GRAY) \
            if frame.ndim == 3 else cv2.resize(frame, None, fx=DOWN, fy=DOWN)
        mask = np.full(g.shape, 255, np.uint8)
        if box is not None and np.isfinite(box).all():
            x0, y0, x1, y1 = np.asarray(box, float) * DOWN
            pad = BOX_PAD * (y1 - y0)
            mask[max(0, int(y0 - pad)):max(0, int(y1 + pad)),
                 max(0, int(x0 - pad)):max(0, int(x1 + pad))] = 0
        return self.orb.detectAndCompute(g, mask)

    def _match(self, kp, des) -> Tuple[Optional[np.ndarray], int]:
        """Similarity transform reference -> this frame (full-size pixels), and its inliers."""
        rkp, rdes, _ = self.ref
        if des is None or rdes is None or len(kp) < MIN_MATCHES:
            return None, 0
        m = self.matcher.match(rdes, des)
        if len(m) < MIN_MATCHES:
            return None, 0
        p0 = np.float32([rkp[x.queryIdx].pt for x in m]) / DOWN
        p1 = np.float32([kp[x.trainIdx].pt for x in m]) / DOWN
        a, keep = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
                                              ransacReprojThreshold=RANSAC_PX / DOWN)
        good = int(keep.sum()) if keep is not None else 0
        return (a, good) if a is not None and good >= MIN_MATCHES else (None, good)

    def update(self, frame: np.ndarray, box: Optional[np.ndarray] = None) -> None:
        i = self.n
        self.n += 1
        if i % self.every:
            return
        kp, des = self._features(frame, box)
        if self.ref is None:
            self.ref = (kp, des, np.eye(3))
            self.out[i] = np.eye(3)[:2]
            return
        a, good = self._match(kp, des)
        if a is None:
            return
        first_to_frame = _mat(a) @ self.ref[2]
        self.out[i] = first_to_frame[:2]
        if good < REKEY_BELOW:
            self.ref = (kp, des, first_to_frame)

    def result(self, n: Optional[int] = None) -> np.ndarray:
        n = self.n if n is None else n
        out = np.full((n, 2, 3), np.nan)
        for i, a in self.out.items():
            if i < n:
                out[i] = a
        return out


def measure_camera(frames: Iterable[np.ndarray], boxes: np.ndarray, fps: float) -> np.ndarray:
    """The camera's motion over `frames` (in order, one per row of `boxes`): (N, 2, 3)."""
    tracker = CameraTracker(fps)
    for frame, box in zip(frames, boxes):
        tracker.update(frame, box)
    return tracker.result(len(boxes))
