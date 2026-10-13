"""A single detected body."""

from dataclasses import dataclass, field

import numpy as np

from .skeleton import Angles


@dataclass
class Person:
    box: np.ndarray                      # xyxy, pixels
    keypoints: np.ndarray                # (K, 2) pixel coordinates, skeleton.KEYPOINT_NAMES order
    scores: np.ndarray                   # (K,) per-keypoint confidence (K = 17 or 26)
    angles: Angles = field(default_factory=dict)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
