"""A single detected body."""

from dataclasses import dataclass, field

import numpy as np

from .skeleton import Angles


@dataclass
class Person:
    box: np.ndarray                      # xyxy, pixels
    keypoints: np.ndarray                # (17, 2) pixel coordinates
    scores: np.ndarray                   # (17,) per-keypoint confidence
    angles: Angles = field(default_factory=dict)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
