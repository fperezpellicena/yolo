"""Angle maths on keypoints."""

from typing import Optional

import numpy as np

from .skeleton import ANGLE_DEFS, KP, Angles


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
    """Interior angle ABC in degrees (0-180), measured at vertex b."""
    ba = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    bc = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    norm_ba = float(np.linalg.norm(ba))
    norm_bc = float(np.linalg.norm(bc))
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return None
    cosine = float(np.dot(ba, bc)) / (norm_ba * norm_bc)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def compute_angles(keypoints: np.ndarray, scores: np.ndarray, min_score: float) -> Angles:
    """All angles in ANGLE_DEFS; None where any of the three keypoints is unreliable."""
    angles: Angles = {}
    for label, a_name, b_name, c_name in ANGLE_DEFS:
        ia, ib, ic = KP[a_name], KP[b_name], KP[c_name]
        if min(scores[ia], scores[ib], scores[ic]) < min_score:
            angles[label] = None
        else:
            angles[label] = joint_angle(keypoints[ia], keypoints[ib], keypoints[ic])
    return angles
