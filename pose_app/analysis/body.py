"""Whole-clip body metrics from the tracked athlete's keypoints.

Everything is vectorised over frames. Lengths are normalised by the athlete's
torso length (shoulder-to-hip), so metrics do not depend on camera distance.
In a side view, metrics use the side nearest the camera, which the model sees
best; the far side is mostly guessed.
"""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from ..skeleton import KP
from .signal import fill_gaps, smooth

SIDE_JOINTS = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")


def angle_series(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ABC in degrees for every frame; inputs are (N, 2)."""
    ba, bc = a - b, c - b
    nba, nbc = np.linalg.norm(ba, axis=1), np.linalg.norm(bc, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(ba * bc, axis=1) / (nba * nbc)
        out = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    out[(nba < 1e-6) | (nbc < 1e-6)] = np.nan
    return out


@dataclass
class BodySeries:
    t: np.ndarray                      # (N,) seconds
    frame_index: np.ndarray            # (N,) index in the source video
    keypoints: np.ndarray              # (N, 17, 2) smoothed, NaN when unreliable
    side: str                          # "left" / "right": side facing the camera
    facing: int                        # +1 athlete faces image-right, -1 image-left
    torso_px: float                    # median torso length in pixels
    view_ratio: float                  # shoulder spread / torso: ~0 side-on, ~1 front-on
    coverage: float                    # fraction of frames with the athlete measured
    metrics: Dict[str, np.ndarray] = field(default_factory=dict)

    def point(self, name: str) -> np.ndarray:
        return self.keypoints[:, KP[name]]

    def near(self, joint: str) -> np.ndarray:
        return self.point(f"{self.side}_{joint}")


def _clean_keypoints(t, kp, scores, min_score, fps) -> np.ndarray:
    """Mask low-confidence points, bridge short gaps, smooth without lag."""
    kp = np.where((scores >= min_score)[..., None], kp, np.nan).astype(float)
    window = max(3, int(round(0.1 * fps)) | 1)          # ~0.1 s, odd
    for j in range(kp.shape[1]):
        for d in range(2):
            kp[:, j, d] = smooth(fill_gaps(kp[:, j, d], t, 0.3), window)
    return kp


def build_body_series(t: np.ndarray, frame_index: np.ndarray, kp: np.ndarray,
                      scores: np.ndarray, min_score: float, fps: float) -> BodySeries:
    present = scores.max(axis=1) > 0
    side_score = {s: float(np.mean([scores[present, KP[f"{s}_{j}"]].mean()
                                    for j in SIDE_JOINTS])) if present.any() else 0.0
                  for s in ("left", "right")}
    side = max(side_score, key=side_score.get)
    clean = _clean_keypoints(t, kp, scores, min_score, fps)

    series = BodySeries(t=t, frame_index=frame_index, keypoints=clean, side=side,
                        facing=1, torso_px=np.nan, view_ratio=np.nan, coverage=0.0)
    sh, hip = series.near("shoulder"), series.near("hip")
    torso = np.linalg.norm(sh - hip, axis=1)
    series.torso_px = float(np.nanmedian(torso)) if np.isfinite(torso).any() else np.nan
    L = series.torso_px

    # Facing: the nose sits ahead of the ear in the direction the athlete faces.
    dx = series.point("nose")[:, 0] - series.near("ear")[:, 0]
    if np.isfinite(dx).any():
        series.facing = 1 if np.nanmedian(dx) >= 0 else -1

    spread = np.abs(series.point("left_shoulder")[:, 0] - series.point("right_shoulder")[:, 0])
    series.view_ratio = float(np.nanmedian(spread) / L) if np.isfinite(spread).any() else np.nan

    m = series.metrics
    n = {j: series.near(j) for j in SIDE_JOINTS}
    m["elbow"] = angle_series(n["shoulder"], n["elbow"], n["wrist"])
    m["shoulder"] = angle_series(n["elbow"], n["shoulder"], n["hip"])
    m["hip"] = angle_series(n["shoulder"], n["hip"], n["knee"])
    m["knee"] = angle_series(n["hip"], n["knee"], n["ankle"])
    v = n["shoulder"] - n["hip"]                          # image y points down
    m["trunk"] = np.degrees(np.arctan2(v[:, 0] * series.facing, -v[:, 1]))  # + = forward lean
    m["wrist_h"] = (n["hip"][:, 1] - n["wrist"][:, 1]) / L       # + = above hip
    m["wrist_fwd"] = (n["wrist"][:, 0] - n["hip"][:, 0]) * series.facing / L
    m["hip_h"] = (n["ankle"][:, 1] - n["hip"][:, 1]) / L
    m["shank"] = np.degrees(np.arctan2((n["knee"][:, 0] - n["ankle"][:, 0]) * series.facing,
                                       n["ankle"][:, 1] - n["knee"][:, 1]))  # 0 = vertical shin
    ls, rs = series.point("left_shoulder"), series.point("right_shoulder")
    m["shoulder_tilt"] = np.degrees(np.arctan2(rs[:, 1] - ls[:, 1], np.abs(rs[:, 0] - ls[:, 0])))

    series.coverage = float(np.mean(np.isfinite(m["hip"]))) if len(t) else 0.0
    return series
