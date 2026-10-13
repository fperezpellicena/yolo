"""Keypoint layout and the joint angles derived from it.

Halpe-26: the COCO-17 points (all Ultralytics pose models), in COCO order, then
head, neck, mid-hip and each foot's big toe, small toe and heel (the RTMPose
body-with-feet models). Every pose pass is stored at this width; models with
fewer points leave the rest unmeasured (NaN, score 0).
"""

from typing import Dict, Optional, Tuple

import numpy as np

COCO17: Tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
HALPE_EXTRA: Tuple[str, ...] = (
    "head", "neck", "mid_hip",
    "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
    "left_heel", "right_heel",
)
KEYPOINT_NAMES: Tuple[str, ...] = COCO17 + HALPE_EXTRA
NUM_KEYPOINTS = len(KEYPOINT_NAMES)
KP: Dict[str, int] = {name: i for i, name in enumerate(KEYPOINT_NAMES)}


def pad_keypoints(keypoints: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(..., K, 2) points and (..., K) scores widened to NUM_KEYPOINTS; the missing
    points are NaN with score 0, so every later step treats them as unmeasured."""
    k = keypoints.shape[-2]
    if k > NUM_KEYPOINTS:
        raise ValueError(f"{k} keypoints: only the {NUM_KEYPOINTS}-point Halpe layout is known")
    if k == NUM_KEYPOINTS:
        return keypoints, scores
    pad = NUM_KEYPOINTS - k
    kp = np.concatenate([keypoints, np.full((*keypoints.shape[:-2], pad, 2), np.nan)], axis=-2)
    sc = np.concatenate([scores, np.zeros((*scores.shape[:-1], pad))], axis=-1)
    return kp, sc

# The limbs drawn as a skeleton (torso, arms, legs).
EDGES: Tuple[Tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    # feet, drawn only where measured (Halpe-26 models)
    ("left_ankle", "left_heel"), ("left_heel", "left_big_toe"), ("left_ankle", "left_big_toe"),
    ("right_ankle", "right_heel"), ("right_heel", "right_big_toe"), ("right_ankle", "right_big_toe"),
)

# (label, point A, vertex B, point C) -> angle ABC, measured at the vertex.
# Add a row here to track a new angle; the rest of the app picks it up.
ANGLE_DEFS: Tuple[Tuple[str, str, str, str], ...] = (
    ("L elbow",    "left_shoulder",  "left_elbow",     "left_wrist"),
    ("R elbow",    "right_shoulder", "right_elbow",    "right_wrist"),
    ("L shoulder", "left_elbow",     "left_shoulder",  "left_hip"),
    ("R shoulder", "right_elbow",    "right_shoulder", "right_hip"),
    ("L hip",      "left_shoulder",  "left_hip",       "left_knee"),
    ("R hip",      "right_shoulder", "right_hip",      "right_knee"),
    ("L knee",     "left_hip",       "left_knee",      "left_ankle"),
    ("R knee",     "right_hip",      "right_knee",     "right_ankle"),
)
ANGLE_LABELS: Tuple[str, ...] = tuple(label for label, *_ in ANGLE_DEFS)

# label -> degrees, or None when the joint could not be measured
Angles = Dict[str, Optional[float]]


def empty_angles() -> Angles:
    return {label: None for label in ANGLE_LABELS}
