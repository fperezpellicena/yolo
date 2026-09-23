"""COCO-17 keypoint layout (used by all Ultralytics pose models) and the joint
angles derived from it."""

from typing import Dict, Optional, Tuple

KEYPOINT_NAMES: Tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
KP: Dict[str, int] = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

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
