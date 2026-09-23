"""YOLO pose inference, turned into Person records plus a skeleton overlay."""

from typing import List, Optional, Tuple

import numpy as np

from .geometry import compute_angles
from .person import Person


class PoseEstimator:
    def __init__(self, weights: str, conf: float, imgsz: int,
                 device: Optional[str], min_keypoint_score: float):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "Ultralytics is not installed. Run:  pip install ultralytics opencv-python"
            ) from exc
        print(f"Loading pose model '{weights}' ...")
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.min_keypoint_score = min_keypoint_score

    def estimate(self, frame: np.ndarray) -> Tuple[List[Person], np.ndarray]:
        """People sorted largest first, and a copy of the frame with skeletons drawn."""
        result = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False,
        )[0]
        return people_from_result(result, self.min_keypoint_score), _overlay(result, frame)

    def detect(self, frame: np.ndarray) -> List[Person]:
        """People only, no overlay (faster; used by offline video analysis)."""
        result = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False,
        )[0]
        return people_from_result(result, self.min_keypoint_score)


def people_from_result(result, min_keypoint_score: float) -> List[Person]:
    """Convert an Ultralytics Results object into Person records."""
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
        return []

    xy = keypoints.xy.cpu().numpy()
    if keypoints.conf is not None:
        conf = keypoints.conf.cpu().numpy()
    else:
        conf = np.ones(xy.shape[:2], dtype=np.float32)

    boxes = result.boxes
    if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) == len(xy):
        xyxy = boxes.xyxy.cpu().numpy()
    else:  # fall back to the keypoint bounding box
        xyxy = np.stack([np.concatenate([p.min(axis=0), p.max(axis=0)]) for p in xy])

    people = [
        Person(box=xyxy[i], keypoints=xy[i], scores=conf[i],
               angles=compute_angles(xy[i], conf[i], min_keypoint_score))
        for i in range(len(xy))
    ]
    people.sort(key=lambda p: p.area, reverse=True)
    return people


def _overlay(result, frame: np.ndarray) -> np.ndarray:
    try:
        return result.plot(img=frame.copy(), boxes=True, labels=False, conf=False)
    except TypeError:          # older/newer signature - fall back
        return result.plot()
