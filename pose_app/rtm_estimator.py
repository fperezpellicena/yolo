"""RTMPose inference through rtmlib (ONNX, no mmcv), turned into Person records.

A drop-in alternative to the Ultralytics PoseEstimator: same `estimate()` and
`detect()`, same Person records. It runs the Halpe-26 body-with-feet models,
whose first 17 keypoints are the COCO-17 points in COCO order; the other nine
(head, neck, mid-hip, big toe, small toe and heel on each side) are dropped
here, so nothing downstream changes. Keeping them is a later step.

Model names (the `--model` / MODEL value):
    rtmpose-s   lightweight: YOLOX-tiny detector, RTMPose-s 256x192
    rtmpose-m   balanced:    YOLOX-m detector,    RTMPose-m 256x192 (also "rtmpose")
    rtmpose-x   performance: YOLOX-x detector,    RTMPose-x 384x288
rtmlib downloads the ONNX files on first use and caches them.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .geometry import compute_angles
from .person import Person
from .skeleton import EDGES, KP

COCO17 = 17

# model name -> rtmlib BodyWithFeet mode
RTM_MODELS = {
    "rtmpose": "balanced",
    "rtmpose-s": "lightweight",
    "rtmpose-m": "balanced",
    "rtmpose-x": "performance",
}


def is_rtm_model(name: str) -> bool:
    return name.strip().lower() in RTM_MODELS


def rtm_device(device: Optional[str]) -> str:
    """Our device strings (None, cpu, 0, cuda:1, mps) as rtmlib's onnxruntime devices."""
    if device is None or device == "":
        try:
            import onnxruntime
            if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                return "cuda"
        except ImportError:  # pragma: no cover
            pass
        return "cpu"
    d = str(device).strip().lower()
    if d.isdigit():
        return f"cuda:{d}"
    return d


class RtmPoseEstimator:
    def __init__(self, model: str, conf: float, imgsz: int,
                 device: Optional[str], min_keypoint_score: float):
        """`imgsz` is accepted for the same signature but unused: each rtmlib model
        has a fixed input size (see the module docstring)."""
        try:
            from rtmlib import BodyWithFeet
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "rtmlib is not installed. Run:  pip install -r requirements-rtm.txt"
            ) from exc
        mode = RTM_MODELS[model.strip().lower()]
        dev = rtm_device(device)
        print(f"Loading pose model '{model}' (rtmlib {mode}, {dev}) ...")
        solution = BodyWithFeet(mode=mode, backend="onnxruntime", device=dev)
        # Called separately: the solution's own __call__ runs the pose model on the
        # whole frame when nothing is detected, which would invent a person.
        self.det_model, self.pose_model = solution.det_model, solution.pose_model
        self.det_model.score_thr = conf
        self.min_keypoint_score = min_keypoint_score

    def detect(self, frame: np.ndarray) -> List[Person]:
        """People sorted largest first (offline video analysis)."""
        boxes = np.asarray(self.det_model(frame), dtype=np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            return []
        keypoints, scores = self.pose_model(frame, bboxes=boxes)
        return people_from_arrays(boxes, keypoints, scores, self.min_keypoint_score)

    def estimate(self, frame: np.ndarray) -> Tuple[List[Person], np.ndarray]:
        """People sorted largest first, and a copy of the frame with skeletons drawn."""
        people = self.detect(frame)
        return people, draw_people(frame.copy(), people, self.min_keypoint_score)


def people_from_arrays(boxes: np.ndarray, keypoints: np.ndarray, scores: np.ndarray,
                       min_keypoint_score: float) -> List[Person]:
    """rtmlib's (N, 4) boxes, (N, K, 2) keypoints and (N, K) scores, K >= 17, as Person
    records holding the COCO-17 points."""
    kp = np.asarray(keypoints, dtype=np.float32)[:, :COCO17]
    sc = np.asarray(scores, dtype=np.float32)[:, :COCO17]
    people = [
        Person(box=np.asarray(boxes[i], dtype=np.float32), keypoints=kp[i], scores=sc[i],
               angles=compute_angles(kp[i], sc[i], min_keypoint_score))
        for i in range(len(kp))
    ]
    people.sort(key=lambda p: p.area, reverse=True)
    return people


def draw_people(img: np.ndarray, people: List[Person], min_score: float) -> np.ndarray:
    """Boxes and skeletons, for the live view (Ultralytics draws its own)."""
    s = max(0.6, img.shape[0] / 720)
    lw = max(2, int(2 * s))
    for p in people:
        x1, y1, x2, y2 = p.box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1, cv2.LINE_AA)
        ok = p.scores >= min_score
        for a, b in EDGES:
            ia, ib = KP[a], KP[b]
            if ok[ia] and ok[ib]:
                cv2.line(img, tuple(p.keypoints[ia].astype(int)),
                         tuple(p.keypoints[ib].astype(int)), (255, 160, 60), lw, cv2.LINE_AA)
        for i in np.flatnonzero(ok):
            cv2.circle(img, tuple(p.keypoints[i].astype(int)), lw + 1, (60, 220, 255), -1,
                       cv2.LINE_AA)
    return img
