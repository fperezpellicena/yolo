#!/usr/bin/env python3
"""
Real-time human pose estimation with joint-angle readout.

Pipeline:
  1. Enumerate the webcams attached to the machine and let the user pick one.
  2. Run a YOLO pose model (Ultralytics) on every frame.
  3. Compute the angle at each major joint from the detected keypoints.
  4. Show, side by side: the raw camera feed | the pose overlay | an angle panel.

Keys while running:
  q / ESC  quit
  m        toggle horizontal mirroring
  s        save a PNG snapshot of the composite view
  space    pause / resume

Requires: ultralytics, opencv-python, numpy  (see requirements.txt)
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# Silence the OpenCV backend chatter produced while probing camera indices.
# Must happen before cv2 is imported.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("YOLO_VERBOSE", "False")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# --------------------------------------------------------------------------- #
# Keypoint / angle definitions (COCO-17 layout used by all Ultralytics pose
# models: yolo11*-pose, yolov8*-pose, ...)
# --------------------------------------------------------------------------- #

KEYPOINT_NAMES: Tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
KP: Dict[str, int] = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# (label, point A, vertex B, point C) -> angle ABC, measured at the vertex.
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

PANEL_WIDTH = 340
FONT = cv2.FONT_HERSHEY_SIMPLEX

# --------------------------------------------------------------------------- #
# Camera discovery
# --------------------------------------------------------------------------- #


@dataclass
class Camera:
    index: int
    name: str
    width: int
    height: int
    backend: int

    def describe(self) -> str:
        res = f"{self.width}x{self.height}" if self.width and self.height else "unknown size"
        return f"[{self.index}] {self.name}  ({res})"


def default_backend() -> int:
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW          # much faster to open than MSMF
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


def _linux_camera_names() -> Dict[int, str]:
    names: Dict[int, str] = {}
    base = "/sys/class/video4linux"
    if not os.path.isdir(base):
        return names
    for entry in sorted(os.listdir(base)):
        if not entry.startswith("video"):
            continue
        try:
            index = int(entry[len("video"):])
        except ValueError:
            continue
        try:
            with open(os.path.join(base, entry, "name"), "r") as fh:
                names[index] = fh.read().strip()
        except OSError:
            names[index] = f"/dev/{entry}"
    return names


def _windows_camera_names() -> Dict[int, str]:
    # Optional: pip install pygrabber  -> gives real device names on Windows.
    try:
        from pygrabber.dshow_graph import FilterGraph  # type: ignore
    except Exception:
        return {}
    try:
        return dict(enumerate(FilterGraph().get_input_devices()))
    except Exception:
        return {}


def _friendly_names() -> Dict[int, str]:
    system = platform.system()
    if system == "Linux":
        return _linux_camera_names()
    if system == "Windows":
        return _windows_camera_names()
    return {}


def _open_capture(index: int, backend: int) -> Optional[cv2.VideoCapture]:
    for candidate in (backend, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, candidate)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def discover_cameras(max_index: int = 8, backend: Optional[int] = None) -> List[Camera]:
    """Probe device indices and keep the ones that actually deliver a frame."""
    backend = default_backend() if backend is None else backend
    names = _friendly_names()

    # On Linux the /sys listing is authoritative, so only probe what exists
    # (this also skips the metadata nodes that cannot produce frames).
    candidates: Sequence[int] = sorted(names) if names and platform.system() == "Linux" \
        else range(max_index + 1)

    cameras: List[Camera] = []
    for index in candidates:
        cap = _open_capture(index, backend)
        if cap is None:
            continue
        ok, frame = cap.read()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        used_backend = int(cap.get(cv2.CAP_PROP_BACKEND)) or backend
        cap.release()
        if not ok or frame is None:
            continue
        if not width or not height:
            height, width = frame.shape[:2]
        cameras.append(
            Camera(index, names.get(index, f"Camera {index}"), width, height, used_backend)
        )
    return cameras


def choose_camera(cameras: List[Camera]) -> Camera:
    """Prompt the user to pick one of the discovered cameras."""
    print("\nAvailable video input devices:")
    for position, cam in enumerate(cameras, start=1):
        print(f"  {position}. {cam.describe()}")

    if len(cameras) == 1:
        print("\nOnly one device found - using it.")
        return cameras[0]

    while True:
        try:
            raw = input(f"\nSelect a device [1-{len(cameras)}] (default 1): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nNo selection made - using the first device.")
            return cameras[0]
        if not raw:
            return cameras[0]
        if raw.isdigit():
            choice = int(raw)
            if 1 <= choice <= len(cameras):
                return cameras[choice - 1]
            # Also accept the raw device index (e.g. "2" for /dev/video2).
            for cam in cameras:
                if cam.index == choice:
                    return cam
        print("Invalid choice, try again.")


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


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


@dataclass
class Person:
    box: np.ndarray                      # xyxy
    keypoints: np.ndarray                # (17, 2) pixel coordinates
    scores: np.ndarray                   # (17,) per-keypoint confidence
    angles: Dict[str, Optional[float]] = field(default_factory=dict)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def compute_angles(keypoints: np.ndarray, scores: np.ndarray,
                   min_score: float) -> Dict[str, Optional[float]]:
    angles: Dict[str, Optional[float]] = {}
    for label, a_name, b_name, c_name in ANGLE_DEFS:
        ia, ib, ic = KP[a_name], KP[b_name], KP[c_name]
        if min(scores[ia], scores[ib], scores[ic]) < min_score:
            angles[label] = None
            continue
        angles[label] = joint_angle(keypoints[ia], keypoints[ib], keypoints[ic])
    return angles


def extract_people(result, min_score: float) -> List[Person]:
    """Turn an Ultralytics Results object into a list of Person records."""
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
        xyxy = np.stack([
            np.concatenate([person.min(axis=0), person.max(axis=0)]) for person in xy
        ]) if len(xy) else np.zeros((0, 4))

    people = [
        Person(box=xyxy[i], keypoints=xy[i], scores=conf[i],
               angles=compute_angles(xy[i], conf[i], min_score))
        for i in range(len(xy))
    ]
    people.sort(key=lambda p: p.area, reverse=True)   # biggest person first
    return people


class AngleSmoother:
    """Exponential moving average so the readout does not jitter."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self._state: Dict[str, float] = {}

    def __call__(self, angles: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        if self.alpha >= 1.0:
            return angles
        out: Dict[str, Optional[float]] = {}
        for label, value in angles.items():
            if value is None:
                self._state.pop(label, None)
                out[label] = None
                continue
            previous = self._state.get(label)
            smoothed = value if previous is None else \
                self.alpha * value + (1.0 - self.alpha) * previous
            self._state[label] = smoothed
            out[label] = smoothed
        return out

    def reset(self) -> None:
        self._state.clear()


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def draw_joint_angles(image: np.ndarray, person: Person, min_score: float) -> None:
    """Write each angle next to the joint it belongs to."""
    for label, _, vertex_name, _ in ANGLE_DEFS:
        value = person.angles.get(label)
        if value is None:
            continue
        vertex = KP[vertex_name]
        if person.scores[vertex] < min_score:
            continue
        x, y = person.keypoints[vertex].astype(int)
        text = f"{value:.0f}"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
        cv2.rectangle(image, (x + 6, y - th - 8), (x + 12 + tw, y - 2), (0, 0, 0), -1)
        cv2.putText(image, text, (x + 9, y - 6), FONT, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


def label_panel(image: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(image, text, (10, 20), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def build_angle_panel(height: int, people: List[Person],
                      angles: Dict[str, Optional[float]],
                      fps: float, paused: bool) -> np.ndarray:
    panel = np.full((height, PANEL_WIDTH, 3), 28, dtype=np.uint8)
    cv2.putText(panel, "JOINT ANGLES", (14, 30), FONT, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(panel, (14, 42), (PANEL_WIDTH - 14, 42), (80, 80, 80), 1)

    subtitle = "no person detected" if not people else \
        f"person 1 of {len(people)} (largest)"
    cv2.putText(panel, subtitle, (14, 62), FONT, 0.45, (170, 170, 170), 1, cv2.LINE_AA)

    y = 96
    row_height = max(26, min(40, (height - 150) // max(1, len(ANGLE_DEFS))))
    for label, _, _, _ in ANGLE_DEFS:
        value = angles.get(label)
        known = value is not None
        colour = (235, 235, 235) if known else (110, 110, 110)
        cv2.putText(panel, label, (14, y), FONT, 0.55, colour, 1, cv2.LINE_AA)
        reading = f"{value:6.1f} deg" if known else "    --"
        cv2.putText(panel, reading, (PANEL_WIDTH - 150, y), FONT, 0.55,
                    (0, 255, 255) if known else (110, 110, 110), 1, cv2.LINE_AA)

        # small bar: 0-180 degrees
        bar_top = y + 6
        cv2.rectangle(panel, (14, bar_top), (PANEL_WIDTH - 14, bar_top + 4), (60, 60, 60), -1)
        if known:
            span = PANEL_WIDTH - 28
            filled = int(span * float(np.clip(value, 0.0, 180.0)) / 180.0)
            cv2.rectangle(panel, (14, bar_top), (14 + filled, bar_top + 4),
                          (0, 200, 200), -1)
        y += row_height

    footer = f"{fps:5.1f} FPS" + ("  [PAUSED]" if paused else "")
    cv2.putText(panel, footer, (14, height - 40), FONT, 0.5, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(panel, "q quit  m mirror  s save  space pause",
                (14, height - 16), FONT, 0.38, (140, 140, 140), 1, cv2.LINE_AA)
    return panel


def fit_height(image: np.ndarray, height: int) -> np.ndarray:
    scale = height / image.shape[0]
    width = max(1, int(round(image.shape[1] * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interp)


def compose(raw: np.ndarray, annotated: np.ndarray, panel: np.ndarray,
            height: int) -> np.ndarray:
    left = label_panel(fit_height(raw, height).copy(), "ORIGINAL")
    middle = label_panel(fit_height(annotated, height).copy(), "POSE ESTIMATION")
    return np.hstack([left, middle, panel])


# --------------------------------------------------------------------------- #
# Model wrapper
# --------------------------------------------------------------------------- #


class PoseEstimator:
    def __init__(self, weights: str, conf: float, imgsz: int, device: Optional[str]):
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

    def __call__(self, frame: np.ndarray):
        results = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, device=self.device,
            verbose=False,
        )
        return results[0]

    @staticmethod
    def overlay(result, frame: np.ndarray) -> np.ndarray:
        """Skeleton overlay drawn on a copy of the frame."""
        try:
            return result.plot(img=frame.copy(), boxes=True, labels=False, conf=False)
        except TypeError:          # older/newer signature - fall back
            return result.plot()


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Webcam pose estimation with joint-angle readout.")
    parser.add_argument("--device", type=int, default=None,
                        help="camera index to use (skips the interactive prompt)")
    parser.add_argument("--list-devices", action="store_true",
                        help="list the detected cameras and exit")
    parser.add_argument("--max-index", type=int, default=8,
                        help="highest camera index to probe (default: 8)")
    parser.add_argument("--model", default="yolo11n-pose.pt",
                        help="Ultralytics pose weights (n/s/m/l/x, default: yolo11n-pose.pt)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="person detection confidence threshold (default: 0.5)")
    parser.add_argument("--kpt-conf", type=float, default=0.5,
                        help="minimum keypoint confidence for an angle (default: 0.5)")
    parser.add_argument("--imgsz", type=int, default=640, help="inference size (default: 640)")
    parser.add_argument("--infer-device", default=None,
                        help="torch device for inference, e.g. cpu, 0, mps")
    parser.add_argument("--capture-width", type=int, default=1280)
    parser.add_argument("--capture-height", type=int, default=720)
    parser.add_argument("--panel-height", type=int, default=480,
                        help="height of each video panel in the composite view")
    parser.add_argument("--smooth", type=float, default=0.4,
                        help="angle smoothing factor, 1.0 disables it (default: 0.4)")
    parser.add_argument("--no-mirror", action="store_true",
                        help="do not mirror the camera image")
    parser.add_argument("--record", default=None,
                        help="write the composite view to this .mp4 file")
    return parser.parse_args(argv)


def open_camera(camera: Camera, width: int, height: int) -> cv2.VideoCapture:
    cap = _open_capture(camera.index, camera.backend)
    if cap is None:
        raise SystemExit(f"Could not open camera index {camera.index}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    print("Scanning for cameras ...")
    cameras = discover_cameras(max_index=args.max_index)
    if not cameras:
        print("No working webcam was found. Check that the device is connected "
              "and not in use by another application.", file=sys.stderr)
        return 1

    if args.list_devices:
        for cam in cameras:
            print(cam.describe())
        return 0

    if args.device is not None:
        selected = next((c for c in cameras if c.index == args.device), None)
        if selected is None:
            print(f"Camera index {args.device} is not available. Detected: "
                  f"{[c.index for c in cameras]}", file=sys.stderr)
            return 1
    else:
        selected = choose_camera(cameras)

    print(f"\nUsing {selected.describe()}")
    estimator = PoseEstimator(args.model, args.conf, args.imgsz, args.infer_device)
    cap = open_camera(selected, args.capture_width, args.capture_height)

    smoother = AngleSmoother(args.smooth)
    mirror = not args.no_mirror
    paused = False
    fps = 0.0
    last_time = time.perf_counter()
    writer: Optional[cv2.VideoWriter] = None
    snapshots = 0
    window = "Pose estimation - original | pose | angles"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    last_composite: Optional[np.ndarray] = None

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("Camera stopped delivering frames.", file=sys.stderr)
                    break
                if mirror:
                    frame = cv2.flip(frame, 1)

                result = estimator(frame)
                people = extract_people(result, args.kpt_conf)
                annotated = PoseEstimator.overlay(result, frame)

                primary_angles: Dict[str, Optional[float]] = {}
                if people:
                    for person in people:
                        draw_joint_angles(annotated, person, args.kpt_conf)
                    primary_angles = smoother(people[0].angles)
                else:
                    smoother.reset()
                    primary_angles = {label: None for label, *_ in ANGLE_DEFS}

                now = time.perf_counter()
                delta = now - last_time
                last_time = now
                if delta > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / delta) if fps else 1.0 / delta

                panel = build_angle_panel(args.panel_height, people, primary_angles,
                                          fps, paused)
                last_composite = compose(frame, annotated, panel, args.panel_height)

                if args.record:
                    if writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        h, w = last_composite.shape[:2]
                        writer = cv2.VideoWriter(args.record, fourcc, 20.0, (w, h))
                        if not writer.isOpened():
                            print(f"Could not open '{args.record}' for writing.",
                                  file=sys.stderr)
                            writer = None
                    if writer is not None:
                        writer.write(last_composite)

            if last_composite is not None:
                cv2.imshow(window, last_composite)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mirror = not mirror
                smoother.reset()
            if key == ord(" "):
                paused = not paused
                last_time = time.perf_counter()
            if key == ord("s") and last_composite is not None:
                snapshots += 1
                path = f"pose_snapshot_{snapshots:03d}.png"
                cv2.imwrite(path, last_composite)
                print(f"Saved {path}")
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Recording written to {args.record}")
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
