#!/usr/bin/env python3
"""
Real-time human pose estimation with joint-angle readout.

Pipeline:
  1. Enumerate the webcams attached to the machine and let the user pick one.
  2. Run a YOLO pose model (Ultralytics) on every frame.
  3. Compute the angle at each major joint from the detected keypoints.
  4. Show, side by side: the raw camera feed | the pose overlay | an angle panel.

The window is responsive: the layout is recomputed every frame for the
current window size (side-by-side, stacked, or with the angle panel as a
sidebar or footer - whichever gives the video the most room), and it can be
switched to full screen at any time.

Keys while running:
  f        toggle full screen
  v        cycle view: auto / pose + angles / pose only
  m        toggle horizontal mirroring
  space    pause / resume
  s        save a PNG snapshot of the current view
  h        show / hide the keyboard help
  Esc      leave full screen (or quit when windowed)
  q        quit

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

FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA
VIEW_MODES: Tuple[str, ...] = ("auto", "pose + angles", "pose only")
WINDOW_NAME = "Pose estimation"

# BGR colours
BG = (18, 18, 18)
PANEL_BG = (30, 30, 30)
TEXT = (235, 235, 235)
MUTED = (150, 150, 150)
DIM = (95, 95, 95)
ACCENT = (0, 215, 215)
TRACK = (58, 58, 58)

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
# Text helpers
# --------------------------------------------------------------------------- #


def _thickness(scale: float) -> int:
    return max(1, int(scale * 1.6 + 0.35))


def _text_size(text: str, scale: float, thickness: int) -> Tuple[int, int, int]:
    (width, height), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    return width, height, baseline


def _fit_scale(text: str, max_width: float, scale: float, min_scale: float = 0.3) -> float:
    """Shrink a font scale until the text fits in max_width pixels."""
    while scale > min_scale and _text_size(text, scale, _thickness(scale))[0] > max_width:
        scale *= 0.92
    return scale


def _darken(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int,
            keep: float = 0.3) -> None:
    """Semi-transparent dark box (keeps `keep` of the underlying brightness)."""
    h, w = canvas.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 > x0 and y1 > y0:
        roi = canvas[y0:y1, x0:x1]
        roi[:] = (roi.astype(np.float32) * keep).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Responsive layout
# --------------------------------------------------------------------------- #

Rect = Tuple[int, int, int, int]  # x, y, width, height


@dataclass
class Layout:
    tiles: List[Rect]
    panel: Optional[Rect]


def compute_layout(width: int, height: int, aspect: float, n_tiles: int,
                   show_panel: bool, gap: int) -> Layout:
    """
    Try every arrangement (tiles side by side or stacked; angle panel as a
    sidebar or as a footer) and keep the one that gives the video tiles the
    most pixels for the current window size. The tiles and the panel are then
    centred as one group, and the panel grows into any space left over.
    """
    best: Optional[Tuple[float, Layout]] = None
    for panel_pos in (("right", "bottom") if show_panel else (None,)):
        pw = int(min(np.clip(width * 0.24, 220, 440), width * 0.45))
        ph = int(min(np.clip(height * 0.24, 120, 280), height * 0.45))
        if panel_pos == "right":
            area_w, area_h = width - pw - gap, height
        elif panel_pos == "bottom":
            area_w, area_h = width, height - ph - gap
        else:
            area_w, area_h = width, height
        if area_w < 16 or area_h < 16:
            continue

        for horizontal in ((True, False) if n_tiles > 1 else (True,)):
            cols, rows = (n_tiles, 1) if horizontal else (1, n_tiles)
            tile_w = min((area_w - gap * (cols - 1)) / cols,
                         (area_h - gap * (rows - 1)) / rows * aspect)
            tile_w = int(max(1.0, tile_w))
            tile_h = int(max(1.0, tile_w / aspect))
            block_w = cols * tile_w + gap * (cols - 1)
            block_h = rows * tile_h + gap * (rows - 1)

            panel: Optional[Rect] = None
            if panel_pos == "right":
                panel_w = max(pw, min(width - block_w - gap, int(pw * 1.3)))
                gx = max(0, (width - (block_w + gap + panel_w)) // 2)
                gy = max(0, (height - block_h) // 2)
                panel = (gx + block_w + gap, gy, panel_w, block_h)
            elif panel_pos == "bottom":
                panel_h = max(ph, min(height - block_h - gap, int(ph * 1.5)))
                gx = max(0, (width - block_w) // 2)
                gy = max(0, (height - (block_h + gap + panel_h)) // 2)
                panel = (gx, gy + block_h + gap, block_w, panel_h)
            else:
                gx = max(0, (width - block_w) // 2)
                gy = max(0, (height - block_h) // 2)

            tiles = [
                (gx + (i if horizontal else 0) * (tile_w + gap),
                 gy + (0 if horizontal else i) * (tile_h + gap),
                 tile_w, tile_h)
                for i in range(n_tiles)
            ]
            score = float(tile_w * tile_h)
            if best is None or score > best[0]:
                best = (score, Layout(tiles, panel))

    return best[1] if best else Layout([], None)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def draw_joint_angles(image: np.ndarray, person: Person, min_score: float) -> None:
    """Write each angle next to the joint it belongs to (at camera resolution)."""
    fs = 0.5 * max(0.6, image.shape[0] / 540.0)
    th = _thickness(fs)
    off = int(8 * fs / 0.5)
    for label, _, vertex_name, _ in ANGLE_DEFS:
        value = person.angles.get(label)
        vertex = KP[vertex_name]
        if value is None or person.scores[vertex] < min_score:
            continue
        x, y = person.keypoints[vertex].astype(int)
        text = f"{value:.0f}"
        tw, tht, _ = _text_size(text, fs, th)
        pad = max(2, off // 3)
        cv2.rectangle(image, (x + off - pad, y - tht - off),
                      (x + off + tw + pad, y - off + 2 * pad), (0, 0, 0), -1)
        cv2.putText(image, text, (x + off, y - off + pad), FONT, fs, ACCENT, th, AA)


def draw_tile_label(canvas: np.ndarray, rect: Rect, text: str) -> None:
    x, y, w, h = rect
    s = float(np.clip(h / 480.0, 0.5, 1.6))
    fs = _fit_scale(text, w - 40 * s, 0.5 * s)
    th = _thickness(fs)
    tw, tht, _ = _text_size(text, fs, th)
    margin, pad_x, pad_y = int(8 * s), int(10 * s), int(7 * s)
    x0, y0 = x + margin, y + margin
    x1, y1 = min(x + w, x0 + tw + 2 * pad_x), min(y + h, y0 + tht + 2 * pad_y)
    _darken(canvas, x0, y0, x1, y1, 0.3)
    cv2.putText(canvas, text, (x0 + pad_x, y0 + pad_y + tht), FONT, fs, TEXT, th, AA)


def render_panel(w: int, h: int, n_people: int, angles: Dict[str, Optional[float]],
                 fps: float, paused: bool) -> np.ndarray:
    """Angle read-out that re-flows into 1, 2, 4 or 8 columns to fit its box."""
    panel = np.full((h, w, 3), PANEL_BG, np.uint8)
    hs = float(np.clip(min(w / 340.0, h / 250.0), 0.55, 1.8))
    pad = max(6, int(12 * hs))

    # ---- header: title, and the subtitle on the same line if there's room
    title = "JOINT ANGLES"
    t_fs = _fit_scale(title, w - 2 * pad, 0.62 * hs)
    t_th = _thickness(t_fs) + 1
    t_w, t_h, _ = _text_size(title, t_fs, t_th)
    y = pad + t_h
    cv2.putText(panel, title, (pad, y), FONT, t_fs, TEXT, t_th, AA)

    subtitle = ("no person detected" if n_people == 0 else
                "1 person" if n_people == 1 else f"largest of {n_people} people")
    s_fs = 0.45 * hs
    s_th = _thickness(s_fs)
    s_w, s_h, _ = _text_size(subtitle, s_fs, s_th)
    if pad + t_w + 2 * pad + s_w <= w - pad:
        cv2.putText(panel, subtitle, (w - pad - s_w, y), FONT, s_fs, MUTED, s_th, AA)
    else:
        s_fs = _fit_scale(subtitle, w - 2 * pad, s_fs)
        s_th = _thickness(s_fs)
        _, s_h, _ = _text_size(subtitle, s_fs, s_th)
        y += s_h + int(8 * hs)
        cv2.putText(panel, subtitle, (pad, y), FONT, s_fs, MUTED, s_th, AA)
    y += int(9 * hs)
    cv2.line(panel, (pad, y), (w - pad, y), (70, 70, 70), 1)
    body_top = y + int(6 * hs)

    # ---- footer: fps on the left, help hint on the right
    left = f"{fps:4.1f} FPS" + ("   PAUSED" if paused else "")
    right = "h: help"
    f_fs = _fit_scale(left, w - 2 * pad, 0.42 * hs)
    f_th = _thickness(f_fs)
    l_w, f_h, _ = _text_size(left, f_fs, f_th)
    r_w, _, _ = _text_size(right, f_fs, f_th)
    fy = h - pad
    cv2.putText(panel, left, (pad, fy), FONT, f_fs, ACCENT if paused else MUTED, f_th, AA)
    if pad + l_w + 2 * pad + r_w <= w - pad:
        cv2.putText(panel, right, (w - pad - r_w, fy), FONT, f_fs, DIM, f_th, AA)
    body_bottom = fy - f_h - int(10 * hs)

    # ---- body: choose the column count that allows the largest text
    body_w, body_h = w - 2 * pad, body_bottom - body_top
    if body_w < 40 or body_h < 10:
        return panel
    labels = [d[0] for d in ANGLE_DEFS]
    col_gap = int(18 * hs)
    best = None
    for cols in (1, 2, 4, 8):
        rows = -(-len(labels) // cols)
        cell_w = (body_w - col_gap * (cols - 1)) / cols
        cell_h = body_h / rows
        if cell_w <= 0:
            continue
        unit = min(cell_w / 290.0, cell_h / 42.0)
        if best is None or unit > best[0] * 1.05:   # favour fewer columns on ties
            best = (unit, cols, cell_w, cell_h)
    unit, cols, cell_w, cell_h = best
    unit = float(np.clip(unit, 0.35, 2.0))

    fs = 0.55 * unit
    th = _thickness(fs)
    bar_h = max(2, int(round(5 * unit)))
    bar_gap = max(3, int(7 * unit))
    for i, label in enumerate(labels):
        row, col = divmod(i, cols)             # row-major: L/R pairs sit together
        cx = pad + int(col * (cell_w + col_gap))
        cy = body_top + int(row * cell_h)
        cw = int(cell_w)
        value = angles.get(label)
        known = value is not None
        reading = f"{value:.1f} deg" if known else "--"
        l_w, l_h, _ = _text_size(label, fs, th)
        v_w, _, _ = _text_size(reading, fs, th)
        if known and l_w + v_w + 10 * unit > cw:
            reading = f"{value:.0f}"
            v_w, _, _ = _text_size(reading, fs, th)
        base = cy + int((cell_h - (l_h + bar_gap + bar_h)) / 2) + l_h
        cv2.putText(panel, label, (cx, base), FONT, fs, TEXT if known else DIM, th, AA)
        cv2.putText(panel, reading, (cx + cw - v_w, base), FONT, fs,
                    ACCENT if known else DIM, th, AA)
        bar_y = base + bar_gap
        cv2.rectangle(panel, (cx, bar_y), (cx + cw, bar_y + bar_h), TRACK, -1)
        if known:
            filled = int(cw * float(np.clip(value, 0.0, 180.0)) / 180.0)
            cv2.rectangle(panel, (cx, bar_y), (cx + filled, bar_y + bar_h), ACCENT, -1)
    return panel


def render_view(width: int, height: int, frame: np.ndarray, annotated: np.ndarray,
                n_people: int, angles: Dict[str, Optional[float]], fps: float,
                paused: bool, view_mode: str) -> np.ndarray:
    """Draw the whole UI at exactly width x height pixels."""
    canvas = np.full((height, width, 3), BG, np.uint8)
    aspect = frame.shape[1] / frame.shape[0]
    gap = max(4, int(min(width, height) * 0.008))

    if view_mode == "auto":
        sources = [(frame, "ORIGINAL"), (annotated, "POSE ESTIMATION")]
        show_panel = True
    elif view_mode == "pose + angles":
        sources = [(annotated, "POSE ESTIMATION")]
        show_panel = True
    else:  # pose only - status chip on the video instead of the panel
        sources = [(annotated, f"POSE  {fps:.0f} FPS" + ("  PAUSED" if paused else ""))]
        show_panel = False

    layout = compute_layout(width, height, aspect, len(sources), show_panel, gap)
    for (image, label), (x, y, w, h) in zip(sources, layout.tiles):
        if w < 2 or h < 2:
            continue
        interp = cv2.INTER_AREA if w < image.shape[1] else cv2.INTER_LINEAR
        canvas[y:y + h, x:x + w] = cv2.resize(image, (w, h), interpolation=interp)
        draw_tile_label(canvas, (x, y, w, h), label)

    if layout.panel is not None:
        x, y, w, h = layout.panel
        canvas[y:y + h, x:x + w] = render_panel(w, h, n_people, angles, fps, paused)
    return canvas


HELP_LINES: Tuple[Tuple[str, str], ...] = (
    ("f", "toggle full screen"),
    ("v", "cycle view: auto / pose + angles / pose only"),
    ("m", "mirror the image"),
    ("space", "pause / resume"),
    ("s", "save a snapshot"),
    ("h", "show / hide this help"),
    ("Esc", "leave full screen (quit when windowed)"),
    ("q", "quit"),
)


def draw_help(canvas: np.ndarray) -> None:
    H, W = canvas.shape[:2]
    s = float(np.clip(min(W / 1280.0, H / 720.0), 0.45, 1.6))
    fs = 0.55 * s
    th = _thickness(fs)
    key_w = max(_text_size(k, fs, th)[0] for k, _ in HELP_LINES)
    desc_w = max(_text_size(d, fs, th)[0] for _, d in HELP_LINES)
    col_gap, pad = int(24 * s), int(22 * s)
    content_w = key_w + col_gap + desc_w
    if content_w + 2 * pad > W * 0.92:                       # shrink to fit
        shrink = (W * 0.92 - 2 * pad) / content_w
        fs *= shrink
        th = _thickness(fs)
        key_w = int(key_w * shrink)
        col_gap = int(col_gap * shrink)
        content_w = int(content_w * shrink)
    _, line_h, _ = _text_size("Hg", fs, th)
    step = int(line_h * 2.1)
    box_w = content_w + 2 * pad
    box_h = step * (len(HELP_LINES) + 1) + 2 * pad
    x0, y0 = (W - box_w) // 2, max(0, (H - box_h) // 2)
    _darken(canvas, x0, y0, x0 + box_w, y0 + box_h, 0.18)
    cv2.rectangle(canvas, (x0, y0), (x0 + box_w, y0 + box_h), ACCENT, 1)
    y = y0 + pad + line_h
    cv2.putText(canvas, "KEYBOARD SHORTCUTS", (x0 + pad, y), FONT, fs, TEXT, th + 1, AA)
    for key, desc in HELP_LINES:
        y += step
        cv2.putText(canvas, key, (x0 + pad, y), FONT, fs, ACCENT, th, AA)
        cv2.putText(canvas, desc, (x0 + pad + key_w + col_gap, y), FONT, fs, TEXT, th, AA)


class Toast:
    """Short status message shown at the top of the window."""

    def __init__(self) -> None:
        self.text = ""
        self.until = 0.0

    def show(self, text: str, seconds: float = 1.8) -> None:
        self.text, self.until = text, time.monotonic() + seconds

    @property
    def active(self) -> bool:
        return bool(self.text) and time.monotonic() < self.until

    def draw(self, canvas: np.ndarray) -> None:
        if not self.active:
            return
        H, W = canvas.shape[:2]
        s = float(np.clip(min(W / 1280.0, H / 720.0), 0.5, 1.6))
        fs = _fit_scale(self.text, W - 60 * s, 0.6 * s)
        th = _thickness(fs)
        tw, tht, _ = _text_size(self.text, fs, th)
        pad_x, pad_y = int(16 * s), int(10 * s)
        x0, y0 = (W - tw - 2 * pad_x) // 2, int(20 * s)
        x1, y1 = x0 + tw + 2 * pad_x, y0 + tht + 2 * pad_y
        _darken(canvas, x0, y0, x1, y1, 0.2)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), ACCENT, 1)
        cv2.putText(canvas, self.text, (x0 + pad_x, y0 + pad_y + tht), FONT, fs, TEXT, th, AA)


def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Fit an image into a fixed size without distortion (used for recording)."""
    ih, iw = image.shape[:2]
    scale = min(width / iw, height / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    out = np.full((height, width, 3), BG, np.uint8)
    x, y = (width - nw) // 2, (height - nh) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


# --------------------------------------------------------------------------- #
# Window management
# --------------------------------------------------------------------------- #


def enable_hidpi() -> None:
    """On Windows, opt out of bitmap scaling so the UI is sharp on HiDPI screens."""
    if platform.system() != "Windows":
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def screen_size() -> Optional[Tuple[int, int]]:
    system = platform.system()
    size: Optional[Tuple[int, int]] = None
    try:
        if system == "Windows":
            import ctypes
            user32 = ctypes.windll.user32
            size = (int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1)))
        elif system != "Darwin":          # Tk and OpenCV's Cocoa loop don't mix
            import tkinter
            root = tkinter.Tk()
            root.withdraw()
            size = (int(root.winfo_screenwidth()), int(root.winfo_screenheight()))
            root.destroy()
    except Exception:
        return None
    # A multi-monitor desktop can report the combined width - ignore that.
    if size and size[0] > 0 and size[1] > 0 and size[0] / size[1] <= 2.4:
        return size
    return None


class Display:
    """OpenCV window whose drawable size is queried every frame."""

    MIN_SIZE = 64

    def __init__(self, name: str, size: Optional[Tuple[int, int]], fullscreen: bool):
        self.name = name
        self.screen = screen_size()
        if size is None:
            if self.screen:
                sw, sh = self.screen
                w = min(1600, int(sw * 0.85))
                size = (w, min(int(sh * 0.8), int(w * 9 / 16)))
            else:
                size = (1280, 720)
        self.windowed_size = size
        self._last = size
        self._seen_visible = False
        self.fullscreen = False

        flags = cv2.WINDOW_NORMAL
        flags |= getattr(cv2, "WINDOW_FREERATIO", 0)      # let the image fill the window
        flags |= getattr(cv2, "WINDOW_GUI_NORMAL", 0)     # no Qt toolbar / status bar
        cv2.namedWindow(name, flags)
        cv2.resizeWindow(name, *size)
        if fullscreen:
            self.toggle_fullscreen()

    def toggle_fullscreen(self) -> None:
        if not self.fullscreen:
            self.windowed_size = self._last
            cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            self.fullscreen = True
        else:
            cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.name, *self.windowed_size)
            self.fullscreen = False

    def canvas_size(self) -> Tuple[int, int]:
        """Current drawable area of the window, with sensible fallbacks."""
        w = h = 0
        try:
            _, _, w, h = cv2.getWindowImageRect(self.name)
        except (cv2.error, AttributeError):
            pass
        if w >= self.MIN_SIZE and h >= self.MIN_SIZE:
            self._last = (int(w), int(h))
            return self._last
        if self.fullscreen and self.screen:
            return self.screen
        return self._last

    def is_open(self) -> bool:
        try:
            visible = cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            return False
        if visible >= 1:
            self._seen_visible = True
            return True
        # Some backends never report visibility; only treat it as closed once
        # we have seen it open.
        return not self._seen_visible


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


def _size_arg(text: str) -> Tuple[int, int]:
    try:
        w, h = (int(v) for v in text.lower().split("x"))
        if w <= 0 or h <= 0:
            raise ValueError
        return w, h
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, got '{text}'")


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
    parser.add_argument("--fullscreen", action="store_true", help="start in full screen")
    parser.add_argument("--window-size", type=_size_arg, default=None,
                        help="initial window size, e.g. 1280x720 (default: fit the screen)")
    parser.add_argument("--view", choices=VIEW_MODES, default="auto",
                        help="initial view mode (default: auto)")
    parser.add_argument("--smooth", type=float, default=0.4,
                        help="angle smoothing factor, 1.0 disables it (default: 0.4)")
    parser.add_argument("--no-mirror", action="store_true",
                        help="do not mirror the camera image")
    parser.add_argument("--record", default=None,
                        help="write the view to this .mp4 file")
    parser.add_argument("--record-size", type=_size_arg, default=(1280, 720),
                        help="fixed frame size of the recording (default: 1280x720)")
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
    enable_hidpi()

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

    display = Display(WINDOW_NAME, args.window_size, args.fullscreen)
    toast = Toast()
    toast.show("Press h for keyboard shortcuts", 4.0)

    smoother = AngleSmoother(args.smooth)
    view = VIEW_MODES.index(args.view)
    mirror = not args.no_mirror
    paused = False
    show_help = False
    fps = 0.0
    last_time = time.perf_counter()
    writer: Optional[cv2.VideoWriter] = None
    snapshots = 0

    frame: Optional[np.ndarray] = None
    annotated: Optional[np.ndarray] = None
    n_people = 0
    angles: Dict[str, Optional[float]] = {label: None for label, *_ in ANGLE_DEFS}

    try:
        while True:
            new_frame = False
            if not paused:
                ok, raw = cap.read()
                if not ok or raw is None:
                    print("Camera stopped delivering frames.", file=sys.stderr)
                    break
                frame = cv2.flip(raw, 1) if mirror else raw

                result = estimator(frame)
                people = extract_people(result, args.kpt_conf)
                annotated = PoseEstimator.overlay(result, frame)
                for person in people:
                    draw_joint_angles(annotated, person, args.kpt_conf)
                n_people = len(people)
                if people:
                    angles = smoother(people[0].angles)
                else:
                    smoother.reset()
                    angles = {label: None for label, *_ in ANGLE_DEFS}

                now = time.perf_counter()
                delta, last_time = now - last_time, now
                if delta > 0:
                    fps = 0.9 * fps + 0.1 / delta if fps else 1.0 / delta
                new_frame = True

            if frame is None or annotated is None:
                if (cv2.waitKey(10) & 0xFF) in (ord("q"), 27):
                    break
                continue

            # Re-render every iteration (even when paused) so resizing and
            # full-screen changes take effect immediately.
            width, height = display.canvas_size()
            canvas = render_view(width, height, frame, annotated, n_people, angles,
                                 fps, paused, VIEW_MODES[view])

            if args.record and new_frame:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.record, fourcc, 20.0, args.record_size)
                    if not writer.isOpened():
                        print(f"Could not open '{args.record}' for writing.", file=sys.stderr)
                        args.record, writer = None, None
                if writer is not None:
                    writer.write(letterbox(canvas, *args.record_size))

            shown = canvas
            if show_help or toast.active:            # overlays never go into recordings
                shown = canvas.copy()
                if show_help:
                    draw_help(shown)
                toast.draw(shown)
            cv2.imshow(display.name, shown)

            key = cv2.waitKey(30 if paused else 1) & 0xFF
            char = chr(key).lower() if key < 128 else ""
            if key == 27:                             # Esc
                if display.fullscreen:
                    display.toggle_fullscreen()
                    toast.show("Windowed")
                else:
                    break
            elif char == "q":
                break
            elif char == "f":
                display.toggle_fullscreen()
                toast.show("Full screen  (Esc to leave)" if display.fullscreen else "Windowed")
            elif char == "v":
                view = (view + 1) % len(VIEW_MODES)
                toast.show(f"View: {VIEW_MODES[view]}")
            elif char == "m":
                mirror = not mirror
                smoother.reset()
                toast.show("Mirror on" if mirror else "Mirror off")
            elif char == " ":
                paused = not paused
                last_time = time.perf_counter()
                toast.show("Paused" if paused else "Resumed")
            elif char == "h":
                show_help = not show_help
            elif char == "s":
                snapshots += 1
                path = f"pose_snapshot_{snapshots:03d}.png"
                cv2.imwrite(path, canvas)
                toast.show(f"Saved {path}")
                print(f"Saved {path}")

            if not display.is_open():
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
