"""Webcam discovery and opening (no user interaction here)."""

import os
import platform
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2


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


def open_capture(index: int, backend: int) -> Optional[cv2.VideoCapture]:
    """Open a device with the preferred backend, falling back to CAP_ANY."""
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
        cap = open_capture(index, backend)
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


def open_camera(camera: Camera, size: Tuple[int, int]) -> cv2.VideoCapture:
    """Open the chosen camera and request a capture resolution."""
    cap = open_capture(camera.index, camera.backend)
    if cap is None:
        raise RuntimeError(f"Could not open camera index {camera.index}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    return cap
