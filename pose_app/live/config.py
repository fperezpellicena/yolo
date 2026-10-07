"""All runtime options, independent of how they were supplied."""

from dataclasses import dataclass
from typing import Optional, Tuple

Size = Tuple[int, int]


@dataclass(frozen=True)
class Settings:
    # model
    model: str = "yolo11n-pose.pt"
    conf: float = 0.5                 # person detection threshold
    kpt_conf: float = 0.5             # minimum keypoint confidence for an angle
    imgsz: int = 640
    infer_device: Optional[str] = None
    # capture
    capture_size: Size = (1280, 720)
    mirror: bool = True
    # display
    fullscreen: bool = False
    window_size: Optional[Size] = None
    view: str = "auto"
    # processing
    smooth: float = 0.4
    # output
    record_path: Optional[str] = None
    record_size: Size = (1280, 720)
