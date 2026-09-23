"""Small labels drawn on top of video: joint angles and tile captions."""

import cv2
import numpy as np

from ..person import Person
from ..skeleton import ANGLE_DEFS, KP
from .layout import Rect
from .text import darken, fit_scale, text_size, thickness
from .theme import AA, ACCENT, FONT, TEXT


def draw_joint_angles(image: np.ndarray, person: Person, min_score: float) -> None:
    """Write each angle next to the joint it belongs to (at camera resolution)."""
    fs = 0.5 * max(0.6, image.shape[0] / 540.0)
    th = thickness(fs)
    off = int(8 * fs / 0.5)
    for label, _, vertex_name, _ in ANGLE_DEFS:
        value = person.angles.get(label)
        vertex = KP[vertex_name]
        if value is None or person.scores[vertex] < min_score:
            continue
        x, y = person.keypoints[vertex].astype(int)
        text = f"{value:.0f}"
        tw, tht, _ = text_size(text, fs, th)
        pad = max(2, off // 3)
        cv2.rectangle(image, (x + off - pad, y - tht - off),
                      (x + off + tw + pad, y - off + 2 * pad), (0, 0, 0), -1)
        cv2.putText(image, text, (x + off, y - off + pad), FONT, fs, ACCENT, th, AA)


def draw_tile_label(canvas: np.ndarray, rect: Rect, text: str) -> None:
    """Caption chip in the top-left corner of a video tile."""
    x, y, w, h = rect
    s = float(np.clip(h / 480.0, 0.5, 1.6))
    fs = fit_scale(text, w - 40 * s, 0.5 * s)
    th = thickness(fs)
    tw, tht, _ = text_size(text, fs, th)
    margin, pad_x, pad_y = int(8 * s), int(10 * s), int(7 * s)
    x0, y0 = x + margin, y + margin
    x1, y1 = min(x + w, x0 + tw + 2 * pad_x), min(y + h, y0 + tht + 2 * pad_y)
    darken(canvas, x0, y0, x1, y1, 0.3)
    cv2.putText(canvas, text, (x0 + pad_x, y0 + pad_y + tht), FONT, fs, TEXT, th, AA)
