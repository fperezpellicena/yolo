"""Short-lived status message at the top of the window."""

import time

import cv2
import numpy as np

from .text import darken, fit_scale, text_size, thickness
from .theme import AA, ACCENT, FONT, TEXT


class Toast:
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
        fs = fit_scale(self.text, W - 60 * s, 0.6 * s)
        th = thickness(fs)
        tw, tht, _ = text_size(self.text, fs, th)
        pad_x, pad_y = int(16 * s), int(10 * s)
        x0, y0 = (W - tw - 2 * pad_x) // 2, int(20 * s)
        x1, y1 = x0 + tw + 2 * pad_x, y0 + tht + 2 * pad_y
        darken(canvas, x0, y0, x1, y1, 0.2)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), ACCENT, 1)
        cv2.putText(canvas, self.text, (x0 + pad_x, y0 + pad_y + tht), FONT, fs, TEXT, th, AA)
