"""Keyboard-shortcut overlay."""

from typing import Tuple

import cv2
import numpy as np

from .text import darken, text_size, thickness
from .theme import AA, ACCENT, FONT, TEXT

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
    th = thickness(fs)
    key_w = max(text_size(k, fs, th)[0] for k, _ in HELP_LINES)
    desc_w = max(text_size(d, fs, th)[0] for _, d in HELP_LINES)
    col_gap, pad = int(24 * s), int(22 * s)
    content_w = key_w + col_gap + desc_w
    if content_w + 2 * pad > W * 0.92:                       # shrink to fit
        shrink = (W * 0.92 - 2 * pad) / content_w
        fs *= shrink
        th = thickness(fs)
        key_w = int(key_w * shrink)
        col_gap = int(col_gap * shrink)
        content_w = int(content_w * shrink)
    _, line_h, _ = text_size("Hg", fs, th)
    step = int(line_h * 2.1)
    box_w = content_w + 2 * pad
    box_h = step * (len(HELP_LINES) + 1) + 2 * pad
    x0, y0 = (W - box_w) // 2, max(0, (H - box_h) // 2)
    darken(canvas, x0, y0, x0 + box_w, y0 + box_h, 0.18)
    cv2.rectangle(canvas, (x0, y0), (x0 + box_w, y0 + box_h), ACCENT, 1)
    y = y0 + pad + line_h
    cv2.putText(canvas, "KEYBOARD SHORTCUTS", (x0 + pad, y), FONT, fs, TEXT, th + 1, AA)
    for key, desc in HELP_LINES:
        y += step
        cv2.putText(canvas, key, (x0 + pad, y), FONT, fs, ACCENT, th, AA)
        cv2.putText(canvas, desc, (x0 + pad + key_w + col_gap, y), FONT, fs, TEXT, th, AA)
