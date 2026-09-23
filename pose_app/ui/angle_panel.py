"""The joint-angle read-out panel."""

import cv2
import numpy as np

from ..skeleton import ANGLE_LABELS, Angles
from .text import fit_scale, text_size, thickness
from .theme import AA, ACCENT, DIM, DIVIDER, FONT, MUTED, PANEL_BG, TEXT, TRACK


def render_angle_panel(w: int, h: int, n_people: int, angles: Angles,
                       fps: float, paused: bool) -> np.ndarray:
    """Draw the panel at w x h; the readings re-flow into 1, 2, 4 or 8 columns."""
    panel = np.full((h, w, 3), PANEL_BG, np.uint8)
    hs = float(np.clip(min(w / 340.0, h / 250.0), 0.55, 1.8))
    pad = max(6, int(12 * hs))
    body_top = _draw_header(panel, n_people, hs, pad)
    body_bottom = _draw_footer(panel, fps, paused, hs, pad)
    _draw_readings(panel, angles, body_top, body_bottom, hs, pad)
    return panel


def _draw_header(panel: np.ndarray, n_people: int, hs: float, pad: int) -> int:
    """Title plus subtitle (same line if it fits). Returns the body's top y."""
    w = panel.shape[1]
    title = "JOINT ANGLES"
    t_fs = fit_scale(title, w - 2 * pad, 0.62 * hs)
    t_th = thickness(t_fs) + 1
    t_w, t_h, _ = text_size(title, t_fs, t_th)
    y = pad + t_h
    cv2.putText(panel, title, (pad, y), FONT, t_fs, TEXT, t_th, AA)

    subtitle = ("no person detected" if n_people == 0 else
                "1 person" if n_people == 1 else f"largest of {n_people} people")
    s_fs = 0.45 * hs
    s_th = thickness(s_fs)
    s_w, _, _ = text_size(subtitle, s_fs, s_th)
    if pad + t_w + 2 * pad + s_w <= w - pad:
        cv2.putText(panel, subtitle, (w - pad - s_w, y), FONT, s_fs, MUTED, s_th, AA)
    else:
        s_fs = fit_scale(subtitle, w - 2 * pad, s_fs)
        s_th = thickness(s_fs)
        _, s_h, _ = text_size(subtitle, s_fs, s_th)
        y += s_h + int(8 * hs)
        cv2.putText(panel, subtitle, (pad, y), FONT, s_fs, MUTED, s_th, AA)
    y += int(9 * hs)
    cv2.line(panel, (pad, y), (w - pad, y), DIVIDER, 1)
    return y + int(6 * hs)


def _draw_footer(panel: np.ndarray, fps: float, paused: bool, hs: float, pad: int) -> int:
    """FPS on the left, help hint on the right. Returns the body's bottom y."""
    h, w = panel.shape[:2]
    left = f"{fps:4.1f} FPS" + ("   PAUSED" if paused else "")
    right = "h: help"
    f_fs = fit_scale(left, w - 2 * pad, 0.42 * hs)
    f_th = thickness(f_fs)
    l_w, f_h, _ = text_size(left, f_fs, f_th)
    r_w, _, _ = text_size(right, f_fs, f_th)
    fy = h - pad
    cv2.putText(panel, left, (pad, fy), FONT, f_fs, ACCENT if paused else MUTED, f_th, AA)
    if pad + l_w + 2 * pad + r_w <= w - pad:
        cv2.putText(panel, right, (w - pad - r_w, fy), FONT, f_fs, DIM, f_th, AA)
    return fy - f_h - int(10 * hs)


def _draw_readings(panel: np.ndarray, angles: Angles, top: int, bottom: int,
                   hs: float, pad: int) -> None:
    body_w, body_h = panel.shape[1] - 2 * pad, bottom - top
    if body_w < 40 or body_h < 10:
        return

    # choose the column count that allows the largest text
    col_gap = int(18 * hs)
    best = None
    for cols in (1, 2, 4, 8):
        rows = -(-len(ANGLE_LABELS) // cols)
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
    th = thickness(fs)
    bar_h = max(2, int(round(5 * unit)))
    bar_gap = max(3, int(7 * unit))
    for i, label in enumerate(ANGLE_LABELS):
        row, col = divmod(i, cols)             # row-major: L/R pairs sit together
        cx = pad + int(col * (cell_w + col_gap))
        cy = top + int(row * cell_h)
        cw = int(cell_w)
        value = angles.get(label)
        known = value is not None
        reading = f"{value:.1f} deg" if known else "--"
        l_w, l_h, _ = text_size(label, fs, th)
        v_w, _, _ = text_size(reading, fs, th)
        if known and l_w + v_w + 10 * unit > cw:
            reading = f"{value:.0f}"
            v_w, _, _ = text_size(reading, fs, th)
        base = cy + int((cell_h - (l_h + bar_gap + bar_h)) / 2) + l_h
        cv2.putText(panel, label, (cx, base), FONT, fs, TEXT if known else DIM, th, AA)
        cv2.putText(panel, reading, (cx + cw - v_w, base), FONT, fs,
                    ACCENT if known else DIM, th, AA)
        bar_y = base + bar_gap
        cv2.rectangle(panel, (cx, bar_y), (cx + cw, bar_y + bar_h), TRACK, -1)
        if known:
            filled = int(cw * float(np.clip(value, 0.0, 180.0)) / 180.0)
            cv2.rectangle(panel, (cx, bar_y), (cx + filled, bar_y + bar_h), ACCENT, -1)
