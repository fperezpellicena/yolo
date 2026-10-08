"""Force lines on the annotated video.

While the cords pull: the cord force as an arrow at the hands (length grows
with the tension), its line of action carried on through the body, and the
moment arms from the shoulder and the elbow to that line, labelled with the
moment per side. Always: the cord exit, the cord itself, and the body's
centre of mass, whose drop is what lends the pull the body's weight.
"""

from typing import Optional

import cv2
import numpy as np

from ...drawing.text import darken, text_size, thickness
from ...drawing.theme import AA, FONT
from .dynamics import MOMENT_NAMES

FORCE = (0, 140, 255)           # BGR orange: the cord force
ARM = (230, 200, 40)            # BGR cyan-blue: moment arms
CORD = (170, 170, 170)
COM = (255, 255, 255)
ARROW_M_PER_N = 0.0010          # arrow length: 1 m per 1000 N
MIN_SHOWN = 0.03                # tensions below this share of the session's peak are not drawn
SHORT = {"flexion": "flex", "extension": "ext"}


def _pt(p: np.ndarray):
    return int(round(p[0])), int(round(p[1]))


def _label(img, text: str, at, fs: float, color) -> None:
    x, y = int(at[0]), int(at[1])
    tw, th, _ = text_size(text, fs, thickness(fs))
    darken(img, x - 3, y - th - 4, x + tw + 4, y + 5, 0.3)
    cv2.putText(img, text, (x, y), FONT, fs, color, thickness(fs), AA)


def draw_force(img: np.ndarray, fa, i: int, s: float) -> None:
    """Draw frame i's force picture; `s` is the text/line scale of the frame."""
    if fa.tension is None or fa.frame is None:
        return
    px = fa.pixels
    exit_px = np.asarray(fa.setup.calibration.cord_exit, float)
    grip, com = px["grip"][i], px["com"][i]
    lw = max(1, int(round(2 * s)))
    cv2.circle(img, _pt(exit_px), max(3, int(5 * s)), CORD, -1, AA)
    if np.isfinite(grip).all():
        cv2.line(img, _pt(grip), _pt(exit_px), CORD, max(1, lw - 1), AA)
    if np.isfinite(com).all():
        r = max(4, int(7 * s))
        cv2.circle(img, _pt(com), r, COM, lw, AA)
        cv2.line(img, _pt(com - [r, 0]), _pt(com + [r, 0]), COM, 1, AA)
        cv2.line(img, _pt(com - [0, r]), _pt(com + [0, r]), COM, 1, AA)
    T = fa.tension[i]
    if not (np.isfinite(T) and T > MIN_SHOWN * fa.peak_tension and np.isfinite(grip).all()):
        return
    d = exit_px - grip
    reach = float(np.linalg.norm(d))
    d = d / reach
    scale = fa.frame.scale
    tip = grip + d * min(T * ARROW_M_PER_N * scale, 0.8 * reach)   # stops short of the exit
    tail = grip - d * 0.9 * scale                              # the line of action, into the body
    cv2.line(img, _pt(tail), _pt(grip), FORCE, lw, AA)
    cv2.arrowedLine(img, _pt(grip), _pt(tip), FORCE, max(2, int(4 * s)), AA, tipLength=0.18)
    fs = 0.55 * s
    side = np.array([-d[1], d[0]])
    side = side if side[0] >= 0 else -side                     # beside the arrow, image-right
    _label(img, f"{T:.0f} N", tip - d * 24 * s + side * 16 * s, fs, FORCE)
    # labels: the shoulder's up and to the left, the elbow's below and to the right
    for joint, tag, at in (("shoulder", "shoulder", (-150, -14)), ("elbow", "elbow", (12, 34))):
        j = px[joint][i]
        if not np.isfinite(j).all():
            continue
        foot = grip + np.dot(j - grip, d) * d
        cv2.line(img, _pt(j), _pt(foot), ARM, max(1, lw - 1), AA)
        m = fa.per_side(joint)[i]
        if np.isfinite(m):
            name = SHORT[MOMENT_NAMES[joint][0 if m >= 0 else 1]]
            _label(img, f"{tag} {abs(m):.0f} Nm {name}", j + np.array(at) * s, 0.45 * s, ARM)


def header_line(fa, metrics: dict, i: Optional[int]) -> str:
    """'cord 312 N  body weight 41%' for the header, '' without force data."""
    parts = []
    if i is not None and fa.tension is not None:
        T = fa.tension[i]
        if np.isfinite(T) and T > 1:
            parts.append(f"cord {T:.0f} N")
    bw = metrics.get("f_bw_share", np.nan)
    if np.isfinite(bw):
        parts.append(f"body weight {bw:.0f}%")
    return "  ".join(parts)
