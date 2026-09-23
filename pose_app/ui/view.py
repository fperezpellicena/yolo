"""Composes the full window image: video tiles + angle panel."""

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from ..skeleton import Angles
from .angle_panel import render_angle_panel
from .annotations import draw_tile_label
from .layout import compute_layout
from .theme import BG

VIEW_MODES: Tuple[str, ...] = ("auto", "pose + angles", "pose only")


@dataclass(frozen=True)
class ViewState:
    """Everything the view needs to draw one frame."""
    frame: np.ndarray        # raw camera image (already mirrored if enabled)
    annotated: np.ndarray    # frame with skeletons and joint angles drawn
    n_people: int
    angles: Angles           # angles of the primary (largest) person
    fps: float = 0.0
    paused: bool = False


def render_view(width: int, height: int, state: ViewState, view_mode: str) -> np.ndarray:
    """Draw the whole UI at exactly width x height pixels."""
    canvas = np.full((height, width, 3), BG, np.uint8)
    aspect = state.frame.shape[1] / state.frame.shape[0]
    gap = max(4, int(min(width, height) * 0.008))

    if view_mode == "auto":
        sources = [(state.frame, "ORIGINAL"), (state.annotated, "POSE ESTIMATION")]
        show_panel = True
    elif view_mode == "pose + angles":
        sources = [(state.annotated, "POSE ESTIMATION")]
        show_panel = True
    else:  # pose only - status chip on the video instead of the panel
        chip = f"POSE  {state.fps:.0f} FPS" + ("  PAUSED" if state.paused else "")
        sources = [(state.annotated, chip)]
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
        canvas[y:y + h, x:x + w] = render_angle_panel(
            w, h, state.n_people, state.angles, state.fps, state.paused)
    return canvas
