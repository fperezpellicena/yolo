"""Responsive placement of the video tiles and the angle panel."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

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
