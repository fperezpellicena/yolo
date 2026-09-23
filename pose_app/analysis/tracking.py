"""Follow one athlete through a clip that may contain other people."""

from typing import List, Optional, Tuple

import numpy as np

from ..person import Person

SELECT_MODES = ("largest", "center")


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


class AthleteTracker:
    """Locks onto one person, then follows them frame to frame.

    Initial pick: the largest person, the one nearest the frame centre, or the
    one containing / nearest to `point` (x, y in pixels). After that the
    athlete is matched by box overlap, falling back to the nearest similarly
    sized box, so a passer-by never steals the track.
    """

    def __init__(self, mode: str = "largest", point: Optional[Tuple[float, float]] = None,
                 frame_size: Optional[Tuple[int, int]] = None, min_iou: float = 0.2):
        if mode not in SELECT_MODES:
            raise ValueError(f"mode must be one of {SELECT_MODES}")
        self.mode, self.point, self.frame_size, self.min_iou = mode, point, frame_size, min_iou
        self.last_box: Optional[np.ndarray] = None

    def _initial(self, people: List[Person]) -> Person:
        if self.point is not None:
            p = np.asarray(self.point, float)
            inside = [q for q in people
                      if q.box[0] <= p[0] <= q.box[2] and q.box[1] <= p[1] <= q.box[3]]
            pool = inside or people
            return min(pool, key=lambda q: float(np.linalg.norm(_center(q.box) - p)))
        if self.mode == "center" and self.frame_size is not None:
            c = np.array(self.frame_size, float) / 2
            return min(people, key=lambda q: float(np.linalg.norm(_center(q.box) - c)))
        return max(people, key=lambda q: q.area)

    def select(self, people: List[Person]) -> Optional[Person]:
        if not people:
            return None
        if self.last_box is None:
            chosen = self._initial(people)
        else:
            best = max(people, key=lambda q: iou(q.box, self.last_box))
            if iou(best.box, self.last_box) >= self.min_iou:
                chosen = best
            else:
                chosen = self._reacquire(people)
        if chosen is not None:
            self.last_box = chosen.box.copy()
        return chosen

    def _reacquire(self, people: List[Person]) -> Optional[Person]:
        last = self.last_box
        last_area = (last[2] - last[0]) * (last[3] - last[1])
        diag = float(np.hypot(last[2] - last[0], last[3] - last[1]))
        candidates = [q for q in people if 0.5 <= q.area / max(last_area, 1.0) <= 2.0
                      and np.linalg.norm(_center(q.box) - _center(last)) < diag]
        if not candidates:
            return None
        return min(candidates, key=lambda q: float(np.linalg.norm(_center(q.box) - _center(last))))
