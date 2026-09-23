"""What every station analyzer provides."""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..body import BodySeries
from ..rules import DriftRule, Rule
from ..signal import Cycle


@dataclass(frozen=True)
class ChartSpec:
    metric: str
    label: str
    unit: str = ""


class Station:
    """Subclass per Hyrox station.

    The driver metric is the signal whose peak -> trough -> peak defines one rep
    (e.g. wrist height on the SkiErg). `summarize` turns one rep into a flat
    dict of numbers; rules then check those numbers.
    """

    key: str = ""
    name: str = ""
    rep_word: str = "rep"
    view: str = "side"                     # camera view the rules assume
    driver: str = ""
    lo_frac: float = 0.35                  # hysteresis inside the driver's range
    hi_frac: float = 0.65
    min_rep_s: float = 0.5
    max_rep_s: float = 6.0
    phases: Tuple[str, str] = ("phase 1", "phase 2")   # start->mid, mid->end
    filming_tips: Sequence[str] = ()
    charts: Sequence[ChartSpec] = ()

    def rules(self) -> List[Rule]:
        raise NotImplementedError

    def drift_rules(self) -> List[DriftRule]:
        return []

    def summarize(self, body: BodySeries, rep: Cycle) -> Dict[str, float]:
        raise NotImplementedError

    # ---- helpers shared by stations
    @staticmethod
    def at(series: np.ndarray, i: int, half: int = 1) -> float:
        """Median around frame i (robust single-frame reading)."""
        chunk = series[max(0, i - half): i + half + 1]
        return float(np.nanmedian(chunk)) if np.isfinite(chunk).any() else float("nan")

    @staticmethod
    def extreme(series: np.ndarray, a: int, b: int, kind: str) -> Tuple[float, int]:
        """(min or max value, frame index) over [a, b]."""
        chunk = series[a:b + 1]
        if not np.isfinite(chunk).any():
            return float("nan"), a
        i = int(np.nanargmin(chunk) if kind == "min" else np.nanargmax(chunk))
        return float(chunk[i]), a + i
