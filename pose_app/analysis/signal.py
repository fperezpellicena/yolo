"""Time-series helpers: gap filling, zero-lag smoothing, cycle detection."""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def fill_gaps(y: np.ndarray, t: np.ndarray, max_gap_s: float) -> np.ndarray:
    """Linearly bridge NaN runs no longer than max_gap_s; longer gaps stay NaN."""
    y = np.asarray(y, float).copy()
    missing = np.isnan(y)
    if not missing.any() or missing.all():
        return y
    good = np.flatnonzero(~missing)
    interp = np.interp(t, t[good], y[good])
    edges = np.diff(np.concatenate([[0], missing.astype(np.int8), [0]]))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if a > 0 and b < len(y) and t[b] - t[a - 1] <= max_gap_s:
            y[a:b] = interp[a:b]
    return y


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average (no lag), ignoring NaNs; NaNs stay NaN."""
    y = np.asarray(y, float)
    if window <= 1:
        return y.copy()
    valid = ~np.isnan(y)
    kernel = np.ones(window)
    num = np.convolve(np.where(valid, y, 0.0), kernel, "same")
    den = np.convolve(valid.astype(float), kernel, "same")
    out = np.full_like(y, np.nan)
    ok = valid & (den > 0)
    out[ok] = num[ok] / den[ok]
    return out


@dataclass(frozen=True)
class Cycle:
    """Frame indices of one rep: start peak -> trough -> end peak."""
    start: int
    mid: int
    end: int


def hysteresis_thresholds(y: np.ndarray, lo_frac: float, hi_frac: float) -> Tuple[float, float]:
    """Thresholds placed inside the signal's robust (10-90th percentile) range."""
    v = y[~np.isnan(y)]
    if v.size == 0:
        return np.nan, np.nan
    p10, p90 = np.percentile(v, [10, 90])
    span = p90 - p10
    return p10 + lo_frac * span, p10 + hi_frac * span


def find_cycles(y: np.ndarray, t: np.ndarray, lo: float, hi: float,
                min_s: float, max_s: float) -> List[Cycle]:
    """Peak -> trough -> peak cycles using hysteresis (robust to jitter).

    A peak is the maximum of a stretch that rose above `hi`, a trough the
    minimum of a stretch that fell below `lo`. Missing data breaks a cycle.
    The same state machine can later run live on a webcam stream.
    """
    extrema: List[Tuple[str, int]] = []
    state, best = None, -1
    for i, v in enumerate(y):
        if np.isnan(v):
            if state is not None:
                extrema.append(("break", i))
            state = None
            continue
        if state == "high":
            if v > y[best]:
                best = i
            if v < lo:
                extrema.append(("peak", best))
                state, best = "low", i
        elif state == "low":
            if v < y[best]:
                best = i
            if v > hi:
                extrema.append(("trough", best))
                state, best = "high", i
        elif v > hi:
            state, best = "high", i
        elif v < lo:
            state, best = "low", i
    if state == "high":
        extrema.append(("peak", best))    # the clip ends after a final catch

    cycles: List[Cycle] = []
    for (k0, a), (k1, b), (k2, c) in zip(extrema, extrema[1:], extrema[2:]):
        if (k0, k1, k2) == ("peak", "trough", "peak") and min_s <= t[c] - t[a] <= max_s:
            cycles.append(Cycle(a, b, c))
    return cycles
