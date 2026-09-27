"""Put video strokes and machine samples on one clock.

`offset` maps video time to machine time: machine_t = video_t + offset.
Equivalently, offset is the machine time (s since the file's first sample)
at video second 0; if the video started 3 s before the app, it is -3.

Two estimates, from what real files actually support (onset preferred):
  * rate: cross-correlate the video's stroke-rate curve with the machine's
    stroke rate. Needs some rate variation (starts, surges, fades). The
    machine smooths its rate, so this can carry a small reporting lag.
  * onset: first video stroke vs first distance increase. Only valid when
    both recordings include the start of the piece.
Per-stroke power updates cannot pin the offset to a single stroke: strokes
are too regular and the app logs on a ~0.5 s grid. Machine values are also
smoothed over several strokes, so +/- one stroke of sync error does not
change any per-window result; the report says which method was used.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .model import Telemetry

MIN_R = 0.45          # rate correlation needed to trust the rate estimate
MIN_PROMINENCE = 0.08  # ...and how far above any peak > 3 s away it must stand
AGREE_S = 2.0          # rate and onset agree within this -> high confidence
MIN_OVERLAP_S = 20.0


@dataclass
class Alignment:
    offset: float = float("nan")
    method: str = "none"             # manual | onset+rate | rate | onset | none
    confidence: str = "none"         # high | medium | low | manual | none
    rate_r: float = float("nan")
    rate_offset: float = float("nan")
    onset_offset: float = float("nan")
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.method != "none" and np.isfinite(self.offset)


def _moving_average(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) < n or n <= 1:
        return y
    k = np.ones(n) / n
    pad = np.concatenate([np.full(n // 2, y[0]), y, np.full(n - 1 - n // 2, y[-1])])
    return np.convolve(pad, k, "valid")


def rate_lag(tel: Telemetry, starts: np.ndarray, durations: np.ndarray):
    """(best offset, correlation, prominence) of video vs machine stroke rate."""
    mids = starts + durations / 2
    rate = _moving_average(60.0 / durations, 5)          # machine rate is smoothed too
    if len(rate) < 10 or np.std(rate) < 0.3:
        return np.nan, np.nan, np.nan
    grid = np.arange(mids[0], mids[-1], 0.5)
    video = np.interp(grid, mids, rate)
    a0, a1 = tel.active_span
    lags = np.arange(a0 - grid[-1], a1 - grid[0], 0.1)
    # most of the shorter recording must overlap, or edge effects win
    need = max(MIN_OVERLAP_S, 0.7 * min(grid[-1] - grid[0], a1 - a0)) / 0.5
    r = np.full(len(lags), np.nan)
    for k, lag in enumerate(lags):
        m = tel.at("spm", grid + lag)
        ok = np.isfinite(m)
        if ok.sum() < need or np.std(m[ok]) == 0:
            continue
        g = grid[ok]                       # remove slow trends (a steady fade matches any lag)
        v = video[ok] - np.polyval(np.polyfit(g, video[ok], 1), g)
        mm = m[ok] - np.polyval(np.polyfit(g, m[ok], 1), g)
        if np.std(v) > 0 and np.std(mm) > 0:
            r[k] = np.corrcoef(v, mm)[0, 1]
    if not np.isfinite(r).any():
        return np.nan, np.nan, np.nan
    best = int(np.nanargmax(r))
    far = np.abs(lags - lags[best]) > 3.0
    rival = np.nanmax(r[far]) if np.isfinite(r[far]).any() else -1.0
    return float(lags[best]), float(r[best]), float(r[best] - rival)


def onset_lag(tel: Telemetry, starts: np.ndarray, video_from_rest: bool) -> float:
    """Offset from the first stroke, if both recordings include the start."""
    on = np.flatnonzero(tel.active)
    if not video_from_rest or not on.size or np.nanmin(tel.distance[on[:5]]) > 10.0:
        return np.nan                  # one of the recordings starts mid-piece
    rise = np.flatnonzero(np.diff(tel.distance) > 0)
    if not rise.size:
        return np.nan
    first_move = tel.t[rise[0]]        # distance rose during the interval after this sample
    return float(first_move - starts[0])


def plausible(tel: Telemetry, starts: np.ndarray, durations: np.ndarray,
              offset: float) -> str:
    """Why an offset cannot be right ('' if it can): coverage and stroke count."""
    ends = starts + durations + offset
    covered = np.isfinite(tel.at("spm", ends)).mean()
    if covered < 0.8:
        return f"only {covered:.0%} of the video's strokes fall in active machine time"
    expected = tel.strokes_between(starts[0] + offset, ends[-1])
    if abs(len(starts) - expected) > max(3.0, 0.15 * expected):
        return (f"the machine implies {expected:.0f} strokes where the video has "
                f"{len(starts)}")
    return ""


def align(tel: Telemetry, starts: np.ndarray, durations: np.ndarray,
          video_from_rest: bool = False, manual: Optional[float] = None) -> Alignment:
    """`video_from_rest`: the athlete is still before the first detected stroke."""
    starts, durations = np.asarray(starts, float), np.asarray(durations, float)
    if manual is not None:
        return Alignment(float(manual), "manual", "manual",
                         notes=[f"Offset set by hand: video 0 s = machine {manual:+.2f} s."])
    if len(starts) < 3:
        return Alignment(notes=["Too few strokes in the video to synchronise."])

    lag, r, prom = rate_lag(tel, starts, durations)
    onset = onset_lag(tel, starts, video_from_rest)
    al = Alignment(rate_r=r, rate_offset=lag, onset_offset=onset)
    rate_ok = np.isfinite(r) and r >= MIN_R and prom >= MIN_PROMINENCE
    if np.isfinite(onset):
        # both recordings include the start: the first stroke is the sharpest anchor
        al.offset, al.method = onset, "onset"
        if rate_ok and abs(onset - lag) <= AGREE_S:
            al.method, al.confidence = "onset+rate", "high"
            al.notes.append(f"Aligned on the first stroke, confirmed by the stroke-rate "
                            f"match (r={r:.2f}, {abs(onset - lag):.1f} s apart).")
        else:
            al.confidence = "medium"
            al.notes.append("Aligned on the first stroke; the stroke-rate match "
                            + (f"points {lag - onset:+.1f} s away (steady or short pieces "
                               "match poorly)." if rate_ok else "was inconclusive."))
    elif rate_ok:
        al.offset, al.method, al.confidence = lag, "rate", "medium"
        al.notes.append(f"Aligned by matching stroke rate (r={r:.2f}); expect +/- 1-2 s. "
                        "Film the start of the piece for a sharper sync.")
    else:
        al.notes.append("Could not synchronise video and machine data; per-stroke machine "
                        "values are left out. Film the start of the piece, or pass "
                        "--telemetry-offset.")
        return al
    problem = plausible(tel, starts, durations, al.offset)
    if problem:
        return Alignment(rate_r=r, rate_offset=lag, onset_offset=onset, notes=[
            f"Could not synchronise video and machine data ({problem}); are they from the "
            "same piece? Per-stroke machine values are left out; pass --telemetry-offset "
            "to set the sync by hand."])
    al.notes.append(f"Video 0 s = machine {al.offset:+.2f} s.")
    return al
