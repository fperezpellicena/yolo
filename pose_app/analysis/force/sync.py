"""Which PM5 stroke is which video stroke, and each force curve onto the video's frames.

Matching: every video stroke has a drive window (from the cord payout when
the setup is calibrated, else from the hands' height), every PM5 stroke the
end of its drive and its drive time, both to 0.01 s. One offset puts the two
clocks together (PM5 log seconds = video seconds + offset); it is found by
scoring how many drive ends line up within a few hundredths of a second,
then the best candidates are told apart by their drive durations, which vary
from stroke to stroke. Strokes are regular, so a whole-stroke shift scores
almost as well: the runner-up's score sets the confidence, and force curves
are used only when the match is clear.

Mapping: the PM5 curve covers its own drive, which the video sees as the cord
paying out from its shortest to its longest. Curves spaced by handle travel
are placed by the fraction of the payout reached at each frame (the video's
payout stretched to the PM5's drive length). Curves spaced in time keep the
PM5's own drive time, centred in the payout window: the hands move slowly as
they turn at the catch and the finish, so the payout window is well defined in
metres but tens of milliseconds wider than the force in time, and stretching
the curve over it would spread the force (and its work) too wide.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..signal import Cycle

SIGMA_S = 0.06          # timing scatter expected between matching events
SIGMA_DRIVE_S = 0.06
TOL_S = 0.20            # a pair further apart than this is not a match
HIGH_RMS_S = 0.06       # matched drive ends must agree this well (RMS) for high confidence
MEDIUM_RMS_S = 0.09
STEP_S = 0.02           # offset grid; the best candidates are refined afterwards
MIN_PAYOUT_M = 0.15


@dataclass
class DriveWindow:
    start: int              # frame where the drive starts (payout begins / hands highest)
    end: int                # frame where it ends (payout complete / hands lowest)
    t_start: float          # the same, to a fraction of a frame
    t_end: float


@dataclass
class StrokeSync:
    offset: float = math.nan            # PM5 log seconds at video second 0
    method: str = "none"                # "strokes", or "none" when no match was found
    confidence: str = "none"            # high | medium | low | none
    pairs: Dict[int, int] = field(default_factory=dict)    # window index -> PM5 stroke index
    residual_s: float = math.nan        # RMS difference of the matched drive ends
    margin: float = math.nan            # best score / runner-up score
    events: str = ""                    # "cord" or "hands": what timed the video's drives
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.confidence in ("high", "medium") and math.isfinite(self.offset)


def _refine(y: np.ndarray, t: np.ndarray, i: int) -> float:
    """Time of the extremum at sample i, to a fraction of a frame (parabola through 3 points)."""
    if 0 < i < len(y) - 1 and np.isfinite(y[i - 1:i + 2]).all():
        den = y[i - 1] - 2 * y[i] + y[i + 1]
        if abs(den) > 1e-12:
            d = float(np.clip(0.5 * (y[i - 1] - y[i + 1]) / den, -0.5, 0.5))
            return float(t[i] + d * (t[i + 1] - t[i - 1]) / 2)
    return float(t[i])


def payout_windows(t: np.ndarray, L: np.ndarray, cycles: Sequence[Cycle],
                   fps: float) -> List[Optional[DriveWindow]]:
    """Drive windows from the cord length: shortest near the catch -> longest near the finish."""
    out: List[Optional[DriveWindow]] = []
    before, after = int(round(0.3 * fps)), int(round(0.4 * fps))
    for c in cycles:
        a = max(0, c.start - before)
        seg = L[a:c.mid + 1]
        if np.isfinite(seg).sum() < 3:
            out.append(None)
            continue
        s = a + int(np.nanargmin(seg))
        hi = min(c.end, c.mid + after, len(L) - 1)
        seg = L[s:hi + 1]
        if np.isfinite(seg).sum() < 3:
            out.append(None)
            continue
        e = s + int(np.nanargmax(seg))
        if e - s < 3 or not L[e] - L[s] >= MIN_PAYOUT_M:
            out.append(None)
            continue
        out.append(DriveWindow(s, e, _refine(-L, t, s), _refine(L, t, e)))
    return out


def hand_windows(t: np.ndarray, wrist_h: np.ndarray,
                 cycles: Sequence[Cycle]) -> List[Optional[DriveWindow]]:
    """Drive windows from the hands' height alone (no calibration): highest -> lowest."""
    return [DriveWindow(c.start, c.mid, _refine(wrist_h, t, c.start), _refine(-wrist_h, t, c.mid))
            for c in cycles]


def match_strokes(windows: Sequence[Optional[DriveWindow]], p_start: np.ndarray,
                  p_end: np.ndarray, prior: Optional[float] = None,
                  prior_window: float = 3.0) -> StrokeSync:
    """`p_start` / `p_end`: PM5 drive start and end times (log clock), in time order.

    `prior`: an offset already known roughly (searched within +/- prior_window).
    """
    idx = np.array([k for k, w in enumerate(windows) if w is not None], int)
    if len(idx) < 3 or len(p_end) < 3:
        return StrokeSync(notes=["Too few strokes to match the video with the PM5."])
    v_start = np.array([windows[k].t_start for k in idx])
    v_end = np.array([windows[k].t_end for k in idx])
    p_start, p_end = np.asarray(p_start, float), np.asarray(p_end, float)
    order = np.argsort(p_end)
    p_start, p_end = p_start[order], p_end[order]
    if prior is not None and math.isfinite(prior):
        lo, hi = prior - prior_window, prior + prior_window
    else:
        lo, hi = p_end[0] - v_end[-1] - 1.0, p_end[-1] - v_end[0] + 1.0
    offsets = np.arange(lo, hi + STEP_S, STEP_S)
    scores = _end_scores(offsets, v_end, p_end)
    cands = _peaks(offsets, scores)
    results = [_assign(o, v_start, v_end, p_start, p_end) for o in cands]
    results = [r for r in results if r is not None]
    if not results:
        return StrokeSync(notes=["No offset lines up the video's strokes with the PM5's."])
    results.sort(key=lambda r: -r["score"])
    best = results[0]
    rivals = [r["score"] for r in results[1:] if abs(r["offset"] - best["offset"]) > 0.3]
    margin = best["score"] / max(max(rivals, default=0.0), 1e-9)
    shifted = v_end + best["offset"]
    overlap = int(np.sum((shifted >= p_end[0] - TOL_S) & (shifted <= p_end[-1] + TOL_S)))
    matched = len(best["pairs"])
    share = matched / max(overlap, 1)
    rms = best["rms"]
    if matched >= 5 and share >= 0.8 and margin >= 1.5 and rms <= HIGH_RMS_S:
        confidence = "high"
    elif matched >= 4 and share >= 0.6 and margin >= 1.2 and rms <= MEDIUM_RMS_S:
        confidence = "medium"
    else:
        confidence = "low"
    pairs = {int(idx[i]): int(order[j]) for i, j in best["pairs"].items()}
    sync = StrokeSync(best["offset"], "strokes", confidence, pairs, best["rms"], margin)
    sync.notes.append(f"Matched {matched} of {len(idx)} video strokes to PM5 strokes one by one "
                      f"(drive ends {1000 * best['rms']:.0f} ms apart on average, runner-up "
                      f"{margin:.1f}x less likely).")
    if not sync.ok:
        sync.notes.append("The match is not clear enough to trust stroke by stroke (strokes "
                          "too regular or too few, or the video and the log are from "
                          "different pieces).")
    return sync


def _nearest(p: np.ndarray, q: np.ndarray):
    """Index of the nearest value of sorted `p` to each of `q`, and the signed gap p - q."""
    i = np.clip(np.searchsorted(p, q), 1, len(p) - 1)
    left = q - p[i - 1] < p[i] - q
    j = np.where(left, i - 1, i)
    return j, p[j] - q


def _end_scores(offsets: np.ndarray, v_end: np.ndarray, p_end: np.ndarray) -> np.ndarray:
    out = np.empty(len(offsets))
    step = max(1, 2_000_000 // len(v_end))
    for a in range(0, len(offsets), step):
        q = v_end[None, :] + offsets[a:a + step, None]
        _, d = _nearest(p_end, q)
        out[a:a + step] = np.exp(-0.5 * (d / SIGMA_S) ** 2).sum(axis=1)
    return out


def _peaks(offsets: np.ndarray, scores: np.ndarray, n: int = 6, apart: float = 0.3) -> List[float]:
    order = np.argsort(-scores)
    picked: List[float] = []
    for k in order:
        if scores[k] <= 0:
            break
        if all(abs(offsets[k] - o) > apart for o in picked):
            picked.append(float(offsets[k]))
            if len(picked) == n:
                break
    return picked


def _assign(offset: float, v_start, v_end, p_start, p_end) -> Optional[dict]:
    """Pairs at this offset (refined), scored on drive ends and on drive durations."""
    for _ in range(3):
        j, d = _nearest(p_end, v_end + offset)
        ok = np.abs(d) < TOL_S
        if ok.sum() < 3:
            return None
        offset += float(np.median(d[ok]))
    j, d = _nearest(p_end, v_end + offset)
    ok = np.abs(d) < TOL_S
    dur = (v_end - v_start) - (p_end[j] - p_start[j])
    bias = float(np.median(dur[ok]))           # the two see the drive start a little differently
    w = np.exp(-0.5 * (d / SIGMA_S) ** 2) * np.exp(-0.5 * ((dur - bias) / SIGMA_DRIVE_S) ** 2)
    pairs: Dict[int, int] = {}
    for i in np.flatnonzero(ok & (np.abs(dur - bias) < TOL_S)):
        k = int(j[i])
        prev = next((a for a, b in pairs.items() if b == k), None)
        if prev is None or abs(d[i]) < abs(d[prev]):
            if prev is not None:
                del pairs[prev]
            pairs[int(i)] = k
    if len(pairs) < 3:
        return None
    sel = np.array(sorted(pairs))
    return {"offset": offset, "score": float(w[sel].sum()), "pairs": pairs,
            "rms": float(np.sqrt(np.mean(d[sel] ** 2)))}


def curve_profile(curve: np.ndarray) -> Optional[np.ndarray]:
    """A PM5 curve from the first force to the release, with zero at both ends."""
    c = np.asarray(curve, float)
    on = np.flatnonzero(c > 0)
    if len(on) < 3:
        return None
    return np.concatenate([[0.0], c[on[0]:on[-1] + 1], [0.0]])


def curve_start(w: DriveWindow, drive_time: float) -> Tuple[float, float]:
    """(start, duration) of a time-spaced curve on the video clock: centred in the window."""
    span = drive_time if drive_time > 0 else w.t_end - w.t_start
    return 0.5 * (w.t_start + w.t_end - span), span


def travel_fraction(x: float, basis: str, t: np.ndarray, L: np.ndarray, w: DriveWindow,
                    drive_time: float = math.nan) -> float:
    """Share of the drive's cord travel reached at position `x` (0-1) along a PM5 curve."""
    if basis == "travel":
        return float(x)
    start, span = curve_start(w, drive_time)
    s, e = w.start, w.end
    when = start + x * span
    length = np.interp(when, t[s:e + 1], L[s:e + 1])
    return float(np.clip((length - L[s]) / (L[e] - L[s]), 0.0, 1.0)) if L[e] > L[s] else math.nan


def map_curve(profile: np.ndarray, basis: str, t: np.ndarray, L: np.ndarray,
              w: DriveWindow, drive_time: float = math.nan) -> np.ndarray:
    """Tension at frames w.start..w.end from one PM5 curve (`drive_time`: the PM5's, s)."""
    s, e = w.start, w.end
    x = np.linspace(0.0, 1.0, len(profile))
    if basis == "travel":
        span = L[e] - L[s]
        u = (L[s:e + 1] - L[s]) / span if span > 0 else np.zeros(e - s + 1)
        u = np.maximum.accumulate(np.clip(np.nan_to_num(u), 0.0, 1.0))
        return np.interp(u, x, profile)
    start, span = curve_start(w, drive_time)
    if not span > 0:
        return np.zeros(e - s + 1)
    return np.interp((t[s:e + 1] - start) / span, x, profile, left=0.0, right=0.0)
