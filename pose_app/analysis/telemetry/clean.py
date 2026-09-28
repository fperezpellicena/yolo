"""Make raw app logs trustworthy before anything is measured from them.

What real Track My Indoor Workout files contain (see the sample analysis):
  * recording gaps (app backgrounded or paused), where the header totals
    still count the time as moving;
  * held values: when the athlete stops, power and stroke rate keep
    repeating the last reading while distance stands still;
  * flatlined channels, e.g. heart rate stuck on one value after a strap
    dropout when the app is set to 'repeat last value'.

Only distance advancing proves the athlete is working the machine, so the
active mask is built from it, and every other channel is blanked outside it.
"""

from dataclasses import replace
from typing import List

import numpy as np

from .model import Telemetry

GAP_S = 2.5          # no samples for longer than this = recording gap
IDLE_S = 2.0         # no distance increase within this = not moving
FLAT_S = {"power": 30.0, "spm": 90.0, "hr": 90.0}   # identical value while active = stale


def _fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def clean(raw: Telemetry) -> Telemetry:
    t = raw.t
    notes: List[str] = list(raw.notes)
    keep = np.concatenate([[True], np.diff(t) > 0])          # drop repeated stamps
    if not keep.all():
        raw = _subset(raw, keep)
        t = raw.t

    # Recording gaps
    dt = np.diff(t)
    gap_after = np.flatnonzero(dt > GAP_S)
    for i in gap_after:
        notes.append(f"Recording gap of {dt[i]:.0f} s at {_fmt(t[i])} (machine time); "
                     "nothing is measured across it.")

    seg = np.concatenate([[0], np.cumsum(dt > GAP_S)])          # recording segment id

    # Active = a distance increase within IDLE_S, not across a gap
    dist = raw.distance
    inc = np.flatnonzero(np.diff(dist) > 0) + 1               # samples where distance rose
    active = np.zeros(len(t), bool)
    if inc.size:
        t_inc = t[inc]
        j = np.searchsorted(t_inc, t)
        prev_gap = np.where(j > 0, t - t_inc[np.maximum(j - 1, 0)], np.inf)
        next_gap = np.where(j < len(t_inc), t_inc[np.minimum(j, len(t_inc) - 1)] - t, np.inf)
        active = np.minimum(prev_gap, next_gap) <= IDLE_S
        # an increase only vouches for samples in the same recording segment
        near = np.where(prev_gap <= next_gap, np.maximum(j - 1, 0), np.minimum(j, len(inc) - 1))
        active &= seg == seg[inc[near]]
    elif not np.isfinite(dist).any():
        # no distance or speed logged: fall back to the machine reporting strokes/power
        moving = np.nan_to_num(raw.spm) > 0
        moving |= np.nan_to_num(raw.power) > 0
        active = moving & ~_stale(raw.spm, raw.power)
    else:
        notes.append("Distance never increases: no active rowing/skiing found.")

    on = np.flatnonzero(active)
    if on.size:
        tail = t[-1] - t[on[-1]]
        if tail > IDLE_S:
            notes.append(f"Last {tail:.0f} s after the athlete stopped were held values; ignored.")
        idle_mid = _runs(~active[on[0]:on[-1] + 1])
        for a, b in idle_mid:
            a, b = a + on[0], b + on[0]
            if t[b - 1] - t[a] > IDLE_S and not any(a - 1 <= g < b for g in gap_after):
                notes.append(f"Machine idle for {t[b - 1] - t[a]:.0f} s at {_fmt(t[a])}.")

    out = replace(raw, notes=notes, active=active)
    for name in ("power", "spm", "speed", "hr"):
        v = out.channel(name).astype(float).copy()
        v[~active] = np.nan
        limit = FLAT_S.get(name)
        if limit:
            for a, b in _runs_equal(v, seg):
                if t[b - 1] - t[a] >= limit:
                    v[a:b] = np.nan
                    notes.append(f"{name.upper()} stuck at {out.channel(name)[a]:g} for "
                                 f"{t[b - 1] - t[a]:.0f} s from {_fmt(t[a])} "
                                 "(sensor dropout?); ignored.")
        setattr(out, name, v)

    total = raw.meta.get("Total Time", "").split(",")[0]
    try:
        header_s = float(total)
        if gap_after.size and abs(header_s - t[-1]) < 3:
            notes.append(f"Header 'Total Time' ({header_s:.0f} s) includes the recording gaps.")
    except ValueError:
        pass
    return out


def _subset(tel: Telemetry, keep: np.ndarray) -> Telemetry:
    arrays = {n: getattr(tel, n)[keep] for n in ("t", "power", "spm", "distance", "speed", "hr")}
    return replace(tel, active=np.ones(int(keep.sum()), bool), **arrays)


def _runs(mask: np.ndarray):
    """[start, end) index pairs of True runs."""
    e = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def _runs_equal(v: np.ndarray, seg: np.ndarray):
    """[start, end) runs of identical finite values within one recording segment."""
    runs, a = [], 0
    for i in range(1, len(v) + 1):
        if i == len(v) or not (v[i] == v[a]) or seg[i] != seg[a]:
            if np.isfinite(v[a]) and i - a > 1:
                runs.append((a, i))
            a = i
    return runs


def _stale(spm: np.ndarray, power: np.ndarray, n: int = 8) -> np.ndarray:
    """Samples at the end of the log where stroke rate and power stop changing."""
    both = np.stack([np.nan_to_num(spm), np.nan_to_num(power)], 1)
    changed = np.flatnonzero(np.any(np.diff(both, axis=0) != 0, axis=1))
    out = np.zeros(len(spm), bool)
    if changed.size and len(spm) - changed[-1] > n:
        out[changed[-1] + 2:] = True
    return out
