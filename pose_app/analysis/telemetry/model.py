"""Machine telemetry in canonical units, independent of the file it came from."""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

CHANNELS = ("power", "spm", "distance", "speed", "hr")
HOLD_S = 2.5         # a logged value is not held longer than this (recording gaps)


@dataclass
class Telemetry:
    """One recorded piece from an erg, sampled as the app logged it.

    Times are seconds since the first sample (`t0_ms` is that sample's wall
    clock). Units: power W, spm strokes/min, distance m (cumulative), speed
    m/s, hr bpm. NaN means unknown. `active` is filled by `clean()`.
    """
    source: str
    t0_ms: float
    t: np.ndarray
    power: np.ndarray
    spm: np.ndarray
    distance: np.ndarray
    speed: np.ndarray
    hr: np.ndarray
    meta: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)      # data-quality actions, for the report
    active: np.ndarray = None                            # (N,) bool: athlete moving the machine

    def __post_init__(self):
        if self.active is None:
            self.active = np.ones(len(self.t), bool)

    def channel(self, name: str) -> np.ndarray:
        return getattr(self, name)

    def at(self, name: str, times: np.ndarray) -> np.ndarray:
        """Sample-and-hold value of a channel at `times` (the app logs held values).

        NaN outside the recording and where the machine was not active.
        """
        times = np.atleast_1d(np.asarray(times, float))
        idx = np.searchsorted(self.t, times, side="right") - 1
        idx = np.clip(idx, 0, len(self.t) - 1)
        ok = (times >= self.t[0]) & (times - self.t[idx] <= HOLD_S)   # not across gaps
        vals = self.channel(name)[idx].astype(float)
        vals[~ok | ~self.active[idx]] = np.nan
        return vals

    @property
    def active_span(self) -> tuple:
        on = np.flatnonzero(self.active)
        return (float(self.t[on[0]]), float(self.t[on[-1]])) if on.size else (np.nan, np.nan)

    def strokes_between(self, a: float, b: float) -> float:
        """Strokes implied by the stroke-rate channel over [a, b] (active time only)."""
        if not b > a:
            return 0.0
        grid = np.arange(a, b, 0.1)
        spm = self.at("spm", grid)
        return float(np.nansum(spm) * 0.1 / 60.0)

    def active_mean(self, name: str) -> float:
        """Time-weighted mean of a channel over active samples."""
        v = self.channel(name).astype(float)
        dt = np.diff(self.t, append=self.t[-1])
        ok = self.active & np.isfinite(v) & (dt > 0)
        return float(np.sum(v[ok] * dt[ok]) / np.sum(dt[ok])) if ok.any() else float("nan")
