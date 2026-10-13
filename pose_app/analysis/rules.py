"""Technique rules: a threshold on one per-rep metric, plus a coaching cue.

Thresholds are starting heuristics. Tune them with a coach by editing a JSON
overrides file (see `--print-thresholds` / `--thresholds`).
"""

import math
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence

SEVERITIES = ("minor", "major")


@dataclass(frozen=True)
class Rule:
    id: str
    metric: str                 # key in the per-rep metrics dict
    op: str                     # "<" or ">": the fault condition
    threshold: float
    severity: str               # "minor" | "major"
    title: str                  # short fault name
    cue: str                    # what to tell the athlete
    frame_key: str = "frame_mid"   # which frame best shows the fault

    def check(self, metrics: Dict[str, float]) -> Optional["Fault"]:
        value = metrics.get(self.metric)
        if value is None or not math.isfinite(value):
            return None
        hit = value < self.threshold if self.op == "<" else value > self.threshold
        if not hit:
            return None
        return Fault(self.id, self.title, self.cue, self.severity, self.metric, value,
                     f"{self.op} {self.threshold:g}", int(metrics.get(self.frame_key, -1)))


@dataclass(frozen=True)
class Fault:
    rule_id: str
    title: str
    cue: str
    severity: str
    metric: str
    value: float
    condition: str
    frame: int                  # source-video frame index for snapshots


@dataclass(frozen=True)
class DriftRule:
    """Compares the first and last third of the set (fatigue), by their medians so one
    odd rep (the first pull from standing still, a misread frame) cannot swing it."""
    id: str
    metric: str
    direction: str              # "increase" | "decrease" is the bad direction
    threshold: float            # absolute change, or relative if `relative`
    title: str
    cue: str
    relative: bool = False


@dataclass(frozen=True)
class Drift:
    rule_id: str
    title: str
    cue: str
    metric: str
    early: float
    late: float
    change: float
    triggered: bool


def evaluate_drift(rules: Sequence[DriftRule], reps: List[Dict[str, float]]) -> List[Drift]:
    if len(reps) < 6:
        return []
    third = len(reps) // 3
    out = []
    for r in rules:
        def typical(chunk):
            vals = sorted(x[r.metric] for x in chunk if math.isfinite(x.get(r.metric, math.nan)))
            if not vals:
                return math.nan
            mid = len(vals) // 2
            return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2
        early, late = typical(reps[:third]), typical(reps[-third:])
        if not (math.isfinite(early) and math.isfinite(late)):
            continue
        change = late - early
        if r.relative:
            change = change / abs(early) if early else math.nan
        bad = change if r.direction == "increase" else -change
        out.append(Drift(r.id, r.title, r.cue, r.metric, early, late, change,
                         bool(math.isfinite(bad) and bad > r.threshold)))
    return out


def thresholds_of(rules: Iterable, drift: Iterable) -> Dict[str, float]:
    return {r.id: r.threshold for r in (*rules, *drift)}


def apply_overrides(rules: Sequence, overrides: Dict[str, float]) -> list:
    return [replace(r, threshold=float(overrides[r.id])) if r.id in overrides else r
            for r in rules]
