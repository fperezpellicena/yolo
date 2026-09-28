"""Join machine telemetry to the video's strokes.

Machine values are smoothed by the monitor over several strokes, so they are
attached per stroke for charts and drift, but conclusions are drawn over
windows (splits, thirds, groups of strokes), never from a single stroke.
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..rules import DriftRule
from .model import HOLD_S, Telemetry
from .sync import Alignment

MACHINE_KEYS = ("m_power", "m_spm", "m_pace", "m_speed", "m_dps", "m_work_j", "m_hr",
                "m_distance")
MIN_GROUP = 8          # strokes per side before comparing faulted vs clean output
MIN_EFFECT = 0.03      # relative difference worth reporting
SETTLE_S = 5.0         # monitors show start-up values (smoothing from zero) at first


def machine_drift_rules() -> List[DriftRule]:
    """Fatigue checks on machine output; skipped automatically without telemetry."""
    return [
        DriftRule("drift_power", "m_power", "decrease", 0.08, "Power fades",
                  "Output dropped late in the piece: pace the start more evenly, or "
                  "check the technique drift above for where the power went.", relative=True),
        DriftRule("drift_dps", "m_dps", "decrease", 0.08, "Strokes get shorter",
                  "Each stroke covers less distance when tired: protect stroke length "
                  "(full reach, full finish) before chasing rate.", relative=True),
    ]


def attach(metrics: Sequence[Dict[str, float]], tel: Telemetry, al: Alignment) -> None:
    """Add m_* keys to every rep (NaN when unsynchronised, machine idle or not logged).

    Each channel stands on its own: a file without power still gives heart rate
    and distance. Heart rate and distance are valid from the first stroke;
    power, rate and speed wait for the monitor's start-up smoothing to settle.
    """
    first_active = tel.active_span[0]
    for m in metrics:
        vals = dict.fromkeys(MACHINE_KEYS, math.nan)
        m.update(vals)
        if not al.ok:
            continue
        end = m["t_start"] + m["duration_s"] + al.offset             # machine time at stroke end
        vals["m_hr"] = float(tel.at("hr", end)[0])
        vals["m_distance"] = float(tel.at("distance", end)[0])
        if end - first_active >= SETTLE_S:
            power, spm, speed = (float(tel.at(c, end)[0]) for c in ("power", "spm", "speed"))
            vals.update(m_power=power, m_spm=spm, m_speed=speed,
                        m_pace=500.0 / speed if speed > 0 else math.nan,
                        m_dps=speed * m["duration_s"],
                        m_work_j=power * m["duration_s"])
        m.update(vals)


@dataclass
class Split:
    start_m: float
    end_m: float
    time_s: float
    power: float
    spm: float
    pace: float
    hr: float
    strokes: int                  # video strokes that ended in this split
    fault_pct: float
    top_fault: str


@dataclass
class Association:
    rule_id: str
    title: str
    n_fault: int
    n_clean: int
    power_diff: float             # relative: faulted vs clean strokes
    dps_diff: float
    p_value: float


@dataclass
class MachineAnalysis:
    telemetry: Telemetry
    alignment: Alignment
    summary: Dict[str, float]
    splits: List[Split]
    associations: List[Association]
    checks: List[str] = field(default_factory=list)       # cross-check warnings

    @property
    def notes(self) -> List[str]:
        return self.telemetry.notes


def piece_summary(tel: Telemetry) -> Dict[str, float]:
    a, b = tel.active_span
    on = tel.active
    dist = tel.distance[on]
    speed = tel.active_mean("speed")
    return {
        "distance_m": float(np.nanmax(dist) - np.nanmin(dist)) if dist.size else math.nan,
        "active_s": float(np.sum(np.diff(tel.t, append=tel.t[-1])[on])),
        "power_w": tel.active_mean("power"),
        "spm": tel.active_mean("spm"),
        "pace_500": 500.0 / speed if speed > 0 else math.nan,
        "hr_avg": tel.active_mean("hr"),
        "hr_max": float(np.nanmax(tel.hr)) if np.isfinite(tel.hr).any() else math.nan,
        "strokes_est": tel.strokes_between(a, b),
    }


def splits(tel: Telemetry, reps, al: Alignment, split_m: float = 250.0) -> List[Split]:
    """Machine splits by distance, each with the technique of the strokes inside it."""
    on = np.flatnonzero(tel.active)
    if not on.size:
        return []
    d0, d1 = np.nanmin(tel.distance[on]), np.nanmax(tel.distance[on])
    edges = list(np.arange(d0, d1, split_m)) + [d1]
    if len(edges) >= 3 and edges[-1] - edges[-2] < 0.2 * split_m:
        edges.pop(-2)                                 # fold a tiny remainder into the last split
    ends = np.array([r.metrics["t_start"] + r.metrics["duration_s"] for r in reps]) + \
        (al.offset if al.ok else np.nan)
    out = []
    for lo, hi in zip(edges, edges[1:]):
        mask = tel.active & (tel.distance >= lo) & (tel.distance <= hi)
        idx = np.flatnonzero(mask)
        if idx.size < 2:
            continue
        ta, tb = tel.t[idx[0]], tel.t[idx[-1]]
        steps = np.diff(tel.t[idx])
        moving_s = float(steps[steps <= HOLD_S].sum())      # recording gaps don't count
        sub = [r for r, e in zip(reps, ends) if ta <= e <= tb]
        faults = Counter(f.title for r in sub for f in r.faults)

        def mean(name):
            v = tel.channel(name)[idx]
            return float(np.nanmean(v)) if np.isfinite(v).any() else math.nan
        speed = mean("speed")
        out.append(Split(float(lo - d0), float(hi - d0), moving_s, mean("power"),
                         mean("spm"), 500.0 / speed if speed > 0 else math.nan, mean("hr"),
                         len(sub),
                         100.0 * sum(1 for r in sub if r.faults) / len(sub) if sub else math.nan,
                         faults.most_common(1)[0][0] if faults else ""))
    return out


def cross_checks(reps, tel: Telemetry, al: Alignment) -> List[str]:
    """Where the machine says the video's stroke detection may be wrong."""
    out: List[str] = []
    if not al.ok or len(reps) < 6:
        return out
    first = reps[0].metrics["t_start"] + al.offset
    last = reps[-1].metrics["t_start"] + reps[-1].metrics["duration_s"] + al.offset
    expected = tel.strokes_between(first, last)
    if expected > 0 and abs(len(reps) - expected) > max(3.0, 0.05 * expected):
        more = "missed" if len(reps) < expected else "split or doubled"
        out.append(f"The video found {len(reps)} strokes where the machine's stroke rate "
                   f"implies about {expected:.0f} over the same time: strokes may have been "
                   f"{more} (occlusion, athlete out of frame, camera angle).")
    covered = np.isfinite([r.metrics["m_power"] for r in reps]).mean()
    if covered < 0.9:
        out.append(f"Machine data covers only {covered:.0%} of the video's strokes.")
    return out


def associations(reps, rules, seed: int = 0) -> List[Association]:
    """Output of strokes with a fault vs strokes without it.

    A permutation test guards against noise; neighbouring strokes are not
    independent (smoothed machine values, faults cluster when tired), so this
    is reported as an association, not a cause.
    """
    rng = np.random.default_rng(seed)
    power = np.array([r.metrics.get("m_power", math.nan) for r in reps], float)
    dps = np.array([r.metrics.get("m_dps", math.nan) for r in reps], float)
    ok = np.isfinite(power) & np.isfinite(dps)
    out = []
    for rule in rules:
        hit = np.array([any(f.rule_id == rule.id for f in r.faults) for r in reps])
        a, b = hit & ok, ~hit & ok
        if a.sum() < MIN_GROUP or b.sum() < MIN_GROUP:
            continue
        pd_ = power[a].mean() / power[b].mean() - 1
        dd = dps[a].mean() / dps[b].mean() - 1
        if max(abs(pd_), abs(dd)) < MIN_EFFECT:
            continue
        vals, n_a = power[ok], int(a.sum())
        observed = abs(pd_)
        perm = [abs(p[:n_a].mean() / p[n_a:].mean() - 1)
                for p in (rng.permutation(vals) for _ in range(2000))]
        p_value = (1 + sum(x >= observed for x in perm)) / 2001
        out.append(Association(rule.id, rule.title, n_a, int(b.sum()), float(pd_),
                               float(dd), float(p_value)))
    return sorted(out, key=lambda x: x.power_diff)


def analyze_machine(tel: Telemetry, al: Alignment, reps, rules,
                    split_m: float = 250.0) -> MachineAnalysis:
    return MachineAnalysis(tel, al, piece_summary(tel), splits(tel, reps, al, split_m),
                           associations(reps, rules) if al.ok else [],
                           cross_checks(reps, tel, al))
