"""Pass 1 (pose extraction, cacheable) and the analysis that runs on it."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from ..person import Person
from .body import BodySeries, build_body_series
from .rules import Drift, Fault, apply_overrides, evaluate_drift
from .signal import Cycle, find_cycles, hysteresis_thresholds
from .stations import Station
from .telemetry import (MachineAnalysis, Telemetry, align, analyze_machine, attach,
                        machine_drift_rules)
from .tracking import AthleteTracker
from .video import VideoReader

CACHE_VERSION = 1


class EmptyClipError(RuntimeError):
    """No frames were read in the requested clip."""


@dataclass
class Extraction:
    """The tracked athlete's raw keypoints for every analysed frame."""
    video_path: str
    fps: float
    frame_size: tuple                 # (width, height)
    model: str
    t: np.ndarray                     # (N,)
    frame_index: np.ndarray           # (N,)
    keypoints: np.ndarray             # (N, 17, 2), NaN when no athlete
    scores: np.ndarray                # (N, 17), 0 when no athlete
    boxes: np.ndarray                 # (N, 4), NaN when no athlete

    def save(self, path: str) -> None:
        np.savez_compressed(
            path, version=CACHE_VERSION, video_path=self.video_path, fps=self.fps,
            frame_size=np.array(self.frame_size), model=self.model, t=self.t,
            frame_index=self.frame_index, keypoints=self.keypoints,
            scores=self.scores, boxes=self.boxes)

    @classmethod
    def load(cls, path: str) -> "Extraction":
        d = np.load(path, allow_pickle=False)
        if int(d["version"]) != CACHE_VERSION:
            raise ValueError("pose cache was written by another version; re-run without --reuse")
        return cls(str(d["video_path"]), float(d["fps"]), tuple(int(v) for v in d["frame_size"]),
                   str(d["model"]), d["t"], d["frame_index"], d["keypoints"],
                   d["scores"], d["boxes"])


def extract(reader: VideoReader, detect: Callable[[np.ndarray], List[Person]],
            tracker: AthleteTracker, model_name: str = "",
            on_progress: Optional[Callable[[int, int], None]] = None) -> Extraction:
    """`on_progress(frame number, frame count)` is called after every frame."""
    ts, idxs, kps, scs, boxes = [], [], [], [], []
    total = reader.info.frame_count
    for index, t, frame in reader.frames():
        athlete = tracker.select(detect(frame))
        ts.append(t)
        idxs.append(index)
        if athlete is None:
            kps.append(np.full((17, 2), np.nan))
            scs.append(np.zeros(17))
            boxes.append(np.full(4, np.nan))
        else:
            kps.append(athlete.keypoints.astype(float))
            scs.append(athlete.scores.astype(float))
            boxes.append(athlete.box.astype(float))
        if on_progress:
            on_progress(index + 1, total)
    if not ts:
        raise EmptyClipError("No frames were read (check the start and end times).")
    info = reader.info
    return Extraction(info.path, info.fps, (info.width, info.height), model_name,
                      np.array(ts), np.array(idxs), np.array(kps), np.array(scs),
                      np.array(boxes))


@dataclass
class RepResult:
    number: int                       # 1-based
    cycle: Cycle
    metrics: Dict[str, float]
    faults: List[Fault]

    @property
    def worst(self) -> Optional[str]:
        if any(f.severity == "major" for f in self.faults):
            return "major"
        return "minor" if self.faults else None


@dataclass
class AnalysisResult:
    station: Station
    body: BodySeries
    reps: List[RepResult]
    drift: List[Drift]
    thresholds: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    machine: Optional[MachineAnalysis] = None       # set when telemetry was given

    def rep_at_frame(self) -> np.ndarray:
        """(N,) rep list position for each analysed frame, -1 outside reps."""
        out = np.full(len(self.body.t), -1)
        for k, rep in enumerate(self.reps):
            out[rep.cycle.start:rep.cycle.end] = k
        return out


def analyze(ex: Extraction, station: Station, min_score: float = 0.5,
            overrides: Optional[Dict[str, float]] = None,
            telemetry: Optional[Telemetry] = None,
            telemetry_offset: Optional[float] = None) -> AnalysisResult:
    """`telemetry` must already be cleaned; `telemetry_offset` overrides the sync."""
    overrides = overrides or {}
    rules = apply_overrides(station.rules(), overrides)
    drift_rules = apply_overrides([*station.drift_rules(), *machine_drift_rules()], overrides)

    body = build_body_series(ex.t, ex.frame_index, ex.keypoints, ex.scores, min_score, ex.fps)
    driver = body.metrics[station.driver]
    lo, hi = hysteresis_thresholds(driver, station.lo_frac, station.hi_frac)
    cycles = find_cycles(driver, body.t, lo, hi, station.min_rep_s, station.max_rep_s) \
        if np.isfinite(lo) else []

    metrics = [station.summarize(body, cycle) for cycle in cycles]
    alignment = None
    if telemetry is not None:
        starts = np.array([m["t_start"] for m in metrics])
        durations = np.array([m["duration_s"] for m in metrics])
        from_rest = _starts_from_rest(driver, body.t, cycles)
        alignment = align(telemetry, starts, durations, from_rest, telemetry_offset)
        attach(metrics, telemetry, alignment)

    reps = []
    for k, (cycle, m) in enumerate(zip(cycles, metrics), 1):
        faults = [f for f in (r.check(m) for r in rules) if f is not None]
        reps.append(RepResult(k, cycle, m, faults))
    drift = evaluate_drift(drift_rules, [r.metrics for r in reps])

    thresholds = {r.id: r.threshold for r in (*rules, *drift_rules)}
    result = AnalysisResult(station, body, reps, drift, thresholds)
    if telemetry is not None:
        result.machine = analyze_machine(telemetry, alignment, reps, rules)
    result.warnings = _warnings(result)
    return result


def _starts_from_rest(driver: np.ndarray, t: np.ndarray, cycles: List[Cycle]) -> bool:
    """True if the clip opens with the athlete still, then the strokes begin.

    Then the first video stroke is the first stroke of the piece and can anchor
    the sync; a clip that starts mid-piece shows movement straight away.
    """
    if len(cycles) < 3 or t[cycles[0].start] - t[0] > 15.0:
        return False
    stroke_span = np.nanmedian([np.nanmax(driver[c.start:c.end + 1]) -
                                np.nanmin(driver[c.start:c.end + 1]) for c in cycles[:10]])
    opening = driver[t <= t[0] + 0.8]
    opening = opening[np.isfinite(opening)]
    return bool(opening.size >= 5 and np.ptp(opening) < 0.25 * stroke_span)


def _warnings(res: AnalysisResult) -> List[str]:
    w, body, st = [], res.body, res.station
    if body.coverage < 0.8:
        w.append(f"The athlete was measured in only {body.coverage:.0%} of frames; "
                 "check framing, lighting and occlusion.")
    if st.view == "side" and np.isfinite(body.view_ratio) and body.view_ratio > 0.5:
        w.append(f"The camera looks front-on (shoulder spread {body.view_ratio:.2f} "
                 f"torso lengths); {st.name} rules assume a side view, so angles may be wrong.")
    if not res.reps:
        w.append(f"No {st.rep_word}s were detected. Is this a {st.name} clip, filmed "
                 "with the full movement in frame?")
    elif len(res.reps) < 6:
        w.append("Fewer than 6 reps: fatigue trends need a longer clip.")
    if res.machine is not None:
        al = res.machine.alignment
        if not al.ok or al.confidence == "low":
            w.extend(al.notes[:1])
        w.extend(res.machine.checks)
    return w
