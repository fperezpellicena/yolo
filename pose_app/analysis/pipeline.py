"""Pass 1 (pose extraction, cacheable) and the analysis that runs on it."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..person import Person
from ..pm5 import Pm5Session
from .body import BodySeries, build_body_series
from .force import (ForceAnalysis, ForceSetup, StrokeSync, analyze_force, attach_force,
                    force_drift_rules, force_rules, sync_by_hands)
from .rules import Drift, Fault, Rule, apply_overrides, evaluate_drift
from .signal import Cycle, find_cycles, hysteresis_thresholds
from .stations import Station
from .telemetry import (Alignment, MachineAnalysis, Telemetry, align, analyze_machine, attach,
                        clean, machine_drift_rules, pm5_telemetry)
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
    """`on_progress(frames done, frames in clip or 0)` is called after every frame."""
    ts, idxs, kps, scs, boxes = [], [], [], [], []
    total = reader.clip_frame_count
    for n, (index, t, frame) in enumerate(reader.frames(), 1):
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
            on_progress(n, total)
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
    fps: float = 0.0                # source video; also the annotated video's frame rate
    force: Optional[ForceAnalysis] = None           # set when a PM5 log and a force setup were given
    rules: List[Rule] = field(default_factory=list) # the rules applied (after overrides)
    pm5: Optional[Pm5Session] = None                # the PM5 log, when one was given
    pm5_sync: Optional[StrokeSync] = None           # its strokes matched to the video's, one by one

    def rule_list(self) -> List[Rule]:
        """Every per-rep rule that was checked, with its applied threshold."""
        return self.rules or self.station.rules()

    def rep_at_frame(self) -> np.ndarray:
        """(N,) rep list position for each analysed frame, -1 outside reps."""
        out = np.full(len(self.body.t), -1)
        for k, rep in enumerate(self.reps):
            out[rep.cycle.start:rep.cycle.end] = k
        return out


def analyze(ex: Extraction, station: Station, min_score: float = 0.5,
            overrides: Optional[Dict[str, float]] = None,
            telemetry: Optional[Telemetry] = None,
            telemetry_offset: Optional[float] = None,
            pm5=None, force_setup: Optional[ForceSetup] = None) -> AnalysisResult:
    """`telemetry` must already be cleaned; `telemetry_offset` overrides the sync.

    `pm5` is a decoded PM5 log (pose_app.pm5.Pm5Session). Without other telemetry
    it is the machine data, synchronised stroke by stroke; with `force_setup` (on
    a station that supports it) it also drives the force analysis.
    """
    overrides = overrides or {}
    body = build_body_series(ex.t, ex.frame_index, ex.keypoints, ex.scores, min_score, ex.fps)
    driver = body.metrics[station.driver]
    lo, hi = hysteresis_thresholds(driver, station.lo_frac, station.hi_frac)
    cycles = find_cycles(driver, body.t, lo, hi, station.min_rep_s, station.max_rep_s) \
        if np.isfinite(lo) else []

    metrics = [station.summarize(body, cycle) for cycle in cycles]
    from_pm5 = pm5 is not None and telemetry is None
    if from_pm5:
        telemetry = clean(pm5_telemetry(pm5))
    alignment = None
    if telemetry is not None:
        starts = np.array([m["t_start"] for m in metrics])
        durations = np.array([m["duration_s"] for m in metrics])
        from_rest = _starts_from_rest(driver, body.t, cycles)
        alignment = align(telemetry, starts, durations, from_rest, telemetry_offset)

    force, sync = None, None
    if pm5 is not None:
        prior, window = _sync_prior(alignment, telemetry_offset, from_pm5)
        if force_setup is not None and station.force:
            force = analyze_force(ex.t, ex.keypoints, ex.scores, body, cycles, ex.fps, pm5,
                                  force_setup, min_score, prior, window)
            sync = force.sync
        else:
            sync = sync_by_hands(body, cycles, pm5, prior, window)
        if from_pm5 and telemetry_offset is None and sync.ok:
            alignment = _stroke_alignment(sync, alignment)
    if telemetry is not None:
        attach(metrics, telemetry, alignment)
    if force is not None:
        attach_force(metrics, force)

    extra = force is not None
    rules = apply_overrides([*station.rules(), *(force_rules() if extra else [])], overrides)
    drift_rules = apply_overrides([*station.drift_rules(), *machine_drift_rules(),
                                   *(force_drift_rules() if extra else [])], overrides)
    reps = []
    for k, (cycle, m) in enumerate(zip(cycles, metrics), 1):
        faults = [f for f in (r.check(m) for r in rules) if f is not None]
        reps.append(RepResult(k, cycle, m, faults))
    drift = evaluate_drift(drift_rules, [r.metrics for r in reps])

    thresholds = {r.id: r.threshold for r in (*rules, *drift_rules)}
    result = AnalysisResult(station, body, reps, drift, thresholds, fps=ex.fps, force=force,
                            rules=rules, pm5=pm5, pm5_sync=sync)
    if telemetry is not None:
        result.machine = analyze_machine(telemetry, alignment, reps, rules)
    result.warnings = _warnings(result)
    if pm5 is not None and force is None and station.force and pm5.curves:
        result.warnings.append("The PM5 log has force curves, but no athlete profile and "
                               "calibration were given, so the force analysis was skipped.")
    return result


def _sync_prior(alignment: Optional[Alignment], manual: Optional[float],
                same_clock: bool) -> Tuple[Optional[float], float]:
    """Where to look for the stroke-by-stroke offset: around a hand-set offset, around the
    coarse sync of the same log, or everywhere. Both are on the PM5 log's clock only when
    the log is the machine data (`same_clock`); another file's offset says nothing about it."""
    if not same_clock:
        return None, 0.0
    if manual is not None:
        return float(manual), 1.0
    if alignment is not None and alignment.ok:
        return float(alignment.offset), 3.0
    return None, 0.0


def _stroke_alignment(sync: StrokeSync, coarse: Optional[Alignment]) -> Alignment:
    al = Alignment(sync.offset, "strokes", sync.confidence)
    if coarse is not None:
        al.rate_r, al.rate_offset, al.onset_offset = coarse.rate_r, coarse.rate_offset, \
            coarse.onset_offset
    al.notes = [f"Aligned stroke by stroke on the PM5's own stroke times "
                f"({len(sync.pairs)} strokes matched).", f"Video 0 s = machine {sync.offset:+.2f} s."]
    return al


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
    if res.force is not None:
        w.extend(res.force.checks)
    return w
