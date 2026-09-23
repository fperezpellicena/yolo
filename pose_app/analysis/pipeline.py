"""Pass 1 (pose extraction, cacheable) and the analysis that runs on it."""

import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from ..person import Person
from .body import BodySeries, build_body_series
from .rules import Drift, Fault, apply_overrides, evaluate_drift
from .signal import Cycle, find_cycles, hysteresis_thresholds
from .stations import Station
from .tracking import AthleteTracker
from .video import VideoReader

CACHE_VERSION = 1


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
            tracker: AthleteTracker, model_name: str = "", progress: bool = True) -> Extraction:
    ts, idxs, kps, scs, boxes = [], [], [], [], []
    total = reader.info.frame_count
    started = time.perf_counter()
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
        if progress and n % 25 == 0:
            rate = n / (time.perf_counter() - started)
            print(f"\r  pose: frame {index + 1}/{total or '?'}  ({rate:.1f} fps)",
                  end="", file=sys.stderr, flush=True)
    if progress:
        print(file=sys.stderr)
    if not ts:
        raise RuntimeError("No frames were read (check --start/--end).")
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

    def rep_at_frame(self) -> np.ndarray:
        """(N,) rep list position for each analysed frame, -1 outside reps."""
        out = np.full(len(self.body.t), -1)
        for k, rep in enumerate(self.reps):
            out[rep.cycle.start:rep.cycle.end] = k
        return out


def analyze(ex: Extraction, station: Station, min_score: float = 0.5,
            overrides: Optional[Dict[str, float]] = None) -> AnalysisResult:
    overrides = overrides or {}
    rules = apply_overrides(station.rules(), overrides)
    drift_rules = apply_overrides(station.drift_rules(), overrides)

    body = build_body_series(ex.t, ex.frame_index, ex.keypoints, ex.scores, min_score, ex.fps)
    driver = body.metrics[station.driver]
    lo, hi = hysteresis_thresholds(driver, station.lo_frac, station.hi_frac)
    cycles = find_cycles(driver, body.t, lo, hi, station.min_rep_s, station.max_rep_s) \
        if np.isfinite(lo) else []

    reps = []
    for k, cycle in enumerate(cycles, 1):
        metrics = station.summarize(body, cycle)
        faults = [f for f in (r.check(metrics) for r in rules) if f is not None]
        reps.append(RepResult(k, cycle, metrics, faults))
    drift = evaluate_drift(drift_rules, [r.metrics for r in reps])

    thresholds = {r.id: r.threshold for r in (*rules, *drift_rules)}
    result = AnalysisResult(station, body, reps, drift, thresholds)
    result.warnings = _warnings(result)
    return result


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
    return w
