"""Programmatic entry point: one call runs a whole analysis into a folder.

Used by the CLI and by anything that drives analyses as jobs (e.g. a queue
worker). It never prints: status lines go to the `pose_app` logger, frame
progress to an optional callback, and problems with the inputs raise
AnalysisError with a message that is safe to show to the athlete or coach.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .pipeline import AnalysisResult, EmptyClipError, Extraction, analyze, extract
from .report import write_csv, write_html, write_json
from .stations import STATIONS
from .telemetry import Telemetry, machine_drift_rules
from .tracking import AthleteTracker
from .video import VideoInfo, VideoReader

log = logging.getLogger(__name__)

# (stage, frames done, frames total or 0 if unknown); stage is "pose" or "video"
ProgressFn = Callable[[str, int, int], None]


class AnalysisError(Exception):
    """The inputs cannot be analysed (unreadable video, bad telemetry, ...)."""


@dataclass
class AnalysisOptions:
    station: str = "skierg"
    start: float = 0.0                              # clip, in seconds
    end: Optional[float] = None
    rotate: int = 0                                 # 0, 90, 180 or 270
    athlete: str = "largest"                        # "largest" or "center"
    athlete_point: Optional[Tuple[float, float]] = None
    telemetry: Optional[str] = None                 # machine data CSV path
    telemetry_offset: Optional[float] = None        # machine s at video 0 s
    thresholds: Dict[str, float] = field(default_factory=dict)   # rule id -> value
    model: str = "yolo11m-pose.pt"
    imgsz: int = 960
    conf: float = 0.4
    kpt_conf: float = 0.5
    device: Optional[str] = None
    video: bool = True                              # write annotated.mp4
    video_width: int = 1280
    reuse: bool = False                             # reuse pose_cache.npz if present
    title: Optional[str] = None                     # report heading


@dataclass
class AnalysisOutcome:
    result: AnalysisResult
    video: VideoInfo
    files: Dict[str, str]           # "report", "summary", "reps", "video", "pose_cache" -> path
    warnings: List[str]             # option problems first, then the analysis' own


def default_thresholds(station: str) -> Dict[str, float]:
    """Every tunable rule id for the station with its default value."""
    st = _station(station)
    return {r.id: r.threshold for r in (*st.rules(), *st.drift_rules(), *machine_drift_rules())}


def run_analysis(video_path: str, out_dir: str, opts: Optional[AnalysisOptions] = None,
                 estimator=None, on_progress: Optional[ProgressFn] = None) -> AnalysisOutcome:
    """Analyse `video_path` and write report, CSV, JSON, video and pose cache to `out_dir`.

    `estimator` is a loaded PoseEstimator (anything with `.detect(frame)`) so a
    long-lived caller loads the model once; when omitted one is built from
    `opts`. Its settings must then match `opts.model` and `opts.kpt_conf`.
    """
    opts = opts or AnalysisOptions()
    station = _station(opts.station)
    unknown = set(opts.thresholds) - set(default_thresholds(opts.station))
    warnings = [f"Unknown threshold ids ignored: {sorted(unknown)}"] if unknown else []

    telemetry = _load_telemetry(opts.telemetry) if opts.telemetry else None
    try:
        reader = VideoReader(video_path, opts.start, opts.end, opts.rotate)
    except (RuntimeError, ValueError) as exc:
        raise AnalysisError(str(exc)) from exc
    info = reader.info
    os.makedirs(out_dir, exist_ok=True)
    files = {"pose_cache": os.path.join(out_dir, "pose_cache.npz"),
             "reps": os.path.join(out_dir, "reps.csv"),
             "summary": os.path.join(out_dir, "summary.json"),
             "report": os.path.join(out_dir, "report.html")}
    log.info(f"{info.path}: {info.width}x{info.height}, {info.fps:.1f} fps, "
             f"{info.duration:.1f} s  ->  {out_dir}")

    ex = _extraction(reader, opts, files["pose_cache"], estimator, on_progress)
    res = analyze(ex, station, opts.kpt_conf, opts.thresholds, telemetry, opts.telemetry_offset)
    if res.machine is not None:
        al = res.machine.alignment
        log.info(f"Sync: {al.method} ({al.confidence})" +
                 (f", video 0 s = machine {al.offset:+.2f} s." if al.ok else "."))
    log.info(f"{len(res.reps)} {station.rep_word}s, "
             f"{sum(1 for r in res.reps if not r.faults)} clean.")

    write_csv(res, files["reps"])
    write_json(res, files["summary"])
    title = opts.title or f"{os.path.basename(video_path)}  ·  {info.duration:.0f} s"
    write_html(res, files["report"], reader, title=title)
    if opts.video:
        from .render import render_video
        files["video"] = os.path.join(out_dir, "annotated.mp4")
        render_video(reader, res, files["video"], opts.video_width,
                     _stage(on_progress, "video"))
    return AnalysisOutcome(res, info, files, warnings + res.warnings)


def _station(key: str):
    try:
        return STATIONS[key]
    except KeyError:
        raise AnalysisError(f"Unknown station '{key}'; "
                            f"expected one of {sorted(STATIONS)}.") from None


def _load_telemetry(path: str) -> Telemetry:
    from .telemetry import clean, load_telemetry
    try:
        telemetry = clean(load_telemetry(path))
    except (OSError, ValueError) as exc:
        raise AnalysisError(str(exc)) from exc
    a, b = telemetry.active_span
    log.info(f"Machine data: {telemetry.source}, active {b - a:.0f} s.")
    return telemetry


def _extraction(reader: VideoReader, opts: AnalysisOptions, cache: str, estimator,
                on_progress: Optional[ProgressFn]) -> Extraction:
    if opts.reuse and os.path.exists(cache):
        log.info("Reusing cached pose pass.")
        try:
            return Extraction.load(cache)
        except ValueError as exc:
            raise AnalysisError(str(exc)) from exc
    if estimator is None:
        from ..estimator import PoseEstimator
        estimator = PoseEstimator(opts.model, opts.conf, opts.imgsz, opts.device, opts.kpt_conf)
    info = reader.info
    tracker = AthleteTracker(opts.athlete, opts.athlete_point, (info.width, info.height))
    try:
        ex = extract(reader, estimator.detect, tracker, opts.model, _stage(on_progress, "pose"))
    except EmptyClipError as exc:
        raise AnalysisError(str(exc)) from exc
    ex.save(cache)
    return ex


def _stage(on_progress: Optional[ProgressFn], stage: str) -> Optional[Callable[[int, int], None]]:
    if on_progress is None:
        return None
    return lambda done, total: on_progress(stage, done, total)
