"""Programmatic entry point: one call runs a whole analysis into a folder.

Used by the CLI and by anything that drives analyses as jobs (e.g. a queue
worker). It never prints: status lines go to the `pose_app` logger, frame
progress to an optional callback, and problems with the inputs raise
AnalysisError with a code from ERROR_CODES, which callers turn into their own
(e.g. translated) message, and an English message with the details.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .force import ForceSetup, force_drift_rules, force_rules
from .force.frames import write_force_frames
from .pipeline import AnalysisResult, EmptyClipError, Extraction, analyze, extract
from .report import write_csv, write_html, write_json
from .stations import STATIONS
from .telemetry import Telemetry, machine_drift_rules
from .tracking import AthleteTracker
from .video import VideoInfo, VideoReader

log = logging.getLogger(__name__)

# What run_analysis writes into its output folder ("video", "report" and "reps"
# unless turned off in the options).
OUTPUT_FILES = {"pose_cache": "pose_cache.npz", 
                "reps": "reps.csv", 
                "summary": "summary.json",
                "forces": "forces.json",
                "report": "report.html", 
                "video": "annotated.mp4"}

# (stage, frames done, frames in the clip or 0 if unknown); stage is "pose" or "video"
ProgressFn = Callable[[str, int, int], None]

# What the pose pass depends on, kept in pose_cache.npz: the frames it saw, the athlete it
# followed, and the model settings that found the keypoints
CLIP_KEYS = ("start", "end", "rotate", "athlete", "athlete_point")
MODEL_KEYS = ("model", "imgsz", "conf", "kpt_conf")


# Why the inputs cannot be analysed. The codes are a contract with callers such
# as the web app, which maps each one to a message of its own: keep them stable.
ERROR_CODES = {
    "unreadable_video": "the video cannot be opened or has no readable frames",
    "unreadable_telemetry": "the telemetry file or the PM5 log cannot be read",
    "empty_clip": "no frames between the start and end times",
    "unsupported_station": "the station is not analysed yet",
    "invalid_job": "the request itself is wrong (options, folder or file names): "
                   "the caller's fault, not the athlete's",
}


class AnalysisError(Exception):
    """The inputs cannot be analysed (unreadable video, bad telemetry, ...).

    `code` is one of ERROR_CODES; the message gives the details, in English.
    """

    def __init__(self, code: str, message: str):
        if code not in ERROR_CODES:
            raise ValueError(f"unknown analysis error code '{code}'")
        super().__init__(message)
        self.code = code


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
    pm5: Optional[str] = None                       # PM5 Bluetooth log (pm5_log.py) path
    force: Optional[ForceSetup] = None              # athlete + calibration: force analysis
    thresholds: Dict[str, float] = field(default_factory=dict)   # rule id -> value
    model: str = "yolo11m-pose.pt"
    imgsz: int = 960
    conf: float = 0.4
    kpt_conf: float = 0.5
    device: Optional[str] = None
    video: bool = True                              # write annotated.mp4
    report: bool = True                             # write report.html
    csv: bool = True                                # write reps.csv
    video_width: int = 1280
    reuse: bool = False                             # reuse out_dir's pose_cache.npz if present
    # A previous run's pose_cache.npz (e.g. a job run again with other options), reused
    # only when it was made from the same video with the same clip, athlete and model settings
    pose_cache: Optional[str] = None
    title: Optional[str] = None                     # report heading


@dataclass
class AnalysisOutcome:
    result: AnalysisResult
    video: VideoInfo
    files: Dict[str, str]           # "report", "summary", "reps", "video", "pose_cache", "forces" -> path
    warnings: List[str]             # option problems first, then the analysis' own


def default_thresholds(station: str) -> Dict[str, float]:
    """Every tunable rule id for the station with its default value."""
    st = _station(station)
    force = [*force_rules(), *force_drift_rules()] if st.force else []
    return {r.id: r.threshold for r in (*st.rules(), *st.drift_rules(), *machine_drift_rules(),
                                        *force)}


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
    if opts.force is not None:
        problems = opts.force.problems()
        if problems:
            raise AnalysisError("invalid_job", "Invalid force setup: " + "; ".join(problems) + ".")
        if not station.force:
            raise AnalysisError("invalid_job", f"Force analysis is not available for {station.name}.")
        if not opts.pm5:
            raise AnalysisError("invalid_job", "Force analysis needs the PM5 log of the piece.")
    pm5 = _load_pm5(opts.pm5) if opts.pm5 else None
    try:
        reader = VideoReader(video_path, opts.start, opts.end, opts.rotate)
    except RuntimeError as exc:
        raise AnalysisError("unreadable_video", str(exc)) from exc
    except ValueError as exc:                       # a rotation it does not support
        raise AnalysisError("invalid_job", str(exc)) from exc
    info = reader.info
    os.makedirs(out_dir, exist_ok=True)
    files = {name: os.path.join(out_dir, OUTPUT_FILES[name])
             for name in ("pose_cache", "summary")}
    log.info(f"{info.path}: {info.width}x{info.height}, {info.fps:.1f} fps, "
             f"{info.duration:.1f} s  ->  {out_dir}")

    ex = _extraction(reader, opts, files["pose_cache"], estimator, on_progress,
                     pose_params(video_path, opts))
    res = analyze(ex, station, opts.kpt_conf, opts.thresholds, telemetry, opts.telemetry_offset,
                  pm5, opts.force)
    if res.machine is not None:
        al = res.machine.alignment
        log.info(f"Sync: {al.method} ({al.confidence})" +
                 (f", video 0 s = machine {al.offset:+.2f} s." if al.ok else "."))
    if res.force is not None:
        fs = res.force.summary
        log.info(f"Force: {fs.get('strokes_with_curve', 0):.0f} strokes with force curves "
                 f"(stroke match {res.force.sync.confidence}).")
    log.info(f"{len(res.reps)} {station.rep_word}s, "
             f"{sum(1 for r in res.reps if not r.faults)} clean.")

    write_json(res, files["summary"])
    # The force model frame by frame, for a page to draw over the video
    if res.force is not None:
        forces = os.path.join(out_dir, OUTPUT_FILES["forces"])
        if write_force_frames(res.force, forces, ex.fps, ex.frame_size):
            files["forces"] = forces
    if opts.csv:
        files["reps"] = os.path.join(out_dir, OUTPUT_FILES["reps"])
        write_csv(res, files["reps"])
    if opts.report:
        files["report"] = os.path.join(out_dir, OUTPUT_FILES["report"])
        title = opts.title or f"{os.path.basename(video_path)}  ·  {info.duration:.0f} s"
        write_html(res, files["report"], reader, title=title)
    if opts.video:
        from .render import render_video
        files["video"] = os.path.join(out_dir, OUTPUT_FILES["video"])
        render_video(reader, res, files["video"], opts.video_width,
                     _stage(on_progress, "video"))
    return AnalysisOutcome(res, info, files, warnings + res.warnings)


def _station(key: str):
    try:
        return STATIONS[key]
    except KeyError:
        raise AnalysisError("unsupported_station",
                            f"Unknown station '{key}'; expected one of {sorted(STATIONS)}.") from None


def _load_telemetry(path: str) -> Telemetry:
    from .telemetry import clean, load_telemetry
    try:
        telemetry = clean(load_telemetry(path))
    except (OSError, ValueError) as exc:
        raise AnalysisError("unreadable_telemetry", str(exc)) from exc
    a, b = telemetry.active_span
    log.info(f"Machine data: {telemetry.source}, active {b - a:.0f} s.")
    return telemetry


def _load_pm5(path: str):
    from ..pm5 import load_pm5_session
    try:
        session = load_pm5_session(path)
    except (OSError, ValueError) as exc:
        raise AnalysisError("unreadable_telemetry", str(exc)) from exc
    log.info(f"PM5 log: {session.describe()}.")
    return session


def pose_params(video_path: str, opts: AnalysisOptions) -> Dict[str, Any]:
    """What a pose pass of `video_path` with `opts` depends on, as JSON-ready values."""
    params: Dict[str, Any] = {"video": os.path.basename(video_path),
                              "video_bytes": os.path.getsize(video_path)}
    for key in CLIP_KEYS + MODEL_KEYS:
        value = getattr(opts, key)
        params[key] = [float(v) for v in value] if isinstance(value, (tuple, list)) else value
    return params


def _extraction(reader: VideoReader, opts: AnalysisOptions, cache: str, estimator,
                on_progress: Optional[ProgressFn], params: Dict[str, Any]) -> Extraction:
    # --reuse asks for the output folder's own cache whatever model made it, so only its video
    # and clip must match; a previous run's cache must match in everything
    clip = ("video", "video_bytes", *CLIP_KEYS)
    if opts.reuse:
        if not os.path.exists(cache):
            log.info("No pose cache to reuse yet.")
        else:
            try:
                ex = Extraction.load(cache)
            except ValueError as exc:               # a cache from another version
                raise AnalysisError("invalid_job", str(exc)) from exc
            changed = _changed(ex.params, params, clip) if ex.params else []   # {}: an older cache
            if not changed:
                log.info("Reusing cached pose pass.")
                return ex
            log.info("Not reusing the pose cache: made with another " + ", ".join(changed) + ".")
    if opts.pose_cache and os.path.exists(opts.pose_cache):
        try:
            ex = Extraction.load(opts.pose_cache)
        except ValueError as exc:
            log.info(f"Not reusing the previous pose pass: {exc}")
        else:
            changed = _changed(ex.params, params, tuple(params))
            if not changed:
                log.info("Reusing the previous run's pose pass.")
                if os.path.abspath(opts.pose_cache) != os.path.abspath(cache):
                    ex.save(cache)
                return ex
            log.info("Not reusing the previous pose pass: made with another " +
                     ", ".join(changed) + ".")
    if estimator is None:
        from ..estimator import PoseEstimator
        estimator = PoseEstimator(opts.model, opts.conf, opts.imgsz, opts.device, opts.kpt_conf)
    info = reader.info
    tracker = AthleteTracker(opts.athlete, opts.athlete_point, (info.width, info.height))
    try:
        ex = extract(reader, estimator.detect, tracker, opts.model, _stage(on_progress, "pose"))
    except EmptyClipError as exc:
        raise AnalysisError("empty_clip", str(exc)) from exc
    ex.params = params
    ex.save(cache)
    return ex


def _changed(cached: Dict[str, Any], wanted: Dict[str, Any], keys) -> List[str]:
    """Which of `keys` differ between a cached pose pass and this run ([] = reusable)."""
    return [k for k in keys if k not in cached or cached[k] != wanted.get(k)]


def _stage(on_progress: Optional[ProgressFn], stage: str) -> Optional[Callable[[int, int], None]]:
    if on_progress is None:
        return None
    return lambda done, total: on_progress(stage, done, total)
