"""`python hyrox_analyze.py VIDEO --station skierg` : offline technique analysis."""

import argparse
import json
import logging
import os
import sys
import time
from typing import Optional, Sequence, Tuple

from .api import AnalysisError, AnalysisOptions, default_thresholds, run_analysis
from .stations import STATIONS


def _point(text: str) -> Tuple[float, float]:
    try:
        x, y = (float(v) for v in text.split(","))
        return x, y
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected X,Y in pixels, got '{text}'")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hyrox technique analysis of a recorded video.")
    p.add_argument("video", nargs="?", help="input video file")
    p.add_argument("--station", choices=sorted(STATIONS), default="skierg")
    p.add_argument("--out", default=None, help="output folder (default: <video>_analysis)")
    p.add_argument("--list-stations", action="store_true")
    p.add_argument("--print-thresholds", action="store_true",
                   help="print the station's default thresholds as JSON and exit")
    p.add_argument("--thresholds", default=None,
                   help="JSON file of {rule_id: value} overriding default thresholds")

    clip = p.add_argument_group("clip")
    clip.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    clip.add_argument("--end", type=float, default=None, help="end time in seconds")
    clip.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0,
                      help="extra rotation if the video comes out sideways")

    ath = p.add_argument_group("athlete")
    ath.add_argument("--athlete", choices=("largest", "center"), default="largest",
                     help="who to follow when several people are visible")
    ath.add_argument("--athlete-point", type=_point, default=None,
                     help="follow the person at X,Y (pixels) in the first frame")

    tel = p.add_argument_group("machine data")
    tel.add_argument("--telemetry", default=None, metavar="CSV",
                     help="machine data recorded during the clip (Track My Indoor Workout CSV)")
    tel.add_argument("--telemetry-offset", type=float, default=None, metavar="S",
                     help="machine seconds at video second 0, if automatic sync fails "
                          "(e.g. -3 when the video started 3 s before the app)")

    model = p.add_argument_group("model")
    model.add_argument("--model", default="yolo11m-pose.pt",
                       help="pose weights; offline favours accuracy (default: yolo11m-pose.pt)")
    model.add_argument("--imgsz", type=int, default=960, help="inference size (default: 960)")
    model.add_argument("--conf", type=float, default=0.4)
    model.add_argument("--kpt-conf", type=float, default=0.5)
    model.add_argument("--infer-device", default=None, help="e.g. cpu, 0, mps")

    out = p.add_argument_group("output")
    out.add_argument("--no-video", action="store_true", help="skip the annotated video")
    out.add_argument("--video-width", type=int, default=1280)
    out.add_argument("--reuse", action="store_true",
                     help="reuse the cached pose pass (fast re-analysis after tuning thresholds)")
    return p.parse_args(argv)


class _Console(logging.StreamHandler):
    """Status lines on stdout, plus a `\r` progress line on stderr that is
    ended before anything else is printed."""

    def __init__(self):
        super().__init__(sys.stdout)
        self.setFormatter(logging.Formatter("%(message)s"))
        self.started = {}
        self.open = False

    def progress(self, stage: str, done: int, total: int) -> None:
        t0 = self.started.setdefault(stage, time.perf_counter())
        if done % 25:
            return
        line = f"\r  {stage}: frame {done}/{total or '?'}"
        if stage == "pose":
            line += f"  ({done / max(time.perf_counter() - t0, 1e-6):.1f} fps)"
        print(line, end="", file=sys.stderr, flush=True)
        self.open = True

    def end_line(self) -> None:
        if self.open:
            print(file=sys.stderr)
            self.open = False

    def emit(self, record: logging.LogRecord) -> None:
        self.end_line()
        super().emit(record)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.list_stations:
        for key, st in sorted(STATIONS.items()):
            print(f"{key:10s} {st.name}  ({st.view} view)")
        return 0
    if args.print_thresholds:
        print(json.dumps(default_thresholds(args.station), indent=2))
        return 0
    if not args.video:
        print("error: a video file is required", file=sys.stderr)
        return 2

    thresholds = {}
    if args.thresholds:
        with open(args.thresholds) as fh:
            thresholds = json.load(fh)
    stem = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.video)),
                                       f"{stem}_analysis")
    opts = AnalysisOptions(
        station=args.station, start=args.start, end=args.end, rotate=args.rotate,
        athlete=args.athlete, athlete_point=args.athlete_point,
        telemetry=args.telemetry, telemetry_offset=args.telemetry_offset,
        thresholds=thresholds, model=args.model, imgsz=args.imgsz, conf=args.conf,
        kpt_conf=args.kpt_conf, device=args.infer_device, video=not args.no_video,
        video_width=args.video_width, reuse=args.reuse)

    console = _Console()
    logger = logging.getLogger("pose_app")
    logger.addHandler(console)
    logger.setLevel(logging.INFO)
    try:
        outcome = run_analysis(args.video, out_dir, opts, on_progress=console.progress)
    except AnalysisError as exc:
        console.end_line()
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        logger.removeHandler(console)
    console.end_line()
    for w in outcome.warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(f"Done: {outcome.files['report']}")
    return 0
