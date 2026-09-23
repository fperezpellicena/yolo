"""`python hyrox_analyze.py VIDEO --station skierg` : offline technique analysis."""

import argparse
import json
import os
import sys
from typing import Optional, Sequence, Tuple

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


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.list_stations:
        for key, st in sorted(STATIONS.items()):
            print(f"{key:10s} {st.name}  ({st.view} view)")
        return 0
    station = STATIONS[args.station]
    if args.print_thresholds:
        print(json.dumps({r.id: r.threshold for r in (*station.rules(), *station.drift_rules())},
                         indent=2))
        return 0
    if not args.video:
        print("error: a video file is required", file=sys.stderr)
        return 2

    overrides = {}
    if args.thresholds:
        with open(args.thresholds) as fh:
            overrides = json.load(fh)
        known = {r.id for r in (*station.rules(), *station.drift_rules())}
        unknown = set(overrides) - known
        if unknown:
            print(f"warning: unknown threshold ids ignored: {sorted(unknown)}", file=sys.stderr)

    from .pipeline import Extraction, analyze, extract
    from .report import write_csv, write_html, write_json
    from .tracking import AthleteTracker
    from .video import VideoReader

    try:
        reader = VideoReader(args.video, args.start, args.end, args.rotate)
    except (RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    stem = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.video)),
                                       f"{stem}_analysis")
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, "pose_cache.npz")

    info = reader.info
    print(f"{info.path}: {info.width}x{info.height}, {info.fps:.1f} fps, "
          f"{info.duration:.1f} s  ->  {out_dir}")

    if args.reuse and os.path.exists(cache):
        print("Reusing cached pose pass.")
        ex = Extraction.load(cache)
    else:
        from ..estimator import PoseEstimator
        est = PoseEstimator(args.model, args.conf, args.imgsz, args.infer_device, args.kpt_conf)
        tracker = AthleteTracker(args.athlete, args.athlete_point, (info.width, info.height))
        ex = extract(reader, est.detect, tracker, args.model)
        ex.save(cache)

    res = analyze(ex, station, args.kpt_conf, overrides)
    for w in res.warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(f"{len(res.reps)} {station.rep_word}s, "
          f"{sum(1 for r in res.reps if not r.faults)} clean.")

    write_csv(res, os.path.join(out_dir, "reps.csv"))
    write_json(res, os.path.join(out_dir, "summary.json"))
    write_html(res, os.path.join(out_dir, "report.html"), reader,
               title=f"{os.path.basename(args.video)}  ·  {info.duration:.0f} s")
    if not args.no_video:
        from .render import render_video
        render_video(reader, res, os.path.join(out_dir, "annotated.mp4"), args.video_width)
    print(f"Done: {os.path.join(out_dir, 'report.html')}")
    return 0
