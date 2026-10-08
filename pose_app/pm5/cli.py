"""`python pm5_log.py LOG` : record a PM5 over Bluetooth; `--check LOG` : what a log holds."""

import argparse
import asyncio
import logging
import sys
from typing import Optional, Sequence

import numpy as np

from .protocol import LBF_TO_N
from .session import load_pm5_session


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record a Concept2 PM5 (stroke data and force curves) over Bluetooth, "
                    "for the force analysis of hyrox_analyze.py.")
    p.add_argument("log", help="raw log to write (JSON lines), e.g. ski_2026-10-08.pm5.jsonl")
    p.add_argument("--check", action="store_true",
                   help="do not record: summarise an existing log (strokes, force curves, issues)")
    p.add_argument("--address", default=None, help="connect to this Bluetooth address")
    p.add_argument("--minutes", type=float, default=None, help="stop after this many minutes")
    return p.parse_args(argv)


def check(path: str) -> int:
    try:
        session = load_pm5_session(path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(session.describe())
    strokes = session.strokes
    if strokes:
        span = strokes[-1].t_end - strokes[0].t_start
        peak = np.nanmedian([s.peak_force_n for s in strokes])
        length = np.nanmedian([s.drive_length_m for s in strokes])
        print(f"  {span / 60:.1f} min of strokes; median peak force {peak:.0f} N, "
              f"drive length {length:.2f} m")
        if session.curves:
            print(f"  force curves spaced by {session.curve_basis}")
    for note in session.notes:
        print(f"  - {note}")
    return 0 if session.curves else 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.check:
        return check(args.log)
    try:
        from .recorder import record
    except ImportError:
        print("error: recording needs bleak (pip install bleak)", file=sys.stderr)
        return 1
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    def show(stroke):
        print(f"stroke {stroke.count:4d}  peak {stroke.peak_force_lbf * LBF_TO_N:5.0f} N  "
              f"drive {stroke.drive_length_m:.2f} m / {stroke.drive_time_s:.2f} s", flush=True)
    try:
        n = asyncio.run(record(args.log, args.address, args.minutes, on_stroke=show))
    except KeyboardInterrupt:
        print("stopped")
        n = None
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if n is not None:
        print(f"{n} notifications written to {args.log}")
    return check(args.log) if n else 0
