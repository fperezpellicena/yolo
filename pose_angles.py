#!/usr/bin/env python3
"""Entry point: `python pose_angles.py [options]` (same as `python -m pose_app`)."""

from pose_app.live.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
