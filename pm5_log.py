#!/usr/bin/env python3
"""Entry point: `python pm5_log.py LOG` records a PM5 (same as `python -m pose_app.pm5`)."""

from pose_app.pm5.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
