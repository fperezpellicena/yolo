"""Worker settings, read from environment variables."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class WorkerConfig:
    database_url: str                       # mysql://user:password@host:3306/database
    media_root: str                         # folder that analysis_jobs.job_dir is relative to
    model: str = "yolo11m-pose.pt"
    imgsz: int = 960
    device: Optional[str] = None            # e.g. cpu, 0, mps; None lets Ultralytics pick
    poll_s: float = 2.0                      # wait between polls when the queue is empty
    heartbeat_s: float = 10.0
    stale_s: float = 120.0                   # a RUNNING job silent this long is requeued
    max_attempts: int = 3                   # per job, for internal errors and lost workers

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        def env(name: str, default=None):
            return os.environ.get(f"WORKER_{name}", default)
        missing = [n for n in ("DATABASE_URL", "MEDIA_ROOT") if not env(n)]
        if missing:
            raise SystemExit("missing environment variables: " +
                             ", ".join(f"WORKER_{n}" for n in missing))
        d = cls(env("DATABASE_URL"), env("MEDIA_ROOT"))
        return cls(
            database_url=d.database_url, media_root=d.media_root,
            model=env("MODEL", d.model), imgsz=int(env("IMGSZ", d.imgsz)),
            device=env("DEVICE", d.device), poll_s=float(env("POLL_SECONDS", d.poll_s)),
            heartbeat_s=float(env("HEARTBEAT_SECONDS", d.heartbeat_s)),
            stale_s=float(env("STALE_SECONDS", d.stale_s)),
            max_attempts=int(env("MAX_ATTEMPTS", d.max_attempts)))
