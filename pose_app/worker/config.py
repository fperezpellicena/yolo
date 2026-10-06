"""Worker settings, read from environment variables."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class WorkerConfig:
    api_url: str                            # the web app, e.g. http://web:8081 (its internal API port)
    api_token: str                          # shared secret, the web app's WORKER_API_TOKEN
    media_root: str                         # folder the job's file keys are relative to
    model: str = "yolo11m-pose.pt"
    imgsz: int = 960
    device: Optional[str] = None            # e.g. cpu, 0, mps; None lets Ultralytics pick
    poll_s: float = 2.0                      # wait between claims when the queue is empty
    heartbeat_s: float = 10.0                # progress reports; keep well under the web app's hyrox.jobs.stale-after
    api_timeout_s: float = 30.0              # per HTTP call
    api_retry_s: float = 60.0                # how long to retry a call while the web app is unreachable

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        def env(name: str, default=None):
            return os.environ.get(f"WORKER_{name}", default)
        missing = [n for n in ("API_URL", "API_TOKEN", "MEDIA_ROOT") if not env(n)]
        if missing:
            raise SystemExit("missing environment variables: " +
                             ", ".join(f"WORKER_{n}" for n in missing))
        d = cls(env("API_URL"), env("API_TOKEN"), env("MEDIA_ROOT"))
        return cls(
            api_url=d.api_url, api_token=d.api_token, media_root=d.media_root,
            model=env("MODEL", d.model), imgsz=int(env("IMGSZ", d.imgsz)),
            device=env("DEVICE", d.device), poll_s=float(env("POLL_SECONDS", d.poll_s)),
            heartbeat_s=float(env("HEARTBEAT_SECONDS", d.heartbeat_s)),
            api_timeout_s=float(env("API_TIMEOUT_SECONDS", d.api_timeout_s)),
            api_retry_s=float(env("API_RETRY_SECONDS", d.api_retry_s)))
