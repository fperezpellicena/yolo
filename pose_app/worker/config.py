"""Worker settings, read from environment variables."""

import logging
import os
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Optional


def _env(name: str, default: object = "") -> str:
    """WORKER_<name>, or `default` when unset or empty."""
    return os.environ.get(f"WORKER_{name}") or str(default)


@dataclass
class LogConfig:
    level: str = "INFO"
    file: Optional[str] = None              # None logs to stderr
    max_mb: float = 10.0                    # the file rolls over at this size; 0 never rolls over
    backups: int = 5                        # old files kept: worker.log.1 (newest) ... worker.log.5
    format: str = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"

    @classmethod
    def from_env(cls) -> "LogConfig":
        log_config = cls()
        level = _env("LOG_LEVEL", log_config.level).upper()
        if not isinstance(logging.getLevelName(level), int):
            raise SystemExit(f"WORKER_LOG_LEVEL: unknown level {level!r}")
        return cls(
            level=level, file=_env("LOG_FILE") or None,
            max_mb=float(_env("LOG_MAX_MB", log_config.max_mb)),
            backups=int(_env("LOG_BACKUPS", log_config.backups)),
            format=_env("LOG_FORMAT", log_config.format),
            datefmt=_env("LOG_DATEFMT", log_config.datefmt))


def configure_logging(cfg: LogConfig) -> None:
    """Root logging: to cfg.file, rotated by size, if set; otherwise stderr."""
    handler: logging.Handler = logging.StreamHandler()
    if cfg.file:
        handler = RotatingFileHandler(cfg.file, encoding="utf-8",
                                      maxBytes=int(cfg.max_mb * 1024 * 1024),
                                      backupCount=cfg.backups)
    logging.basicConfig(level=cfg.level, handlers=[handler],
                        format=cfg.format, datefmt=cfg.datefmt)


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
        missing = [n for n in ("API_URL", "API_TOKEN", "MEDIA_ROOT") if not _env(n)]
        if missing:
            raise SystemExit("missing environment variables: " +
                             ", ".join(f"WORKER_{n}" for n in missing))
        worker_config = cls(_env("API_URL"), _env("API_TOKEN"), _env("MEDIA_ROOT"))
        return cls(
            api_url=worker_config.api_url, api_token=worker_config.api_token,
            media_root=worker_config.media_root,
            model=_env("MODEL", worker_config.model),
            imgsz=int(_env("IMGSZ", worker_config.imgsz)),
            device=_env("DEVICE") or None,
            poll_s=float(_env("POLL_SECONDS", worker_config.poll_s)),
            heartbeat_s=float(_env("HEARTBEAT_SECONDS", worker_config.heartbeat_s)),
            api_timeout_s=float(_env("API_TIMEOUT_SECONDS", worker_config.api_timeout_s)),
            api_retry_s=float(_env("API_RETRY_SECONDS", worker_config.api_retry_s)))
