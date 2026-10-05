"""The loop: claim a job, analyse the files in its folder, write the outputs there."""

import glob
import logging
import os
import shutil
import socket
import tempfile
import threading
from typing import Any, Dict, Optional

from ..analysis.api import (OUTPUT_FILES, AnalysisError, AnalysisOptions, AnalysisOutcome,
                            run_analysis)
from ..analysis.tracking import SELECT_MODES
from .config import WorkerConfig
from .jobs import JOB_OPTIONS, Job, JobStore

log = logging.getLogger(__name__)

SCRATCH_PREFIX = ".analysis-tmp-"       # work folder inside job_dir while a job runs


class JobLost(Exception):
    """The job was cancelled or handed to another worker while we ran it."""


class Heartbeat:
    """Writes the job's heartbeat and progress from a background thread.

    It has its own database connection, so the analysis thread never waits on
    the database; the analysis thread learns that the job was lost through
    `progress()`, which then raises JobLost to stop the work early.
    """

    def __init__(self, cfg: WorkerConfig, worker_id: str, job: Job):
        self.cfg, self.worker_id, self.job = cfg, worker_id, job
        self.stage: Optional[str] = None
        self.pct: Optional[int] = None
        self.lost = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"heartbeat-{job.id}",
                                        daemon=True)

    def __enter__(self) -> "Heartbeat":
        self.store = JobStore(self.cfg.database_url)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self.store.close()

    def _run(self) -> None:
        while not self._stop.wait(self.cfg.heartbeat_s):
            try:
                if not self.store.heartbeat(self.job, self.worker_id, self.stage, self.pct):
                    self.lost = True
                    return
            except Exception:
                log.exception("job %s: heartbeat failed", self.job.id)

    def check(self) -> None:
        if self.lost:
            raise JobLost()

    def progress(self, stage: str, done: int, total: int) -> None:
        self.check()
        self.stage, self.pct = stage, min(100, done * 100 // total) if total else None


class Worker:
    def __init__(self, cfg: WorkerConfig, jobs: JobStore, estimator=None,
                 worker_id: Optional[str] = None):
        """`estimator` defaults to a PoseEstimator built from `cfg`, loaded once."""
        if not os.path.isdir(cfg.media_root):
            raise SystemExit(f"media root '{cfg.media_root}' is not a folder")
        self.cfg, self.jobs = cfg, jobs
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"
        if estimator is None:
            from ..estimator import PoseEstimator
            defaults = AnalysisOptions()
            estimator = PoseEstimator(cfg.model, defaults.conf, cfg.imgsz, cfg.device,
                                      defaults.kpt_conf)
        self.estimator = estimator

    def run(self, stop: threading.Event) -> None:
        """Process jobs until `stop` is set (checked between jobs)."""
        log.info("worker %s polling for jobs", self.worker_id)
        while not stop.is_set():
            try:
                busy = self.run_once()
            except Exception:
                log.exception("queue unavailable; retrying")
                busy = False
            if not busy:
                stop.wait(self.cfg.poll_s)

    def run_once(self) -> bool:
        """Process one job; False if the queue was empty."""
        recovered = self.jobs.recover_stale(self.cfg.stale_s, self.cfg.max_attempts)
        if recovered:
            log.warning("recovered %d job(s) from unresponsive workers", recovered)
        job = self.jobs.claim(self.worker_id)
        if job is None:
            return False
        log.info("job %s: %s/%s, attempt %d", job.id, job.job_dir, job.video_file,
                 job.attempts)
        self.process(job)
        return True

    def process(self, job: Job) -> None:
        with Heartbeat(self.cfg, self.worker_id, job) as hb:
            try:
                result = self._analyse(job, hb)
            except JobLost:
                log.warning("job %s: cancelled or reassigned; dropped", job.id)
                return
            except AnalysisError as exc:
                log.info("job %s: input rejected: %s", job.id, exc)
                self.jobs.fail(job, self.worker_id, "input", exc.code, str(exc), retry=False)
                return
            except Exception as exc:
                retry = job.attempts < self.cfg.max_attempts
                log.exception("job %s: failed%s", job.id, "; will retry" if retry else "")
                self.jobs.fail(job, self.worker_id, "internal", None,
                               f"{type(exc).__name__}: {exc}", retry)
                return
        if self.jobs.succeed(job, self.worker_id, result):
            log.info("job %s: done, %d reps", job.id, result["reps"])
        else:
            log.warning("job %s: finished, but it was cancelled or reassigned meanwhile", job.id)

    def _analyse(self, job: Job, hb: Heartbeat) -> Dict[str, Any]:
        """Analyse in a scratch folder inside job_dir, then move the outputs next to
        the inputs: a move within one disk is atomic, so the web app never serves a
        half-written file, and a failed or cancelled job leaves nothing behind."""
        opts = job_options(job, self.cfg)
        job_dir = self._job_dir(job)
        video = _input(job_dir, job.video_file)
        if job.telemetry_file:
            opts.telemetry = _input(job_dir, job.telemetry_file)
        for old in glob.glob(os.path.join(job_dir, SCRATCH_PREFIX + "*")):
            shutil.rmtree(old, ignore_errors=True)      # left by a worker that died
        work = tempfile.mkdtemp(prefix=SCRATCH_PREFIX, dir=job_dir)
        try:
            outcome = run_analysis(video, work, opts, self.estimator, hb.progress)
            hb.check()
            files = {}
            for name, path in outcome.files.items():
                files[name] = os.path.basename(path)
                os.replace(path, os.path.join(job_dir, files[name]))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return self._result(outcome, files)

    def _job_dir(self, job: Job) -> str:
        root = os.path.realpath(self.cfg.media_root)
        if not os.path.isdir(root):                     # e.g. a network mount is down: retry
            raise RuntimeError(f"media root '{root}' is not available")
        path = os.path.realpath(os.path.join(root, job.job_dir))
        if not path.startswith(root + os.sep):
            raise AnalysisError("invalid_job", f"Analysis folder '{job.job_dir}' is outside the media root.")
        if not os.path.isdir(path):
            raise AnalysisError("invalid_job", f"Analysis folder '{job.job_dir}' not found.")
        return path

    @staticmethod
    def _result(outcome: AnalysisOutcome, files: Dict[str, str]) -> Dict[str, Any]:
        """What the web app needs to list and link a finished analysis; `files` are
        names in job_dir, and the full numbers are in the `summary` file."""
        reps, t, video = outcome.result.reps, outcome.result.body.t, outcome.video
        return {"files": files, "reps": len(reps),
                "clean_reps": sum(1 for r in reps if not r.faults),
                "analysed_s": round(float(t[-1] - t[0]), 1),      # the clip, not the file
                # the whole uploaded file, its size as displayed (rotation applied)
                "video": {"duration_s": round(video.duration, 3),
                          "width": video.width, "height": video.height},
                "warnings": outcome.warnings}


# Converts each JSON value in analysis_job.options to what AnalysisOptions expects.
_PARSE = {
    "start": float, "end": float, "telemetry_offset": float,
    "rotate": lambda v: _one_of(int(v), (0, 90, 180, 270)),
    "athlete": lambda v: _one_of(v, SELECT_MODES),
    "athlete_point": lambda v: tuple(float(x) for x in v[:2]) if len(v) == 2 else _bad(v),
    "thresholds": lambda v: {str(k): float(x) for k, x in v.items()},
    "video": lambda v: v if isinstance(v, bool) else _bad(v),
}
assert set(_PARSE) == set(JOB_OPTIONS)


def job_options(job: Job, cfg: WorkerConfig) -> AnalysisOptions:
    unknown = set(job.options) - set(JOB_OPTIONS)
    if unknown:
        raise AnalysisError("invalid_job", f"Unknown job options: {sorted(unknown)}.")
    parsed = {}
    for key, value in job.options.items():
        if value is None:
            continue
        try:
            parsed[key] = _PARSE[key](value)
        except (TypeError, ValueError, AttributeError):
            raise AnalysisError("invalid_job", f"Invalid job option {key}={value!r}: "
                                               f"expected {JOB_OPTIONS[key]}.") from None
    # No report.html or reps.csv: the web app builds its report from summary.json.
    return AnalysisOptions(station=job.station, model=cfg.model, imgsz=cfg.imgsz,
                           device=cfg.device, report=False, csv=False, **parsed)


def _one_of(value, allowed):
    return value if value in allowed else _bad(value)


def _bad(value):
    raise ValueError(value)


def _input(job_dir: str, name: str) -> str:
    """Path of an uploaded file, which must be a plain file name in job_dir."""
    if os.path.basename(name) != name or name in ("", ".", ".."):
        raise AnalysisError("invalid_job", f"'{name}' is not a file name.")
    if name in OUTPUT_FILES.values() or name.startswith(SCRATCH_PREFIX):
        raise AnalysisError("invalid_job", f"'{name}' is reserved for the analysis outputs; "
                                           "store the upload under another name.")
    path = os.path.join(job_dir, name)
    if not os.path.isfile(path):
        raise AnalysisError("invalid_job", f"Uploaded file '{name}' not found.")
    return path
