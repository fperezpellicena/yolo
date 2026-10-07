"""The loop: claim a job, analyse its uploads, write the outputs next to them and
report the outcome to the web app."""

import glob
import json
import logging
import os
import posixpath
import shutil
import socket
import tempfile
import threading
from typing import Any, Dict, Optional, Tuple

from ..analysis.api import (OUTPUT_FILES, AnalysisError, AnalysisOptions, AnalysisOutcome,
                            run_analysis)
from .config import WorkerConfig
from .jobs import ApiError, Job, JobClient
from .options import build_analysis_options

log = logging.getLogger(__name__)

SCRATCH_PREFIX = ".analysis-tmp-"       # work folder inside the output folder while a job runs

# Outputs moved next to the uploads, by their names in AnalysisOutcome.files and
# in the web app. The summary is not among them: it is sent as the report, and
# the web app stores it once it has checked it.
KEPT_OUTPUTS = {"video": "annotated_video", "pose_cache": "pose_cache"}


class JobLost(Exception):
    """The job was cancelled or handed to another worker while we ran it."""


class Heartbeat:
    """Reports the job's progress from a background thread, which also tells the
    web app that the worker is alive.

    The analysis thread never waits on the web app; it learns that the job was
    lost through `progress()`, which then raises JobLost to stop the work early.
    """

    def __init__(self, cfg: WorkerConfig, jobs: JobClient, worker_id: str, job: Job):
        self.cfg, self.jobs, self.worker_id, self.job = cfg, jobs, worker_id, job
        self.stage: Optional[str] = None
        self.pct: Optional[int] = None
        self.lost = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"heartbeat-{job.id}",
                                        daemon=True)

    def __enter__(self) -> "Heartbeat":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.wait(self.cfg.heartbeat_s):
            try:
                if not self.jobs.progress(self.job, self.worker_id, self.stage, self.pct):
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
    def __init__(self, cfg: WorkerConfig, jobs: JobClient, estimator=None,
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
        job = self.jobs.claim_next(self.worker_id)
        if job is None:
            return False
        log.info("job %s: %s, attempt %d", job.id, job.video, job.attempt)
        self.process(job)
        return True

    def process(self, job: Job) -> None:
        """Run the job and report its outcome. The web app decides whether a failed
        job is retried."""
        with Heartbeat(self.cfg, self.jobs, self.worker_id, job) as hb:
            try:
                outputs, result, report = self._analyse(job, hb)
            except JobLost:
                log.warning("job %s: cancelled or reassigned; dropped", job.id)
                return
            except AnalysisError as exc:
                log.info("job %s: input rejected: %s", job.id, exc)
                self._fail(job, exc.code, str(exc), retryable=False)
                return
            except Exception as exc:
                log.exception("job %s: failed", job.id)
                self._fail(job, None, f"{type(exc).__name__}: {exc}", retryable=True)
                return
        try:
            completed = self.jobs.complete(job, self.worker_id, outputs, result, report)
        except ApiError as exc:                         # e.g. a report it cannot read
            log.error("job %s: the web app refused the outcome: %s", job.id, exc)
            self._fail(job, None, f"Outcome refused: {exc}", retryable=True)
            return
        if completed:
            log.info("job %s: done, %d reps", job.id, result["reps"])
        else:
            log.warning("job %s: finished, but it was cancelled or reassigned meanwhile", job.id)

    def _fail(self, job: Job, code: Optional[str], message: str, retryable: bool) -> None:
        if not self.jobs.fail(job, self.worker_id, code, message, retryable):
            log.warning("job %s: failed, but it was cancelled or reassigned meanwhile", job.id)

    def _analyse(self, job: Job, hb: Heartbeat) -> Tuple[Dict[str, str], Dict[str, Any],
                                                            Dict[str, Any]]:
        """Analyse in a scratch folder inside the output folder, then move the outputs
        in: a move within one disk is atomic, so the web app never serves a
        half-written file, and a failed or cancelled job leaves nothing behind.

        Returns the keys of the outputs, the result and the report (summary.json)."""
        opts = build_analysis_options(job, self.cfg)
        root = self._media_root()
        out_dir = _resolve(root, job.output_dir, "Output folder")
        if not os.path.isdir(out_dir):
            raise AnalysisError("invalid_job", f"Output folder '{job.output_dir}' not found.")
        video = _input(root, job.video)
        if job.telemetry:
            opts.telemetry = _input(root, job.telemetry)
        for old in glob.glob(os.path.join(out_dir, SCRATCH_PREFIX + "*")):
            shutil.rmtree(old, ignore_errors=True)      # left by a worker that died
        work = tempfile.mkdtemp(prefix=SCRATCH_PREFIX, dir=out_dir)
        try:
            outcome = run_analysis(video, work, opts, self.estimator, hb.progress)
            hb.check()
            with open(outcome.files["summary"], encoding="utf-8") as fh:
                report = json.load(fh)
            outputs = {}
            for name, key in KEPT_OUTPUTS.items():
                if name in outcome.files:
                    file_name = os.path.basename(outcome.files[name])
                    os.replace(outcome.files[name], os.path.join(out_dir, file_name))
                    outputs[key] = posixpath.join(job.output_dir, file_name)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return outputs, self._result(outcome), report

    def _media_root(self) -> str:
        root = os.path.realpath(self.cfg.media_root)
        if not os.path.isdir(root):                     # e.g. a network mount is down: retry
            raise RuntimeError(f"media root '{root}' is not available")
        return root

    @staticmethod
    def _result(outcome: AnalysisOutcome) -> Dict[str, Any]:
        """What the web app needs to list a finished analysis; the full numbers are
        in the report."""
        reps, t, video = outcome.result.reps, outcome.result.body.t, outcome.video
        return {"reps": len(reps),
                "clean_reps": sum(1 for r in reps if not r.faults),
                "analysed_s": round(float(t[-1] - t[0]), 1),      # the clip, not the file
                # the whole uploaded file, its size as displayed (rotation applied)
                "video": {"duration_s": round(video.duration, 3),
                          "width": video.width, "height": video.height},
                "warnings": outcome.warnings}


def _resolve(root: str, key: str, what: str) -> str:
    """Path of a file key, which must stay inside the media root."""
    path = os.path.realpath(os.path.join(root, key))
    if not key or os.path.isabs(key) or not path.startswith(root + os.sep):
        raise AnalysisError("invalid_job", f"{what} '{key}' is outside the media root.")
    return path


def _input(root: str, key: str) -> str:
    """Path of an uploaded file, whose name must not be one the outputs take."""
    name = posixpath.basename(key)
    if name in OUTPUT_FILES.values() or name.startswith(SCRATCH_PREFIX):
        raise AnalysisError("invalid_job", f"'{name}' is reserved for the analysis outputs; "
                                           "store the upload under another name.")
    path = _resolve(root, key, "Uploaded file")
    if not os.path.isfile(path):
        raise AnalysisError("invalid_job", f"Uploaded file '{key}' not found.")
    return path
