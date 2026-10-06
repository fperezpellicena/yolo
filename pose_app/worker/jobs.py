"""The web app's job queue, as the worker sees it.

The worker only knows the JobClient interface: claim the next job, report its
progress, then complete or fail it. HttpJobClient implements it with the
internal API of the web app (Spring), which alone owns the queue and its
database: the worker needs no database driver or credentials, and no
knowledge of the schema. Another transport, such as a message broker, would
be a second implementation of the same interface.

    POST {api}/internal/jobs/claim            {worker}                     -> 200 job | 204
    POST {api}/internal/jobs/{id}/progress    {worker, stage, progress}    -> 204 | 409
    POST {api}/internal/jobs/{id}/complete    {worker, outputs, result, report} -> 204 | 409 | 422
    POST {api}/internal/jobs/{id}/fail        {worker, code, message, retryable} -> 204 | 409

409 means the job is no longer ours (cancelled, or given to another worker
after we went silent): the worker drops it. Progress doubles as the
heartbeat; the web app requeues a job that stays silent too long, and decides
whether a failed job is retried.
"""

import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import requests
from tenacity import (Retrying, before_sleep_log, retry_if_exception, stop_after_delay,
                      wait_exponential)

log = logging.getLogger(__name__)

# Keys the web app may put in a job's options, all optional. Model and
# hardware settings are the worker's own (see config.py), not per job.
JOB_OPTIONS = {
    "start": "clip start in seconds",
    "end": "clip end in seconds",
    "rotate": "0, 90, 180 or 270",
    "athlete": "'largest' or 'center'",
    "athlete_point": "[x, y] in pixels: follow the person there in the first frame",
    "telemetry_offset": "machine seconds at video second 0, if automatic sync fails",
    "thresholds": "{rule_id: value} overriding the station's defaults",
    "video": "false to skip the annotated video",
}


@dataclass
class Job:
    id: int
    station: str
    video: str                      # key of the uploaded video: a path relative to the media root
    telemetry: Optional[str]        # key of the machine data CSV, if any
    output_dir: str                 # key of the folder to write the outputs in
    options: Dict[str, Any]
    attempt: int                    # 1 on the first run


class JobClient(Protocol):
    """The queue. The calls about a job return False when it is no longer ours."""

    def claim_next(self, worker_id: str) -> Optional[Job]:
        """Take the oldest queued job, or None if there is none."""

    def progress(self, job: Job, worker_id: str, stage: Optional[str], pct: Optional[int]) -> bool:
        """Record progress; also tells the queue that the worker is alive."""

    def complete(self, job: Job, worker_id: str, outputs: Dict[str, str],
                 result: Dict[str, Any], report: Dict[str, Any]) -> bool:
        """`outputs` maps output names to keys; `report` is the summary.json content."""

    def fail(self, job: Job, worker_id: str, code: Optional[str], message: str,
             retryable: bool) -> bool:
        """`code` is the AnalysisError code of an unusable input (then not
        `retryable`), None for the worker's own failures."""


class ApiError(Exception):
    """The web app refused a call: a bad token, an outcome it cannot take, a bug."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status


class _Unavailable(Exception):
    """A 5xx answer, as while the web app restarts: worth retrying."""


def _transient(exc: BaseException) -> bool:
    return isinstance(exc, (requests.ConnectionError, requests.Timeout, _Unavailable))


class HttpJobClient:
    """JobClient over the web app's internal API. Safe to use from several threads
    (the heartbeat runs in its own): each thread gets its own HTTP session.

    Calls that fail because the web app is unreachable or answers 5xx are
    retried with exponential backoff for up to `retry_s` seconds, so a restart
    of the web app goes unnoticed; after that the error is raised.
    """

    def __init__(self, base_url: str, token: str, timeout_s: float = 30.0, retry_s: float = 60.0):
        self.base_url = base_url.rstrip("/") + "/internal/jobs"
        self.token = token
        self.timeout_s = timeout_s
        self.retry_s = retry_s
        self._local = threading.local()

    def claim_next(self, worker_id: str) -> Optional[Job]:
        status, body = self._post("/claim", {"worker": worker_id})
        if status == 204:
            return None
        inputs = body["inputs"]
        return Job(id=body["id"], station=body["station"], video=inputs["video"],
                   telemetry=inputs.get("telemetry"), output_dir=body["output_dir"],
                   options=body.get("options") or {}, attempt=body["attempt"])

    def progress(self, job: Job, worker_id: str, stage: Optional[str], pct: Optional[int]) -> bool:
        return self._report(job, "progress", {"worker": worker_id, "stage": stage, "progress": pct})

    def complete(self, job: Job, worker_id: str, outputs: Dict[str, str],
                 result: Dict[str, Any], report: Dict[str, Any]) -> bool:
        return self._report(job, "complete", {"worker": worker_id, "outputs": outputs,
                                              "result": result, "report": report})

    def fail(self, job: Job, worker_id: str, code: Optional[str], message: str,
             retryable: bool) -> bool:
        return self._report(job, "fail", {"worker": worker_id, "code": code,
                                          "message": message, "retryable": retryable})

    def close(self) -> None:
        session = getattr(self._local, "session", None)
        if session is not None:
            session.close()

    def _report(self, job: Job, action: str, body: Dict[str, Any]) -> bool:
        status, _ = self._post(f"/{job.id}/{action}", body)
        return status != 409

    def _post(self, path: str, body: Dict[str, Any]):
        """(status, parsed body or None). 409 is returned; other 4xx raise ApiError."""
        retrying = Retrying(retry=retry_if_exception(_transient), reraise=True,
                            wait=wait_exponential(multiplier=0.5, max=10),
                            stop=stop_after_delay(self.retry_s),
                            before_sleep=before_sleep_log(log, logging.WARNING))
        return retrying(self._post_once, path, body)

    def _post_once(self, path: str, body: Dict[str, Any]):
        # json.dumps writes NaN for values Python could not measure; requests'
        # own json= refuses them. The web app reads them.
        response = self._session().post(self.base_url + path, data=json.dumps(body),
                                        timeout=self.timeout_s,
                                        headers={"Content-Type": "application/json"})
        if response.status_code >= 500:
            raise _Unavailable(f"HTTP {response.status_code} on {path}")
        if response.status_code in (200, 201):
            return response.status_code, response.json()
        if response.status_code in (204, 409):
            return response.status_code, None
        raise ApiError(response.status_code, response.text[:1000] or response.reason)

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = self._local.session = requests.Session()
            session.headers["Authorization"] = f"Bearer {self.token}"
        return session
