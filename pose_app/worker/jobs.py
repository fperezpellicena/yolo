"""The analysis_job table (schema.sql) used as a queue.

Workers claim jobs with SELECT ... FOR UPDATE SKIP LOCKED, so any number of
them can poll the same table without taking the same job twice. A running job
carries a heartbeat; a job whose heartbeat stops (worker killed, machine lost)
is put back in the queue, or failed once it has used up its attempts.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlsplit

import pymysql

# Keys the web app may put in analysis_job.options, all optional. Model and
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
    job_dir: str                    # relative to the media root
    video_file: str                  # in job_dir
    telemetry_file: Optional[str]
    options: Dict[str, Any]
    attempts: int                   # including this one


def connect(url: str) -> pymysql.connections.Connection:
    """`mysql://user:password@host:3306/database` (a `jdbc:` prefix is ignored)."""
    u = urlsplit(url[5:] if url.startswith("jdbc:") else url)
    if u.scheme not in ("mysql", "mysql+pymysql"):
        raise ValueError(f"expected a mysql:// database URL, got '{u.scheme}://'")
    print(f"Connecting to MySQL database {u.path.lstrip('/')} at {u.hostname}:{u.port or 3306}, with user {u.username} and password {u.password}...");
    return pymysql.connect(host=u.hostname or "localhost", port=u.port or 3306,
                           user=unquote(u.username or ""), password=unquote(u.password or ""),
                           database=u.path.lstrip("/"), charset="utf8mb4", autocommit=False)


class JobStore:
    """One connection, used from one thread at a time."""

    def __init__(self, url: str):
        self.url = url
        self.conn = connect(url)

    def _execute(self, sql: str, args=()) -> int:
        self.conn.ping(reconnect=True)
        try:
            with self.conn.cursor() as cur:
                n = cur.execute(sql, args)
            self.conn.commit()
            return n
        except Exception:
            self.conn.rollback()
            raise

    def claim(self, worker_id: str) -> Optional[Job]:
        """Take the oldest QUEUED job, or None if there is none."""
        self.conn.ping(reconnect=True)
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT id, station, job_dir, video_file, telemetry_file, options, attempts "
                            "FROM analysis_job WHERE status = 'QUEUED' "
                            "ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED")
                row = cur.fetchone()
                if row is not None:
                    cur.execute("UPDATE analysis_job SET status = 'RUNNING', worker_id = %s, "
                                "attempts = attempts + 1, started_at = NOW(3), "
                                "heartbeat_at = NOW(3), progress_stage = NULL, "
                                "progress_pct = NULL WHERE id = %s", (worker_id, row[0]))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        if row is None:
            return None
        job_id, station, job_dir, video_file, telemetry_file, options, attempts = row
        return Job(job_id, station, job_dir, video_file, telemetry_file,
                   json.loads(options) if options else {}, attempts + 1)

    def heartbeat(self, job: Job, worker_id: str, stage: Optional[str],
                  pct: Optional[int]) -> bool:
        """Record progress; False if the job is no longer ours (cancelled or reclaimed)."""
        return self._execute(
            "UPDATE analysis_job SET heartbeat_at = NOW(3), progress_stage = %s, "
            "progress_pct = %s WHERE id = %s AND worker_id = %s AND status = 'RUNNING'",
            (stage, pct, job.id, worker_id)) == 1

    def succeed(self, job: Job, worker_id: str, result: Dict[str, Any]) -> bool:
        return self._execute(
            "UPDATE analysis_job SET status = 'SUCCEEDED', result = %s, progress_stage = NULL, "
            "progress_pct = NULL, error_kind = NULL, error_message = NULL, "
            "finished_at = NOW(3) WHERE id = %s AND worker_id = %s AND status = 'RUNNING'",
            (json.dumps(result), job.id, worker_id)) == 1

    def fail(self, job: Job, worker_id: str, kind: str, message: str, retry: bool) -> bool:
        """Fail the job; with `retry` it goes back in the queue while attempts remain."""
        status = "QUEUED" if retry else "FAILED"
        return self._execute(
            "UPDATE analysis_job SET status = %s, error_kind = %s, error_message = %s, "
            "worker_id = IF(%s = 'QUEUED', NULL, worker_id), progress_stage = NULL, "
            "progress_pct = NULL, finished_at = IF(%s = 'FAILED', NOW(3), NULL) "
            "WHERE id = %s AND worker_id = %s AND status = 'RUNNING'",
            (status, kind, message[:10000], status, status, job.id, worker_id)) == 1

    def recover_stale(self, stale_s: float, max_attempts: int) -> int:
        """Requeue RUNNING jobs whose heartbeat stopped; fail those out of attempts."""
        stale = ("status = 'RUNNING' AND heartbeat_at < NOW(3) - INTERVAL %s SECOND "
                 "AND attempts {} %s")
        failed = self._execute(
            "UPDATE analysis_job SET status = 'FAILED', error_kind = 'internal', "
            "error_message = 'The worker stopped responding.', finished_at = NOW(3), "
            "progress_stage = NULL, progress_pct = NULL WHERE " + stale.format(">="),
            (stale_s, max_attempts))
        requeued = self._execute(
            "UPDATE analysis_job SET status = 'QUEUED', worker_id = NULL, "
            "progress_stage = NULL, progress_pct = NULL WHERE " + stale.format("<"),
            (stale_s, max_attempts))
        return failed + requeued

    def close(self) -> None:
        self.conn.close()
