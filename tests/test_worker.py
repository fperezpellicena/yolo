"""Queue worker against a real MySQL 8 server (no model or GPU needed).

Needs WORKER_TEST_DATABASE_URL, e.g. mysql://root@127.0.0.1:3306 ; the tests
create and use the database `pose_worker_test`, dropping it first. Skipped
when the variable is not set. Run: python tests/test_worker.py (or pytest).
"""
import json, os, shutil, sys, tempfile, threading, time, unittest, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_skierg import FPS, FRAMES, H, W, FakeEstimator, KINDS, write_video
from pose_app.worker.config import WorkerConfig
from pose_app.worker.jobs import JobStore, connect
from pose_app.worker.worker import SCRATCH_PREFIX, Worker

DB = "pose_worker_test"
SCHEMA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "pose_app", "worker", "schema.sql")
_state = {}


def _setup():
    """Fresh table and media root with one upload folder; the clip is written once.

    Returns the config, a connection and the upload folder's path relative to
    the media root (what the web app stores in job_dir).
    """
    server = os.environ.get("WORKER_TEST_DATABASE_URL")
    if not server:
        raise unittest.SkipTest("set WORKER_TEST_DATABASE_URL to run the worker tests")
    if not _state:
        conn = connect(server)
        with conn.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {DB}")
            cur.execute(f"CREATE DATABASE {DB}")
        conn.close()
        _state["url"] = server.rstrip("/") + "/" + DB
        conn = connect(_state["url"])
        with conn.cursor() as cur:
            with open(SCHEMA) as fh:
                cur.execute(fh.read().strip().rstrip(";"))
        conn.close()
        _state["tmp"] = tempfile.mkdtemp()
        _state["clip"] = os.path.join(_state["tmp"], "clip.mp4")
        write_video(_state["clip"])
    conn = connect(_state["url"])
    with conn.cursor() as cur:
        cur.execute("TRUNCATE analysis_job")
    conn.commit()
    root = os.path.join(_state["tmp"], "media")
    shutil.rmtree(root, ignore_errors=True)
    job_dir = os.path.join("analysis", "alice", str(uuid.uuid4()))
    os.makedirs(os.path.join(root, job_dir))
    shutil.copyfile(_state["clip"], os.path.join(root, job_dir, "clip.mp4"))
    cfg = WorkerConfig(_state["url"], root, heartbeat_s=0.05, max_attempts=3)
    return cfg, conn, job_dir


def _insert(conn, job_dir, video_file="clip.mp4", options=None, **cols) -> int:
    cols = {"station": "skierg", "job_dir": job_dir, "video_file": video_file,
            "options": json.dumps(options) if options is not None else None, **cols}
    with conn.cursor() as cur:
        cur.execute(f"INSERT INTO analysis_job ({', '.join(cols)}) "
                    f"VALUES ({', '.join(['%s'] * len(cols))})", list(cols.values()))
        job_id = cur.lastrowid
    conn.commit()
    return job_id


def _row(conn, job_id):
    conn.commit()                                   # end the snapshot: see other sessions' writes
    with conn.cursor() as cur:
        cur.execute("SELECT status, attempts, error_kind, error_code, error_message, result, "
                    "worker_id FROM analysis_job WHERE id = %s", (job_id,))
        status, attempts, kind, code, message, result, worker = cur.fetchone()
    return {"status": status, "attempts": attempts, "error_kind": kind, "error_code": code,
            "error": message,
            "result": json.loads(result) if result else None, "worker": worker}


def _run_once(cfg, estimator=None) -> bool:
    """A fresh worker per job: FakeEstimator replays the clip from its first frame."""
    jobs = JobStore(cfg.database_url)
    try:
        return Worker(cfg, jobs, estimator or FakeEstimator(), "test-worker").run_once()
    finally:
        jobs.close()


def _listing(cfg, job_dir):
    return sorted(os.listdir(os.path.join(cfg.media_root, job_dir)))


def test_job_writes_outputs_next_to_the_upload():
    cfg, conn, job_dir = _setup()
    job = _insert(conn, job_dir, options={"thresholds": {"early_arm_pull": 0.2}})
    assert _run_once(cfg)
    row = _row(conn, job)
    assert row["status"] == "SUCCEEDED" and row["attempts"] == 1, row
    result = row["result"]
    assert result["reps"] == len(KINDS) and result["clean_reps"] < len(KINDS), result
    assert result["files"] == {"summary": "summary.json", "video": "annotated.mp4",
                               "pose_cache": "pose_cache.npz"}, result["files"]
    assert result["video"] == {"duration_s": round(len(FRAMES) / FPS, 3), "width": W,
                               "height": H}, result["video"]
    assert _listing(cfg, job_dir) == sorted(["clip.mp4", *result["files"].values()])
    with open(os.path.join(cfg.media_root, job_dir, "clip.mp4"), "rb") as fh, \
            open(_state["clip"], "rb") as orig:
        assert fh.read() == orig.read()             # the upload is untouched
    assert not _run_once(cfg)                       # queue is empty now


def test_unusable_inputs_fail_without_retry():
    cfg, conn, job_dir = _setup()
    with open(os.path.join(cfg.media_root, job_dir, "junk.mp4"), "w") as fh:
        fh.write("not a video")
    shutil.copyfile(_state["clip"], os.path.join(cfg.media_root, job_dir, "annotated.mp4"))
    cases = [
        (_insert(conn, job_dir, "junk.mp4"), "unreadable_video", "Could not open"),
        (_insert(conn, job_dir, station="sled_push"), "unsupported_station", "Unknown station"),
        (_insert(conn, job_dir, "missing.mp4"), "invalid_job", "not found"),
        (_insert(conn, job_dir, "annotated.mp4"), "invalid_job", "reserved"),
        (_insert(conn, job_dir, "../clip.mp4"), "invalid_job", "not a file name"),
        (_insert(conn, "analysis/alice/nope"), "invalid_job", "not found"),
        (_insert(conn, "../../etc"), "invalid_job", "outside the media root"),
        (_insert(conn, job_dir, options={"rotate": 45}), "invalid_job", "rotate"),
        (_insert(conn, job_dir, options={"model": "yolo11x-pose.pt"}), "invalid_job",
         "Unknown job options"),
    ]
    while _run_once(cfg):
        pass
    for job, code, text in cases:
        row = _row(conn, job)
        assert row["status"] == "FAILED" and row["error_kind"] == "input", row
        assert row["error_code"] == code and row["attempts"] == 1 and text in row["error"], \
            (code, text, row)
    assert _listing(cfg, job_dir) == ["annotated.mp4", "clip.mp4", "junk.mp4"]


class _BrokenEstimator(FakeEstimator):
    def detect(self, frame):
        raise RuntimeError("CUDA error: device lost")


def test_internal_errors_retry_then_fail():
    cfg, conn, job_dir = _setup()
    job = _insert(conn, job_dir)
    for attempt in (1, 2):
        assert _run_once(cfg, _BrokenEstimator())
        row = _row(conn, job)
        assert row["status"] == "QUEUED" and row["attempts"] == attempt, row
        assert row["error_kind"] == "internal" and row["error_code"] is None, row
        assert row["worker"] is None, row
    assert _run_once(cfg, _BrokenEstimator())
    row = _row(conn, job)
    assert row["status"] == "FAILED" and "device lost" in row["error"], row
    assert _listing(cfg, job_dir) == ["clip.mp4"]  # no scratch folder left behind


def test_jobs_of_unresponsive_workers_are_recovered():
    cfg, conn, job_dir = _setup()
    os.makedirs(os.path.join(cfg.media_root, job_dir, SCRATCH_PREFIX + "dead"))
    old = "2000-01-01 00:00:00"
    retried = _insert(conn, job_dir, status="RUNNING", attempts=1, worker_id="dead",
                      heartbeat_at=old)
    exhausted = _insert(conn, job_dir, status="RUNNING", attempts=3, worker_id="dead",
                        heartbeat_at=old)
    alive = _insert(conn, job_dir, status="RUNNING", attempts=1, worker_id="busy",
                    heartbeat_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    assert _run_once(cfg)
    assert _row(conn, retried)["status"] == "SUCCEEDED"
    assert _row(conn, retried)["attempts"] == 2
    row = _row(conn, exhausted)
    assert row["status"] == "FAILED" and "stopped responding" in row["error"], row
    assert _row(conn, alive)["status"] == "RUNNING"
    assert not any(n.startswith(SCRATCH_PREFIX) for n in _listing(cfg, job_dir))


class _CancellingEstimator(FakeEstimator):
    """Cancels its own job part-way through the pose pass, as the web app would."""

    def __init__(self, url, job):
        super().__init__()
        self.url, self.job = url, job

    def detect(self, frame):
        if self.i == 100:
            conn = connect(self.url)
            with conn.cursor() as cur:
                cur.execute("UPDATE analysis_job SET status = 'CANCELLED' WHERE id = %s",
                            (self.job,))
            conn.commit()
            conn.close()
        if self.i > 100:
            time.sleep(0.002)                       # leave the heartbeat time to notice
        return super().detect(frame)


def test_cancelled_job_is_dropped():
    cfg, conn, job_dir = _setup()
    job = _insert(conn, job_dir)
    estimator = _CancellingEstimator(cfg.database_url, job)
    assert _run_once(cfg, estimator)
    row = _row(conn, job)
    assert row["status"] == "CANCELLED" and row["result"] is None, row
    assert estimator.i < len(KINDS) * 50, estimator.i       # stopped well before the end
    assert _listing(cfg, job_dir) == ["clip.mp4"]


def test_concurrent_claims_never_share_a_job():
    cfg, conn, job_dir = _setup()
    ids = {_insert(conn, job_dir) for _ in range(40)}
    claimed, lock = [], threading.Lock()

    def claim_all(name):
        jobs = JobStore(cfg.database_url)
        while True:
            job = jobs.claim(name)
            if job is None:
                break
            with lock:
                claimed.append(job.id)
        jobs.close()
    threads = [threading.Thread(target=claim_all, args=(f"w{n}",)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(claimed) == sorted(ids), (len(claimed), len(ids))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"{name}: ok")
            except unittest.SkipTest as exc:
                print(f"{name}: skipped ({exc})")
