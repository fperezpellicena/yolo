"""Queue worker against a fake of the web app's internal API (no model, GPU or
database needed). Run: python tests/test_worker.py (or pytest).

FakeApi keeps the queue in memory and answers like the web app: the oldest
queued job on claim, 409 for a job that is no longer the caller's, retries of
the worker's own failures while attempts remain.
"""
import json, math, os, shutil, sys, tempfile, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_skierg import FPS, FRAMES, H, W, FakeEstimator, KINDS, write_video
from pose_app.worker.config import WorkerConfig
from pose_app.worker.jobs import ApiError, HttpJobClient
from pose_app.worker.worker import SCRATCH_PREFIX, Worker

TOKEN = "s3cret"
_state = {}


class FakeApi:
    """The web app's side of the contract, in memory."""

    def __init__(self, max_attempts=3):
        self.max_attempts = max_attempts
        self.jobs = {}                  # id -> dict
        self.calls = []                 # (action, job id, body)
        self.unavailable = 0            # answer 503 to this many calls first
        self.refuse_complete = False    # answer 422 to complete
        self.lock = threading.Lock()
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                status, answer = api.handle(self.path, self.headers.get("Authorization"), body)
                payload = json.dumps(answer).encode() if answer is not None else b""
                self.send_response(status)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def queue(self, video, output_dir, station="skierg", telemetry=None, options=None,
              pm5=None) -> int:
        with self.lock:
            job_id = len(self.jobs) + 1
            self.jobs[job_id] = {"status": "QUEUED", "station": station, "video": video,
                                 "telemetry": telemetry, "output_dir": output_dir,
                                 "options": options, "attempts": 0, "worker": None, "pm5": pm5}
        return job_id

    def cancel(self, job_id):
        with self.lock:
            self.jobs[job_id]["status"] = "CANCELLED"

    def handle(self, path, authorization, body):
        with self.lock:
            if authorization != f"Bearer {TOKEN}":
                return 401, None
            if self.unavailable:
                self.unavailable -= 1
                return 503, None
            parts = path.strip("/").split("/")          # internal, jobs, claim | id, action
            if parts[2] == "claim":
                return self._claim(body["worker"])
            job_id, action = int(parts[2]), parts[3]
            self.calls.append((action, job_id, body))
            job = self.jobs[job_id]
            if job["status"] != "RUNNING" or job["worker"] != body["worker"]:
                return 409, None
            if action == "progress":
                job["progress"] = (body["stage"], body["progress"])
            elif action == "complete":
                if self.refuse_complete:
                    return 422, {"detail": "unreadable report"}
                job.update(status="SUCCEEDED", outcome=body)
            elif action == "fail":
                retry = body["retryable"] and job["attempts"] < self.max_attempts
                job.update(status="QUEUED" if retry else "FAILED", worker=None, outcome=body)
            return 204, None

    def _claim(self, worker):
        queued = [i for i, j in sorted(self.jobs.items()) if j["status"] == "QUEUED"]
        if not queued:
            return 204, None
        job_id = queued[0]
        job = self.jobs[job_id]
        job.update(status="RUNNING", worker=worker, attempts=job["attempts"] + 1)
        inputs = {"video": job["video"]}
        if job["telemetry"]:
            inputs["telemetry"] = job["telemetry"]
        if job["pm5"]:
            inputs["pm5"] = job["pm5"]
        return 200, {"id": job_id, "station": job["station"], "options": job["options"] or {},
                     "inputs": inputs, "output_dir": job["output_dir"], "attempt": job["attempts"]}


def _setup(**api_args):
    """A fresh fake API and media root with one upload folder; the clip is written once.

    Returns the config, the API and the upload folder's key (what the web app
    hands out as output_dir).
    """
    if not _state:
        _state["tmp"] = tempfile.mkdtemp()
        _state["clip"] = os.path.join(_state["tmp"], "clip.mp4")
        write_video(_state["clip"])
    if "api" in _state:
        _state["api"].close()
    api = _state["api"] = FakeApi(**api_args)
    root = os.path.join(_state["tmp"], "media")
    shutil.rmtree(root, ignore_errors=True)
    folder = f"alice/{uuid.uuid4()}"
    os.makedirs(os.path.join(root, folder))
    shutil.copyfile(_state["clip"], os.path.join(root, folder, "clip.mp4"))
    cfg = WorkerConfig(api.url, TOKEN, root, heartbeat_s=0.05, api_retry_s=5)
    return cfg, api, folder


def _client(cfg):
    return HttpJobClient(cfg.api_url, cfg.api_token, timeout_s=5, retry_s=cfg.api_retry_s)


def _run_once(cfg, estimator=None) -> bool:
    """A fresh worker per job: FakeEstimator replays the clip from its first frame."""
    jobs = _client(cfg)
    try:
        return Worker(cfg, jobs, estimator or FakeEstimator(), "test-worker").run_once()
    finally:
        jobs.close()


def _listing(cfg, folder):
    return sorted(os.listdir(os.path.join(cfg.media_root, folder)))


def test_job_writes_outputs_next_to_the_upload_and_sends_the_report():
    cfg, api, folder = _setup()
    job = api.queue(f"{folder}/clip.mp4", folder, options={"thresholds": {"early_arm_pull": 0.2}})
    assert _run_once(cfg)
    state = api.jobs[job]
    assert state["status"] == "SUCCEEDED" and state["attempts"] == 1, state
    outcome = state["outcome"]
    assert outcome["worker"] == "test-worker"
    assert outcome["outputs"] == {"annotated_video": f"{folder}/annotated.mp4",
                                  "pose_cache": f"{folder}/pose_cache.npz"}, outcome["outputs"]
    result = outcome["result"]
    assert result["reps"] == len(KINDS) and result["clean_reps"] < len(KINDS), result
    assert result["video"] == {"duration_s": round(len(FRAMES) / FPS, 3), "width": W,
                               "height": H}, result["video"]
    report = outcome["report"]
    assert report["schema_version"] == 1 and report["reps"] == len(KINDS), report.keys()
    rule = next(r for r in report["rules"] if r["id"] == "early_arm_pull")
    assert rule["threshold"] == 0.2, rule
    # The summary is the web app's to store; only the other outputs land in the folder
    assert _listing(cfg, folder) == ["annotated.mp4", "clip.mp4", "pose_cache.npz"]
    with open(os.path.join(cfg.media_root, folder, "clip.mp4"), "rb") as fh, \
            open(_state["clip"], "rb") as orig:
        assert fh.read() == orig.read()             # the upload is untouched
    assert any(action == "progress" for action, _, _ in api.calls)
    assert not _run_once(cfg)                       # queue is empty now


def test_unusable_inputs_fail_without_retry():
    cfg, api, folder = _setup()
    with open(os.path.join(cfg.media_root, folder, "junk.mp4"), "w") as fh:
        fh.write("not a video")
    shutil.copyfile(_state["clip"], os.path.join(cfg.media_root, folder, "annotated.mp4"))
    cases = [
        (api.queue(f"{folder}/junk.mp4", folder), "unreadable_video", "Could not open"),
        (api.queue(f"{folder}/clip.mp4", folder, station="sled_push"), "unsupported_station",
         "Unknown station"),
        (api.queue(f"{folder}/missing.mp4", folder), "invalid_job", "not found"),
        (api.queue(f"{folder}/annotated.mp4", folder), "invalid_job", "reserved"),
        (api.queue(f"{folder}/../../../clip.mp4", folder), "invalid_job", "outside the media root"),
        (api.queue("/etc/passwd", folder), "invalid_job", "outside the media root"),
        (api.queue(f"{folder}/clip.mp4", "alice/nope"), "invalid_job", "not found"),
        (api.queue(f"{folder}/clip.mp4", "../../etc"), "invalid_job", "outside the media root"),
        (api.queue(f"{folder}/clip.mp4", folder, options={"rotate": 45}), "invalid_job", "rotate"),
        (api.queue(f"{folder}/clip.mp4", folder, options={"model": "yolo11x-pose.pt"}),
         "invalid_job", "Unknown job options"),
    ]
    while _run_once(cfg):
        pass
    for job, code, text in cases:
        state = api.jobs[job]
        outcome = state["outcome"]
        assert state["status"] == "FAILED" and state["attempts"] == 1, (code, state)
        assert outcome["code"] == code and not outcome["retryable"], (code, outcome)
        assert text in outcome["message"], (text, outcome)
    assert _listing(cfg, folder) == ["annotated.mp4", "clip.mp4", "junk.mp4"]


def test_force_job_reads_the_pm5_log_and_reports_the_force():
    import synthetic_force as sf
    cfg, api, folder = _setup()
    session = sf.Session(n_strokes=12)
    base = os.path.join(cfg.media_root, folder)
    session.write_video(os.path.join(base, "ski.mp4"))
    session.write_pm5_log(os.path.join(base, "ski.pm5.jsonl"))
    job = api.queue(f"{folder}/ski.mp4", folder, pm5=f"{folder}/ski.pm5.jsonl",
                    options={"force": session.setup().to_dict(), "video": False})
    jobs = _client(cfg)
    try:
        assert Worker(cfg, jobs, session.estimator(), "test-worker").run_once()
    finally:
        jobs.close()
    state = api.jobs[job]
    assert state["status"] == "SUCCEEDED", state.get("outcome")
    report = state["outcome"]["report"]
    assert report["force"]["sync"]["confidence"] == "high", report["force"]["sync"]
    assert report["force"]["summary"]["strokes_with_curve"] >= 8
    assert report["machine"]["sync"]["method"] == "strokes"
    bad = api.queue(f"{folder}/ski.mp4", folder, options={"force": session.setup().to_dict()})
    assert _run_once(cfg, session.estimator())
    outcome = api.jobs[bad]["outcome"]
    assert outcome["code"] == "invalid_job" and "PM5 log" in outcome["message"], outcome


class _BrokenEstimator(FakeEstimator):
    def detect(self, frame):
        raise RuntimeError("CUDA error: device lost")


def test_own_failures_are_reported_as_retryable():
    cfg, api, folder = _setup(max_attempts=2)
    job = api.queue(f"{folder}/clip.mp4", folder)
    assert _run_once(cfg, _BrokenEstimator())
    assert api.jobs[job]["status"] == "QUEUED"      # the web app retries it
    assert _run_once(cfg, _BrokenEstimator())
    state = api.jobs[job]
    assert state["status"] == "FAILED" and state["attempts"] == 2, state
    outcome = state["outcome"]
    assert outcome["retryable"] and outcome["code"] is None, outcome
    assert "device lost" in outcome["message"], outcome
    assert _listing(cfg, folder) == ["clip.mp4"]    # no scratch folder left behind


def test_a_refused_outcome_fails_the_job():
    cfg, api, folder = _setup()
    api.refuse_complete = True
    job = api.queue(f"{folder}/clip.mp4", folder)
    assert _run_once(cfg)
    outcome = api.jobs[job]["outcome"]
    assert api.jobs[job]["status"] == "QUEUED" and outcome["retryable"], api.jobs[job]
    assert "Outcome refused" in outcome["message"] and "422" in outcome["message"], outcome


def test_scratch_left_by_a_dead_worker_is_cleared():
    cfg, api, folder = _setup()
    os.makedirs(os.path.join(cfg.media_root, folder, SCRATCH_PREFIX + "dead"))
    api.queue(f"{folder}/clip.mp4", folder)
    assert _run_once(cfg)
    assert not any(n.startswith(SCRATCH_PREFIX) for n in _listing(cfg, folder))


class _CancellingEstimator(FakeEstimator):
    """Cancels its own job part-way through the pose pass, as the web app would."""

    def __init__(self, api, job):
        super().__init__()
        self.api, self.job = api, job

    def detect(self, frame):
        if self.i == 100:
            self.api.cancel(self.job)
        if self.i > 100:
            time.sleep(0.002)                       # leave the heartbeat time to notice
        return super().detect(frame)


def test_cancelled_job_is_dropped():
    cfg, api, folder = _setup()
    job = api.queue(f"{folder}/clip.mp4", folder)
    estimator = _CancellingEstimator(api, job)
    assert _run_once(cfg, estimator)
    assert api.jobs[job]["status"] == "CANCELLED"
    assert not any(action in ("complete", "fail") for action, _, _ in api.calls), api.calls
    assert estimator.i < len(KINDS) * 50, estimator.i       # stopped well before the end
    assert _listing(cfg, folder) == ["clip.mp4"]


def test_calls_are_retried_while_the_web_app_restarts():
    cfg, api, folder = _setup()
    api.queue(f"{folder}/clip.mp4", folder)
    api.unavailable = 3
    jobs = _client(cfg)
    job = jobs.claim_next("test-worker")
    assert job is not None and job.video == f"{folder}/clip.mp4" and job.attempt == 1
    assert api.unavailable == 0
    assert jobs.progress(job, "test-worker", "pose", 5)
    assert not jobs.progress(job, "other-worker", "pose", 5)      # 409: not ours
    jobs.close()


def test_gives_up_when_the_web_app_stays_away():
    cfg, api, _ = _setup()
    api.close()
    jobs = HttpJobClient(api.url, TOKEN, timeout_s=1, retry_s=0.5)
    started = time.monotonic()
    try:
        jobs.claim_next("test-worker")
        raise AssertionError("expected a connection error")
    except Exception as exc:
        assert "Connection" in type(exc).__name__, repr(exc)
    assert time.monotonic() - started < 10


def test_a_wrong_token_is_an_error():
    cfg, api, _ = _setup()
    jobs = HttpJobClient(api.url, "guess", timeout_s=5, retry_s=1)
    try:
        jobs.claim_next("test-worker")
        raise AssertionError("expected ApiError")
    except ApiError as exc:
        assert exc.status == 401, exc


def test_values_python_could_not_measure_are_sent_as_nan():
    cfg, api, folder = _setup()
    job_id = api.queue(f"{folder}/clip.mp4", folder)
    jobs = _client(cfg)
    job = jobs.claim_next("test-worker")
    assert jobs.complete(job, "test-worker", {}, {"reps": 0}, {"clean_pct": float("nan")})
    assert math.isnan(api.jobs[job_id]["outcome"]["report"]["clean_pct"])
    jobs.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
