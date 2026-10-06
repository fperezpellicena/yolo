# Pose estimation with joint angles + Hyrox technique analysis

Two tools share one package:

* `pose_angles.py`: live webcam pose estimation with joint angles.
* `hyrox_analyze.py`: offline technique analysis of a recorded station video.

## Hyrox technique analysis (recorded video)

    python hyrox_analyze.py clip.mp4 --station skierg
    python hyrox_analyze.py clip.mp4 --station skierg --start 12 --end 250
    python hyrox_analyze.py --list-stations

Writes `<clip>_analysis/` containing:

    report.html      summary, faults with cues, fatigue checks, example frames, per-rep charts
    annotated.mp4    skeleton, rep counter, phase and the cue for each rep
    reps.csv         every metric for every rep
    summary.json     machine-readable summary
    pose_cache.npz   the pose pass, so re-analysis is instant

Tuning thresholds with a coach (no re-running the model):

    python hyrox_analyze.py --station skierg --print-thresholds > ski.json
    # edit ski.json
    python hyrox_analyze.py clip.mp4 --station skierg --thresholds ski.json --reuse --no-video

### Adding machine data (SkiErg / rower)

Record the piece with Track My Indoor Workout (FTMS) at the same time as the
video, export the workout as CSV, and pass it in:

    python hyrox_analyze.py clip.mp4 --station skierg --telemetry workout.csv

The report gains a "Machine data" section (power over the piece with the
faulty strokes marked, 250 m splits showing pace, power, rate and HR next to
the technique of the strokes in each split, and how output differed on strokes
with each fault), fatigue checks on power and distance per stroke, `m_*`
columns in reps.csv, and heart rate, distance, watts and pace in the annotated
video (each shown only when the file has it). Telemetry is not
part of the pose cache, so `--reuse` still re-analyses instantly.

Sync is automatic. It is sharpest when both recordings include the start:
start filming, stand still for a moment, then start the piece. Without the
start it matches stroke rate (+/- 1-2 s); if that is inconclusive the machine
data is summarised but not attached per stroke, and you can set the sync by
hand with `--telemetry-offset S` (machine seconds at video second 0).
Machine values are smoothed by the monitor over several strokes, so compare
them over splits and groups of strokes, not single strokes.

Recording tips: keep the app in the foreground on its own phone (film on a
second device), set heart-rate gap handling to nulls, and leave power and
calorie tuning at 100%.

If other people are in shot, the largest person is followed by default; use
`--athlete center` or `--athlete-point X,Y` to pick someone else. Offline
defaults favour accuracy (`yolo11m-pose.pt`, `--imgsz 960`); on a CPU-only
machine use `--model yolo11s-pose.pt --imgsz 640` for speed.

### Adding a station

Subclass `Station` in `pose_app/analysis/stations/`, set the driver metric
whose peak -> trough -> peak is one rep, implement `summarize()` (one rep ->
dict of numbers) and `rules()`, then register it in `stations/__init__.py`.
Body metrics available per frame are in `analysis/body.py`.

### Calling it from code

`hyrox_analyze.py` is a thin wrapper around one function, which is also what a
job worker should call:

    from pose_app.analysis.api import AnalysisError, AnalysisOptions, run_analysis

    outcome = run_analysis("clip.mp4", "out/", AnalysisOptions(station="skierg"),
                           estimator=None,                  # or a PoseEstimator loaded once
                           on_progress=lambda stage, done, total: ...)
    outcome.files       # {"report": ..., "summary": ..., "reps": ..., "video": ..., "pose_cache": ...}
    outcome.warnings    # messages for the athlete / coach

It never prints. Status lines go to the `pose_app` logger, and unusable inputs
(unreadable video, bad telemetry, empty clip, unknown station) raise
`AnalysisError`, whose message is safe to show to the athlete or coach. Any
other exception is a bug or an environment problem.

### Queue worker (for the web app)

The web app saves the uploads in a folder per analysis and inserts a row into
the `analysis_jobs` table; the worker analyses the files in place and writes
its outputs into the same folder:

    analysis_jobs QUEUED --claim--> run_analysis() on <media root>/<job_dir>/<video_file>
        -> summary.json, annotated.mp4, pose_cache.npz written next to it
        -> SUCCEEDED / FAILED

    pip install -r requirements-worker.txt
    export WORKER_DATABASE_URL=mysql://user:password@db:3306/hyrox
    export WORKER_MEDIA_ROOT=/srv/media      # where the web app's media folder is mounted
    python -m pose_app.worker

The settings can also go in a `.env` file (same `NAME=value` lines) in the
folder you start the worker from, or a parent folder; variables already set
in the environment take precedence. `.env` is git-ignored.

* Table contract: `pose_app/worker/schema.sql` (MySQL 8). The web app owns
  the table through its migrations and may add columns, e.g. `user_id`.
* `job_dir` is relative to the media root (e.g. `analysis/alice/<uuid>`), and
  `video_file` / `telemetry_file` are file names in it. Each side sets its own
  root, so the web app and the worker can mount the disk in different places.
  Uploads must not use the output names above. There is no report.html or
  reps.csv: the web app draws the report from summary.json (see below).
* Outputs appear all at once: the worker writes them in a hidden
  `.analysis-tmp-*` folder inside `job_dir` and moves them in at the end, so a
  failed or cancelled job leaves nothing behind. The worker needs write access
  to `job_dir`, and its files must be readable by the web app (same user, or a
  shared group with a suitable umask).
* Job options (`options` JSON column): clip start/end, rotation, which
  athlete to follow, telemetry offset, threshold overrides, no video; see `JOB_OPTIONS` in `pose_app/worker/jobs.py`. The model and hardware
  are the worker's settings, not per job.
* While running, the worker writes `progress_stage` (pose, video) and
  `progress_pct` every few seconds. On success `result` lists the output file
  names, rep counts and warnings.
* Failures: `error_kind = 'input'` means the upload can't be analysed and the
  message can be shown to the user (no retry); `'internal'` is retried up to
  `WORKER_MAX_ATTEMPTS` times. Setting `status = 'CANCELLED'` stops a running
  job within a few seconds.
* Several workers can poll the same table safely (`FOR UPDATE SKIP LOCKED`).
  If a worker dies, its job is requeued once its heartbeat is
  `WORKER_STALE_SECONDS` old.
* Other settings (`pose_app/worker/config.py`): `WORKER_MODEL`, `WORKER_IMGSZ`,
  `WORKER_DEVICE`, `WORKER_POLL_SECONDS`, `WORKER_HEARTBEAT_SECONDS`. In a
  container, set `PYTHONUNBUFFERED=1` so logs appear immediately.

### summary.json

Everything the report shows, for the web app to draw its own
(`schema_version` changes when a consumer would need to):

* `station`: key, name, rep word, camera view, phase names, filming tips.
* `reps`, `clean_reps`, `clean_pct`, `averages` (per chart metric), `camera`,
  `warnings`.
* `rules`: every technique rule with its title, severity, metric, fault
  condition (`op` + `threshold`, after overrides) and coaching cue.
  `thresholds` also lists the fatigue rules' values.
* `faults`: how often each rule fired; `drift`: early vs late fatigue checks.
* `charts`: which per-rep metrics to plot, with label and unit; draw each
  rule's threshold on the chart of its metric.
* `examples`: for each fault type, the rep where it was worst, with the value
  and when it happened.
* `per_rep`: every metric of every rep, its faults (rule, value, time), and
  `video_t_start` / `video_t_end`.
* `machine`: telemetry sync, splits and output vs technique (when given).

Times: `t` / `t_start` are seconds in the uploaded video; `video_t*` are
seconds in annotated.mp4 (which starts at the clip start), so the web app can
seek the player straight to a rep or a fault.

Regression tests (no model needed): `python tests/test_skierg.py`,
`python tests/test_telemetry.py` and `python tests/test_api.py`.
`python tests/test_worker.py` needs a MySQL 8 server:
`WORKER_TEST_DATABASE_URL=mysql://root@127.0.0.1:3306` (it creates the
`pose_worker_test` database).

Real-time webcam pose estimation (Ultralytics YOLO) with a responsive,
full-screen-capable view: original feed | pose overlay | joint-angle panel.

    pip install -r requirements.txt
    python pose_angles.py            # or: python -m pose_app
    python pose_angles.py --help     # all options

## Project layout

    pose_angles.py              live webcam app entry point
    hyrox_analyze.py            offline analysis entry point
    tests/                      synthetic regression tests
    pose_app/
        cli.py                  parse options, pick the camera, start the app
        app.py                  PoseApp: main loop + keyboard actions
        config.py               Settings dataclass (all runtime options)
        cameras.py              webcam discovery / opening
        camera_selection.py     interactive camera prompt
        estimator.py            PoseEstimator: YOLO -> Person records + overlay
        person.py               Person dataclass
        skeleton.py             COCO-17 keypoints and ANGLE_DEFS
        geometry.py             angle maths
        smoothing.py            AngleSmoother (EMA)
        fps.py                  FpsMeter
        recorder.py             Recorder: fixed-size video output
        analysis/               offline Hyrox technique analysis
            api.py              run_analysis(): one call per analysis (CLI, workers)
            cli.py              hyrox_analyze.py options
            video.py            VideoReader: file frames with real timestamps
            tracking.py         AthleteTracker: follow one person
            body.py             normalised body metrics for the whole clip
            signal.py           gap filling, zero-lag smoothing, rep detection
            rules.py            Rule / DriftRule: thresholds + coaching cues
            pipeline.py         pose pass (cached) + analysis
            render.py           annotated video
            report.py           HTML / CSV / JSON
            stations/           one analyzer per station (skierg.py so far)
            telemetry/          machine data: readers, cleaning, sync, fusion
        worker/                 queue worker for the web app
            schema.sql          analysis_jobs table (the contract with the web app)
            jobs.py             claim / heartbeat / finish jobs in MySQL
            worker.py           Worker: one job from claim to outputs in its folder
            config.py           WORKER_* environment variables
        ui/
            display.py          window, full screen, drawable size
            view.py             ViewState + render_view (composes the screen)
            layout.py           responsive tile/panel placement
            angle_panel.py      the angle read-out panel
            annotations.py      joint-angle labels and tile captions
            help_overlay.py     keyboard shortcut overlay
            toast.py            transient status messages
            text.py             scalable text + translucent box helpers
            theme.py            fonts and colours

## Common changes

* Track another angle: add a row to `ANGLE_DEFS` in `skeleton.py`.
* Add a key binding: add a method to `PoseApp` and register it in `_actions`.
* Restyle: edit `ui/theme.py`.
