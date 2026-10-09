# Pose estimation with joint angles + Hyrox technique analysis

Four tools share one package, `pose_app`, each in its own subpackage:

* `pose_angles.py` (`pose_app/live`): live webcam pose estimation with joint angles.
* `hyrox_analyze.py` (`pose_app/analysis`): offline technique analysis of a recorded station video.
* `python -m pose_app.worker` (`pose_app/worker`): runs the analyses queued by the web app.
* `pm5_log.py` (`pose_app/pm5`): records a Concept2 PM5 over Bluetooth for the force analysis.

## Hyrox technique analysis (recorded video)

    python hyrox_analyze.py clip.mp4 --station skierg
    python hyrox_analyze.py clip.mp4 --station skierg --start 12 --end 250
    python hyrox_analyze.py --list-stations

Writes `<clip>_analysis/` containing:

    report.html      summary, faults with cues, fatigue checks, example frames, per-rep charts
    annotated.mp4    skeleton, rep counter, phase and the cue for each rep
    reps.csv         every metric for every rep
    summary.json     machine-readable summary
    forces.json      the force model frame by frame (with the force analysis, below)
    pose_cache.npz   the pose pass, so re-analysis is instant

Tuning thresholds with a coach (no re-running the model):

    python hyrox_analyze.py --station skierg --print-thresholds > ski.json
    # edit ski.json
    python hyrox_analyze.py clip.mp4 --station skierg --thresholds ski.json --reuse --no-video

The cache records what the pose pass depended on (the video, clip, rotation,
athlete choice and model settings). `--reuse` takes the output folder's cache
whatever model made it, but runs the pass again if the clip or the athlete
choice changed.

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


### Force analysis (SkiErg with a Concept2 PM5)

With the PM5's Bluetooth log of the same piece, the athlete's mass and height
and two clicks of calibration, the analysis follows the force from the cords
into the body. The PM5 gives the tension (its force curve, both cords
together); the video gives the cord's direction and every body segment.

    python pm5_log.py ski.pm5.jsonl          # record the piece (pip install bleak)
    python pm5_log.py ski.pm5.jsonl --check  # strokes, force curves, issues
    python hyrox_analyze.py clip.mp4 --calibration-frame first.jpg   # read pixels from it
    python hyrox_analyze.py clip.mp4 --station skierg --pm5 ski.pm5.jsonl \
        --mass 80 --height 1.80 --sex male --cord-exit 612,88 --scale 402,905,398,602,1.0

`--scale X1,Y1,X2,Y2,M` is a known length filmed in the athlete's plane (a 1 m
stick held at the midline before the piece) and `--cord-exit X,Y` the point
where the cords leave the machine, both in pixels of the first analysed frame
(after `--rotate`). Without `--scale` the image scale comes from the athlete's
height, which is less accurate (the report says so). `--force-setup setup.json`
takes the same as JSON:

    {"athlete": {"mass_kg": 80, "height_m": 1.80, "sex": "male"},
     "calibration": {"cord_exit": [612, 88],
                     "scale": {"points": [[402, 905], [398, 602]], "length_m": 1.0}}}

What it adds:

* annotated.mp4: while the cords pull, the cord force as an arrow at the hands,
  its line of action through the body, and the moment arms from the shoulder and
  the elbow with their moments per side; always the cord, the cord exit and the
  body's centre of mass; "cord 312 N  body weight 41%" in the header.
* forces.json: the force model of every analysed frame, for a page to draw
  over the video (see "forces.json" below).
* report.html: a "Force analysis" section (averages, the mean force curve early
  and late in the piece, a peak-force frame, data checks) and force charts.
  A PM5 log alone, without the setup, already gives a "PM5 force curves"
  section (every stroke's curve over the piece's mean) and, in summary.json,
  every stroke with its curve placed on the video (`pm5`, below).
* per stroke (`f_*` columns in reps.csv and `per_rep` in summary.json):

  | Key | Meaning |
  |---|---|
  | `f_peak_n`, `f_avg_n`, `f_work_j` | PM5 peak and average drive force (N), work per stroke (J) |
  | `f_drive_len_m`, `f_drive_time_s`, `f_payout_m` | PM5 drive length and time; cord payout seen in the video |
  | `f_bw_share` | % of the stroke's work handed to the cords by the body's drop (m g Δh of the centre of mass in the drive / PM5 work) |
  | `f_com_drop_cm`, `f_com_drop_after_cm` | centre-of-mass drop during the drive, and after the force fades |
  | `f_cord_angle` | cord angle from horizontal at peak force (°; 90 = vertical) |
  | `f_friction` | floor friction the feet need to hold the cord's pull at peak force, statically |
  | `f_capacity_n` | static force capacity: the most tension the peak-force posture holds before the pressure under the feet reaches the heel or toe |
  | `f_peak_pos`, `f_catch70_pos` | % of the cord's travel before the force peaks / first reaches 70% of its peak |
  | `f_shoulder_ext_nm`, `f_shoulder_ext_max_nm` | shoulder extension moment per side at peak force, and the drive's highest |
  | `f_elbow_flex_nm` | elbow moment per side at peak force (+ flexion, - extension) |
  | `f_shoulder_arm_cm`, `f_elbow_arm_cm` | distance from the joint to the cord's line at peak force |
  | `f_cord_work_j`, `f_energy_err` | work of the mapped curve on the video's payout; joint work vs PM5 work (%) |
  | `f_t_peak`, `frame_force_peak` | when the force peaked (video seconds / source frame) |

  "At peak force" is a tension-weighted average over the frames above 80% of
  the stroke's peak, so a value does not hinge on one frame.
* rules (minor, tunable like the others): `low_bodyweight_share` (< 25%),
  `shallow_cord` (< 65°, standing too far from the machine), `late_force_peak`
  (> 45% of the travel); fatigue checks `drift_bodyweight` and
  `drift_peak_force`. The thresholds come from the synthetic reference stroke of
  the requirement analysis; tune them with a coach.

How it works (`pose_app/analysis/force`): keypoints are filtered with a
zero-lag 4th-order Butterworth at 6 Hz and put in metres (x towards the
machine, y up, origin on the floor under the static ankle). Segments follow
de Leva's (1996) masses, centres of mass and radii of gyration by sex (forearm
and hand as one segment, the grip a hand length beyond the wrist, the head's
centre of mass at the ear, the feet static). Each video stroke is paired with a
PM5 stroke by the drive ends and drive times (to 0.01 s on the PM5's own
clock); the pairing's confidence is reported, and force is only used when it
is clear. Curves spaced by handle travel are placed by the fraction of the cord
payout reached; curves spaced in time keep the PM5's drive time, centred in the
payout window. Newton-Euler inverse dynamics, from the hands to the floor, then
gives the joint moments; over each stroke the joints' work must match the
PM5's work (the energy check).

Hip, knee and ankle moments and the floor reaction are computed but not
reported: one side-on camera does not measure them well enough until a
validation study (handle load cells, balance board) says otherwise. Moments are
technique measures, not injury risk.

Checks the report raises: strokes the PM5 could not be matched to, the video's
cord payout against the PM5's drive length (calibration), the curves' work
against the PM5's, the energy check, the stick against the athlete's height,
and keypoint jitter at the (static) ankle.

Capture protocol: tripod side-on and square to the athlete at hip height, 3-4
m away, the whole body and the cord exit in frame, 1080p at 30 fps or more
(60 preferred); film the 1 m stick at the midline first; log the PM5 for the
whole piece (force curves need a PM5 from late 2016 or later, hardware 600+).

Recording the PM5: `pip install bleak`, close ErgData (the PM5 takes one app
connection), wake the monitor, run `python pm5_log.py LOG` and row; it stops
when the piece ends on the monitor, after 10 minutes without strokes, or on
Ctrl+C. The log keeps every notification as it arrived (JSON lines with `t`,
`uuid`, `hex`), so it can be re-read by a better parser later. The raw logs of
the open-source [pm5-force-logger](https://github.com/jonstraveladventures/pm5-force-logger)
(its `raw/<start>.jsonl`, including its in-browser recorder) are the same
format and work too. A PM5 log also serves as the machine data (no Track My
Indoor Workout CSV needed), synchronised stroke by stroke; `--telemetry
ski.pm5.jsonl` reads it as machine data alone.

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
    outcome.files       # {"report": ..., "summary": ..., "reps": ..., "video": ..., "pose_cache": ...,
                        #  "forces": ... (with the force analysis)}
    outcome.warnings    # messages for the athlete / coach

It never prints. Status lines go to the `pose_app` logger, and unusable inputs
(unreadable video, bad telemetry, empty clip, unknown station) raise
`AnalysisError`, whose message is safe to show to the athlete or coach. Any
other exception is a bug or an environment problem.

### Queue worker (for the web app)

The web app (Spring) owns the job queue and its database. The worker never
touches the database: it is a plain HTTP client of the web app's internal
API, and the two share only the media folder, where the web app saves the
uploads and the worker writes its outputs next to them:

    POST /internal/jobs/claim --> run_analysis() on <media root>/<inputs.video>
        -> annotated.mp4, pose_cache.npz (and forces.json with the force analysis)
           written in <media root>/<output_dir>
        -> POST /internal/jobs/{id}/complete with the report (summary.json) / .../fail

The complete call lists the files in `outputs`: `annotated_video`,
`pose_cache` and, with the force analysis, `force_frames`. A run that does not
write forces.json removes the one an earlier run left, so the folder never
pairs new results with an old force model; a failed run leaves every file as
it was.

A job run again on the same uploads, as when the athlete adds the force
analysis' setup, reuses the last run's pose_cache.npz if the video, clip,
rotation, athlete choice and the worker's model settings are the same, so it
skips straight to the analysis and the annotated video.

    pip install -r requirements-worker.txt
    export WORKER_API_URL=http://web:8081      # the web app's internal API port
    export WORKER_API_TOKEN=...                # the web app's WORKER_API_TOKEN
    export WORKER_MEDIA_ROOT=/srv/media        # where the web app's media folder is mounted
    python -m pose_app.worker

The settings can also go in a `.env` file (same `NAME=value` lines) in the
folder you start the worker from, or a parent folder; variables already set
in the environment take precedence. `.env` is git-ignored.

* API contract: `pose_app/worker/jobs.py`. The worker codes against the
  `JobClient` interface (`claim_next`, `progress`, `complete`, `fail`);
  `HttpJobClient` implements it over HTTP with a shared token. Moving to a
  message broker later means a second implementation; the worker loop and
  `run_analysis()` stay as they are.
* A claimed job names its files by key, a path relative to the media root
  (e.g. `inputs.video = alice/<uuid>/video.mp4`, `output_dir = alice/<uuid>`).
  Each side sets its own root, so the web app and the worker can mount the
  disk in different places. Uploads must not use the output names above.
* Outputs appear all at once: the worker writes them in a hidden
  `.analysis-tmp-*` folder inside `output_dir` and moves them in at the end,
  so a failed or cancelled job leaves nothing behind. The worker needs write
  access to `output_dir`, and its files must be readable by the web app (same
  user, or a shared group with a suitable umask). The summary is not written
  there: it is sent as the report, and the web app stores it after checking it.
* Job options: clip start/end, rotation, which athlete to follow, telemetry
  offset, threshold overrides, no video, and the force analysis' setup
  (`force`, the JSON shown under "Force analysis"); see `JOB_OPTIONS` in
  `pose_app/worker/options.py`. The model and hardware are the worker's
  settings, not per job.
* Inputs: `inputs.video`, and optionally `inputs.telemetry` (machine data CSV)
  and `inputs.pm5` (PM5 Bluetooth log, needed by the `force` option; without a
  telemetry CSV it is also the machine data). An unreadable PM5 log fails the
  job with `unreadable_telemetry`.
* While running, the worker reports `stage` (pose, video) and `progress`
  every `WORKER_HEARTBEAT_SECONDS`; that is also its heartbeat. The web app
  requeues a job that stays silent too long (`hyrox.jobs.stale-after`).
* Failures: an unusable upload is failed with its `code` (see `ERROR_CODES`
  in `pose_app/analysis/api.py`) and `retryable: false`; the worker's own
  errors are `retryable: true`, and the web app decides whether to run the
  job again (`hyrox.jobs.max-attempts`). A `409` answer means the job was
  cancelled or given to another worker: the worker stops and drops it.
* If the web app is restarting, calls are retried with backoff for
  `WORKER_API_RETRY_SECONDS` (60 by default), so the worker does not notice.
* Several workers can run at once; the web app hands each job to one of them.
* Other settings (`pose_app/worker/config.py`): `WORKER_MODEL`, `WORKER_IMGSZ`,
  `WORKER_DEVICE`, `WORKER_POLL_SECONDS`, `WORKER_API_TIMEOUT_SECONDS`.
* Logging: `WORKER_LOG_LEVEL` (`INFO` by default) and `WORKER_LOG_FILE`
  (unset: stderr, which suits a container or systemd). The file rolls over
  at `WORKER_LOG_MAX_MB` (10) keeping `WORKER_LOG_BACKUPS` (5) old files;
  give each worker on a machine its own file, as rotation is per process.
  `WORKER_LOG_FORMAT` and `WORKER_LOG_DATEFMT` override the line and
  timestamp format. In a
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
* `force` (null without a force analysis): athlete, calibration (scale and its
  source, cord exit in pixels and metres), PM5 log summary and curve spacing,
  stroke `sync` (method, confidence, offset, matched strokes), `summary`
  (session averages of the `f_*` metrics, energy check, keypoint jitter),
  `curve` (mean force in N over the drive: `x` 0-1 with `all`, `early` and
  `late`), `reported_joints`, an `example` peak-force frame and `checks`.
  Force rules and charts appear in `rules` and `charts` only when it ran.
* `pm5` (null without a PM5 log; no calibration needed): every stroke the PM5
  logged, so a page can show each stroke's force curve. `curve_spacing` (time or
  travel), `curves` (strokes with one), `sync` (method, confidence, offset,
  `matched`, `events` hands or cord, and `linked`: whether strokes carry video
  times), `mean_curve` (as `force.curve`, over the whole log) and `notes`.
  Each of `strokes` has `n` (1-based, whole log), `piece`, `count` (the PM5's
  stroke number in the piece), `rep` (the video rep it was matched to, or
  null), `t_start` / `t_end` (the PM5's drive on the uploaded video's clock,
  null unless `linked`; within about 0.1 s on the synthetic test sessions),
  `distance_m`, `drive_time_s`, `drive_length_m`,
  `recovery_time_s`, `peak_n`, `avg_n`, `work_j`, `power_w`, and `curve`: the
  force in whole N from the first force to the release, padded with a zero at
  each end and evenly spaced by `spacing` (null when the curve was lost),
  with `peak_pct`, where along it the force peaks (0-100).

### forces.json

The force model of every analysed frame (`pose_app/analysis/force/frames.py`),
for a page to draw over the uploaded video as it plays. Positions are in the
analysed frame's pixels (the uploaded video as displayed: x right, y down),
forces in newtons in the same directions, both sides of the body together;
pairs are flattened `[x0, y0, x1, y1, ...]` and a missing value is null.

* Once: `version` (1), `fps`, `frame_size`, `px_per_m`, `facing`, `mass_kg`,
  `weight_n`, `cord_exit`, `floor_y`, `fixed_points` (ankle, heel and toe: the
  feet are flat and still in the model), `feet` (mass and centre of mass),
  `segments` (name, the points it joins, its mass, its centre of mass as a
  fraction from `from` to `to`, and the points to draw it between),
  `joints` (elbow, shoulder, hip, knee, ankle), `reported` (elbow and
  shoulder) and `moment_names` (the anatomical name of a positive and a
  negative moment at each joint).
* Per frame: `t` (seconds in the uploaded video), `points` (grip, wrist, elbow,
  shoulder, ear, hip, knee), `body_com`, `tension` (the cords' pull on the
  hands along the grip -> cord exit line; 0 between drives), `passed` (at each
  joint, the force the body above it passes on to the body below it, towards
  the floor), `floor` (the floor's force on the feet) and `cop_x` (where it
  acts on the floor line), and `moment` (per side, positive as the first of
  `moment_names`).

The hip, knee and ankle values and the floor reaction are estimates, as in
the report: one side-on camera does not measure them well enough until a
validation study says otherwise. Their sum checks out: the still feet take
what the ankle passes on, their weight and the floor's force, to within a
newton. About 190 bytes a frame: some 1.4 MB for four minutes at 30 fps.

Times: `t` / `t_start` are seconds in the uploaded video; `video_t*` are
seconds in annotated.mp4 (which starts at the clip start), so the web app can
seek the player straight to a rep or a fault.

Regression tests (no model needed): `python tests/test_skierg.py`,
`python tests/test_telemetry.py`, `python tests/test_api.py`,
`python tests/test_pm5.py`, `python tests/test_force.py` (a synthetic athlete
with known physics and its PM5 log; see `tests/synthetic_force.py`) and
`python tests/test_worker.py` (against an in-memory fake of the web app's API).

Real-time webcam pose estimation (Ultralytics YOLO) with a responsive,
full-screen-capable view: original feed | pose overlay | joint-angle panel.

    pip install -r requirements.txt
    python pose_angles.py            # or: python -m pose_app.live (or pose_app)
    python pose_angles.py --help     # all options

## Benchmarking pose models

`tools/bench_pose.py` compares pose models on real videos before a model,
crop or hosting change: YOLO11 against YOLO26, CPU against GPU, PyTorch
against ONNX or OpenVINO, the full frame against a crop around the athlete,
and every frame against every second frame. It needs no ground truth.

    python tools/bench_pose.py videos/*.mp4 --device cpu
    python tools/bench_pose.py session.mp4 \
        --models yolo11m-pose.pt yolo26s-pose.pt yolo26m-pose.pt \
        --reference yolo26x-pose.pt --formats pt openvino --max-seconds 90 --save-keypoints

For each run it reports milliseconds per frame, the projected pose-pass time
for a 4.5-minute piece, detection rate, keypoint noise (corrected for frame
rate), the noise left after the force model's own filter, and agreement with
the reference run (the largest model on every full frame). Results go to
`bench_results/results.csv` and `results.json`; `--save-keypoints` also keeps
each run's keypoints as `.npz`. Defaults match the worker (960 px input,
6 Hz filter). `python tools/bench_pose.py --help` lists every option.

## Project layout

    pose_angles.py              live webcam app entry point
    hyrox_analyze.py            offline analysis entry point
    pm5_log.py                  PM5 Bluetooth recorder entry point
    tests/                      synthetic regression tests
    tools/
        bench_pose.py           pose model benchmark: speed, jitter, agreement
    pose_app/
        estimator.py            shared: PoseEstimator: YOLO -> Person records + overlay
        person.py               shared: Person dataclass
        skeleton.py             shared: COCO-17 keypoints and ANGLE_DEFS
        geometry.py             shared: angle maths
        drawing/                shared: OpenCV drawing helpers
            text.py             scalable text + translucent box helpers
            theme.py            fonts and colours
        live/                   live webcam app
            cli.py              parse options, pick the camera, start the app
            app.py              PoseApp: main loop + keyboard actions
            config.py           Settings dataclass (all runtime options)
            cameras.py          webcam discovery / opening
            camera_selection.py interactive camera prompt
            smoothing.py        AngleSmoother (EMA)
            fps.py              FpsMeter
            recorder.py         Recorder: fixed-size video output
            ui/
                display.py      window, full screen, drawable size
                view.py         ViewState + render_view (composes the screen)
                layout.py       responsive tile/panel placement
                angle_panel.py  the angle read-out panel
                annotations.py  joint-angle labels and tile captions
                help_overlay.py keyboard shortcut overlay
                toast.py        transient status messages
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
            force/              force analysis with a PM5 log (SkiErg)
                setup.py        athlete profile + calibration
                model.py        de Leva body segments
                kinematics.py   zero-lag filter, pixels -> metres, segment motion
                dynamics.py     joint moments and forces, power, energy, force capacity
                sync.py         PM5 strokes <-> video strokes, curves onto frames
                analysis.py     the chain + per-stroke f_* metrics and checks
                rules.py        force rules, fatigue checks, charts
                overlay.py      force lines on the annotated video
                frames.py       the force model frame by frame (forces.json)
        pm5/                    Concept2 PM5 over Bluetooth
            protocol.py         UUIDs, byte layouts, force-curve packets
            log.py              raw log (JSON lines) reader / writer
            session.py          raw log -> strokes with force curves
            recorder.py         Bluetooth recorder (bleak)
            cli.py              pm5_log.py options
        worker/                 queue worker for the web app
            jobs.py             JobClient interface + HttpJobClient (the web app's API)
            worker.py           Worker: one job from claim to outputs and report
            options.py          JOB_OPTIONS: a job's options -> AnalysisOptions
            config.py           WORKER_* environment variables

## Common changes

* Track another angle: add a row to `ANGLE_DEFS` in `skeleton.py`.
* Add a key binding: add a method to `PoseApp` and register it in `_actions`.
* Restyle: edit `drawing/theme.py`.
