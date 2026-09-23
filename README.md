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

If other people are in shot, the largest person is followed by default; use
`--athlete center` or `--athlete-point X,Y` to pick someone else. Offline
defaults favour accuracy (`yolo11m-pose.pt`, `--imgsz 960`); on a CPU-only
machine use `--model yolo11s-pose.pt --imgsz 640` for speed.

### Adding a station

Subclass `Station` in `pose_app/analysis/stations/`, set the driver metric
whose peak -> trough -> peak is one rep, implement `summarize()` (one rep ->
dict of numbers) and `rules()`, then register it in `stations/__init__.py`.
Body metrics available per frame are in `analysis/body.py`.

Regression test (no model needed): `python tests/test_skierg.py`.

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
