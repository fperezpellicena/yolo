"""A job's options (JSON from the web app) turned into AnalysisOptions; no model,
video or web app needed. Run: python tests/test_options.py (or pytest).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pose_app.analysis.api import AnalysisError
from pose_app.worker.config import WorkerConfig
from pose_app.worker.jobs import Job
from pose_app.worker.options import build_analysis_options

CFG = WorkerConfig("http://web:8081", "s3cret", "/media", model="m.pt", imgsz=640, device="cpu")


def _options(**options):
    job = Job(id=1, station="skierg", video="alice/1/video.mp4", telemetry=None,
              output_dir="alice/1", options=options, attempt=1)
    return build_analysis_options(job, CFG)


def _rejected(**options) -> AnalysisError:
    try:
        _options(**options)
    except AnalysisError as exc:
        assert exc.code == "invalid_job"
        return exc
    raise AssertionError(f"accepted {options}")


def test_no_options_use_the_job_and_worker_settings():
    opts = _options()
    assert (opts.station, opts.model, opts.imgsz, opts.device) == ("skierg", "m.pt", 640, "cpu")
    assert opts.report is False and opts.csv is False     # the web app builds its own report
    assert opts.video is True and opts.rotate == 0 and opts.athlete == "largest"


def test_values_are_converted_to_what_the_analysis_expects():
    opts = _options(start=1, end="12.5", telemetry_offset=-3, rotate=90.0, athlete="center",
                    athlete_point=[640, 360], thresholds={"catch_angle": 95}, video=False)
    assert (opts.start, opts.end, opts.telemetry_offset) == (1.0, 12.5, -3.0)
    assert opts.rotate == 90 and isinstance(opts.rotate, int)
    assert opts.athlete == "center"
    assert opts.athlete_point == (640.0, 360.0)
    assert opts.thresholds == {"catch_angle": 95.0}
    assert opts.video is False


def test_null_options_keep_the_defaults():
    opts = _options(start=None, end=None, rotate=None, athlete_point=None, video=None)
    assert (opts.start, opts.end, opts.rotate, opts.athlete_point) == (0.0, None, 0, None)
    assert opts.video is True


def test_unknown_options_are_rejected():
    exc = _rejected(start=2, speed=2, model="other.pt")
    assert "['model', 'speed']" in str(exc)


def test_invalid_values_are_rejected():
    for key, value in [("start", "soon"), ("end", [10]), ("telemetry_offset", {}),
                       ("rotate", 45), ("rotate", "upside down"),
                       ("athlete", "tallest"), ("athlete", 1),
                       ("athlete_point", [1]), ("athlete_point", [1, 2, 3]),
                       ("athlete_point", 5), ("athlete_point", ["x", 2]),
                       ("thresholds", [95]), ("thresholds", {"catch_angle": "steep"}),
                       ("video", "false"), ("video", 0)]:
        _rejected(**{key: value})


def test_athlete_point_must_be_a_list():
    # A two-character string has length 2 too; it once passed as the point (1.0, 2.0).
    for value in ["12", {"x": 1, "y": 2}]:
        _rejected(athlete_point=value)


def test_the_error_says_what_was_expected():
    exc = _rejected(rotate=45)
    assert str(exc) == "Invalid job option rotate=45: expected 0, 90, 180 or 270."


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
