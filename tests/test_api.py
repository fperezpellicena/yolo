"""run_analysis(): the entry point the CLI and a job worker share.

Run: python tests/test_api.py (or pytest). No model or GPU needed.
"""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_skierg import FakeEstimator, KINDS, STROKE_STARTS, period_of, write_video
from synthetic_telemetry import make_tmiw_csv
from pose_app.analysis.api import (AnalysisError, AnalysisOptions, default_thresholds,
                                   run_analysis)
from pose_app.analysis.cli import main


def check_summary(s):
    """summary.json carries everything the web app needs to draw the report."""
    assert s["schema_version"] == 1
    assert s["station"]["name"] == "SkiErg" and s["station"]["filming_tips"]
    rules = {r["id"]: r for r in s["rules"]}
    assert rules["early_arm_pull"]["threshold"] == 0.2          # the override, not the default
    assert {"title", "severity", "metric", "op", "cue"} <= set(rules["early_arm_pull"])
    assert s["charts"] and all(c["label"] for c in s["charts"])
    assert {e["rule"] for e in s["examples"]} == {f["rule"] for f in s["faults"]}
    reps = {r["n"]: r for r in s["per_rep"]}
    for e in s["examples"]:
        rep = reps[e["rep"]]
        assert e["rule"] in {f["rule"] for f in rep["faults"]}
        assert rep["t_start"] <= e["t"] <= rep["t_start"] + rep["duration_s"], (e, rep)
        assert rep["video_t_start"] <= e["video_t"] <= rep["video_t_end"], e
        assert abs(e["t"] - e["video_t"]) < 0.05          # constant frame rate clip from 0 s
    assert all(isinstance(f["value"], float) for r in s["per_rep"] for f in r["faults"])
    assert s["pm5"] is None and s["force"] is None          # no PM5 log given


def test_run_analysis_writes_everything():
    with tempfile.TemporaryDirectory() as d:
        video = os.path.join(d, "ski.mp4")
        write_video(video)
        out = os.path.join(d, "out")
        seen = []
        opts = AnalysisOptions(model="fake", thresholds={"no_such_rule": 1.0,
                                                         "early_arm_pull": 0.2})
        outcome = run_analysis(video, out, opts, FakeEstimator(),
                               lambda stage, done, total: seen.append(stage))

        assert len(outcome.result.reps) == len(KINDS)
        assert set(outcome.files) == {"report", "summary", "reps", "video", "pose_cache"}
        assert all(os.path.getsize(p) > 0 for p in outcome.files.values())
        assert {"pose", "video"} <= set(seen)
        assert "no_such_rule" in outcome.warnings[0]
        with open(outcome.files["summary"]) as fh:
            check_summary(json.load(fh))

        # Reuse the cached pose pass without any estimator, as the CLI does.
        again = run_analysis(video, out, AnalysisOptions(reuse=True, video=False))
        assert len(again.result.reps) == len(KINDS) and "video" not in again.files
        bare = run_analysis(video, os.path.join(d, "bare"),
                            AnalysisOptions(video=False, report=False, csv=False),
                            FakeEstimator())
        assert set(bare.files) == {"pose_cache", "summary"}
        assert sorted(os.listdir(os.path.join(d, "bare"))) == ["pose_cache.npz", "summary.json"]
        assert main([video, "--out", out, "--reuse", "--no-video"]) == 0


def test_run_analysis_with_telemetry():
    with tempfile.TemporaryDirectory() as d:
        video = os.path.join(d, "ski.mp4")
        write_video(video)
        csv = os.path.join(d, "t.csv")
        make_tmiw_csv(csv, STROKE_STARTS, [int(period_of(n) * 30) / 30 for n in range(len(KINDS))],
                      KINDS, log_start=STROKE_STARTS[0] - 0.3)
        outcome = run_analysis(video, os.path.join(d, "out"),
                               AnalysisOptions(telemetry=csv, video=False), FakeEstimator())
        assert outcome.result.machine is not None


def test_bad_inputs_raise_analysis_error():
    with tempfile.TemporaryDirectory() as d:
        video = os.path.join(d, "ski.mp4")
        write_video(video)
        bad_csv = os.path.join(d, "bad.csv")
        with open(bad_csv, "w") as fh:
            fh.write("not,telemetry\n1,2\n")
        cases = [
            (os.path.join(d, "missing.mp4"), AnalysisOptions(), "unreadable_video"),
            (video, AnalysisOptions(station="no_such_station"), "unsupported_station"),
            (video, AnalysisOptions(telemetry=bad_csv), "unreadable_telemetry"),
            (video, AnalysisOptions(start=10_000), "empty_clip"),
            (video, AnalysisOptions(rotate=45), "invalid_job"),
        ]
        for path, opts, code in cases:
            try:
                run_analysis(path, os.path.join(d, "out"), opts, FakeEstimator())
            except AnalysisError as exc:
                assert exc.code == code, (opts, exc.code, exc)
                continue
            raise AssertionError(f"no AnalysisError for {opts}")
    assert "early_arm_pull" in default_thresholds("skierg")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
