"""Regression test: synthetic SkiErg clip with known faults.

Strokes 1-8 good, 9-14 early arm pull, 15-20 squat pattern; the rate fades
throughout and a bystander is in the background. Run: python tests/test_skierg.py
(or pytest). No model or GPU needed.
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_skierg import FakeEstimator, KINDS, write_video
from pose_app.analysis.pipeline import Extraction, analyze, extract
from pose_app.analysis.report import write_csv, write_html, write_json
from pose_app.analysis.stations import STATIONS
from pose_app.analysis.tracking import AthleteTracker
from pose_app.analysis.video import VideoReader


def test_skierg_pipeline():
    with tempfile.TemporaryDirectory() as d:
        video = os.path.join(d, "ski.mp4")
        write_video(video)
        reader = VideoReader(video)
        ex = extract(reader, FakeEstimator().detect,
                     AthleteTracker("largest", frame_size=(960, 720)), "fake", progress=False)
        ex.save(os.path.join(d, "cache.npz"))
        ex = Extraction.load(os.path.join(d, "cache.npz"))
        res = analyze(ex, STATIONS["skierg"])

        assert res.body.side == "left" and res.body.facing == 1
        assert len(res.reps) == len(KINDS)
        for rep, kind in zip(res.reps, KINDS):
            ids = {f.rule_id for f in rep.faults}
            assert ("early_arm_pull" in ids) == (kind == "arms"), (rep.number, ids)
            assert ("squat_pattern" in ids) == (kind == "squat"), (rep.number, ids)
        drift = {x.rule_id: x.triggered for x in res.drift}
        assert drift["drift_rate"] and drift["drift_squat"]

        write_csv(res, os.path.join(d, "reps.csv"))
        write_json(res, os.path.join(d, "summary.json"))
        write_html(res, os.path.join(d, "report.html"), reader)


if __name__ == "__main__":
    test_skierg_pipeline()
    print("ok")
