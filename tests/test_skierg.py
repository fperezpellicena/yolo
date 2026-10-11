"""Regression test: synthetic SkiErg clip with known faults.

Strokes 1-8 good, 9-14 early arm pull, 15-20 squat pattern; the rate fades
throughout and a bystander is in the background. Run: python tests/test_skierg.py
(or pytest). No model or GPU needed.
"""
import math, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from synthetic_skierg import FakeEstimator, KINDS, write_video
from pose_app.analysis.pipeline import Extraction, analyze, extract
from pose_app.analysis.report import write_csv, write_html, write_json
from pose_app.analysis.rules import DriftRule, evaluate_drift
from pose_app.analysis.stations import STATIONS
from pose_app.analysis.stations.skierg import arm_timing
from pose_app.analysis.tracking import AthleteTracker
from pose_app.analysis.video import VideoReader


def test_skierg_pipeline():
    with tempfile.TemporaryDirectory() as d:
        video = os.path.join(d, "ski.mp4")
        write_video(video)
        reader = VideoReader(video)
        ex = extract(reader, FakeEstimator().detect,
                     AthleteTracker("largest", frame_size=(960, 720)), "fake")
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


def _drive(fps, hip_start, elbow_start, elbow_range=60.0, n=0.6):
    """Hip and elbow over one drive: each eases over 0.3 s from its own start time."""
    t = np.arange(0, n, 1 / fps)
    def ease(t0, a, b):
        u = np.clip((t - t0) / 0.3, 0, 1)
        return a + (b - a) * (1 - np.cos(np.pi * u)) / 2
    return t, ease(hip_start, 170.0, 80.0), ease(elbow_start, 165.0, 165.0 - elbow_range)


def test_arm_timing_measures_who_starts_first():
    for fps in (25, 60):
        for lead in (-0.08, 0.0, 0.06):             # + = elbows first, seconds
            t, hip, elbow = _drive(fps, 0.1, 0.1 - lead)
            m = arm_timing(t, hip, elbow, 0, len(t) - 1, len(t) - 1)
            # both joints follow the same curve, so the onsets differ by exactly `lead`;
            # interpolation keeps 25 fps within a few ms, not 40 ms steps
            assert abs(m["arm_lead_ms"] - 1000 * lead) < 6, (fps, lead, m)
        t, hip, elbow = _drive(fps, 0.1, 0.1)
        m = arm_timing(t, hip, elbow, 0, len(t) - 1, len(t) - 1)
        assert abs(m["arm_bend_at_half_hinge"] - 50) < 3, m     # halfway together
        assert hip[int(m["hip_onset"])] <= 170 - 0.25 * 90 + 1e-9
    t, hip, elbow = _drive(60, 0.1, 0.1, elbow_range=5.0)           # arms barely bend
    m = arm_timing(t, hip, elbow, 0, len(t) - 1, len(t) - 1)
    assert math.isnan(m["arm_lead_ms"]) and math.isnan(m["arm_bend_at_half_hinge"])


def test_drift_is_not_swung_by_one_odd_rep():
    rule = DriftRule("drift_x", "x", "increase", 10, "x grows", "cue")
    steady = [{"x": 0.0}] * 12
    first_from_standing = [{"x": -300.0}] + steady[1:]       # the first pull from rest
    assert not evaluate_drift([rule], first_from_standing)[0].triggered
    fading = [{"x": 0.0}] * 4 + [{"x": 5.0}] * 4 + [{"x": 20.0}] * 4
    assert evaluate_drift([rule], fading)[0].triggered


if __name__ == "__main__":
    test_skierg_pipeline()
    test_arm_timing_measures_who_starts_first()
    test_drift_is_not_swung_by_one_odd_rep()
    print("ok")
