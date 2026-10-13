"""Heel lift in the SkiErg technique analysis (body-with-feet models), without a force setup.

The synthetic session (synthetic_force.py) lifts the heel 6 cm at chosen strokes: late in
the recovery (rising onto the toes for the next catch) or at the finish (tipping onto the
toes), up and down over about half a second. Run: python tests/test_feet.py (or pytest). No model or GPU needed.
"""
import json, math, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import synthetic_force as sf
from pose_app.analysis import feet
from pose_app.analysis.api import AnalysisOptions, run_analysis
from pose_app.analysis.stations.skierg import heel_windows

FEET = {"heel": 0.025, "toe": 0.100}
WIDTH = 0.15                      # of a stroke either side of the lift: ~0.5 s up and down
_state = {}


def _video():
    if "video" not in _state:
        _state["dir"] = tempfile.mkdtemp()
        _state["session"] = sf.Session(n_strokes=16)
        _state["video"] = os.path.join(_state["dir"], "ski.mp4")
        _state["session"].write_video(_state["video"])
    return _state["session"], _state["video"]


def _run(out, kp, sc, **opts):
    _, video = _video()
    return run_analysis(video, os.path.join(_state["dir"], out),
                        AnalysisOptions(model="fake", video=False, **opts), sf.FakeEstimator(kp, sc))


def _lift_strokes(res, session, strokes, at):
    """1-based numbers of the video strokes whose catch or finish window holds each lift."""
    out, t = set(), res.body.t
    for k in strokes:
        x = float(np.interp(k + at, session.phase_fine, session.t_fine))
        for r in res.reps:
            catch, finish = heel_windows(r.cycle.start, r.cycle.mid, r.cycle.end)
            (a, b) = catch if at > sf.DRIVE + 0.2 else finish
            if t[a] <= x <= t[b]:
                out.add(r.number)
    return out


def test_heel_lift_is_the_heel_over_the_toe_beyond_flat():
    n = 100
    toe = np.tile([500.0, 700.0], (n, 1))
    heel = np.tile([440.0, 702.0], (n, 1))                # flat: 2 px below the toe point
    heel[40:50, 1] -= 30                                   # 30 px up
    lift = feet.heel_lift(heel, toe, 300.0)
    assert np.allclose(lift[:40], 0) and np.allclose(lift[40:50], 10.0)   # 30 px at 300 px/m
    knee, ankle = np.tile([0.0, 0.0], (n, 1)), np.tile([0.0, 0.246 * 1.75 * 300], (n, 1))
    assert abs(feet.px_per_m(knee, ankle) - 300.0) < 1e-6              # nominal 1.75 m
    assert abs(feet.px_per_m(knee, ankle, 1.60) - 300.0 * 1.75 / 1.60) < 1e-6
    assert np.isnan(feet.heel_lift(np.full((n, 2), np.nan), toe, 300.0)).all()
    toe[60:, 0] += 30                                      # a 10 cm step
    assert feet.travel_cm(toe, 0, 99, 300.0) > feet.STEP_CM > feet.travel_cm(toe, 0, 50, 300.0)


def test_catch_and_finish_lifts_are_told_apart():
    session, _ = _video()
    catch_k, finish_k = [3, 4], [8, 9]
    kp, sc = session.keypoints(feet=FEET, heel_lift={"strokes": catch_k, "height_m": 0.06,
                                                     "width": WIDTH}, seed=3)
    kp2, _ = session.keypoints(feet=FEET, heel_lift={"strokes": finish_k, "height_m": 0.06,
                                                     "at": sf.DRIVE, "width": WIDTH}, seed=3)
    for j in ("left_heel", "right_heel"):                 # both lifts in one clip
        i = sf.KP[j]
        kp[:, i, 1] = np.minimum(kp[:, i, 1], kp2[:, i, 1])
    out = _run("both", kp, sc)
    res = out.result
    at_catch = _lift_strokes(res, session, catch_k, 0.9)
    at_finish = _lift_strokes(res, session, finish_k, sf.DRIVE)
    assert len(at_catch) == 2 and len(at_finish) == 2 and not at_catch & at_finish
    last = len(res.reps)
    for r in res.reps:
        c, f = r.metrics["heel_lift_catch"], r.metrics["heel_lift_finish"]
        ids = {x.rule_id for x in r.faults}
        if r.number in (1, last):                         # stepping on and off: not judged
            assert math.isnan(c) and math.isnan(f) and "heels_up_finish" not in ids
            continue
        assert (c > feet.LIFT_CM) == (r.number in at_catch), (r.number, c)
        assert (f > feet.LIFT_CM) == (r.number in at_finish), (r.number, f)
        assert ("heels_up_finish" in ids) == (r.number in at_finish), (r.number, ids)
        if r.number in at_catch | at_finish:              # 5.8 cm at the nominal 1.75 m; pixel
            assert 5.2 < max(c, f) < 6.6, (r.number, c, f)  # noise and a missed frame: +-0.5
        else:
            assert max(c, f) < 2.0, (r.number, c, f)
    with open(out.files["summary"]) as fh:
        s = json.load(fh)
    charts = {c["metric"] for c in s["charts"]}
    assert {"heel_lift_catch", "heel_lift_finish"} <= charts
    assert any(f["rule"] == "heels_up_finish" and f["count"] == 2 for f in s["faults"])
    assert not any("step" in w for w in res.warnings)     # no step mid-set


def test_the_athletes_height_scales_the_lift():
    session, _ = _video()
    kp, sc = session.keypoints(feet=FEET, heel_lift={"strokes": [4], "height_m": 0.06,
                                                     "width": WIDTH}, seed=3)
    nominal = _run("nominal", kp, sc).result
    setup = session.setup()                                # 1.80 m athlete: the true scale
    told = _run("told", kp, sc, force=setup).result
    a = max(r.metrics["heel_lift_catch"] for r in nominal.reps[1:-1])
    b = max(r.metrics["heel_lift_catch"] for r in told.reps[1:-1])
    assert abs(a / b - 1.75 / 1.80) < 0.01, (a, b)       # shorter: more px per m, fewer cm


def test_a_step_mid_set_is_not_judged():
    session, _ = _video()
    kp, sc = session.keypoints(feet=FEET, step={"stroke": 7, "back_m": 0.15}, seed=3)
    res = _run("step", kp, sc).result
    stepped = [r.number for r in res.reps if math.isnan(r.metrics["heel_lift_catch"])]
    assert stepped[0] == 1 and stepped[-1] == len(res.reps) and len(stepped) == 3, stepped
    note = next(w for w in res.warnings if w.startswith("The feet moved (a step)"))
    assert f"stroke {stepped[1]}," in note


def test_a_drifting_phone_is_not_a_step():
    """A short clip whose camera drifts ~8 cm per stroke: measured on the background in the
    pose pass, the drift is taken out and no stroke looks like a step. Without the camera's
    motion (a pose cache from before it was measured), every stroke would."""
    from pose_app.analysis.pipeline import Extraction, analyze
    from pose_app.analysis.stations import STATIONS
    session = sf.Session(n_strokes=5)
    drift = (115.0, 105.0)                                 # as far as the synthetic room allows
    d = tempfile.mkdtemp()
    video = os.path.join(d, "drift.mp4")
    session.write_video(video, drift=drift, texture=True)
    kp, sc = session.keypoints(feet=FEET, drift=drift, seed=3)
    out = run_analysis(video, os.path.join(d, "out"), AnalysisOptions(model="fake", video=False),
                       sf.FakeEstimator(kp, sc))
    res = out.result
    assert res.body.camera is not None
    assert not any(w.startswith("The feet moved") for w in res.warnings), res.warnings
    middle = [r.metrics["heel_lift_catch"] for r in res.reps[1:-1]]
    assert len(middle) >= 2 and all(math.isfinite(v) and v < 2.0 for v in middle), middle
    ex = Extraction.load(out.files["pose_cache"])
    ex.camera = None
    blind = analyze(ex, STATIONS["skierg"])
    assert any(w.startswith("The feet moved") for w in blind.warnings)


def test_17_point_models_have_no_heel_metrics():
    session, _ = _video()
    kp, sc = session.keypoints(seed=3)                     # no feet
    out = _run("plain", kp, sc)
    assert all(math.isnan(r.metrics["heel_lift_catch"]) for r in out.result.reps)
    assert not any(w.startswith("The feet moved") for w in out.result.warnings)
    with open(out.files["summary"]) as fh:
        s = json.load(fh)
    assert not {"heel_lift_catch", "heel_lift_finish"} & {c["metric"] for c in s["charts"]}
    assert s["averages"]["heel_lift_catch"] is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
