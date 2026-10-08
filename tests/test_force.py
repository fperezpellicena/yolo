"""Force analysis (SkiErg + PM5): filter, body model, dynamics, stroke matching, end to end.

The synthetic session (synthetic_force.py) moves a body model through smooth
strokes with a known cord tension, so the analysis of its noisy 30 fps
keypoints and PM5 log can be checked against the exact values.
Run: python tests/test_force.py (or pytest). No model or GPU needed.
"""
import csv, json, math, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import synthetic_force as sf
from pose_app.analysis.api import AnalysisError, AnalysisOptions, default_thresholds, run_analysis
from pose_app.analysis.force import ForceSetup
from pose_app.analysis.force.dynamics import cross, static_capacity
from pose_app.analysis.force.kinematics import _pass, _pass_numpy, lowpass
from pose_app.analysis.force.model import BodyModel
from pose_app.analysis.force.setup import AthleteProfile, Calibration
from pose_app.analysis.force.sync import DriveWindow, match_strokes

_state = {}


def _inputs(travel=False):
    """The synthetic clip and its PM5 log, written once per run."""
    key = "travel" if travel else "time"
    if key not in _state:
        if "dir" not in _state:
            _state["dir"] = tempfile.mkdtemp()
        session = sf.Session(n_strokes=24, travel=travel)
        video = os.path.join(_state["dir"], "ski.mp4")
        if not os.path.exists(video):
            session.write_video(video)
        log = os.path.join(_state["dir"], f"ski_{key}.pm5.jsonl")
        session.write_pm5_log(log)
        _state[key] = (session, video, log, session.truth())
    return _state[key]


def _run(setup, log, out="out", **opts):
    session, video, _, _ = _inputs()
    return run_analysis(video, os.path.join(_state["dir"], out),
                        AnalysisOptions(model="fake", pm5=log, force=setup, video=False, **opts),
                        session.estimator())


# ---------------------------------------------------------------- building blocks

def test_lowpass_keeps_the_stroke_without_lag():
    fs = 30.0
    t = np.arange(0, 20, 1 / fs)
    slow, fast = np.sin(2 * np.pi * 0.8 * t), 0.5 * np.sin(2 * np.pi * 12 * t)
    y = lowpass(np.column_stack([slow + fast + 3.0, slow]), fs, 6.0)
    mid = slice(60, -60)
    assert np.max(np.abs(y[mid, 0] - (slow + 3.0)[mid])) < 0.03       # 12 Hz gone, DC kept
    lags = [np.dot(np.roll(y[mid, 1], k), slow[mid]) for k in range(-3, 4)]
    assert int(np.argmax(lags)) == 3                                 # no lag
    x = np.random.default_rng(0).normal(size=(200, 3))
    assert np.allclose(_pass(x, *_coefficients()), _pass_numpy(x, *_coefficients()))


def _coefficients():
    wc = np.tan(np.pi * 6.0 / (0.802 * 30.0))
    k1, k2 = np.sqrt(2) * wc, wc * wc
    a0 = 1 + k1 + k2
    return np.array([k2, 2 * k2, k2]) / a0, np.array([1, 2 * (k2 - 1) / a0, (1 - k1 + k2) / a0])


def test_setup_from_json_and_its_checks():
    good = {"athlete": {"mass_kg": 80, "height_m": 1.8, "sex": "female"},
            "calibration": {"cord_exit": [612, 88],
                            "scale": {"points": [[400, 900], [400, 600]], "length_m": 1.0}}}
    setup = ForceSetup.from_dict(good)
    assert setup.calibration.stick_px_per_m == 300.0 and ForceSetup.from_dict(setup.to_dict()) == setup
    assert ForceSetup.from_dict({"athlete": {"mass_kg": "75", "height_m": 1.7},
                                 "calibration": {"cord_exit": [1, 2]}}).athlete.sex == "unspecified"
    bad = [({"athlete": {"mass_kg": 80, "height_m": 180}}, "metres, not cm"),
           ({"athlete": {"mass_kg": 80, "height_m": 1.8, "sex": "x"}}, "sex must be"),
           ({"athlete": {"mass_kg": 80, "height_m": 1.8, "weight": 1}}, "unknown fields"),
           ({"calibration": {"cord_exit": [612]}}, "list of 2"),
           ({"calibration": {"cord_exit": [612, 88], "scale": {"points": [[0, 0], [0, 5]],
                                                                "length_m": 1}}}, "20 px"),
           ({"calibration": {"cord_exit": [612, 88], "scale": {"length_m": 1}}}, "scale must")]
    for change, text in bad:
        d = json.loads(json.dumps(good))
        for k, v in change.items():
            d[k] = v
        try:
            ForceSetup.from_dict(d)
        except (TypeError, ValueError) as exc:
            assert text in str(exc), (text, exc)
            continue
        raise AssertionError(f"accepted {change}")


def test_body_model_adds_up_to_the_athlete():
    lengths = {"forearm_hand": 0.26, "upper_arm": 0.33, "trunk": 0.52, "thigh": 0.44, "shank": 0.44}
    for sex in ("male", "female", "unspecified"):
        model = BodyModel(AthleteProfile(72.0, 1.75, sex), lengths)
        assert abs(model.total_mass - 72.0) < 0.05, (sex, model.total_mass)
        fh = model.segments["forearm_hand"]
        assert 0.6 < fh.com < 0.75 and fh.inertia > 0


def test_static_force_capacity_puts_the_pressure_on_the_foot_edge():
    model = BodyModel(AthleteProfile(80.0, 1.80, "male"),
                      {"forearm_hand": 0.26, "upper_arm": 0.33, "trunk": 0.52, "thigh": 0.44,
                       "shank": 0.44})
    W, grip, u = model.weight, np.array([0.30, 1.50]), np.array([0.0, 1.0])
    T, edge = static_capacity(0.02, grip, u, model)
    cop = (W * 0.02 - T * cross(grip, u)) / (W - T * u[1])
    assert edge == "heel" and abs(cop + model.heel) < 1e-9, (T, edge, cop)
    assert math.isnan(static_capacity(-0.2, grip, u, model)[0])       # already past the heel
    tilted = np.array([np.cos(np.radians(50)), np.sin(np.radians(50))])
    T2, edge2 = static_capacity(0.05, np.array([0.6, 1.5]), tilted, model)
    assert edge2 in ("toe", "heel", "lift") and T2 > 0


def test_joint_work_balances_the_energy_exactly_on_exact_motion():
    """Newton-Euler moments x joint speeds = body energy change + work on the cord."""
    _, _, _, truth = _inputs()
    errors = [abs(s["energy_err"]) for s in truth]
    assert max(errors) < 0.1, errors                                  # percent of the stroke's work


def test_matching_tells_neighbouring_strokes_apart():
    rng = np.random.default_rng(3)
    periods = 1.5 * (1 + 0.04 * rng.standard_normal(40))
    ends = np.cumsum(periods) + 30.0
    drive = 0.55 + 0.03 * rng.standard_normal(40)
    true_offset = 4.27
    v_end = ends[5:35] - true_offset + 0.01 * rng.standard_normal(30)
    v_start = v_end - drive[5:35] - 0.06 + 0.01 * rng.standard_normal(30)
    windows = [DriveWindow(0, 0, a, b) for a, b in zip(v_start, v_end)]
    windows[7] = None                                                 # a stroke the video missed
    sync = match_strokes(windows, ends - drive, ends)
    assert sync.ok and sync.confidence == "high" and abs(sync.offset - true_offset) < 0.02, sync
    assert all(j == k + 5 for k, j in sync.pairs.items()) and 7 not in sync.pairs
    lost = match_strokes(windows, ends - drive + 200, ends + 200, prior=0.0, prior_window=3.0)
    assert not lost.ok


# ---------------------------------------------------------------- end to end

TOLERANCE = {"f_peak_n": ("peak_n", 6.0), "f_bw_share": ("bw_share", 2.0),
             "f_cord_angle": ("cord_angle", 2.0), "f_shoulder_ext_nm": ("shoulder_ext", 5.0),
             "f_elbow_flex_nm": ("elbow_flex", 4.0), "f_shoulder_arm_cm": ("shoulder_arm", 2.0),
             "f_friction": ("friction", 0.04), "f_capacity_n": ("capacity", 60.0)}


def _check_against_truth(outcome, session, truth):
    res, fa = outcome.result, outcome.result.force
    assert fa.sync.ok and fa.sync.confidence == "high", fa.sync
    # machine time = video time + offset; the log's clock starts with its first notification
    assert abs(fa.sync.offset - (sf.LOG_OFFSET - 0.2)) < 0.05, fa.sync.offset
    assert res.machine.alignment.method == "strokes"
    true_ends = np.array([s["t_end"] for s in session.strokes])
    with_curve = [s for s in fa.strokes if s.profile is not None]
    assert len(with_curve) >= 21, len(with_curve)
    errors = {k: [] for k in TOLERANCE}
    for s in fa.strokes:
        k = int(np.argmin(np.abs(true_ends - s.window.t_end)))
        assert s.pm5.count == k + 1                                   # the right PM5 stroke
        if s.profile is None:
            continue
        for key, (name, _) in TOLERANCE.items():
            errors[key].append(s.metrics[key] - truth[k][name])
    for key, (name, tol) in TOLERANCE.items():
        median = float(np.median(np.abs(errors[key])))
        assert median < tol, (key, median, tol)
    assert fa.summary["energy_err_median"] < 10, fa.summary
    assert not fa.checks, fa.checks


def test_force_analysis_recovers_the_truth_time_curves():
    session, _, log, truth = _inputs()
    outcome = _run(session.setup(), log)
    _check_against_truth(outcome, session, truth)
    res = outcome.result
    assert all(r.metrics["f_bw_share"] > 30 for r in res.reps[1:])
    assert not any(f.rule_id in ("low_bodyweight_share", "shallow_cord", "late_force_peak")
                   for r in res.reps for f in r.faults)
    with open(outcome.files["summary"]) as fh:
        s = json.load(fh)
    f = s["force"]
    assert s["schema_version"] == 1 and f["sync"]["confidence"] == "high"
    assert f["pm5"]["curve_spacing"] == "time" and f["reported_joints"] == ["elbow", "shoulder"]
    assert len(f["curve"]["x"]) == len(f["curve"]["all"]) == len(f["curve"]["late"])
    assert {"f_bw_share", "f_cord_angle", "f_shoulder_ext_nm"} <= set(s["per_rep"][3])
    assert {"low_bodyweight_share", "shallow_cord"} <= {r["id"] for r in s["rules"]}
    assert "f_bw_share" in {c["metric"] for c in s["charts"]}
    assert f["example"]["rep"] >= 1 and f["example"]["video_t"] is not None
    with open(outcome.files["reps"]) as fh:
        header = next(csv.reader(fh))
    assert "f_peak_n" in header and "frame_force_peak" not in header
    with open(outcome.files["report"], encoding="utf-8") as fh:
        html = fh.read()
    assert "Force analysis" in html and "Force curve" in html and "<img" in html


def test_force_analysis_recovers_the_truth_travel_curves():
    session, _, log, truth = _inputs(travel=True)
    outcome = _run(session.setup(), log, out="travel")
    _check_against_truth(outcome, session, truth)
    assert outcome.result.force.pm5.curve_basis == "travel"


def test_overlay_draws_the_force_line():
    session, _, log, _ = _inputs()
    outcome = _run(session.setup(), log)
    from pose_app.analysis.render import Annotator
    res = outcome.result
    stroke = next(s for s in res.force.strokes if s.profile is not None)
    frame = int(res.body.frame_index[stroke.peak])
    blank = np.zeros((sf.H, sf.W, 3), np.uint8)

    def orange(img):                    # the force colour, away from the header and the cues
        part = img[150:560, 400:]
        return int(((part[..., 2] > 200) & (part[..., 1] > 100) & (part[..., 1] < 180) &
                    (part[..., 0] < 60)).sum())
    assert orange(Annotator(res).draw(blank.copy(), frame)) > 500     # arrow + line of action
    recovery = int(res.body.frame_index[stroke.stop - 3])
    assert orange(Annotator(res).draw(blank.copy(), recovery)) == 0


def test_bad_calibration_is_flagged():
    session, _, log, _ = _inputs()
    setup = session.setup()
    (x0, y0), (x1, y1) = setup.calibration.scale_points
    wrong = ForceSetup(setup.athlete, Calibration(setup.calibration.cord_exit,
                                                  ((x0, y0), (x1, y1)), 1.25))
    checks = " ".join(_run(wrong, log, out="wrong").result.force.checks)
    assert "calibration length gives 240 px/m" in checks, checks
    assert "cord payout seen in the video" in checks, checks


def test_standing_far_from_the_machine_fires_the_cord_rule():
    session, _, log, _ = _inputs()
    far = session.setup(cord_exit=sf.CORD_EXIT + [0.55, 0.0])
    res = _run(far, log, out="far").result
    angles = [r.metrics["f_cord_angle"] for r in res.reps if math.isfinite(r.metrics["f_cord_angle"])]
    assert np.median(angles) < 65, angles
    hits = sum(any(f.rule_id == "shallow_cord" for f in r.faults) for r in res.reps)
    assert hits >= 15, hits


def test_a_log_of_another_piece_leaves_force_out():
    session, _, _, _ = _inputs()
    other = sf.Session(n_strokes=24, seed=5, pattern=(7.0, 3.1, 2.5))
    path = os.path.join(_state["dir"], "other.pm5.jsonl")
    other.write_pm5_log(path)
    res = _run(session.setup(), path, out="other").result
    assert not res.force.sync.ok and not res.force.strokes
    assert all(math.isnan(r.metrics["f_bw_share"]) for r in res.reps)
    assert any("could not be matched" in w for w in res.warnings), res.warnings


def test_pm5_log_without_setup_syncs_the_machine_data():
    _, _, log, _ = _inputs()
    res = _run(None, log, out="plain").result
    assert res.force is None and res.machine.alignment.method == "strokes"
    assert abs(res.machine.alignment.offset - (sf.LOG_OFFSET - 0.2)) < 0.1
    assert any("force analysis was skipped" in w for w in res.warnings)
    assert all("f_bw_share" not in r.metrics for r in res.reps)


def test_force_options_are_checked():
    session, video, log, _ = _inputs()
    setup = session.setup()
    cases = [(AnalysisOptions(force=setup), "invalid_job", "needs the PM5 log"),
             (AnalysisOptions(force=ForceSetup(AthleteProfile(80, 180), setup.calibration), pm5=log),
              "invalid_job", "metres, not cm"),
             (AnalysisOptions(pm5=os.path.join(_state["dir"], "none.jsonl")),
              "unreadable_telemetry", "not found")]
    for opts, code, text in cases:
        try:
            run_analysis(video, os.path.join(_state["dir"], "bad"), opts, session.estimator())
        except AnalysisError as exc:
            assert exc.code == code and text in str(exc), (code, exc)
            continue
        raise AssertionError(f"no AnalysisError for {opts}")
    assert {"low_bodyweight_share", "drift_bodyweight"} <= set(default_thresholds("skierg"))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
