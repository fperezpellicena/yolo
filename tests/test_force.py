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
from pose_app.analysis.feet import LIFT_CM
from pose_app.analysis.stations.skierg import heel_windows
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


def _run(setup, log, out="out", estimator=None, **opts):
    session, video, _, _ = _inputs()
    return run_analysis(video, os.path.join(_state["dir"], out),
                        AnalysisOptions(model="fake", pm5=log, force=setup, video=False, **opts),
                        estimator or session.estimator())


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


def test_a_hand_set_offset_only_steers_the_log_it_belongs_to():
    from pose_app.analysis.pipeline import _sync_prior
    assert _sync_prior(None, 5.0, same_clock=True) == (5.0, 1.0)
    assert _sync_prior(None, 5.0, same_clock=False) == (None, 0.0)     # another file's offset


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


def _check_against_truth(outcome, session, truth, allowed=()):
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
    assert not [c for c in fa.checks if not c.startswith(allowed)], fa.checks


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
    assert s["pm5"]["sync"]["events"] == "cord" and s["pm5"]["sync"]["linked"]
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


def test_the_force_model_goes_frame_by_frame_into_forces_json():
    session, _, log, _ = _inputs()
    outcome = _run(session.setup(), log)
    res = outcome.result
    with open(outcome.files["forces"]) as fh:
        f = json.load(fh)
    n = len(res.body.t)
    arr = lambda a: np.array([np.nan if v is None else v for v in a], float)
    pairs = lambda a: arr(a).reshape(-1, 2)
    assert f["version"] == 1 and f["frame_size"] == [sf.W, sf.H] and f["facing"] == 1
    assert f["reported"] == ["elbow", "shoulder"]
    assert f["joints"] == ["elbow", "shoulder", "hip", "knee", "ankle"]
    assert len(f["t"]) == len(f["tension"]) == len(f["cop_x"]) == n
    assert all(len(v) == 2 * n for v in [*f["points"].values(), *f["passed"].values(), f["floor"]])
    # In the video's pixels: the model's points sit on the athlete, the floor under the ankle
    truth = session.points_at(arr(f["t"]))
    to_px = lambda m: np.stack([sf.ORIGIN[0] + m[..., 0] * sf.SCALE,
                                sf.ORIGIN[1] - m[..., 1] * sf.SCALE], -1)
    for name in ("elbow", "shoulder", "hip", "knee"):
        err = np.linalg.norm(pairs(f["points"][name]) - to_px(truth[name]), axis=1)
        assert np.nanmedian(err) < 3, (name, np.nanmedian(err))
    assert abs(f["floor_y"] - sf.ORIGIN[1]) < 2 and f["fixed_points"]["ankle"][1] < f["floor_y"]
    assert abs(f["px_per_m"] - sf.SCALE) < 1 and f["cord_exit"] == list(session.setup().calibration.cord_exit)
    # The chain adds up: the still feet take what the ankle passes on, their weight and the floor
    feet = pairs(f["floor"]) + pairs(f["passed"]["ankle"]) + [0, f["feet"]["mass_kg"] * 9.81]
    assert np.nanmax(np.abs(feet)) < 2, np.nanmax(np.abs(feet))
    # At peak force the arm passes the cord's pull on towards the cord exit
    T = arr(f["tension"])
    k = int(np.nanargmax(T))
    to_exit = np.array(f["cord_exit"]) - pairs(f["points"]["grip"])[k]
    for joint in ("elbow", "shoulder"):
        v = pairs(f["passed"][joint])[k]
        cos = np.dot(v, to_exit) / np.linalg.norm(v) / np.linalg.norm(to_exit)
        assert cos > math.cos(math.radians(10)), (joint, v)
    # Between drives the hips carry the upper body's weight, straight down
    upper = sum(s["mass_kg"] for s in f["segments"]
                if s["name"] in ("forearm_hand", "upper_arm", "head", "trunk")) * 9.81
    hip = pairs(f["passed"]["hip"])[T == 0]
    assert abs(np.nanmedian(hip[:, 1]) / upper - 1) < 0.1 and abs(np.nanmedian(hip[:, 0])) < 30
    # The moments are the analysis' own, per side; the force is unknown where the analysis has none
    fa = res.force
    assert abs(f["moment"]["shoulder"][k] - fa.per_side("shoulder")[k]) < 0.06
    assert all((v is None) == (not np.isfinite(t)) for v, t in zip(f["tension"], fa.tension))
    # The geometry of the pull: the shoulder's distance to the cord's line, the centre of mass's height
    from pose_app.analysis.force.dynamics import moment_arm
    arm = 100 * moment_arm(fa.kin.points["shoulder"], fa.kin.points["grip"], fa.cord_dir)
    assert np.nanmax(np.abs(arr(f["arm"]["shoulder"]) - arm)) < 0.06
    assert np.nanmax(np.abs(arr(f["com_height"]) - 100 * fa.kin.body_com[:, 1])) < 0.06
    # The typical stroke, catch to catch, over every stroke of the analysis
    assert f["strokes"] == [[r.cycle.start, r.cycle.end] for r in res.reps]
    typical = f["typical"]
    assert typical["x"][0] == 0 and typical["x"][-1] == 100 and len(typical["x"]) == 51
    assert typical["strokes"] == len(res.reps)
    pull = arr(typical["series"]["tension"]["median"])
    assert 5 < typical["x"][int(np.nanargmax(pull))] < 45, pull     # the drive opens the stroke
    assert np.nanmax(pull[30:]) < 0.05 * np.nanmax(pull)            # slack in the recovery
    for name, band in typical["series"].items():
        lo, mid, hi = (arr(band[k]) for k in ("low", "median", "high"))
        assert np.isfinite(mid).all(), name                         # the PM5 knows every point
        assert (lo <= mid + 0.11).all() and (mid <= hi + 0.11).all(), name


# ---------------------------------------------------------------- the camera and the feet

FEET = {"heel": 0.025, "toe": 0.100}       # this athlete's own, unlike the stature fractions
DRIFT = (-40.0, 60.0)                      # px by the end of the clip: a phone sliding down


def test_camera_tracker_measures_the_background_not_the_athlete():
    import cv2
    from pose_app.analysis.camera import measure_camera
    room = sf._room()
    centre = (sf.W / 2, sf.H / 2)
    truth, frames, boxes = [], [], []
    for k in range(40):
        a = cv2.getRotationMatrix2D(centre, 0.02 * k, 1 + 0.0004 * k)   # a slow turn and zoom...
        a[:, 2] += [-0.8 * k, 1.5 * k]                                  # ...and slide
        m = np.vstack([a, [0, 0, 1]]) @ np.array([[1, 0, -sf.ROOM_MARGIN], [0, 1, -sf.ROOM_MARGIN],
                                                    [0, 0, 1]])
        img = cv2.warpAffine(room, m[:2], (sf.W, sf.H))
        box = np.array([380 + 6 * k, 150, 620 + 6 * k, 700])            # a moving athlete
        cv2.rectangle(img, (int(box[0]), 150), (int(box[2]), 700), (255, 255, 255), -1)
        cv2.circle(img, (int(box[0]) + 60 + 9 * k % 100, 400), 30, (0, 0, 0), -1)
        frames.append(img)
        boxes.append(box)
        truth.append(m @ np.linalg.inv(np.array([[1, 0, -sf.ROOM_MARGIN],
                                                  [0, 1, -sf.ROOM_MARGIN], [0, 0, 1]])))
    cam = measure_camera(frames, np.array(boxes), fps=8.0)               # samples every other frame
    sampled = [k for k in range(40) if np.isfinite(cam[k]).all()]
    assert sampled[0] == 0 and len(sampled) >= 18, sampled
    first = np.linalg.inv(truth[0])
    for k in sampled:
        want = (truth[k] @ first)[:2]
        for p in ((100, 100), (330, 700), (900, 650)):                   # corners and the feet
            got, exp = cam[k] @ [*p, 1], want @ [*p, 1]
            assert np.linalg.norm(got - exp) < 1.5, (k, p, got, exp)


def test_a_drifting_camera_is_steadied():
    """The phone slides 7 cm down and 13 cm sideways over the piece; the force analysis still
    finds the truth, and draws the model where the athlete is in each frame."""
    session, _, log, truth = _inputs()
    video = os.path.join(_state["dir"], "drift.mp4")
    if not os.path.exists(video):
        session.write_video(video, drift=DRIFT, texture=True)
    kp, sc = session.keypoints(feet=FEET, drift=DRIFT, seed=11)
    outcome = run_analysis(video, os.path.join(_state["dir"], "drift"),
                           AnalysisOptions(model="fake", pm5=log, force=session.setup(), video=False),
                           sf.FakeEstimator(kp, sc))
    _check_against_truth(outcome, session, truth, allowed=("The camera moved",))
    fa = outcome.result.force
    assert any(c.startswith("The camera moved during the clip, shifting the floor by up to 2")
               for c in fa.checks), fa.checks
    assert not fa.stepping
    with open(outcome.files["forces"]) as fh:
        f = json.load(fh)
    arr = lambda a: np.array([np.nan if v is None else v for v in a], float).reshape(-1, 2)
    t = np.array(f["t"])
    moved = (t / session.duration)[:, None] * DRIFT
    floor = np.array([sf.ORIGIN[0], sf.ORIGIN[1], 1.0])
    seen = np.array([np.reshape(m, (2, 3)) @ floor for m in f["camera"]]) - sf.ORIGIN
    assert np.max(np.abs(seen - moved)[15:-15]) < 3, np.max(np.abs(seen - moved)[15:-15])
    truth_px = session.points_at(t)
    to_px = lambda m: np.stack([sf.ORIGIN[0] + m[..., 0] * sf.SCALE,
                                sf.ORIGIN[1] - m[..., 1] * sf.SCALE], -1)
    for name in ("elbow", "shoulder", "hip"):
        err = np.linalg.norm(arr(f["points"][name]) - (to_px(truth_px[name]) + moved), axis=1)
        assert np.nanmedian(err) < 3, (name, np.nanmedian(err))
    # the first-frame references stay where the calibration was clicked
    assert abs(f["floor_y"] - sf.ORIGIN[1]) < 2
    assert f["cord_exit"] == list(session.setup().calibration.cord_exit)
    # the old pose cache had no camera: it is measured from the video again, and kept
    cache = outcome.files["pose_cache"]
    with np.load(cache) as d:
        stripped = {k: d[k] for k in d.files if k != "camera"}
    np.savez_compressed(cache, **stripped)
    again = run_analysis(video, os.path.join(_state["dir"], "drift"),
                         AnalysisOptions(model="fake", pm5=log, force=session.setup(), video=False,
                                         reuse=True))
    assert np.allclose(again.result.force.camera, fa.camera, atol=1e-6)
    with np.load(cache) as d:
        assert "camera" in d.files


def test_stepping_into_place_is_not_a_moving_camera():
    """The athlete steps back 15 cm during the first stroke, the camera still: nothing moves the
    machine, and the heel check leaves that stroke out."""
    session, video, log, _ = _inputs()
    kp, sc = session.keypoints(feet=FEET, step={"stroke": 0, "back_m": 0.15}, seed=11)
    res = _run(session.setup(), log, out="step", estimator=sf.FakeEstimator(kp, sc)).result
    fa = res.force
    assert fa.stepping == [0], fa.stepping
    assert np.allclose(fa.camera, np.eye(3))                  # the camera was not moved
    assert not any(c.startswith("The camera moved") for c in fa.checks)
    assert any(c.startswith("The feet moved (stepping) during stroke 1.") for c in fa.checks)
    heel = [r.metrics["heel_lift_catch"] for r in res.reps]
    assert math.isnan(heel[0]) and math.isfinite(heel[1])
    assert math.isnan(heel[-1])                                  # the last stroke: stepping out


def test_measured_feet_set_the_floor_and_the_base_of_support():
    session, _, log, _ = _inputs()
    kp, sc = session.keypoints(feet=FEET, seed=11)
    fa = _run(session.setup(), log, out="feet", estimator=sf.FakeEstimator(kp, sc)).result.force
    m, H = fa.model, sf.HEIGHT
    assert m.feet_measured and fa.feet_source == "measured"
    assert abs(m.heel - FEET["heel"] * H) < 0.005 and abs(m.toe - FEET["toe"] * H) < 0.005
    assert abs(m.ankle_h - sf.LEN["ankle_h"] * H) < 0.005             # floor where the soles are
    assert fa.summary["feet_measured"] == 1 and abs(fa.summary["heel_cm"] - 4.5) < 0.5
    assert not any(c.startswith(("The camera moved", "Heels", "The feet")) for c in fa.checks)
    plain_res = _run(session.setup(), log).result                         # 17 points: no feet
    plain = plain_res.force
    assert not plain.model.feet_measured and plain.feet_source == "height"
    assert plain.summary["feet_measured"] == 0 and math.isnan(plain.summary["heel_lift_strokes"])
    assert all(math.isnan(r.metrics["heel_lift_catch"]) for r in plain_res.reps)
    # a shorter base of support holds less before the pressure reaches its edge
    cap = lambda fa: np.nanmedian([s.metrics["f_capacity_n"] for s in fa.strokes])
    assert cap(fa) < cap(plain), (cap(fa), cap(plain))


def test_implausible_feet_fall_back_to_the_stature_fractions():
    session, _, log, _ = _inputs()
    kp, sc = session.keypoints(feet={"heel": 0.025, "toe": 0.30}, seed=11)        # a 54 cm toe
    fa = _run(session.setup(), log, out="bigfoot", estimator=sf.FakeEstimator(kp, sc)).result.force
    assert not fa.model.feet_measured
    assert any(c.startswith("The feet were measured but do not look right") for c in fa.checks)


def test_heels_off_the_floor_are_flagged():
    """The heel rises 6 cm late in the recovery of three strokes: the technique analysis reads
    it around the next catch (its heel metrics), and the force model, which takes the feet as
    flat, says so and exports the same lift frame by frame."""
    session, _, log, _ = _inputs()
    lifted = [5, 6, 7]
    kp, sc = session.keypoints(feet=FEET, heel_lift={"strokes": lifted, "height_m": 0.06}, seed=11)
    outcome = _run(session.setup(), log, out="heels", estimator=sf.FakeEstimator(kp, sc))
    res, fa = outcome.result, outcome.result.force
    lift_t = [float(np.interp(k + 0.9, session.phase_fine, session.t_fine)) for k in lifted]
    flagged, t = [], res.body.t
    for r in res.reps:
        (a, b), _ = heel_windows(r.cycle.start, r.cycle.mid, r.cycle.end)
        mine = any(t[a] <= x <= t[b] for x in lift_t)
        v = r.metrics["heel_lift_catch"]
        if r.number in (1, len(res.reps)):                  # stepping in and out: not judged
            assert math.isnan(v), (r.number, v)
            continue
        assert (v > LIFT_CM) == mine, (r.number, v, mine)
        assert r.metrics["heel_lift_finish"] < 2.0, r.metrics["heel_lift_finish"]   # pixel noise
        if mine:
            # scaled by the force setup's height; smoothing takes ~8% off this 0.3 s bump
            assert 5.2 < v < 6.3, v
            flagged.append(r.number)
    assert len(flagged) == len(lifted)
    assert not any(f.rule_id == "heels_up_finish" for r in res.reps for f in r.faults)
    note = next(c for c in fa.checks if c.startswith("Heels came off the floor"))
    assert f"on 3 of the {len(res.reps) - 2} strokes checked" in note
    assert note in res.warnings and fa.summary["heel_lift_strokes"] == 3
    with open(outcome.files["forces"]) as fh:
        f = json.load(fh)
    lift = np.array([np.nan if v is None else v for v in f["heel_lift_cm"]], float)
    assert f["feet_measured"] and 5.2 < np.nanmax(lift) < 6.3
    with open(outcome.files["reps"]) as fh:
        head = next(csv.reader(fh))
        assert "heel_lift_catch" in head and "f_heel_lift_cm" not in head


def test_without_the_setup_there_is_no_force_model():
    _, _, log, _ = _inputs()
    assert "forces" not in _run(None, log, out="plain").files


def test_without_the_pm5_log_the_force_model_is_geometry():
    session, video, log, _ = _inputs()
    kp, sc = session.keypoints()                   # both runs see the same keypoints
    runs = {}
    for name, pm5 in (("pm5", log), ("geometry", None)):
        runs[name] = run_analysis(video, os.path.join(_state["dir"], f"model-{name}"),
                                  AnalysisOptions(model="fake", pm5=pm5, force=session.setup(),
                                                  video=False), sf.FakeEstimator(kp, sc))
    geometry = runs["geometry"]
    assert geometry.result.force is None and geometry.result.pm5 is None
    with open(geometry.files["summary"]) as fh:
        assert json.load(fh)["force"] is None       # no force values to report
    frames = {}
    for name, outcome in runs.items():
        with open(outcome.files["forces"]) as fh:
            frames[name] = json.load(fh)
    f, p = frames["geometry"], frames["pm5"]
    assert f["mode"] == "geometry" and p["mode"] == "pm5"
    assert f["points"] == p["points"] and f["segments"] == p["segments"]
    # The pull is unknown wherever the cords pull, and so are the forces it changes
    n = len(f["t"])
    pulling = [k for k in range(n) if (p["tension"][k] or 0) > 20]
    assert pulling and all(f["tension"][k] is None for k in pulling)
    unknown = [k for k in range(n) if f["tension"][k] is None]
    assert all(f["floor"][2 * k] is None and f["moment"]["hip"][k] is None and
               f["passed"]["elbow"][2 * k] is None for k in unknown)
    # Between drives the cords are slack: the forces are the PM5 run's own
    slack = [k for k in range(n) if f["tension"][k] == 0 and p["tension"][k] == 0]
    assert len(slack) > n / 3, len(slack)
    for k in slack:
        for a, b in ((f["floor"], p["floor"]), (f["passed"]["hip"], p["passed"]["hip"])):
            assert abs(a[2 * k] - b[2 * k]) <= 1 and abs(a[2 * k + 1] - b[2 * k + 1]) <= 1, k
        assert abs(f["moment"]["shoulder"][k] - p["moment"]["shoulder"][k]) <= 0.1, k
    # The geometry is the same; the typical stroke knows the moments only in the recovery
    assert f["arm"] == p["arm"] and f["com_height"] == p["com_height"]
    series = f["typical"]["series"]
    assert series["arm_shoulder"] == p["typical"]["series"]["arm_shoulder"]
    x = f["typical"]["x"]
    assert series["shoulder"]["median"][x.index(20)] is None        # mid-drive
    assert series["shoulder"]["median"][x.index(70)] is not None    # recovery


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
    outcome = _run(session.setup(), path, out="other")
    res = outcome.result
    assert not res.force.sync.ok and not res.force.strokes
    assert all(math.isnan(r.metrics["f_bw_share"]) for r in res.reps)
    assert any("could not be matched" in w for w in res.warnings), res.warnings
    with open(outcome.files["summary"]) as fh:        # the curves stay, off the video
        p = json.load(fh)["pm5"]
    assert not p["sync"]["linked"] and p["curves"] == 23
    assert all(x["t_start"] is None and x["rep"] is None for x in p["strokes"])


def test_pm5_log_without_setup_syncs_the_machine_data():
    _, _, log, _ = _inputs()
    res = _run(None, log, out="plain").result
    assert res.force is None and res.machine.alignment.method == "strokes"
    assert abs(res.machine.alignment.offset - (sf.LOG_OFFSET - 0.2)) < 0.1
    assert all("f_bw_share" not in r.metrics for r in res.reps)
    # Not a problem with the recording: the command line says how to turn it on
    assert not any("force" in w for w in res.warnings), res.warnings
    import contextlib, io
    from pose_app.analysis.cli import main
    _, video, _, _ = _inputs()
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        assert main([video, "--out", os.path.join(_state["dir"], "plain"), "--reuse",
                     "--no-video", "--pm5", log]) == 0
    assert "add --mass, --height and --cord-exit" in err.getvalue(), err.getvalue()


def test_every_pm5_stroke_and_its_curve_go_in_the_summary():
    session, _, log, _ = _inputs()
    outcome = _run(None, log, out="plain")
    with open(outcome.files["summary"]) as fh:
        s = json.load(fh)
    p = s["pm5"]
    assert p["curve_spacing"] == "time" and p["curves"] == 23
    assert p["sync"]["linked"] and p["sync"]["events"] == "hands" and p["sync"]["matched"] >= 20
    strokes = p["strokes"]
    assert [x["n"] for x in strokes] == list(range(1, 25))
    assert strokes[8]["curve"] is None and strokes[8]["peak_n"] > 0     # a curve packet was lost
    reps = {r["n"]: r for r in s["per_rep"]}
    matched = [x for x in strokes if x["rep"] is not None]
    assert len(matched) == p["sync"]["matched"]
    assert all(a["rep"] < b["rep"] for a, b in zip(matched, matched[1:]))
    for x, truth in zip(strokes, session.strokes):
        assert abs(x["t_start"] - truth["t_start"]) < 0.12, (x, truth["t_start"])
        assert abs(x["t_end"] - truth["t_end"]) < 0.12, (x, truth["t_end"])
        if x["rep"] is not None:
            assert abs(reps[x["rep"]]["t_start"] - truth["t_start"]) < 0.2
        if x["curve"] is not None:
            c = x["curve"]
            assert c[0] == c[-1] == 0 and all(isinstance(v, int) for v in c)
            assert abs(max(c) - x["peak_n"]) < 5      # whole lbf on the curve, 0.1 lbf per stroke
            assert 20 < x["peak_pct"] < 60
    m = p["mean_curve"]
    assert len(m["x"]) == len(m["all"]) == len(m["early"]) == len(m["late"]) == 51
    with open(outcome.files["report"], encoding="utf-8") as fh:
        html = fh.read()
    assert "PM5 force curves" in html and html.count('class="stroke"') == 23


def test_force_options_are_checked():
    session, video, log, _ = _inputs()
    setup = session.setup()
    cases = [(AnalysisOptions(force=ForceSetup(AthleteProfile(80, 180), setup.calibration), pm5=log),
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
