"""Force analysis of a SkiErg clip with its PM5 log: from the cord to the joints.

    keypoints -> metres (calibration) -> segment kinematics
    PM5 strokes <-> video strokes; each force curve onto its drive's frames
    cord force (PM5 tension along the cord's direction in the video)
        -> net joint moments, joint power, the energy check
        -> per-stroke metrics (f_* keys) and session summary

Hip, knee and ankle moments and the floor reaction are computed (the energy
check needs them) but not reported: one side-on camera does not measure them
well enough until a validation study says otherwise.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...pm5.session import Pm5Session, Pm5Stroke
from ..body import BodySeries
from ..signal import Cycle
from .dynamics import Dynamics, inverse_dynamics, moment_arm, static_capacity
from .kinematics import (ImageFrame, Kinematics, body_points, fit_frame, keypoint_tracks,
                         kinematics, lowpass, pixel_lengths)
from .model import G, BodyModel
from .setup import ForceSetup
from .sync import (DriveWindow, StrokeSync, curve_profile, hand_windows, map_curve,
                   match_strokes, payout_windows, travel_fraction)

# Per-stroke metrics added to each rep (NaN where the stroke has no force data)
FORCE_KEYS = ("f_peak_n", "f_avg_n", "f_work_j", "f_drive_len_m", "f_drive_time_s",
              "f_payout_m", "f_peak_pos", "f_catch70_pos", "f_bw_share", "f_com_drop_cm",
              "f_com_drop_after_cm", "f_cord_angle", "f_friction", "f_capacity_n",
              "f_shoulder_ext_nm", "f_shoulder_ext_max_nm", "f_elbow_flex_nm",
              "f_shoulder_arm_cm", "f_elbow_arm_cm", "f_cord_work_j", "f_energy_err",
              "f_t_peak")
CHECK_PAYOUT = 0.10           # video payout vs PM5 drive length
CHECK_WORK = 0.10             # cord work from the curve vs PM5 work per stroke
CHECK_ENERGY = 0.15           # joint work vs PM5 work per stroke
CHECK_JITTER_MM = 15.0        # above this, even the upper-body moments get noisy
PEAK_ZONE = 0.8               # "at peak force": frames above this share of the stroke's peak
CURVE_GRID = np.linspace(0.0, 1.0, 51)


@dataclass
class ForceStroke:
    rep: int                       # index into the analysis' reps
    pm5: Pm5Stroke
    window: DriveWindow
    peak: int                      # frame of peak tension (window.end when there is no curve)
    stop: int                      # frame where the next drive starts (end of the energy window)
    profile: Optional[np.ndarray]  # PM5 curve, first force to release, newtons
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ForceAnalysis:
    setup: ForceSetup
    pm5: Pm5Session
    sync: StrokeSync
    frame: Optional[ImageFrame] = None
    model: Optional[BodyModel] = None
    t: Optional[np.ndarray] = None
    frame_index: Optional[np.ndarray] = None      # (N,) source-video frame of each sample
    tension: Optional[np.ndarray] = None          # (N,) N, NaN where unknown
    peak_tension: float = math.nan                # the session's highest tension, N
    cord_dir: Optional[np.ndarray] = None         # (N, 2) unit, grip -> cord exit
    exit_m: Optional[np.ndarray] = None
    kin: Optional[Kinematics] = None
    dyn: Optional[Dynamics] = None
    pixels: Dict[str, np.ndarray] = field(default_factory=dict)   # filtered tracks + grip, com
    strokes: List[ForceStroke] = field(default_factory=list)
    checks: List[str] = field(default_factory=list)
    summary: Dict[str, float] = field(default_factory=dict)
    curves: Dict[str, List[float]] = field(default_factory=dict)  # mean curves on CURVE_GRID

    @property
    def by_rep(self) -> Dict[int, ForceStroke]:
        return {s.rep: s for s in self.strokes}

    def per_side(self, joint: str) -> np.ndarray:
        """Net moment at a joint for one side (N m, counter-clockwise positive)."""
        return self.dyn.moment[joint] / 2.0


def pm5_drives(pm5: Pm5Session) -> Tuple[np.ndarray, np.ndarray]:
    """Drive start and end of every PM5 stroke, log clock."""
    return (np.array([s.t_start for s in pm5.strokes]), np.array([s.t_end for s in pm5.strokes]))


def sync_by_hands(body: BodySeries, cycles: Sequence[Cycle], pm5: Pm5Session,
                  prior: Optional[float], prior_window: float) -> StrokeSync:
    """Stroke-by-stroke sync without calibration (for the machine data alone)."""
    windows = hand_windows(body.t, body.metrics["wrist_h"], cycles)
    sync = match_strokes(windows, *pm5_drives(pm5), prior, prior_window)
    sync.events = "hands"
    return sync


def analyze_geometry(t: np.ndarray, keypoints: np.ndarray, scores: np.ndarray, body: BodySeries,
                     cycles: Sequence[Cycle], fps: float, setup: ForceSetup,
                     min_score: float = 0.5) -> ForceAnalysis:
    """The force model from the video alone, without a PM5 log: the body model, the cord's
    line and the moment arms are geometry. The cords' pull is unknown during each drive
    (found from the cord's payout), so the forces there are too; between drives the cords
    are slack and the forces follow from gravity and the body's motion. `pm5` and `sync`
    are None; `checks` says what the video could not give."""
    fa = ForceAnalysis(setup, None, None)
    tracks = keypoint_tracks(t, keypoints, scores, body.side, min_score, fps)
    frame, notes = fit_frame(setup.calibration, setup.athlete, tracks, body.facing)
    fa.checks = list(notes)
    if frame is None:
        fa.checks.append("Force lines skipped: the athlete's feet were not measured.")
        return fa
    lengths_px = pixel_lengths(tracks)
    lengths = {"forearm_hand": lengths_px["forearm"], **{k: v for k, v in lengths_px.items()
                                                         if k != "forearm"}}
    model = BodyModel(setup.athlete, {k: v / frame.scale for k, v in lengths.items()})
    points = body_points(tracks, frame, model)
    exit_m = frame.to_m(np.array(setup.calibration.cord_exit, float))
    to_exit = exit_m[None, :] - points["grip"]
    L = np.linalg.norm(to_exit, axis=1)
    u = to_exit / L[:, None]
    tension = np.zeros(len(t))
    tension[~np.isfinite(L)] = np.nan
    for cycle, w in zip(cycles, payout_windows(t, L, cycles, fps)):
        # Unknown while the cords pull: the drive seen in the payout, or the cycle's first half
        a, b = (w.start, w.end) if w is not None else (cycle.start, cycle.mid)
        tension[a:b + 1] = np.nan
    kin = kinematics(points, t, model)
    dyn = inverse_dynamics(kin, model, np.nan_to_num(tension, nan=0.0)[:, None] * u)
    unknown = ~np.isfinite(tension)
    for values in (dyn.floor, dyn.cop_x, *dyn.moment.values(), *dyn.joint_force.values()):
        values[unknown] = np.nan
    fa.frame, fa.model, fa.t, fa.frame_index = frame, model, t, body.frame_index
    fa.exit_m, fa.cord_dir, fa.tension, fa.kin, fa.dyn = exit_m, u, tension, kin, dyn
    fa.pixels = {**tracks, "grip": frame.to_px(points["grip"]), "com": frame.to_px(kin.body_com)}
    return fa


def analyze_force(t: np.ndarray, keypoints: np.ndarray, scores: np.ndarray, body: BodySeries,
                  cycles: Sequence[Cycle], fps: float, pm5: Pm5Session, setup: ForceSetup,
                  min_score: float = 0.5, prior: Optional[float] = None,
                  prior_window: float = 3.0) -> ForceAnalysis:
    tracks = keypoint_tracks(t, keypoints, scores, body.side, min_score, fps)
    frame, notes = fit_frame(setup.calibration, setup.athlete, tracks, body.facing)
    if frame is None:
        fa = ForceAnalysis(setup, pm5, sync_by_hands(body, cycles, pm5, prior, prior_window))
        fa.checks = notes + ["Force analysis skipped: the athlete's feet were not measured."]
        return fa
    lengths_px = pixel_lengths(tracks)
    lengths = {"forearm_hand": lengths_px["forearm"], **{k: v for k, v in lengths_px.items()
                                                         if k != "forearm"}}
    model = BodyModel(setup.athlete, {k: v / frame.scale for k, v in lengths.items()})
    points = body_points(tracks, frame, model)
    exit_m = frame.to_m(np.array(setup.calibration.cord_exit, float))
    to_exit = exit_m[None, :] - points["grip"]
    L = np.linalg.norm(to_exit, axis=1)
    u = to_exit / L[:, None]

    windows = payout_windows(t, L, cycles, fps)
    sync = match_strokes(windows, *pm5_drives(pm5), prior, prior_window)
    sync.events = "cord"
    fa = ForceAnalysis(setup, pm5, sync, frame, model, t, body.frame_index, exit_m=exit_m,
                       cord_dir=u)
    fa.checks.extend(notes)

    tension = np.zeros(len(t))
    tension[~np.isfinite(L)] = np.nan
    strokes: List[ForceStroke] = []
    starts = sorted(w.start for w in windows if w is not None)
    for k, w in enumerate(windows):
        if w is None:
            continue
        j = sync.pairs.get(k) if sync.ok else None
        if j is None:
            tension[w.start:w.end + 1] = np.nan          # a drive without its force
            continue
        stroke = pm5.strokes[j]
        profile = curve_profile(stroke.curve_n) if stroke.curve_n is not None else None
        mapped = None
        if profile is not None:
            mapped = map_curve(profile, stroke.curve_basis, t, L, w, stroke.drive_time_s)
            if not np.nanmax(mapped) > 0:                 # the curve fell outside the window
                profile = mapped = None
        if mapped is None:
            tension[w.start:w.end + 1] = np.nan
            peak = w.end
        else:
            tension[w.start:w.end + 1] = mapped
            peak = w.start + int(np.nanargmax(mapped))
        stop = next((s for s in starts if s > w.start), cycles[k].end)
        strokes.append(ForceStroke(k, stroke, w, peak, stop, profile))

    kin = kinematics(points, t, model)
    dyn = inverse_dynamics(kin, model, tension[:, None] * u)
    fa.tension, fa.kin, fa.dyn = tension, kin, dyn
    fa.peak_tension = float(np.nanmax(tension)) if np.isfinite(tension).any() else math.nan
    fa.pixels = {**tracks, "grip": frame.to_px(points["grip"]),
                 "com": frame.to_px(kin.body_com)}
    for s in strokes:
        s.metrics = _stroke_metrics(fa, s, L)
    fa.strokes = strokes
    fa.summary, fa.curves = _summary(fa, len(cycles)), _mean_curves(strokes)
    fa.summary["jitter_mm"] = _ankle_jitter(t, keypoints, scores, body.side, min_score, frame, fps)
    fa.checks.extend(_checks(fa, len(cycles)))
    return fa


def _integral(y: np.ndarray, t: np.ndarray) -> float:
    ok = np.isfinite(y)
    if ok.sum() < 2 or not ok.all():
        return math.nan
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(t)))


def _stroke_metrics(fa: ForceAnalysis, s: ForceStroke, L: np.ndarray) -> Dict[str, float]:
    kin, dyn, model, w, p = fa.kin, fa.dyn, fa.model, s.window, s.pm5
    t, T = fa.t, fa.tension
    a, b, k = w.start, w.end, s.peak
    y = kin.body_com[:, 1]
    work = p.work_j if p.work_j > 0 else math.nan
    m: Dict[str, float] = dict.fromkeys(("f_peak_pos", "f_catch70_pos", "f_cord_angle",
                                         "f_friction", "f_capacity_n", "f_shoulder_ext_nm",
                                         "f_shoulder_ext_max_nm", "f_elbow_flex_nm",
                                         "f_shoulder_arm_cm", "f_elbow_arm_cm",
                                         "f_cord_work_j", "f_energy_err", "f_t_peak"), math.nan)
    m.update(f_peak_n=p.peak_force_n, f_avg_n=p.avg_force_n, f_work_j=work,
             f_drive_len_m=p.drive_length_m, f_drive_time_s=p.drive_time_s,
             f_payout_m=float(L[b] - L[a]))
    drop = y[a] - y[b]
    m["f_com_drop_cm"] = 100.0 * drop
    m["f_bw_share"] = 100.0 * model.mass * G * drop / work
    after = y[b:max(s.stop, b + 1)]
    m["f_com_drop_after_cm"] = 100.0 * (y[b] - np.nanmin(after)) if np.isfinite(after).any() \
        else math.nan
    m["frame_force_peak"] = float(fa.frame_index[k])
    if s.profile is None:
        return m

    prof = s.profile
    top = int(np.argmax(prof))
    m["f_peak_n"] = float(prof[top])
    # where along the cord's travel the force peaks / first reaches 70% of its peak
    where = lambda i: 100.0 * travel_fraction(i / (len(prof) - 1), p.curve_basis, t, L, w,
                                              p.drive_time_s)
    m["f_peak_pos"] = where(top)
    m["f_catch70_pos"] = where(int(np.argmax(prof >= 0.7 * prof[top])))
    m["f_t_peak"] = float(t[k])
    # "At peak force" = averaged over the frames above PEAK_ZONE of the peak, weighted by the
    # tension: a single frame would make every value hinge on which frame the peak fell on.
    zone = np.arange(a, b + 1)
    zone = zone[np.nan_to_num(T[a:b + 1]) >= PEAK_ZONE * T[k]]
    wts = T[zone] / T[zone].sum()
    u, grip = fa.cord_dir[zone], kin.points["grip"][zone]

    def at_peak(values) -> float:
        v = np.asarray(values, float)
        ok = np.isfinite(v)
        return float(np.sum(v[ok] * wts[ok]) / wts[ok].sum()) if ok.any() else math.nan
    m["f_cord_angle"] = at_peak(np.degrees(np.arctan2(u[:, 1], u[:, 0])))
    lift = model.weight - T[zone] * u[:, 1]
    m["f_friction"] = at_peak(np.where(lift > 0, T[zone] * np.abs(u[:, 0]) / lift, np.nan))
    m["f_capacity_n"] = at_peak([static_capacity(float(kin.body_com[i, 0]), kin.points["grip"][i],
                                                 fa.cord_dir[i], model)[0] for i in zone])
    m["f_shoulder_ext_nm"] = at_peak(-fa.per_side("shoulder")[zone])
    m["f_shoulder_ext_max_nm"] = float(np.nanmax(-fa.per_side("shoulder")[a:b + 1]))
    m["f_elbow_flex_nm"] = at_peak(fa.per_side("elbow")[zone])
    m["f_shoulder_arm_cm"] = 100.0 * at_peak(moment_arm(kin.points["shoulder"][zone], grip, u))
    m["f_elbow_arm_cm"] = 100.0 * at_peak(moment_arm(kin.points["elbow"][zone], grip, u))
    m["f_cord_work_j"] = float(np.sum(0.5 * (T[a + 1:b + 1] + T[a:b]) * np.diff(L[a:b + 1])))
    e = slice(a, s.stop)                 # one whole stroke: this drive to the next one
    joints = _integral(dyn.joint_power[e], t[e])
    gained = dyn.energy[s.stop - 1] - dyn.energy[a]
    m["f_energy_err"] = 100.0 * (joints - gained - work) / work
    return m


def _mean(strokes: List[ForceStroke], key: str) -> float:
    v = [s.metrics.get(key, math.nan) for s in strokes]
    v = [x for x in v if math.isfinite(x)]
    return float(np.mean(v)) if v else math.nan


def _summary(fa: ForceAnalysis, video_strokes: int) -> Dict[str, float]:
    strokes = fa.strokes
    keys = ("f_peak_n", "f_avg_n", "f_work_j", "f_bw_share", "f_cord_angle", "f_friction",
            "f_capacity_n", "f_shoulder_ext_nm", "f_shoulder_arm_cm", "f_elbow_flex_nm",
            "f_peak_pos", "f_catch70_pos", "f_com_drop_after_cm", "f_drive_len_m", "f_payout_m")
    out = {k: _mean(strokes, k) for k in keys}
    errs = [abs(s.metrics.get("f_energy_err", math.nan)) for s in strokes]
    errs = [e for e in errs if math.isfinite(e)]
    out["energy_err_median"] = float(np.median(errs)) if errs else math.nan
    out["strokes_matched"] = float(len(strokes))
    out["strokes_with_curve"] = float(sum(1 for s in strokes if s.profile is not None))
    out["video_strokes"] = float(video_strokes)
    return out


def _mean_curves(strokes: List[ForceStroke]) -> Dict[str, List[float]]:
    return mean_curves([s.profile for s in strokes if s.profile is not None])


def mean_curves(profiles: Sequence[np.ndarray]) -> Dict[str, List[float]]:
    """Mean force curve (N) on CURVE_GRID, 0-1 of the way along the drive: over all
    curves, and over the first and last thirds when there are 6 or more. {} without curves.
    `profiles` are in time order, each from the first force to the release (curve_profile)."""
    rows = [np.interp(CURVE_GRID, np.linspace(0, 1, len(p)), p) for p in profiles]
    if not rows:
        return {}
    rows = np.array(rows)
    out = {"x": [round(float(x), 3) for x in CURVE_GRID],
           "all": [round(float(v), 1) for v in rows.mean(axis=0)]}
    if len(rows) >= 6:
        third = len(rows) // 3
        out["early"] = [round(float(v), 1) for v in rows[:third].mean(axis=0)]
        out["late"] = [round(float(v), 1) for v in rows[-third:].mean(axis=0)]
    return out


def _ankle_jitter(t, keypoints, scores, side, min_score, frame: ImageFrame, fps) -> float:
    """Frame-to-frame scatter of the (static) near ankle, mm: what the camera adds as noise."""
    from ...skeleton import KP
    k = KP[f"{side}_ankle"]
    xy = np.where((scores[:, k] >= min_score)[:, None], keypoints[:, k], np.nan)
    ok = np.isfinite(xy).all(axis=1)
    if ok.sum() < 30:
        return math.nan
    xy = xy[ok]
    slow = lowpass(xy, fps, 1.0)
    return float(1000.0 * np.sqrt(np.mean(np.sum((xy - slow) ** 2, axis=1) / 2)) / frame.scale)


def _ratio_note(values: List[float], limit: float) -> Optional[float]:
    v = [x for x in values if math.isfinite(x) and x > 0]
    if len(v) < 3:
        return None
    r = float(np.median(v))
    return r if abs(r - 1) > limit else None


def _checks(fa: ForceAnalysis, video_strokes: int) -> List[str]:
    out = []
    if not fa.pm5.curves:
        out.append("The PM5 log has no force curves, so no force could be put on the cords: "
                   "they need a PM5 from late 2016 or later (hardware 600+), and a recorder "
                   "that subscribes to them (pm5_log.py does).")
    if not fa.sync.ok:
        out.append("Force values are left out: the video's strokes could not be matched one by "
                   "one with the PM5 log. Record the same piece on both, ideally from its start.")
        out.extend(fa.sync.notes)
        return out
    n = len(fa.strokes)
    with_curve = sum(1 for s in fa.strokes if s.profile is not None)
    if video_strokes and with_curve < 0.8 * video_strokes:
        out.append(f"Force data for {with_curve} of {video_strokes} strokes: the others had no "
                   "matching PM5 stroke or force curve.")
    ms = [s.metrics for s in fa.strokes]
    r = _ratio_note([m["f_payout_m"] / m["f_drive_len_m"] for m in ms], CHECK_PAYOUT)
    if r is not None:
        out.append(f"The cord payout seen in the video is {abs(r - 1):.0%} "
                   f"{'longer' if r > 1 else 'shorter'} than the PM5's drive length: check the "
                   "calibration length and the cord-exit point.")
    r = _ratio_note([m["f_cord_work_j"] / m["f_work_j"] for m in ms], CHECK_WORK)
    if r is not None:
        out.append(f"The force curves applied to the video's cord payout give {abs(r - 1):.0%} "
                   f"{'more' if r > 1 else 'less'} work than the PM5 reports per stroke.")
    e = fa.summary.get("energy_err_median", math.nan)
    if math.isfinite(e) and e > 100 * CHECK_ENERGY:
        out.append(f"Joint work and the PM5's work per stroke differ by {e:.0f}% (median): "
                   "noisy keypoints or a camera that is not square to the athlete; read the "
                   "moments with care.")
    jit = fa.summary.get("jitter_mm", math.nan)
    if math.isfinite(jit) and jit > CHECK_JITTER_MM:
        out.append(f"Keypoints jitter by {jit:.0f} mm at the ankle (the feet do not move): "
                   "use a tripod, better light or 60 fps.")
    if not n:
        out.append("No stroke had force data.")
    return out


def attach_force(metrics: Sequence[Dict[str, float]], fa: ForceAnalysis) -> None:
    """Add the f_* keys (NaN without force data) and frame_force_peak to every rep."""
    by_rep = fa.by_rep
    for k, m in enumerate(metrics):
        m.update(dict.fromkeys(FORCE_KEYS, math.nan))
        m["frame_force_peak"] = m.get("frame_mid", -1.0)
        s = by_rep.get(k)
        if s is not None:
            vals = dict(s.metrics)
            m.update(vals)
