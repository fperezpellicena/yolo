"""SkiErg: side view, one rep = one stroke (hands high -> hands low -> hands high).

Efficient pattern the rules look for:
  * Tall catch with long, nearly straight arms.
  * Pull initiated by hinging the hips: the hips start moving before the elbows
    bend (lats and trunk move the handles first, not the biceps).
  * Hip-dominant hinge with a soft knee, not a squat.
  * Hands finish at or past the hips.
  * Recovery at least as long as the drive (breathe, reset, re-load).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..body import BodySeries
from ..rules import DriftRule, Rule
from ..signal import Cycle
from .base import ChartSpec, Station

# A joint "starts" moving when it has covered this share of its range in the drive.
ONSET = 0.25
# Below these ranges (degrees) a joint barely moves and its timing means nothing.
MIN_HIP_RANGE, MIN_ELBOW_RANGE = 10.0, 10.0
# Elbows starting this much before the hips is an early arm pull (provisional).
ARM_LEAD_MS = 40.0


def _crossing(t: np.ndarray, x: np.ndarray, a: int, b: int,
              level: float) -> Tuple[float, Optional[int]]:
    """When `x` first falls to `level` in frames [a, b]: (seconds, interpolated between
    frames so 25 fps timing is not stuck on 40 ms steps; first frame at or below)."""
    for i in range(a, b + 1):
        if np.isfinite(x[i]) and x[i] <= level:
            if i > a and np.isfinite(x[i - 1]) and x[i - 1] > level:
                f = (x[i - 1] - level) / (x[i - 1] - x[i])
                return float(t[i - 1] + f * (t[i] - t[i - 1])), i
            return float(t[i]), i
    return float("nan"), None


def arm_timing(t: np.ndarray, hip: np.ndarray, elbow: np.ndarray, s: int,
               hip_end: int, elbow_end: int) -> Dict[str, float]:
    """When the elbows start bending compared with the hips, over one drive.

    arm_lead_ms: how long before the hips the elbows start (each joint's start being
        ONSET of its range); positive = arms first, negative = hips first.
    arm_bend_at_half_hinge: share of the drive's elbow bend done by the time the hips
        are halfway through their hinge, in percent.
    hip_onset: frame where the hips start (shows the arms already bent when they lead).
    Ranges: hip from the catch to its lowest by `hip_end`, elbow to its most bent by
    `elbow_end`. NaN when either joint barely moves.
    """
    nan = float("nan")
    out = {"arm_lead_ms": nan, "arm_bend_at_half_hinge": nan, "hip_onset": s}
    h0, e0 = hip[s], elbow[s]
    if not (np.isfinite(h0) and np.isfinite(e0)):
        return out
    hip_win, elb_win = hip[s:hip_end + 1], elbow[s:elbow_end + 1]
    if not (np.isfinite(hip_win).any() and np.isfinite(elb_win).any()):
        return out
    h_low = s + int(np.nanargmin(hip_win))
    e_low = s + int(np.nanargmin(elb_win))
    h_range, e_range = h0 - hip[h_low], e0 - elbow[e_low]
    if h_range < MIN_HIP_RANGE or e_range < MIN_ELBOW_RANGE:
        return out
    t_hip, i_hip = _crossing(t, hip, s, h_low, h0 - ONSET * h_range)
    t_elb, _ = _crossing(t, elbow, s, e_low, e0 - ONSET * e_range)
    t_half, _ = _crossing(t, hip, s, h_low, h0 - 0.5 * h_range)
    if np.isfinite(t_hip) and np.isfinite(t_elb):
        out["arm_lead_ms"] = (t_hip - t_elb) * 1000.0
    if np.isfinite(t_half):
        seg = slice(s, e_low + 1)
        ok = np.isfinite(elbow[seg])
        e_half = np.interp(t_half, t[seg][ok], elbow[seg][ok])
        out["arm_bend_at_half_hinge"] = 100.0 * (e0 - e_half) / e_range
    if i_hip is not None:
        out["hip_onset"] = i_hip
    return out


class SkiErg(Station):
    key = "skierg"
    name = "SkiErg"
    rep_word = "stroke"
    view = "side"
    driver = "wrist_h"
    min_rep_s, max_rep_s = 0.7, 5.0
    phases = ("drive", "recovery")
    force = True
    filming_tips = (
        "Film side-on, camera square to the athlete, at roughly hip height, 3-4 m away.",
        "Keep the whole body and the handles at full reach overhead in frame.",
        "Landscape orientation, 60 fps if your phone supports it.",
        "Nobody walking between the camera and the athlete.",
    )
    charts = (
        ChartSpec("rate_spm", "Stroke rate", "spm"),
        ChartSpec("arm_lead_ms", "Arms ahead of the hips (- = hips first)", "ms"),
        ChartSpec("hip_min", "Hip angle at bottom", "°"),
        ChartSpec("knee_min", "Knee angle at bottom", "°"),
        ChartSpec("wrist_bottom", "Hand height at finish", "torso lengths"),
    )

    def rules(self) -> List[Rule]:
        return [
            Rule("early_arm_pull", "arm_lead_ms", ">", ARM_LEAD_MS, "major",
                 "Arms pull before the hips",
                 "Start each pull by driving the hips back and down; let the elbows "
                 "bend once the hips are moving. The arms finish the stroke.",
                 "frame_hip_onset"),
            Rule("squat_pattern", "knee_min", "<", 110, "major",
                 "Squatting the handles down",
                 "Keep a soft knee and hinge at the hips. Power comes from the hips "
                 "and trunk, not from dropping into a squat.", "frame_bottom"),
            Rule("knee_dominant", "hinge_ratio", "<", 1.2, "minor",
                 "Knees bend as much as hips",
                 "Push the hips back more and keep the shins nearly vertical.",
                 "frame_bottom"),
            Rule("shallow_hinge", "hip_min", ">", 135, "minor",
                 "Too little hip hinge",
                 "Hinge further, chest toward the thighs, to use the posterior chain "
                 "instead of only the arms.", "frame_bottom"),
            Rule("over_crunch", "hip_min", "<", 60, "minor",
                 "Over-crunching at the bottom",
                 "Stop the hinge around hip height and let the arms finish; folding "
                 "deeper costs energy and slows the recovery.", "frame_bottom"),
            Rule("bent_arm_catch", "elbow_catch", "<", 130, "minor",
                 "Short reach at the catch",
                 "Reach long at the top with soft, nearly straight elbows to "
                 "lengthen the stroke.", "frame_catch"),
            Rule("low_catch", "wrist_top", "<", 1.1, "minor",
                 "Catch starts too low",
                 "Stand tall and start the pull with the hands above head height.",
                 "frame_catch"),
            Rule("short_finish", "wrist_bottom", ">", 0.25, "minor",
                 "Pull stops above the hips",
                 "Drive the hands down to or past the hips before recovering.",
                 "frame_finish"),
            Rule("rushed_recovery", "recovery_ratio", "<", 1.0, "minor",
                 "Recovery rushed",
                 "Pull fast, return with control: the recovery is where you breathe "
                 "and load the next stroke.", "frame_finish"),
        ]

    def drift_rules(self) -> List[DriftRule]:
        return [
            DriftRule("drift_rate", "rate_spm", "decrease", 0.10,
                      "Stroke rate fades", "Pacing: start slightly more conservatively "
                      "or hold rate with a shorter, sharper hinge.", relative=True),
            DriftRule("drift_hinge", "hip_min", "increase", 10,
                      "Hinge gets shallower when tired",
                      "Late in the set, keep driving the hips; don't let the arms take over."),
            DriftRule("drift_arms", "arm_lead_ms", "increase", ARM_LEAD_MS,
                      "Arms take over when tired",
                      "Re-focus on starting each pull with the hips, not the arms."),
            DriftRule("drift_squat", "knee_min", "decrease", 10,
                      "Knees bend more when tired",
                      "Keep the soft-knee hinge; squatting deeper burns the legs "
                      "you need for the run."),
        ]

    def summarize(self, body: BodySeries, rep: Cycle) -> Dict[str, float]:
        m, t = body.metrics, body.t
        s, mid, e = rep.start, rep.mid, rep.end
        early_end = s + max(1, (mid - s) // 2)
        bottom_end = mid + max(1, (e - mid) // 3)      # bottom can lag the hands

        elbow_catch = self.at(m["elbow"], s)
        elbow_early_min, f_early = self.extreme(m["elbow"], s, early_end, "min")
        hip_min, f_bottom = self.extreme(m["hip"], s, bottom_end, "min")
        knee_min, _ = self.extreme(m["knee"], s, bottom_end, "min")
        trunk_max, _ = self.extreme(m["trunk"], s, bottom_end, "max")

        drive = t[mid] - t[s]
        recovery = t[e] - t[mid]
        duration = t[e] - t[s]
        hip_flex, knee_flex = 180 - hip_min, 180 - knee_min
        timing = arm_timing(t, m["hip"], m["elbow"], s, bottom_end, mid)
        idx = body.frame_index
        return {
            "t_start": float(t[s]),
            "duration_s": float(duration),
            "rate_spm": 60.0 / duration if duration > 0 else np.nan,
            "drive_s": float(drive),
            "recovery_s": float(recovery),
            "recovery_ratio": recovery / drive if drive > 0 else np.nan,
            "elbow_catch": elbow_catch,
            "elbow_drop_early": elbow_catch - elbow_early_min,
            "arm_lead_ms": timing["arm_lead_ms"],
            "arm_bend_at_half_hinge": timing["arm_bend_at_half_hinge"],
            "hip_min": hip_min,
            "knee_min": knee_min,
            "hinge_ratio": hip_flex / max(knee_flex, 5.0),
            "trunk_max": trunk_max,
            "wrist_top": self.at(m["wrist_h"], s),
            "wrist_bottom": self.at(m["wrist_h"], mid),
            "frame_catch": float(idx[s]),
            "frame_early": float(idx[f_early]),
            "frame_hip_onset": float(idx[timing["hip_onset"]]),
            "frame_bottom": float(idx[f_bottom]),
            "frame_finish": float(idx[mid]),
            "frame_mid": float(idx[mid]),
        }
