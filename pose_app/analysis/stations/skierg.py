"""SkiErg: side view, one rep = one stroke (hands high -> hands low -> hands high).

Efficient pattern the rules look for:
  * Tall catch with long, nearly straight arms.
  * Pull initiated by hinging the hips; the elbow angle stays locked for the
    first part of the drive (lats and trunk move the handles, not the biceps).
  * Hip-dominant hinge with a soft knee, not a squat.
  * Hands finish at or past the hips.
  * Recovery at least as long as the drive (breathe, reset, re-load).
"""

from typing import Dict, List

import numpy as np

from ..body import BodySeries
from ..rules import DriftRule, Rule
from ..signal import Cycle
from .base import ChartSpec, Station


class SkiErg(Station):
    key = "skierg"
    name = "SkiErg"
    rep_word = "stroke"
    view = "side"
    driver = "wrist_h"
    min_rep_s, max_rep_s = 0.7, 5.0
    phases = ("drive", "recovery")
    filming_tips = (
        "Film side-on, camera square to the athlete, at roughly hip height, 3-4 m away.",
        "Keep the whole body and the handles at full reach overhead in frame.",
        "Landscape orientation, 60 fps if your phone supports it.",
        "Nobody walking between the camera and the athlete.",
    )
    charts = (
        ChartSpec("rate_spm", "Stroke rate", "spm"),
        ChartSpec("elbow_drop_early", "Elbow bend in first half of pull", "°"),
        ChartSpec("hip_min", "Hip angle at bottom", "°"),
        ChartSpec("knee_min", "Knee angle at bottom", "°"),
        ChartSpec("wrist_bottom", "Hand height at finish", "torso lengths"),
    )

    def rules(self) -> List[Rule]:
        return [
            Rule("early_arm_pull", "elbow_drop_early", ">", 35, "major",
                 "Arms pull before the hips",
                 "Keep the elbow angle locked for the first half of the pull and "
                 "start it by driving the hips back and down; the arms finish the stroke.",
                 "frame_early"),
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
            DriftRule("drift_arms", "elbow_drop_early", "increase", 10,
                      "Arms take over when tired",
                      "Re-focus on locked elbows in the first half of the pull."),
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
            "hip_min": hip_min,
            "knee_min": knee_min,
            "hinge_ratio": hip_flex / max(knee_flex, 5.0),
            "trunk_max": trunk_max,
            "wrist_top": self.at(m["wrist_h"], s),
            "wrist_bottom": self.at(m["wrist_h"], mid),
            "frame_catch": float(idx[s]),
            "frame_early": float(idx[f_early]),
            "frame_bottom": float(idx[f_bottom]),
            "frame_finish": float(idx[mid]),
            "frame_mid": float(idx[mid]),
        }
