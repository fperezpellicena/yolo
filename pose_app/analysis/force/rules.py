"""Force-based technique rules, fatigue checks and charts (SkiErg with a PM5).

Thresholds are starting heuristics from the synthetic reference stroke in the
requirement analysis (a hinge-led stroke takes about 40% of its work from the
body's drop, with the cords about 78 degrees from horizontal at peak force,
and its force peaks within the first third of the cord's travel); there is no
published SkiErg reference yet, so tune them with a coach like any other rule.
"""

from typing import List

from ..rules import DriftRule, Rule
from ..stations.base import ChartSpec

FRAME = "frame_force_peak"


def force_rules() -> List[Rule]:
    return [
        Rule("low_bodyweight_share", "f_bw_share", "<", 25, "minor",
             "Not using body weight",
             "Hinge first and hang on the handles: let the body's drop load the cords, "
             "and the arms finish the pull.", FRAME),
        Rule("shallow_cord", "f_cord_angle", "<", 65, "minor",
             "Standing too far from the machine",
             "Step closer until the cords hang nearly vertical at peak force: a shallow cord "
             "pulls you forward and makes you sit back out of the hinge.", FRAME),
        Rule("late_force_peak", "f_peak_pos", ">", 45, "minor",
             "Force peaks late in the pull",
             "Fall into the catch: let the force build early in the pull instead of "
             "saving it for the finish.", FRAME),
    ]


def force_drift_rules() -> List[DriftRule]:
    return [
        DriftRule("drift_bodyweight", "f_bw_share", "decrease", 8,
                  "Less body weight when tired",
                  "Late in the piece, keep hinging and hanging on the handles; don't let "
                  "the arms take over the pull."),
        DriftRule("drift_peak_force", "f_peak_n", "decrease", 0.10, "Peak force fades",
                  "The pull gets weaker late in the piece: protect the hinge and the catch, "
                  "or pace the start more evenly.", relative=True),
    ]


FORCE_CHARTS = (ChartSpec("f_peak_n", "Peak cord force", "N"),
                ChartSpec("f_bw_share", "Body-weight share of the work", "%"),
                ChartSpec("f_cord_angle", "Cord angle at peak force", "°"),
                ChartSpec("f_shoulder_ext_nm", "Shoulder moment at peak force, per side", "N·m"))
