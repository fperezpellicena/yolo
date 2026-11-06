"""RowErg (Concept2 rower): side view, one rep = one stroke (catch -> finish -> catch).

The athlete faces the flywheel; the feet sit still on the footplate, so positions are
read ahead of the ankle (body.py's x_* metrics, torso lengths): the seat (hip), the
shoulders and the handle (wrist). Kleshnev's decomposition splits the handle's travel
into legs (hip on the ankle), trunk (shoulder on the hip) and arms (wrist on the
shoulder).

Efficient pattern the rules look for (Concept2, British Rowing, Kleshnev):
  * Catch: shins about vertical, not past it; trunk leaning forward from the hips;
    arms long; handle about knee height.
  * Drive: legs first; the trunk opens once the knees pass about 90 degrees; the arms
    finish. The seat and the handle move off together (no "shooting the slide").
  * Finish: legs down, trunk slightly back, handle to the lower ribs.
  * Recovery: hands away and past the knees before the knees bend; the trunk swings
    forward by halfway up the slide (no "diving" at the catch); the slide slower than
    the drive. The drive's share of the stroke grows with the rate (Kleshnev 2005).

Thresholds start looser than the world-class ranges they come from (2026-10-18), to be
tightened with real Hyrox clips. Heels rising at the catch is normal on the rower
(Concept2), so the heels are reported, never judged. Sources:
claude/rowing-technique-research.md in the project.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..body import BodySeries
from ..rules import DriftRule, Rule
from ..signal import Cycle
from .base import ChartSpec, Station

# Drive share of the stroke on a Concept2: 41.7% at 20.7 spm, 51.9% at 32.1 (Kleshnev 2005)
RHYTHM_AT, RHYTHM_RATE, RHYTHM_SLOPE = 41.7, 20.7, 0.9
RHYTHM_RANGE = (35.0, 60.0)
WINDOW = 0.10           # catch / finish windows: this share of the stroke on each side
RSF_SHARE = 0.20        # Kleshnev's Rowing Style Factor: over the first 20% of the handle's travel
TRANSITION_KNEE = 90.0  # the trunk should hold its angle until the knees pass this
ARM_ONSET = 0.25        # elbows "start" bending when they have done this share of their bend
KNEE_ONSET = 0.15       # knees "start" bending in the recovery at this share of their bend
MIN_ELBOW_RANGE = 20.0  # degrees; less and the arms barely bend: their timing means nothing
MIN_KNEE_RANGE = 30.0
PAUSE_S = 0.3           # the handle moving less than STILL_TOL of the stroke length in this
STILL_TOL = 0.015       # long is still (a pause at the catch), not turning


def rhythm_reference(rate_spm: float) -> float:
    """Usual drive share of the stroke (%) at this rate on a Concept2."""
    ref = RHYTHM_AT + RHYTHM_SLOPE * (rate_spm - RHYTHM_RATE)
    return float(np.clip(ref, *RHYTHM_RANGE))


def crossing(t: np.ndarray, x: np.ndarray, a: int, b: int, level: float,
             rising: bool) -> Tuple[float, Optional[int]]:
    """When `x` first reaches `level` in frames [a, b], going up (`rising`) or down:
    (seconds, interpolated between frames; first frame at or past the level)."""
    sign = 1.0 if rising else -1.0
    for i in range(max(a, 0), min(b, len(x) - 1) + 1):
        if np.isfinite(x[i]) and sign * (x[i] - level) >= 0:
            if i > a and np.isfinite(x[i - 1]) and sign * (x[i - 1] - level) < 0:
                f = (level - x[i - 1]) / (x[i] - x[i - 1])
                return float(t[i - 1] + f * (t[i] - t[i - 1])), i
            return float(t[i]), i
    return float("nan"), None


def value_at(t: np.ndarray, x: np.ndarray, a: int, b: int, when: float) -> float:
    """`x` interpolated at time `when` within frames [a, b]."""
    if not math.isfinite(when):
        return float("nan")
    seg = slice(max(a, 0), min(b, len(x) - 1) + 1)
    ok = np.isfinite(x[seg])
    if ok.sum() < 2:
        return float("nan")
    return float(np.interp(when, t[seg][ok], x[seg][ok]))


def velocity(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.gradient(x, t) if len(x) > 1 else np.full_like(x, np.nan)


class RowErg(Station):
    key = "rowerg"
    name = "RowErg"
    rep_word = "stroke"
    view = "side"
    driver = "x_wrist"            # the handle ahead of the ankle: furthest at the catch
    min_rep_s, max_rep_s = 1.2, 6.0
    phases = ("drive", "recovery")
    force = False
    filming_tips = (
        "Film side-on, camera square to the rower at about seat height, far enough to keep "
        "the whole machine in frame: the flywheel and the athlete at both ends of the slide.",
        "Film the side of the athlete nearest the camera unblocked by the monorail.",
        "Landscape orientation, 60 fps if your phone supports it (the hands move fast at "
        "the finish).",
        "Start filming before the first stroke and sit still for a moment at the catch.",
    )
    charts = (
        ChartSpec("rate_spm", "Stroke rate", "spm"),
        ChartSpec("stroke_len", "Stroke length (handle travel)", "torso lengths"),
        ChartSpec("rhythm_excess", "Drive share of the stroke beyond the usual for the rate",
                  "points", (0.0, "Usual for the rate")),
        ChartSpec("slide_lead", "Seat travel / handle travel early in the drive", "",
                  (0.95, "Top of the efficient range")),
        ChartSpec("legs_at_arm_bend", "Legs' drive done when the arms bend", "%"),
        ChartSpec("trunk_open_at_90", "Trunk opened before the knees pass 90°", "°"),
        ChartSpec("knee_catch", "Knee angle at the catch", "°", (47.0, "World-class average")),
        ChartSpec("trunk_catch", "Forward lean at the catch", "°", (24.5, "World-class average")),
        ChartSpec("trunk_finish", "Layback at the finish", "°", (26.3, "World-class average")),
        ChartSpec("legs_pct", "Legs' share of the stroke length", "%"),
        ChartSpec("knees_after_hands_ms", "Knees bend after the hands pass them", "ms"),
        # shown only when a body-with-feet model measured the feet
        ChartSpec("heel_lift_catch", "Heels up around the catch", "cm"),
    )

    def rules(self) -> List[Rule]:
        return [
            Rule("arms_early", "legs_at_arm_bend", "<", 50, "major",
                 "Arms bend before the legs have driven",
                 "Keep the arms long like ropes while the legs push; the arms only draw the "
                 "handle in once the legs are nearly down.", "frame_arm_bend"),
            Rule("body_opens_early", "trunk_open_at_90", ">", 15, "major",
                 "Back opens before the legs",
                 "Hold the forward lean while the legs push; swing the back open only once "
                 "the handle has passed the knees.", "frame_knee90"),
            Rule("shooting_slide", "slide_lead", ">", 1.20, "major",
                 "Shooting the slide",
                 "Seat and handle should move off together: brace the core at the catch so "
                 "the leg push reaches the handle instead of only moving the seat.",
                 "frame_rsf"),
            Rule("knees_early", "knees_after_hands_ms", "<", -100, "minor",
                 "Knees rise before the hands pass them",
                 "Hands away first, then the body over, then the knees: clear the knees with "
                 "the handle before you slide.", "frame_knees_up"),
            Rule("diving_catch", "trunk_dive", ">", 12, "minor",
                 "Diving into the catch",
                 "Swing the body forward early in the recovery, by halfway up the slide, then "
                 "keep that angle into the catch.", "frame_next_catch"),
            Rule("over_compression", "shin_catch", ">", 20, "minor",
                 "Over-compressed at the catch",
                 "Stop the slide with the shins about vertical; going further shortens the "
                 "push and folds the body.", "frame_catch"),
            Rule("short_catch", "knee_catch", ">", 85, "minor",
                 "Short slide at the catch",
                 "Come further up the slide, shins to vertical, to use the legs' full "
                 "range.", "frame_catch"),
            Rule("upright_catch", "trunk_catch", "<", 5, "minor",
                 "No forward lean at the catch",
                 "Rock the body forward from the hips in the recovery so the catch starts "
                 "with the shoulders ahead of the hips.", "frame_catch"),
            Rule("over_reach", "trunk_catch", ">", 45, "minor",
                 "Over-reaching at the catch",
                 "Lean forward from the hips but stop short of lunging; reaching further "
                 "rounds the back.", "frame_catch"),
            Rule("over_layback", "trunk_finish", ">", 45, "minor",
                 "Leaning back too far",
                 "Finish slightly past vertical, held by the core; leaning further costs time "
                 "and loads the back.", "frame_finish"),
            Rule("no_layback", "trunk_finish", "<", 0, "minor",
                 "No layback at the finish",
                 "Swing the body slightly past vertical at the finish to lengthen the "
                 "stroke.", "frame_finish"),
            Rule("legs_not_down", "knee_finish", "<", 150, "minor",
                 "Legs not pushed down",
                 "Drive the legs all the way down before the finish.", "frame_finish"),
            Rule("bent_arm_catch", "elbow_catch", "<", 145, "minor",
                 "Arms bent at the catch",
                 "Reach with long, relaxed arms at the catch; bent arms pull before the legs "
                 "and shorten the stroke.", "frame_catch"),
            Rule("handle_drop_catch", "handle_catch_h", "<", -0.35, "minor",
                 "Handle dropped at the catch",
                 "Let the handle rise to about knee height as you come forward instead of "
                 "dropping it in front of the shins.", "frame_catch"),
            Rule("high_finish", "handle_finish_pos", ">", 0.9, "minor",
                 "Handle drawn too high",
                 "Draw the handle to the lower ribs, elbows past the body, shoulders low.",
                 "frame_finish"),
            Rule("low_finish", "handle_finish_pos", "<", 0.15, "minor",
                 "Handle drawn to the lap",
                 "Draw the handle to the lower ribs rather than the lap.", "frame_finish"),
            Rule("rushed_recovery", "rhythm_excess", ">", 12, "minor",
                 "Recovery rushed",
                 "Drive hard, come forward with control: at this rate the recovery should "
                 "take longer.", "frame_finish"),
            Rule("rushing_slide", "slide_speed_ratio", ">", 1.1, "minor",
                 "Rushing up the slide",
                 "Come up the slide slower than you drive back; let the flywheel run.",
                 "frame_half_slide"),
        ]

    def drift_rules(self) -> List[DriftRule]:
        return [
            DriftRule("drift_rate", "rate_spm", "decrease", 0.10,
                      "Stroke rate fades", "Pacing: start slightly more conservatively, or "
                      "hold the rate with a quicker hands-away.", relative=True),
            DriftRule("drift_length", "stroke_len", "decrease", 0.08,
                      "Strokes get shorter when tired",
                      "Keep coming all the way up the slide and finishing with the legs down "
                      "and the body back.", relative=True),
            DriftRule("drift_catch_lean", "trunk_catch", "increase", 6,
                      "Leaning further forward at the catch when tired",
                      "Late in the piece, sit tall and swing from the hips; a back that rounds "
                      "to reach the catch takes the load off the legs."),
            DriftRule("drift_sequence", "slide_lead", "increase", 0.15,
                      "Legs run ahead of the handle when tired",
                      "Re-connect the catch: brace before you push so seat and handle move "
                      "together."),
            DriftRule("drift_layback", "trunk_finish", "increase", 6,
                      "Leaning back further when tired",
                      "Keep the finish compact; leaning back to find length costs time."),
        ]

    @staticmethod
    def moving(hx: np.ndarray, t: np.ndarray, rep: Cycle) -> Tuple[int, int, int]:
        """The stroke's catches where the handle actually turns. The athlete sitting still
        at the catch (before the first stroke, after the last, or resting mid-piece) puts
        the cycle's peak anywhere in that pause: a pause next to the peak is cut, the start
        moved to where the handle leaves it and the end to where it arrives. Still means the
        handle moving less than STILL_TOL of the stroke length within PAUSE_S; the ordinary
        turn of the handle at a catch moves more, so stroke times keep their full length."""
        s, mid, e = rep.start, rep.mid, rep.end
        seg = hx[s:e + 1]
        if not np.isfinite(seg).any():
            return s, mid, e
        span = float(np.nanmax(seg) - np.nanmin(seg))
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
        if not (span > 0 and dt > 0):
            return s, mid, e
        w = max(1, int(round(PAUSE_S / 2 / dt)))

        def still(i: int) -> bool:
            chunk = hx[max(0, i - w): i + w + 1]
            chunk = chunk[np.isfinite(chunk)]
            return chunk.size > w and float(np.ptp(chunk)) < STILL_TOL * span

        if still(s):
            i = s
            while i + 1 < mid and still(i + 1):
                i += 1
            s = min(i + w, mid - 1)
        if still(e):
            i = e
            while i - 1 > mid and still(i - 1):
                i -= 1
            e = max(i - w, mid + 1)
        return s, mid, e

    def summarize(self, body: BodySeries, rep: Cycle) -> Dict[str, float]:
        m, t, idx = body.metrics, body.t, body.frame_index
        last = len(t) - 1
        hx, sx, shx, kx = m["x_wrist"], m["x_hip"], m["x_shoulder"], m["x_knee"]
        s, mid, e = self.moving(hx, t, rep)
        k = max(1, int(round(WINDOW * (e - s))))
        cw = (max(0, s - k), min(last, s + max(1, k // 2)))   # around this stroke's catch
        before = (max(0, s - k), min(last, s + 1))           # into the catch, for the lean
        fw = (max(0, mid - k), min(last, mid + k))           # around the finish
        nw = (max(0, e - k), min(last, e + 1))               # into the next catch
        nan = float("nan")

        duration, drive, recovery = t[e] - t[s], t[mid] - t[s], t[e] - t[mid]
        rate = 60.0 / duration if duration > 0 else nan
        rhythm = 100.0 * drive / duration if duration > 0 else nan

        knee, trunk, elbow = m["knee"], m["trunk"], m["elbow"]

        # Stroke length and each segment's part of it (Kleshnev)
        hip_c, hip_f = self.at(sx, s), self.at(sx, mid)
        sh_c, sh_f = self.at(shx, s), self.at(shx, mid)
        hand_c, hand_f = self.at(hx, s), self.at(hx, mid)
        legs, trunk_d, arms = hip_c - hip_f, (sh_c - hip_c) - (sh_f - hip_f), \
            (hand_c - sh_c) - (hand_f - sh_f)
        length = legs + trunk_d + arms
        share = (lambda v: 100.0 * v / length) if length > 0.1 else (lambda v: nan)

        # Catch and finish posture
        knee_catch, f_knee_catch = self.extreme(knee, *cw, "min")
        shin_catch, _ = self.extreme(m["shank"], *cw, "max")
        trunk_catch, _ = self.extreme(trunk, *before, "max")
        knee_finish, f_knee_finish = self.extreme(knee, *fw, "max")
        lay, _ = self.extreme(trunk, *fw, "min")
        L = body.torso_px
        kp_knee, kp_wrist = body.near("knee"), body.near("wrist")
        kp_hip, kp_sh = body.near("hip"), body.near("shoulder")
        handle_h = (kp_knee[:, 1] - kp_wrist[:, 1]) / L          # + = handle above the knee
        torso = kp_sh - kp_hip
        with np.errstate(invalid="ignore", divide="ignore"):
            along = np.sum((kp_wrist - kp_hip) * torso, axis=1) / np.sum(torso * torso, axis=1)

        # Seat vs handle early in the drive (Rowing Style Factor)
        rsf, f_rsf = nan, s
        if length > 0.1:
            when, i = crossing(t, hx, s, mid, hx[s] - RSF_SHARE * length, rising=False)
            seat = value_at(t, sx, s, mid, when)
            if math.isfinite(seat):
                rsf, f_rsf = (sx[s] - seat) / (RSF_SHARE * length), i

        # Trunk opening before the knees pass 90 degrees
        open90, f90 = nan, f_knee_catch
        if math.isfinite(knee_catch) and knee_catch < TRANSITION_KNEE:
            when, i = crossing(t, knee, f_knee_catch, mid, TRANSITION_KNEE, rising=True)
            tr = value_at(t, trunk, s, mid, when)
            if math.isfinite(tr):
                open90, f90 = trunk_catch - tr, i

        # How much of the legs' drive is done when the elbows start bending
        legs_at_bend, f_bend = nan, s
        e0 = self.at(elbow, s)
        e_low, f_low = self.extreme(elbow, s, fw[1], "min")
        if math.isfinite(e0) and e0 - e_low >= MIN_ELBOW_RANGE and abs(hip_c - hip_f) > 0.05:
            when, i = crossing(t, elbow, s, f_low, e0 - ARM_ONSET * (e0 - e_low), rising=False)
            seat = value_at(t, sx, s, f_low, when)
            if math.isfinite(seat):
                legs_at_bend, f_bend = 100.0 * (sx[s] - seat) / (sx[s] - sx[mid]), i

        # Segment velocities (torso lengths / s) and when each peaks in the drive
        v = {"legs": -velocity(sx, t), "trunk": -velocity(shx - sx, t),
             "arms": -velocity(hx - shx, t), "handle": -velocity(hx, t)}
        seq: Dict[str, float] = {}
        for name, series in v.items():
            peak, f_peak = self.extreme(series, s, mid, "max")
            seq[f"v_{name}_max"] = peak
            if name != "handle":
                seq[f"seq_{name}_ms"] = 1000.0 * (t[f_peak] - t[s]) if math.isfinite(peak) else nan

        # Recovery: the hands pass the knees, then the knees bend
        knees_after, f_knees = nan, mid
        t_hands, _ = crossing(t, hx - kx, mid, e, 0.0, rising=True)
        knee_next, _ = self.extreme(knee, *nw, "min")
        if math.isfinite(knee_finish) and knee_finish - knee_next >= MIN_KNEE_RANGE:
            level = knee_finish - KNEE_ONSET * (knee_finish - knee_next)
            t_knees, i = crossing(t, knee, f_knee_finish, e, level, rising=False)
            if math.isfinite(t_hands) and math.isfinite(t_knees):
                knees_after, f_knees = 1000.0 * (t_knees - t_hands), i

        # Recovery: the trunk's forward swing left for the second half of the slide
        dive, f_half = nan, mid
        seat_fin, _ = self.extreme(sx, *fw, "min")
        seat_next, _ = self.extreme(sx, *nw, "max")
        trunk_next, _ = self.extreme(trunk, *nw, "max")
        if math.isfinite(seat_fin) and math.isfinite(seat_next) and seat_next - seat_fin > 0.1:
            when, i = crossing(t, sx, mid, e, (seat_fin + seat_next) / 2, rising=True)
            tr = value_at(t, trunk, mid, e, when)
            if math.isfinite(tr):
                dive, f_half = trunk_next - tr, i

        slide_back, _ = self.extreme(v["legs"], s, mid, "max")
        slide_fwd, _ = self.extreme(-v["legs"], mid, e, "max")
        heel_catch, _ = self.extreme(m["heel_lift"], *cw, "max")

        def frame(i: Optional[int]) -> float:
            return float(idx[min(max(int(i if i is not None else s), 0), last)])

        return {
            "t_start": float(t[s]),
            "duration_s": float(duration),
            "rate_spm": rate,
            "drive_s": float(drive),
            "recovery_s": float(recovery),
            "rhythm_pct": rhythm,
            "rhythm_excess": rhythm - rhythm_reference(rate) if math.isfinite(rate) else nan,
            "stroke_len": length,
            "legs_pct": share(legs),
            "trunk_pct": share(trunk_d),
            "arms_pct": share(arms),
            "knee_catch": knee_catch,
            "shin_catch": shin_catch,
            "trunk_catch": trunk_catch,
            "elbow_catch": e0,
            "handle_catch_h": self.at(handle_h, s),
            "knee_finish": knee_finish,
            "trunk_finish": -lay,
            "handle_finish_pos": self.at(along, mid),
            "slide_lead": rsf,
            "trunk_open_at_90": open90,
            "legs_at_arm_bend": legs_at_bend,
            **seq,
            "knees_after_hands_ms": knees_after,
            "trunk_dive": dive,
            "slide_speed_ratio": slide_fwd / slide_back if slide_back > 0 else nan,
            "heel_lift_catch": heel_catch,            # cm, reported only
            "frame_catch": frame(s),
            "frame_finish": frame(mid),
            "frame_mid": frame(mid),
            "frame_rsf": frame(f_rsf),
            "frame_knee90": frame(f90),
            "frame_arm_bend": frame(f_bend),
            "frame_knees_up": frame(f_knees),
            "frame_half_slide": frame(f_half),
            "frame_next_catch": frame(e),
        }
