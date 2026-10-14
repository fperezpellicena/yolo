"""The feet, with a body-with-feet model (Halpe-26): when the heels leave the floor.

`heel_lift` is how far the near heel stands above the big toe, beyond its usual
height in the clip (the median: heels down), in cm, frame by frame. Heel and toe
are read in the same frame, so a drifting camera cancels out, and only their
height difference counts, so the view does not need to be exactly side-on. The
scale is the athlete's own shank (knee to ankle, 0.246 of stature; Drillis and
Contini), with the athlete's height when it is known (a force setup's) and
NOMINAL_HEIGHT_M otherwise: lifts are then within about 10%.

What the points can tell. On five real clips (2026-10-13, RTMPose-m) the heel
and toe points slide over a flat shoe as the posture changes, an apparent lift
of up to 3.7 cm every stroke, mostly at the catch. Real lifts on the same clips
(athletes stepping on and off the machine) read 6-14 cm. LIFT_CM sits between
the two (provisional). The median baseline assumes the heels are down for most
of the clip.

17-point models measure no feet: every value is NaN.
"""

from typing import Optional

import numpy as np

NOMINAL_HEIGHT_M = 1.75
SHANK_OF_STATURE = 0.246   # knee to ankle (as force/model.py's SEGMENT_H)
LIFT_CM = 5.0              # a heel this far above its flat height is off the floor
STEP_CM = 6.0              # the toe travelling more than this within a stroke: the athlete stepped


def px_per_m(knee: np.ndarray, ankle: np.ndarray, height_m: Optional[float] = None) -> float:
    """Pixels per metre at the feet, from the median shank length; NaN without one."""
    shank = np.linalg.norm(knee - ankle, axis=1)
    if not np.isfinite(shank).any():
        return float("nan")
    return float(np.nanmedian(shank)) / (SHANK_OF_STATURE * (height_m or NOMINAL_HEIGHT_M))


def heel_lift(heel: np.ndarray, toe: np.ndarray, scale: float) -> np.ndarray:
    """(N,) cm: the heel's height above the toe, less its median; NaN where unmeasured."""
    rise = toe[:, 1] - heel[:, 1]                     # + : heel above the toe (y grows down)
    if not (np.isfinite(rise).any() and np.isfinite(scale) and scale > 0):
        return np.full(len(rise), np.nan)
    return 100.0 * (rise - np.nanmedian(rise)) / scale


def travel_cm(p: np.ndarray, a: int, b: int, scale: float) -> float:
    """How far a point moved over frames [a, b], 5th to 95th percentile, in cm."""
    seg = p[a:b + 1]
    seg = seg[np.isfinite(seg).all(axis=1)]
    if len(seg) < 3 or not (np.isfinite(scale) and scale > 0):
        return float("nan")
    spread = np.percentile(seg, 95, axis=0) - np.percentile(seg, 5, axis=0)
    return 100.0 * float(np.hypot(*spread)) / scale
