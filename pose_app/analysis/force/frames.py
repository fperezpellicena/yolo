"""The force model frame by frame, for a page to draw over the video (forces.json).

One sample per analysed frame, at `t` seconds in the uploaded video. Positions
are in the analysed frame's pixels (x right, y down: the uploaded video as it
is displayed), forces in newtons in the same directions, both sides of the
body together:

  * `points`: the model's moving points (grip, wrist, elbow, shoulder, ear,
    hip, knee). The feet are flat and still in the model, so the ankle, heel
    and toe are fixed and listed once in `fixed_points`, on `floor_y`.
  * `segments`: the model's rigid segments between those points, each with
    its mass and where its centre of mass lies (a fraction of the way from
    `from` to `to`), so a page can draw them; `body_com` is the whole body's
    centre of mass, feet included.
  * `tension`: the cords' pull on the hands, along the grip -> `cord_exit`
    line; 0 between drives, null where the force is not known.
  * `passed`: at each joint of `joints`, the force the body above it (the
    hand side) passes on to the body below it, towards the floor.
  * `floor` and `cop_x`: the floor's force on the feet and where it acts on
    the floor line (the centre of pressure).
  * `moment`: the net moment at each joint, per side, positive as the first
    of its `moment_names`.

`reported` names the joints whose values the report quotes; the others and the
floor are estimates until a validation study supports them (see the README).
Pairs are flattened, x then y; a missing value is null.
"""

import json
import math
from typing import Dict, List, Optional

import numpy as np

from .analysis import ForceAnalysis
from .dynamics import MOMENT_NAMES
from .model import G, REPORTED_JOINTS, SEGMENTS

VERSION = 1
POINTS = ("grip", "wrist", "elbow", "shoulder", "ear", "hip", "knee")
JOINTS = ("elbow", "shoulder", "hip", "knee", "ankle")    # the chain from the cord to the floor
# The model's segments as drawn: the forearm and hand reach the grip; the rest as in SEGMENTS
DRAWN = {"forearm_hand": ("elbow", "grip")}


def _round(v: float, digits: int) -> Optional[float]:
    if not math.isfinite(v):
        return None
    r = round(float(v), digits)
    return int(r) if digits == 0 else r


def _scalars(a: np.ndarray, digits: int) -> List[Optional[float]]:
    return [_round(v, digits) for v in np.asarray(a, float)]


def _pairs(a: np.ndarray, digits: int) -> List[Optional[float]]:
    """(N, 2) -> [x0, y0, x1, y1, ...], both null where either is missing."""
    out: List[Optional[float]] = []
    for x, y in np.asarray(a, float):
        if math.isfinite(x) and math.isfinite(y):
            out += [_round(x, digits), _round(y, digits)]
        else:
            out += [None, None]
    return out


def force_frames(fa: ForceAnalysis, fps: float, frame_size) -> Optional[Dict]:
    """The export, or None when the force analysis did not get as far as the dynamics."""
    if fa.dyn is None or fa.kin is None or fa.frame is None:
        return None
    frame, model, kin, dyn = fa.frame, fa.model, fa.kin, fa.dyn
    # A force in the model's frame (x towards the machine, y up) -> image directions
    image = np.array([frame.facing, -1.0])
    px = lambda m: frame.to_px(np.asarray(m, float))
    feet = {"ankle": px([0.0, model.ankle_h]), "heel": px([-model.heel, 0.0]),
            "toe": px([model.toe, 0.0])}
    segments = []
    for name, (start, end, _) in SEGMENTS.items():
        seg = model.segments[name]
        drawn_start, drawn_end = DRAWN.get(name, (start, end))
        segments.append({"name": name, "from": start, "to": end, "com": round(seg.com, 4),
                         "mass_kg": round(seg.mass, 2),
                         "drawn": [drawn_start, drawn_end]})
    floor_y = float(px([0.0, 0.0])[1])
    cop = np.where(np.isfinite(dyn.cop_x), dyn.cop_x, np.nan)
    cop_px = frame.origin[0] + frame.facing * cop * frame.scale
    return {
        "version": VERSION,
        "fps": _round(fps, 3),
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
        "px_per_m": _round(frame.scale, 3),
        "facing": int(frame.facing),
        "mass_kg": model.mass,
        "weight_n": _round(model.mass * G, 1),
        "cord_exit": [_round(v, 1) for v in fa.setup.calibration.cord_exit],
        "floor_y": _round(floor_y, 1),
        "fixed_points": {k: [_round(v, 1) for v in p] for k, p in feet.items()},
        "feet": {"mass_kg": round(model.feet_mass, 2),
                 "com": [_round(v, 1) for v in px(model.feet_com)]},
        "segments": segments,
        "joints": list(JOINTS),
        "reported": list(REPORTED_JOINTS),
        "moment_names": {j: list(MOMENT_NAMES[j]) for j in JOINTS},
        "t": _scalars(fa.t, 3),
        "points": {name: _pairs(px(kin.points[name]), 1) for name in POINTS},
        "body_com": _pairs(px(kin.body_com), 1),
        "tension": _scalars(fa.tension, 0),
        "passed": {j: _pairs(-dyn.joint_force[j] * image, 0) for j in JOINTS},
        "floor": _pairs(dyn.floor * image, 0),
        "cop_x": _scalars(cop_px, 1),
        "moment": {j: _scalars(fa.per_side(j), 1) for j in JOINTS},
    }


def write_force_frames(fa: ForceAnalysis, path: str, fps: float, frame_size) -> bool:
    """Writes forces.json; False (and nothing written) without the dynamics."""
    data = force_frames(fa, fps, frame_size)
    if data is None:
        return False
    with open(path, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    return True
