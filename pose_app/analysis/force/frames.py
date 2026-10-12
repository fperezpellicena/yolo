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
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np

from .analysis import ForceAnalysis
from .dynamics import MOMENT_NAMES, moment_arm
from .model import G, REPORTED_JOINTS, SEGMENTS

VERSION = 1
POINTS = ("grip", "wrist", "elbow", "shoulder", "ear", "hip", "knee")
JOINTS = ("elbow", "shoulder", "hip", "knee", "ankle")    # the chain from the cord to the floor
# The model's segments as drawn: the forearm and hand reach the grip; the rest as in SEGMENTS
DRAWN = {"forearm_hand": ("elbow", "grip")}
ARM_JOINTS = ("shoulder", "elbow")      # their distance to the cord's line: the geometry of the pull
TYPICAL_GRID = np.linspace(0.0, 100.0, 51)
MIN_STROKES = 3                         # a typical value needs this many strokes behind it


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


def _typical(series: Dict[str, np.ndarray], strokes: Sequence[Sequence[int]],
             digits: Dict[str, int]) -> Dict:
    """Each series through the typical stroke: resampled from 0 to 100% of every stroke, then
    the median and the middle half (25th to 75th percentile) of the strokes at each point, so
    one odd stroke does not bend it. Null
    where fewer than MIN_STROKES strokes have a value, as during the drives without the PM5."""
    out = {}
    for name, values in series.items():
        values = np.asarray(values, float)
        rows = np.full((len(strokes), len(TYPICAL_GRID)), np.nan)
        for k, (a, b) in enumerate(strokes):
            pos = a + TYPICAL_GRID / 100.0 * (b - a)
            i = np.minimum(np.floor(pos).astype(int), b - 1)
            w = pos - i
            rows[k] = values[i] * (1 - w) + values[i + 1] * w
        median = low = high = np.full(len(TYPICAL_GRID), np.nan)
        enough = np.isfinite(rows).sum(axis=0) >= MIN_STROKES
        if enough.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN columns
                median = np.where(enough, np.nanmedian(rows, axis=0), np.nan)
                low = np.where(enough, np.nanpercentile(rows, 25, axis=0), np.nan)
                high = np.where(enough, np.nanpercentile(rows, 75, axis=0), np.nan)
        out[name] = {"median": _scalars(median, digits[name]), "low": _scalars(low, digits[name]),
                     "high": _scalars(high, digits[name])}
    return out


def force_frames(fa: ForceAnalysis, fps: float, frame_size,
                 cycles: Sequence = ()) -> Optional[Dict]:
    """The export, or None when the force analysis did not get as far as the dynamics.
    `cycles` are the analysis' strokes, for the typical stroke."""
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
    grip = kin.points["grip"]
    arms = {j: 100.0 * moment_arm(kin.points[j], grip, fa.cord_dir) for j in ARM_JOINTS}
    com_height = 100.0 * kin.body_com[:, 1]
    n = len(fa.t)
    strokes = [[int(c.start), int(min(c.end, n - 1))] for c in cycles
               if 0 <= c.start and min(c.end, n - 1) - c.start >= 4]
    series = {**{j: fa.per_side(j) for j in JOINTS}, "tension": fa.tension,
              "floor": np.linalg.norm(dyn.floor, axis=1),
              **{f"arm_{j}": arms[j] for j in ARM_JOINTS}, "com_height": com_height}
    digits = {**dict.fromkeys(JOINTS, 1), "tension": 0, "floor": 0,
              **{f"arm_{j}": 1 for j in ARM_JOINTS}, "com_height": 1}
    return {
        "version": VERSION,
        # "pm5": the cords' pull from the PM5 log; "geometry": the video alone, so the forces
        # are null while the cords pull and only the geometry is known there
        "mode": "pm5" if fa.pm5 is not None else "geometry",
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
        "arm": {j: _scalars(arms[j], 1) for j in ARM_JOINTS},
        "com_height": _scalars(com_height, 1),
        "strokes": strokes,
        "typical": {"x": [round(float(v), 1) for v in TYPICAL_GRID], "strokes": len(strokes),
                    "series": _typical(series, strokes, digits)},
    }


def write_force_frames(fa: ForceAnalysis, path: str, fps: float, frame_size,
                       cycles: Sequence = ()) -> bool:
    """Writes forces.json; False (and nothing written) without the dynamics."""
    data = force_frames(fa, fps, frame_size, cycles)
    if data is None:
        return False
    with open(path, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    return True
