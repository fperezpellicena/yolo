"""What a force analysis needs besides the video and the PM5 log.

The athlete's profile scales the body model, and the calibration places the
video in metres: the cord exit (where the cords leave the top of the machine)
and, ideally, a known length filmed in the athlete's plane (a 1 m stick held
at the midline before the piece). Pixel positions are in the analysed frame,
after any --rotate, as the first frame of the clip shows it.

As JSON (CLI --force-setup file, or a job's "force" option):

    {"athlete": {"mass_kg": 80, "height_m": 1.80, "sex": "male"},
     "calibration": {"cord_exit": [612, 88],
                     "scale": {"points": [[402, 905], [398, 602]], "length_m": 1.0}}}

"scale" is optional: without it the image scale is estimated from the
athlete's height, which is less accurate (the report says so).
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

SEXES = ("male", "female", "unspecified")
Point = Tuple[float, float]


@dataclass(frozen=True)
class AthleteProfile:
    mass_kg: float
    height_m: float
    sex: str = "unspecified"            # picks the segment table; "unspecified" averages both


@dataclass(frozen=True)
class Calibration:
    cord_exit: Point                                    # px
    scale_points: Optional[Tuple[Point, Point]] = None  # px, the ends of a known length
    scale_length_m: Optional[float] = None

    @property
    def stick_px_per_m(self) -> Optional[float]:
        if self.scale_points is None or not self.scale_length_m:
            return None
        (x0, y0), (x1, y1) = self.scale_points
        return math.hypot(x1 - x0, y1 - y0) / self.scale_length_m


@dataclass(frozen=True)
class ForceSetup:
    athlete: AthleteProfile
    calibration: Calibration

    def problems(self) -> List[str]:
        """Why this setup cannot be used (empty if it can)."""
        a, c, out = self.athlete, self.calibration, []
        if not (_finite(a.mass_kg) and 30 <= a.mass_kg <= 250):
            out.append(f"athlete mass {a.mass_kg!r} kg is outside 30-250")
        if not (_finite(a.height_m) and 1.2 <= a.height_m <= 2.3):
            out.append(f"athlete height {a.height_m!r} m is outside 1.2-2.3 (metres, not cm)")
        if a.sex not in SEXES:
            out.append(f"sex must be one of {', '.join(SEXES)}")
        if not (len(c.cord_exit) == 2 and all(_finite(v) for v in c.cord_exit)):
            out.append("cord_exit must be [x, y] in pixels")
        if (c.scale_points is None) != (c.scale_length_m is None):
            out.append("scale needs both its two points and its length")
        elif c.scale_points is not None:
            pts = [v for p in c.scale_points for v in p]
            if len(c.scale_points) != 2 or len(pts) != 4 or not all(_finite(v) for v in pts):
                out.append("scale points must be [[x, y], [x, y]] in pixels")
            elif not (_finite(c.scale_length_m) and 0.1 <= c.scale_length_m <= 5):
                out.append(f"scale length {c.scale_length_m!r} m is outside 0.1-5")
            elif c.stick_px_per_m * c.scale_length_m < 20:
                out.append("the scale points are less than 20 px apart")
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ForceSetup":
        """Parse the JSON form; ValueError (or TypeError) if its shape is wrong."""
        if not isinstance(d, dict) or set(d) - {"athlete", "calibration"}:
            raise ValueError("expected {athlete: {...}, calibration: {...}}")
        a, c = d.get("athlete"), d.get("calibration")
        if not isinstance(a, dict) or not isinstance(c, dict):
            raise ValueError("expected {athlete: {...}, calibration: {...}}")
        unknown = (set(a) - {"mass_kg", "height_m", "sex"}) | (set(c) - {"cord_exit", "scale"})
        if unknown:
            raise ValueError(f"unknown fields {sorted(unknown)}")
        athlete = AthleteProfile(_number(a.get("mass_kg")), _number(a.get("height_m")),
                                 str(a.get("sex") or "unspecified"))
        points = length = None
        scale = c.get("scale")
        if scale is not None:
            if not isinstance(scale, dict) or set(scale) != {"points", "length_m"}:
                raise ValueError("scale must be {points: [[x, y], [x, y]], length_m: L}")
            points = tuple(_point(p) for p in _list(scale["points"], 2))
            length = _number(scale["length_m"])
        setup = cls(athlete, Calibration(_point(c.get("cord_exit")), points, length))
        problems = setup.problems()
        if problems:
            raise ValueError("; ".join(problems))
        return setup

    def to_dict(self) -> Dict[str, Any]:
        a, c = self.athlete, self.calibration
        cal: Dict[str, Any] = {"cord_exit": list(c.cord_exit)}
        if c.scale_points is not None:
            cal["scale"] = {"points": [list(p) for p in c.scale_points],
                            "length_m": c.scale_length_m}
        return {"athlete": {"mass_kg": a.mass_kg, "height_m": a.height_m, "sex": a.sex},
                "calibration": cal}


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _number(v) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise ValueError(f"expected a number, got {v!r}")
    return float(v)


def _list(v, n: int) -> list:
    if not isinstance(v, (list, tuple)) or len(v) != n:
        raise ValueError(f"expected a list of {n}, got {v!r}")
    return list(v)


def _point(v) -> Point:
    x, y = _list(v, 2)
    return _number(x), _number(y)
