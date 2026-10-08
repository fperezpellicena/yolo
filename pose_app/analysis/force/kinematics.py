"""Keypoints -> metres -> segment positions, velocities and accelerations.

Accelerations amplify keypoint noise, so the raw keypoints are low-pass
filtered first with a zero-lag Butterworth filter: a second-order section run
forwards and backwards (fourth order, no lag), its cut-off corrected for the
two passes (Winter 2009). 6 Hz keeps the stroke and drops frame-to-frame
jitter. Positions are in metres in the athlete's plane: x towards the
machine, y up, origin on the floor under the ankle.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ...skeleton import KP
from ..signal import fill_gaps
from .model import ANKLE_H, SEGMENT_H, SEGMENTS, BodyModel
from .setup import AthleteProfile, Calibration

CUTOFF_HZ = 6.0
MAX_GAP_S = 0.3              # bridge missed detections up to this long
MIN_RUN_S = 0.5              # shorter measured stretches are too short to filter
TRACKED = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "ear")
STICK_VS_HEIGHT = 0.15       # disagreement between the stick and the height estimate to flag


def lowpass(x: np.ndarray, fs: float, fc: float = CUTOFF_HZ) -> np.ndarray:
    """Zero-lag fourth-order Butterworth low-pass of each column of `x` (no NaNs)."""
    x = np.asarray(x, float)
    one = x.ndim == 1
    x2 = x[:, None] if one else x
    n = len(x2)
    if n < 4:
        return x.copy()
    fc = min(fc, 0.3 * fs)
    wc = np.tan(np.pi * fc / (0.802 * fs))            # 0.802: two passes, same -3 dB point
    k1, k2 = np.sqrt(2.0) * wc, wc * wc
    a0 = 1.0 + k1 + k2
    b = np.array([k2, 2.0 * k2, k2]) / a0
    a = np.array([1.0, 2.0 * (k2 - 1.0) / a0, (1.0 - k1 + k2) / a0])
    pad = min(n - 1, int(3 * fs / fc))
    y = np.concatenate([2 * x2[0] - x2[pad:0:-1], x2, 2 * x2[-1] - x2[-2:-pad - 2:-1]])
    y = _pass(_pass(y, b, a)[::-1], b, a)[::-1]
    out = y[pad:pad + n]
    return out[:, 0] if one else out


def _pass(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """One causal pass, starting at rest on the first sample (no step at the start)."""
    try:
        from scipy.signal import lfilter, lfilter_zi
    except ImportError:
        return _pass_numpy(x, b, a)
    zi = lfilter_zi(b, a)[:, None] * x[0][None, :]
    return lfilter(b, a, x, axis=0, zi=zi)[0]


def _pass_numpy(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    x1 = x2 = y1 = y2 = x[0]
    for i in range(len(x)):
        xi = x[i]
        yi = b[0] * xi + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        y[i] = yi
        x2, x1, y2, y1 = x1, xi, y1, yi
    return y


def filtered_track(t: np.ndarray, xy: np.ndarray, fs: float, fc: float = CUTOFF_HZ) -> np.ndarray:
    """Bridge short gaps, then filter each measured stretch on its own; NaN elsewhere."""
    xy = np.column_stack([fill_gaps(xy[:, d], t, MAX_GAP_S) for d in range(xy.shape[1])])
    out = np.full_like(xy, np.nan)
    ok = np.isfinite(xy).all(axis=1)
    edges = np.diff(np.concatenate([[0], ok.astype(np.int8), [0]]))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if t[b - 1] - t[a] >= MIN_RUN_S:
            out[a:b] = lowpass(xy[a:b], fs, fc)
    return out


def keypoint_tracks(t: np.ndarray, keypoints: np.ndarray, scores: np.ndarray, side: str,
                    min_score: float, fps: float) -> Dict[str, np.ndarray]:
    """Filtered pixel tracks (N, 2) of the near side's joints and ear."""
    other = "right" if side == "left" else "left"
    out = {}
    for name in TRACKED:
        k = KP[f"{side}_{name}"]
        xy = np.where((scores[:, k] >= min_score)[:, None], keypoints[:, k], np.nan)
        if name == "ear":                       # the far ear when the near one is hidden
            k2 = KP[f"{other}_ear"]
            far = np.where((scores[:, k2] >= min_score)[:, None], keypoints[:, k2], np.nan)
            xy = np.where(np.isfinite(xy), xy, far)
        out[name] = filtered_track(t, xy.astype(float), fps)
    return out


def _median_length(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a - b, axis=1)
    return float(np.nanmedian(d)) if np.isfinite(d).any() else np.nan


def pixel_lengths(tracks: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Median segment lengths in pixels (shank, thigh, trunk, upper_arm, forearm)."""
    pairs = {"shank": ("knee", "ankle"), "thigh": ("hip", "knee"), "trunk": ("hip", "shoulder"),
             "upper_arm": ("shoulder", "elbow"), "forearm": ("elbow", "wrist")}
    return {k: _median_length(tracks[a], tracks[b]) for k, (a, b) in pairs.items()}


@dataclass(frozen=True)
class ImageFrame:
    """Pixels <-> metres in the athlete's plane."""
    scale: float                  # pixels per metre
    origin: Tuple[float, float]   # pixels: the floor under the ankle
    facing: int                   # +1 when the athlete faces image-right
    source: str                   # "stick" or "height": where the scale came from

    def to_m(self, px: np.ndarray) -> np.ndarray:
        px = np.asarray(px, float)
        return np.stack([self.facing * (px[..., 0] - self.origin[0]) / self.scale,
                         (self.origin[1] - px[..., 1]) / self.scale], axis=-1)

    def to_px(self, m: np.ndarray) -> np.ndarray:
        m = np.asarray(m, float)
        return np.stack([self.origin[0] + self.facing * m[..., 0] * self.scale,
                         self.origin[1] - m[..., 1] * self.scale], axis=-1)


def fit_frame(cal: Calibration, athlete: AthleteProfile, tracks: Dict[str, np.ndarray],
              facing: int) -> Tuple[Optional[ImageFrame], list]:
    """The image frame and notes about it; None if the ankle or scale cannot be found."""
    notes = []
    lengths = pixel_lengths(tracks)
    by_height = [v / (SEGMENT_H[k] * athlete.height_m) for k, v in lengths.items() if np.isfinite(v)]
    height_scale = float(np.median(by_height)) if by_height else np.nan
    scale = cal.stick_px_per_m
    if scale:
        source = "stick"
        if np.isfinite(height_scale) and abs(height_scale / scale - 1) > STICK_VS_HEIGHT:
            notes.append(f"The calibration length gives {scale:.0f} px/m but the athlete's "
                         f"height suggests {height_scale:.0f} px/m: check the two scale points, "
                         "the athlete's height, and that the length was held in the athlete's "
                         "plane.")
    else:
        scale, source = height_scale, "height"
        notes.append("Image scale estimated from the athlete's height (no calibration "
                     "length): distances, angles and moments may be off by about 10%.")
    ankle = tracks["ankle"]
    if not np.isfinite(scale) or not np.isfinite(ankle).any():
        return None, notes + ["The ankle was not measured, so the floor cannot be placed."]
    ax, ay = np.nanmedian(ankle[:, 0]), np.nanmedian(ankle[:, 1])
    origin = (float(ax), float(ay + ANKLE_H * athlete.height_m * scale))
    return ImageFrame(float(scale), origin, int(facing), source), notes


def _unwrap(a: np.ndarray) -> np.ndarray:
    ok = np.isfinite(a)
    out = np.full_like(a, np.nan)
    if ok.any():
        out[ok] = np.unwrap(a[ok])
    return out


def _derivative(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.gradient(x, t, axis=0) if len(t) > 2 else np.zeros_like(x)


@dataclass
class Kinematics:
    t: np.ndarray
    points: Dict[str, np.ndarray]      # metres (N, 2), incl. "grip"
    com: Dict[str, np.ndarray]         # segment centres of mass (N, 2)
    com_vel: Dict[str, np.ndarray]
    com_acc: Dict[str, np.ndarray]
    angle: Dict[str, np.ndarray]       # rad, of the proximal -> distal line
    omega: Dict[str, np.ndarray]
    alpha: Dict[str, np.ndarray]
    body_com: np.ndarray               # whole body, feet included (N, 2)


def body_points(tracks: Dict[str, np.ndarray], frame: ImageFrame,
                model: BodyModel) -> Dict[str, np.ndarray]:
    """Joint positions in metres; the feet are static, so the ankle is fixed."""
    pts = {name: frame.to_m(xy) for name, xy in tracks.items()}
    n = len(next(iter(pts.values())))
    pts["ankle"] = np.tile([0.0, model.ankle_h], (n, 1))
    ear = pts["ear"]
    trunk = pts["shoulder"] - pts["hip"]
    up = trunk / np.linalg.norm(trunk, axis=1, keepdims=True)
    pts["ear"] = np.where(np.isfinite(ear), ear, pts["shoulder"] + model.neck * up)
    fore = pts["wrist"] - pts["elbow"]
    pts["grip"] = pts["wrist"] + model.grip * fore / np.linalg.norm(fore, axis=1, keepdims=True)
    return pts


def kinematics(points: Dict[str, np.ndarray], t: np.ndarray, model: BodyModel) -> Kinematics:
    com, vel, acc, ang, om, al = {}, {}, {}, {}, {}, {}
    body = model.feet_mass * model.feet_com
    for name, (p, d, _) in SEGMENTS.items():
        seg = model.segments[name]
        c = points[p] + seg.com * (points[d] - points[p])
        com[name] = c
        vel[name] = _derivative(c, t)
        acc[name] = _derivative(vel[name], t)
        v = points[d] - points[p]
        ang[name] = _unwrap(np.arctan2(v[:, 1], v[:, 0]))
        om[name] = _derivative(ang[name], t)
        al[name] = _derivative(om[name], t)
        body = body + seg.mass * c
    return Kinematics(t, points, com, vel, acc, ang, om, al, body / model.total_mass)


def segment_lengths(points: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Session-median lengths in metres, for the body model's inertias."""
    out = {name: _median_length(points[p], points[d]) for name, (p, d, _) in SEGMENTS.items()}
    return out
