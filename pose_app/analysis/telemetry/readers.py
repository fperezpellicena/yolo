"""Telemetry file readers. Each returns a raw (uncleaned) Telemetry."""

import csv
import os
from typing import Dict, List

import numpy as np

from .model import Telemetry

_DISTANCE_UNITS = {"M": 1.0, "KM": 1000.0, "MI": 1609.344, "YD": 0.9144}


def load_telemetry(path: str) -> Telemetry:
    """Read a telemetry file, picking the reader from its content."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"telemetry file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".fit", ".tcx", ".gz"):
        raise ValueError(f"{ext} telemetry is not supported yet; export the workout as CSV "
                         "from Track My Indoor Workout.")
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        head = fh.read(4096)
    if head.startswith("TMIW") or "RIDE DATA" in head:
        return read_tmiw_csv(path)
    raise ValueError(f"{path}: unrecognised telemetry CSV (expected a Track My Indoor "
                     "Workout export with 'RIDE SUMMARY' / 'RIDE DATA' sections).")


def read_tmiw_csv(path: str) -> Telemetry:
    """Track My Indoor Workout CSV: a key/value summary, then the sample table.

    The summary is kept as metadata only: its totals include pauses and its
    'Start Time' can be a resume time, so nothing downstream relies on it.
    Samples are stamped by the phone on arrival (epoch ms); ELAPSED is ignored.
    """
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        rows = [[c.strip() for c in r] for r in csv.reader(fh)]
    try:
        data_at = next(i for i, r in enumerate(rows) if r and r[0].upper() == "RIDE DATA")
    except StopIteration:
        raise ValueError(f"{path}: no 'RIDE DATA' section") from None

    meta: Dict[str, str] = {}
    for r in rows[:data_at]:
        if len(r) >= 2 and r[0] and r[0].upper() not in ("RIDE SUMMARY",):
            meta[r[0]] = ",".join(v for v in r[1:] if v)

    header = [h.upper() for h in rows[data_at + 1] if h]
    col = {name: header.index(name) for name in header}
    if "TIME_STAMP" not in col:
        raise ValueError(f"{path}: RIDE DATA has no TIME_STAMP column")

    def column(name: str, body: List[List[str]]) -> np.ndarray:
        if name not in col:
            return np.full(len(body), np.nan)
        out = np.full(len(body), np.nan)
        for i, r in enumerate(body):
            try:
                out[i] = float(r[col[name]])
            except (IndexError, ValueError):
                pass
        return out

    body = [r for r in rows[data_at + 2:] if len(r) > col["TIME_STAMP"] and r[col["TIME_STAMP"]]]
    if len(body) < 2:
        raise ValueError(f"{path}: RIDE DATA has fewer than two samples")
    stamp = column("TIME_STAMP", body)
    keep = np.isfinite(stamp)
    order = np.argsort(stamp[keep], kind="stable")

    def take(name: str) -> np.ndarray:
        return column(name, body)[keep][order]

    stamp = stamp[keep][order]
    dist_unit = meta.get("Total Distance", "").split(",")[-1].strip().upper() or "M"
    distance = take("DISTANCE") * _DISTANCE_UNITS.get(dist_unit, 1.0)
    t = (stamp - stamp[0]) / 1000.0
    notes: List[str] = []
    speed, unit = _speed_to_mps(take("SPEED"), distance, t)
    if not np.isfinite(distance).any():
        distance = _distance_from_speed(speed, t)
        notes.append("No distance in the file; distance integrated from speed (km/h assumed)."
                     if np.isfinite(distance).any() else
                     "No distance or speed in the file; activity taken from stroke rate/power.")
    elif unit != "km/h":
        notes.append(f"SPEED column read as {unit} (inferred from the distance channel).")

    hr = take("HR")
    hr[(hr <= 0) | (hr > 250)] = np.nan          # 'zeros' / 'nulls' gap handling in the app
    if not np.isfinite(hr).any():
        notes.append("No heart-rate data in the file (no strap paired, or it never connected).")
    return Telemetry(
        source=f"Track My Indoor Workout CSV ({os.path.basename(path)})",
        t0_ms=float(stamp[0]), t=t, power=take("POWER"), spm=take("RPM"),
        distance=distance, speed=speed, hr=hr, meta=meta, notes=notes)


def _speed_to_mps(speed: np.ndarray, distance: np.ndarray, t: np.ndarray):
    """Pick the speed unit that agrees with the distance channel (default km/h)."""
    factors = {"km/h": 1 / 3.6, "mph": 0.44704, "m/s": 1.0}
    moving = np.isfinite(speed) & (speed > 0)
    if moving.sum() > 20 and np.isfinite(distance).any():
        dd, dt = np.diff(distance), np.diff(t)
        step = np.isfinite(dd) & (dd >= 0) & (dt > 0) & (dt < 2.0) & moving[1:]
        span, dist = dt[step].sum(), dd[step].sum()   # contiguous, moving samples only
        if span > 10 and dist > 0:
            true_mps = dist / span
            median_raw = float(np.median(speed[moving]))
            unit = min(factors, key=lambda u: abs(np.log(median_raw * factors[u] / true_mps)))
            return speed * factors[unit], unit
    return speed * factors["km/h"], "km/h"


def _distance_from_speed(speed: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Cumulative metres from speed (m/s); NaN if there is no speed either."""
    if not np.isfinite(speed).any():
        return np.full(len(t), np.nan)
    dt = np.minimum(np.diff(t, prepend=t[0]), 2.5)          # never integrate across gaps
    v = np.nan_to_num(np.concatenate([[0.0], speed[:-1]]))  # held value over the interval
    return np.cumsum(v * dt)
