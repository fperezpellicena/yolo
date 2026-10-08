"""A recorded PM5 log turned into one record per stroke, with its force curve.

What a raw log contains, and how it is handled:
  * stroke data (0x0035) arrives twice per stroke, at the end of the drive and
    again at the end of the recovery; the second copy carries that stroke's
    recovery time (the first carries the previous stroke's). Records are keyed
    on the stroke count. Count 0 is the reset at a piece's start or end and is
    dropped; counts starting again from 1 mean a new piece on the monitor.
  * Bluetooth sometimes delivers the same notification twice: identical
    copies are counted and dropped.
  * a force curve arrives just after its drive, so it is given to the stroke
    whose stroke data arrived closest to it; curves with a lost packet are
    dropped by the assembler.
  * stroke times use the PM5's own clock (0.01 s, no radio jitter), placed on
    the recording machine's clock by the median difference between the two.

Times in a Pm5Session are seconds since the log's first notification.
"""

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .log import RawLog, read_raw_log
from .protocol import (CURVE_CHANNELS, FORCE_CURVE, FORCE_CURVE_TRAVEL, LBF_TO_N,
                       AdditionalStatus, AdditionalStrokeData, CurveAssembler, GeneralStatus,
                       StrokeData, parse)

CURVE_MATCH_S = 3.0          # a curve belongs to the stroke whose data arrived within this
CLOCK_JUMP_S = 1.0           # PM5 clock vs recording clock moving by more = clock reset or pause


@dataclass
class Pm5Stroke:
    piece: int                       # 0-based; a new piece starts when the counts restart
    count: int                       # the PM5's stroke count within the piece
    t_end: float                     # end of the drive, s on the log clock
    drive_time_s: float
    drive_length_m: float
    recovery_time_s: float           # NaN if the copy sent after the recovery was lost
    distance_m: float
    peak_force_n: float
    avg_force_n: float
    work_j: float
    t_arrival: float                 # when its stroke data arrived (log clock)
    power_w: float = math.nan
    curve_n: Optional[np.ndarray] = None     # force curve in newtons
    curve_basis: str = ""                    # "time" or "travel": what the points are spaced by

    @property
    def t_start(self) -> float:
        return self.t_end - self.drive_time_s


@dataclass
class StatusSamples:
    """Monitor status as logged (log clock), for machine telemetry."""
    distance_t: np.ndarray
    distance_m: np.ndarray
    status_t: np.ndarray
    speed_mps: np.ndarray
    spm: np.ndarray
    hr: np.ndarray
    power_t: np.ndarray
    power_w: np.ndarray


@dataclass
class Pm5Session:
    source: str
    device: Dict[str, Any]
    t0: float                                    # epoch seconds of the first notification
    strokes: List[Pm5Stroke]
    status: StatusSamples
    notes: List[str] = field(default_factory=list)

    @property
    def curves(self) -> int:
        return sum(1 for s in self.strokes if s.curve_n is not None)

    @property
    def curve_basis(self) -> str:
        """The channel most strokes' curves came from ('' without curves)."""
        kinds = [s.curve_basis for s in self.strokes if s.curve_n is not None]
        return max(set(kinds), key=kinds.count) if kinds else ""

    def describe(self) -> str:
        name = self.device.get("name") or self.device.get("serial") or "PM5"
        fw = self.device.get("firmware_rev")
        return (f"{name}" + (f", firmware {fw}" if fw else "") +
                f": {len(self.strokes)} strokes, {self.curves} force curves")


def load_pm5_session(path: str) -> Pm5Session:
    """Read and decode a raw PM5 log (OSError / ValueError on unreadable files)."""
    log = read_raw_log(path)
    return build_session(log, source=f"PM5 Bluetooth log ({os.path.basename(path)})")


def build_session(log: RawLog, source: str = "PM5 Bluetooth log") -> Pm5Session:
    t0 = log.events[0][0]
    assemblers = {ch: CurveAssembler() for ch in CURVE_CHANNELS}
    records: Dict[Tuple[int, int], Dict[str, Any]] = {}
    powers: Dict[Tuple[int, int], float] = {}
    curves: List[Tuple[float, int, List[int]]] = []
    gs, st, pw = [], [], []
    piece, top, duplicates, pieces_started = 0, 0, 0, 0

    for t_abs, short, payload in log.events:
        t = t_abs - t0
        if short in assemblers:
            curve = assemblers[short].add(payload)
            if curve is not None:
                curves.append((t, short, curve))
            continue
        p = parse(short, payload)
        if isinstance(p, StrokeData):
            n = p.count
            if n == 0:
                continue
            if top and n < top - 1:                      # counts restarted: a new piece
                piece, top, pieces_started = piece + 1, 0, pieces_started + 1
            top = max(top, n)
            key = (piece, n)
            if key not in records:
                prev = records.get((piece, n - 1))
                if prev is not None and math.isnan(prev["recovery"]):
                    prev["recovery"] = p.recovery_time_s     # the first copy carries the previous one
                records[key] = {"t": t, "data": p, "raw": payload, "recovery": math.nan}
            elif payload == records[key]["raw"]:
                duplicates += 1
            elif p.elapsed_s > records[key]["data"].elapsed_s:
                records[key]["recovery"] = p.recovery_time_s      # the copy after the recovery
        elif isinstance(p, AdditionalStrokeData):
            powers.setdefault((piece, p.count), float(p.power_w))
            pw.append((t, float(p.power_w)))
        elif isinstance(p, GeneralStatus):
            gs.append((t, p.distance_m))
        elif isinstance(p, AdditionalStatus):
            st.append((t, p.speed_mps, float(p.spm), math.nan if p.hr is None else float(p.hr)))

    strokes = _strokes(records, powers)
    notes = [f"{len(strokes)} strokes logged."]
    unattached = _attach_curves(strokes, curves)
    broken = sum(a.broken for a in assemblers.values())
    with_curve = sum(1 for s in strokes if s.curve_n is not None)
    if strokes and not with_curve:
        notes.append("No force curves in the log: they need a PM5 from late 2016 or later "
                     "(hardware 600+), and the recorder must subscribe to them.")
    elif with_curve < len(strokes):
        notes.append(f"{len(strokes) - with_curve} strokes have no force curve "
                     "(Bluetooth drop-outs or the start and end of the piece).")
    if broken:
        notes.append(f"{broken} force curves arrived incomplete and were dropped.")
    if unattached:
        notes.append(f"{unattached} force curves could not be matched to a stroke.")
    if duplicates:
        notes.append(f"{duplicates} repeated stroke notifications ignored.")
    if pieces_started:
        notes.append(f"The log holds {pieces_started + 1} pieces (stroke counts restarted).")
    if log.skipped:
        notes.append(f"{log.skipped} unreadable lines in the log were skipped.")

    def arr(rows, k):
        return np.array([r[k] for r in rows], float) if rows else np.zeros(0)
    status = StatusSamples(arr(gs, 0), arr(gs, 1), arr(st, 0), arr(st, 1), arr(st, 2),
                           arr(st, 3), arr(pw, 0), arr(pw, 1))
    device = dict(log.meta.get("device") or {})
    return Pm5Session(source, device, t0, strokes, status, notes)


def _strokes(records: Dict, powers: Dict) -> List[Pm5Stroke]:
    """Stroke records ordered in time, each timed on the PM5's clock moved onto the log clock."""
    keys = sorted(records, key=lambda k: records[k]["t"])
    if not keys:
        return []
    arrival = np.array([records[k]["t"] for k in keys])
    elapsed = np.array([records[k]["data"].elapsed_s for k in keys])
    lag = arrival - elapsed
    # the PM5 clock resets with each piece and may pause: time each stretch on its own
    jumps = np.flatnonzero((np.abs(np.diff(lag)) > CLOCK_JUMP_S) | (np.diff(elapsed) < 0)) + 1
    t_end = np.empty(len(keys))
    for a, b in zip(np.concatenate([[0], jumps]), np.concatenate([jumps, [len(keys)]])):
        t_end[a:b] = elapsed[a:b] + np.median(lag[a:b])
    out = []
    for k, te in zip(keys, t_end):
        d: StrokeData = records[k]["data"]
        out.append(Pm5Stroke(k[0], k[1], float(te), d.drive_time_s, d.drive_length_m,
                             records[k]["recovery"], d.distance_m, d.peak_force_lbf * LBF_TO_N,
                             d.avg_force_lbf * LBF_TO_N, d.work_j, float(records[k]["t"]),
                             powers.get(k, math.nan)))
    return out


def _attach_curves(strokes: List[Pm5Stroke], curves) -> int:
    """Give each curve to the nearest stroke in arrival time; the travel channel wins
    when a stroke has both. Returns how many curves found no stroke."""
    if not strokes:
        return len(curves)
    arrival = np.array([s.t_arrival for s in strokes])
    taken = {FORCE_CURVE: set(), FORCE_CURVE_TRAVEL: set()}
    lost = 0
    for t, channel, points in curves:
        order = np.argsort(np.abs(arrival - t))
        k = next((int(i) for i in order[:3] if abs(arrival[i] - t) < CURVE_MATCH_S
                  and int(i) not in taken[channel]), None)
        if k is None:
            lost += 1
            continue
        taken[channel].add(k)
        s = strokes[k]
        if channel == FORCE_CURVE_TRAVEL or s.curve_basis != "travel":
            s.curve_n = np.asarray(points, float) * LBF_TO_N
            s.curve_basis = CURVE_CHANNELS[channel]
    return lost
