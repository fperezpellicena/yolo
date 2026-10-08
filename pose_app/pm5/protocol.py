"""Concept2 PM5 Bluetooth notifications: UUIDs, byte layouts and force-curve packets.

Layouts follow Concept2's "PM Bluetooth Smart Communication Interface
Definition" (multi-byte values little-endian). Times are 0.01 s, distances
0.1 m unless noted, forces 0.1 lbf in the stroke data and whole lbf in the
force curve (each curve's peak equals the stroke's reported peak force).

Force curves arrive split over several notifications. Byte 0 holds the number
of packets in the curve (high nibble) and the number of 16-bit points in this
packet (low nibble); byte 1 is the packet's index within the curve, so a new
curve starts at 0. Two channels exist:
  * 0x003D, documented: points step in time and a reading often repeats
    2-3 times, with zeros before the force rises;
  * 0x0043, sent by recent firmware right after 0x003D: one point per step of
    handle travel.
"""

from dataclasses import dataclass
from typing import List, Optional

BASE_UUID = "ce06{:04x}-43e5-11e4-916c-0800200c9a66"

DEVICE_INFO_SERVICE = 0x0010
DEVICE_INFO = {0x0011: "model", 0x0012: "serial", 0x0013: "hardware_rev",
               0x0014: "firmware_rev", 0x0015: "manufacturer", 0x0016: "machine_type"}
ROWING_SERVICE = 0x0030
GENERAL_STATUS = 0x0031
ADDITIONAL_STATUS = 0x0032
ADDITIONAL_STATUS_2 = 0x0033
STROKE_DATA = 0x0035
ADDITIONAL_STROKE_DATA = 0x0036
WORKOUT_SUMMARY = 0x0039
FORCE_CURVE = 0x003D            # force against time
FORCE_CURVE_TRAVEL = 0x0043     # force against handle travel (recent firmware)

# channel -> what the points are spaced by
CURVE_CHANNELS = {FORCE_CURVE: "time", FORCE_CURVE_TRAVEL: "travel"}

LBF_TO_N = 4.4482216152605

WORKOUT_STATES = {0: "wait_to_begin", 1: "workout_row", 2: "countdown_pause", 3: "interval_rest",
                  4: "interval_work_time", 5: "interval_work_distance",
                  6: "interval_rest_end_to_work_time", 7: "interval_rest_end_to_work_distance",
                  8: "interval_work_time_to_rest", 9: "interval_work_distance_to_rest",
                  10: "workout_end", 11: "terminate", 12: "workout_logged", 13: "rearm"}


def uuid(short: int) -> str:
    """Full 128-bit UUID of a Concept2 service or characteristic."""
    return BASE_UUID.format(short)


def short_id(text: str) -> int:
    """'0035', '0x0035' or 'ce060035-43e5-...' -> 0x0035."""
    text = text.strip().lower()
    if len(text) == 36 and text.count("-") == 4:
        text = text[4:8]
    return int(text[2:] if text.startswith("0x") else text, 16)


def _u16(b: bytes, i: int) -> int:
    return b[i] | b[i + 1] << 8


def _u24(b: bytes, i: int) -> int:
    return b[i] | b[i + 1] << 8 | b[i + 2] << 16


@dataclass(frozen=True)
class GeneralStatus:
    elapsed_s: float
    distance_m: float
    workout_state: int
    drag_factor: int


@dataclass(frozen=True)
class AdditionalStatus:
    elapsed_s: float
    speed_mps: float
    spm: int
    hr: Optional[int]               # None: no heart-rate strap
    pace_s: float                   # current pace, s per 500 m


@dataclass(frozen=True)
class StrokeData:
    elapsed_s: float
    distance_m: float
    drive_length_m: float
    drive_time_s: float
    recovery_time_s: float          # the previous stroke's in the copy sent at the end of the drive
    stroke_distance_m: float
    peak_force_lbf: float
    avg_force_lbf: float
    work_j: float
    count: int


@dataclass(frozen=True)
class AdditionalStrokeData:
    elapsed_s: float
    power_w: int
    count: int


def parse(short: int, b: bytes):
    """Decode one notification; None for characteristics not used here or short payloads."""
    if short == GENERAL_STATUS and len(b) >= 19:
        return GeneralStatus(_u24(b, 0) / 100, _u24(b, 3) / 10, b[8], b[18])
    if short == ADDITIONAL_STATUS and len(b) >= 9:
        return AdditionalStatus(_u24(b, 0) / 100, _u16(b, 3) / 1000, b[5],
                                None if b[6] in (0, 255) else b[6], _u16(b, 7) / 100)
    if short == STROKE_DATA and len(b) >= 20:
        return StrokeData(_u24(b, 0) / 100, _u24(b, 3) / 10, b[6] / 100, b[7] / 100,
                          _u16(b, 8) / 100, _u16(b, 10) / 100, _u16(b, 12) / 10,
                          _u16(b, 14) / 10, _u16(b, 16) / 10, _u16(b, 18))
    if short == ADDITIONAL_STROKE_DATA and len(b) >= 9:
        return AdditionalStrokeData(_u24(b, 0) / 100, _u16(b, 3), _u16(b, 7))
    return None


class CurveAssembler:
    """Joins one channel's force-curve packets into curves (lists of lbf values).

    A curve is returned once its last packet arrives. A curve with a missing
    or out-of-order packet is dropped (counted in `broken`) rather than
    returned with a hole, and packets before the first index-0 packet (the
    logger joined mid-curve) are ignored.
    """

    def __init__(self):
        self.parts: List[List[int]] = []
        self.expected: Optional[int] = None
        self.broken = 0

    def add(self, b: bytes) -> Optional[List[int]]:
        if len(b) < 2:
            return None
        total, words, index = b[0] >> 4, b[0] & 0x0F, b[1]
        if index == 0:
            if self.expected is not None:
                self.broken += 1            # the previous curve never finished
            self.parts, self.expected = [], total
        if self.expected is None:
            return None
        if index != len(self.parts) or total != self.expected:
            self.broken += 1
            self.parts, self.expected = [], None
            return None
        words = min(words, (len(b) - 2) // 2)
        self.parts.append([_u16(b, 2 + 2 * k) for k in range(words)])
        if len(self.parts) == self.expected:
            curve = [v for part in self.parts for v in part]
            self.parts, self.expected = [], None
            return curve
        return None


def curve_packets(points: List[int], per_packet: int = 9) -> List[bytes]:
    """The notifications a PM5 sends for one curve (used by tests and replays)."""
    chunks = [points[i:i + per_packet] for i in range(0, len(points), per_packet)] or [[]]
    if len(chunks) > 15 or per_packet > 9:
        raise ValueError("a force curve fits in at most 15 packets of 9 points")
    out = []
    for index, chunk in enumerate(chunks):
        body = b"".join(int(v).to_bytes(2, "little") for v in chunk)
        out.append(bytes([(len(chunks) << 4) | len(chunk), index]) + body)
    return out
