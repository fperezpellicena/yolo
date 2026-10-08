"""The raw PM5 log: every Bluetooth notification as it arrived, one JSON object per line.

    {"t": 1791460800.000, "device": {"name": "PM5 431234567 Ski", "serial": "431234567", ...}}
    {"t": 1791460800.812, "uuid": "0035", "hex": "5100002100008e510000bc034c04660266110100"}

`t` is the recording machine's clock (epoch seconds) when the notification
arrived; `uuid` the characteristic's short id (a full UUID is accepted too).
Other lines are ignored. Nothing is decoded while recording, so a better
parser can always re-read an old log. The same format is written by the
open-source pm5-force-logger (its raw/<start>.jsonl files), so those work too.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .protocol import short_id

Event = Tuple[float, int, bytes]          # (arrival time, characteristic short id, payload)


@dataclass
class RawLog:
    path: str
    meta: Dict[str, Any] = field(default_factory=dict)      # "device": {...} and the like
    events: List[Event] = field(default_factory=list)       # in arrival order
    skipped: int = 0                                        # unreadable lines


def read_raw_log(path: str) -> RawLog:
    """Read a raw log; ValueError if it has no notifications at all."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"PM5 log not found: {path}")
    log = RawLog(path)
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                t = float(row["t"])             # also rejects anything but an object
            except (ValueError, KeyError, TypeError, IndexError):
                log.skipped += 1                # e.g. the last line of a log cut short by a crash
                continue
            if "uuid" in row and "hex" in row:
                try:
                    log.events.append((t, short_id(str(row["uuid"])), bytes.fromhex(str(row["hex"]))))
                except ValueError:
                    log.skipped += 1
            else:
                log.meta.update({k: v for k, v in row.items() if k != "t"})
    if not log.events:
        raise ValueError(f"{path}: no PM5 notifications found (expected JSON lines with "
                         "'t', 'uuid' and 'hex')")
    log.events.sort(key=lambda e: e[0])
    return log


class RawLogWriter:
    """Appends notifications to a raw log, one flushed line each, so a crash
    or a pulled plug loses at most the line being written."""

    def __init__(self, path: str, clock=time.time):
        self.path, self.clock = path, clock
        self._fh = open(path, "a", encoding="utf-8")
        self.count = 0

    def meta(self, **fields: Any) -> None:
        self._line({"t": round(self.clock(), 3), **fields})

    def notification(self, short: int, payload: bytes, t: Optional[float] = None) -> None:
        self._line({"t": round(self.clock() if t is None else t, 3), "uuid": f"{short:04x}",
                    "hex": bytes(payload).hex()})
        self.count += 1

    def _line(self, row: Dict[str, Any]) -> None:
        if self._fh.closed:
            return
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "RawLogWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
