"""Concept2 PM5 over Bluetooth: record a piece, then decode it for the analysis.

    protocol.py     UUIDs, byte layouts, force-curve packets
    log.py          the raw log: every notification as it arrived (JSON lines)
    session.py      raw log -> one record per stroke with its force curve
    recorder.py     Bluetooth recorder (bleak), run by `python pm5_log.py LOG`

Only the recorder needs Bluetooth; reading a log needs nothing beyond numpy.
"""

from .log import RawLog, RawLogWriter, read_raw_log
from .session import Pm5Session, Pm5Stroke, build_session, load_pm5_session

__all__ = ["RawLog", "RawLogWriter", "read_raw_log", "Pm5Session", "Pm5Stroke",
           "build_session", "load_pm5_session"]
