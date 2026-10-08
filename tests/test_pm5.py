"""PM5 Bluetooth log: byte layouts, force-curve packets, raw log files, stroke records.

Run: python tests/test_pm5.py (or pytest). No Bluetooth, model or GPU needed.
"""
import json, math, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import synthetic_force as sf
from pose_app.analysis.telemetry import clean, load_telemetry
from pose_app.pm5 import RawLog, RawLogWriter, build_session, load_pm5_session, read_raw_log
from pose_app.pm5.cli import main as pm5_main
from pose_app.pm5.protocol import (ADDITIONAL_STATUS, LBF_TO_N, STROKE_DATA, CurveAssembler,
                                   curve_packets, parse, short_id, uuid)


def _u(v, n):
    return int(round(v)).to_bytes(n, "little")


def _stroke(count, elapsed, recovery=0.9, peak_lbf=110.0, work=445.4):
    return (_u(elapsed * 100, 3) + _u(333, 3) + bytes([142, 81]) + _u(recovery * 100, 2) +
            _u(956, 2) + _u(peak_lbf * 10, 2) + _u(614, 2) + _u(work * 10, 2) + _u(count, 2))


def test_stroke_data_layout():
    s = parse(STROKE_DATA, _stroke(7, 10.5))
    assert (s.elapsed_s, s.distance_m, s.drive_length_m, s.drive_time_s) == (10.5, 33.3, 1.42, 0.81)
    assert (s.recovery_time_s, s.stroke_distance_m, s.peak_force_lbf) == (0.9, 9.56, 110.0)
    assert (s.avg_force_lbf, s.work_j, s.count) == (61.4, 445.4, 7)
    assert parse(STROKE_DATA, _stroke(7, 10.5)[:19]) is None           # short payload
    status = parse(ADDITIONAL_STATUS, _u(1050, 3) + _u(4200, 2) + bytes([31, 255]) + _u(11900, 2))
    assert status.hr is None and status.spm == 31 and abs(status.speed_mps - 4.2) < 1e-9
    assert uuid(0x0035) == "ce060035-43e5-11e4-916c-0800200c9a66"
    assert short_id("0035") == short_id("0x0035") == short_id(uuid(0x0035)) == 0x0035


def test_force_curve_packets():
    points = list(range(1, 41))
    packets = curve_packets(points)
    assert packets[0][:2] == bytes([(5 << 4) | 9, 0]) and packets[-1][0] == (5 << 4) | 4
    a = CurveAssembler()
    assert [a.add(p) for p in packets][-1] == points

    lost = CurveAssembler()                       # a lost packet drops the whole curve
    assert all(lost.add(p) is None for i, p in enumerate(packets) if i != 2)
    assert lost.broken == 1
    assert [lost.add(p) for p in packets][-1] == points      # the next curve is fine

    late = CurveAssembler()                       # joined mid-curve: wait for the next one
    assert all(late.add(p) is None for p in packets[2:])
    assert late.broken == 0 and [late.add(p) for p in packets][-1] == points


def _write(path, rows, device=True):
    with open(path, "w") as fh:
        if device:
            fh.write(json.dumps({"t": 1.0, "device": {"name": "PM5 test"}}) + "\n")
        for t, short, payload in rows:
            fh.write(json.dumps({"t": t, "uuid": f"{short:04x}", "hex": payload.hex()}) + "\n")


def test_session_from_a_synthetic_piece():
    s = sf.Session(n_strokes=8)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "piece.pm5.jsonl")
        s.write_pm5_log(path, duplicate=3, lose_packet=5)
        session = load_pm5_session(path)
    assert session.device["serial"] == "431234567"
    assert [p.count for p in session.strokes] == list(range(1, 9))
    assert session.curves == 7 and session.strokes[4].curve_n is None
    assert session.curve_basis == "time"
    notes = " ".join(session.notes)
    assert "1 repeated stroke notifications" in notes and "incomplete" in notes, notes
    first = 0.2                                   # the log clock starts at its first notification
    for p, true in zip(session.strokes, s.strokes):
        assert abs(p.t_end - (true["t_end"] + sf.LOG_OFFSET - first + 0.035)) < 0.02
        assert abs(p.drive_time_s - true["drive_time"]) <= 0.006
        assert abs(p.peak_force_n - true["peak"]) < 0.06 * LBF_TO_N
        assert abs(p.work_j - true["work"]) <= 0.06
    assert all(math.isfinite(p.recovery_time_s) for p in session.strokes[:-1])
    peak = max(session.strokes[0].curve_n)
    assert abs(peak - s.strokes[0]["peak"]) < 0.02 * s.strokes[0]["peak"]


def test_counts_restarting_start_a_new_piece():
    rows, t = [], 0.0
    for piece, n in ((0, 6), (1, 3)):
        for c in range(1, n + 1):
            t += 1.5
            rows.append((t, STROKE_DATA, _stroke(c, 1.5 * c)))
            rows.append((t + 1.0, STROKE_DATA, _stroke(c, 1.5 * c + 1.0, recovery=1.0)))
    session = build_session(RawLog("x", events=[(a, b, c) for a, b, c in rows]))
    assert [(p.piece, p.count) for p in session.strokes] == \
        [(0, c) for c in range(1, 7)] + [(1, c) for c in range(1, 4)]
    assert any("2 pieces" in n for n in session.notes), session.notes
    # each piece is timed on its own PM5 clock, so stroke times keep increasing
    assert np.all(np.diff([p.t_end for p in session.strokes]) > 0)
    assert session.strokes[0].recovery_time_s == 1.0


def test_raw_log_files():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "log.jsonl")
        clock = iter([10.0, 10.5, 11.0])
        with RawLogWriter(path, clock=lambda: next(clock)) as w:
            w.meta(device={"name": "PM5 1"})
            w.notification(STROKE_DATA, _stroke(1, 2.0))
            w.notification(STROKE_DATA, _stroke(2, 4.0), t=12.0)
        with open(path, "a") as fh:
            fh.write('["not", "a", "row"]\n{"t": 13.0, "uuid": "0035", "hex": "zz"}\n{"t": 14')
        log = read_raw_log(path)
        assert log.meta["device"]["name"] == "PM5 1" and log.skipped == 3
        assert [(t, s) for t, s, _ in log.events] == [(10.5, 0x35), (12.0, 0x35)]
        empty = os.path.join(d, "empty.jsonl")
        _write(empty, [])
        for bad, error in ((empty, ValueError), (os.path.join(d, "missing.jsonl"), FileNotFoundError)):
            try:
                read_raw_log(bad)
            except error:
                continue
            raise AssertionError(f"read {bad}")


def test_pm5_log_as_machine_telemetry():
    s = sf.Session(n_strokes=8)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "piece.pm5.jsonl")
        s.write_pm5_log(path)
        tel = clean(load_telemetry(path))
        assert tel.source.startswith("PM5 Bluetooth log")
        a, b = tel.active_span
        start = s.strokes[0]["t_end"] + sf.LOG_OFFSET - 0.2
        assert a <= start + 1.0 and b >= s.strokes[-1]["t_end"] + sf.LOG_OFFSET - 0.2 - 1.0, (a, b)
        assert np.nanmax(tel.power) > 0 and np.nanmax(tel.spm) == 40
        assert pm5_main(["--check", path]) == 0


def test_check_says_when_there_are_no_force_curves():
    rows = [(1.0 + 1.5 * c, STROKE_DATA, _stroke(c, 1.5 * c)) for c in range(1, 6)]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "old_pm5.jsonl")
        _write(path, rows)
        session = load_pm5_session(path)
        assert session.curves == 0 and any("late 2016" in n for n in session.notes)
        assert pm5_main(["--check", path]) == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
