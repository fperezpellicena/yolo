"""Regression test: synthetic RowErg clip with known faults (tests/synthetic_rowerg.py).

Strokes 1-6 good, 7-9 arms early, 10-12 shooting the slide, 13-15 back opening early,
16-18 knees rising before the hands pass them; the rate fades from 25 to 20 spm and a
bystander is in the background. Also: Track My Indoor Workout and PM5 machine data on
the rower. Run: python tests/test_rowerg.py (or pytest). No model or GPU needed.
"""
import json, math, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from synthetic_rowerg import DRIVE_S, FakeEstimator, KINDS, STROKE_STARTS, period_of, write_video
from synthetic_telemetry import make_tmiw_csv
from pose_app.analysis.api import default_thresholds
from pose_app.analysis.pipeline import analyze, extract
from pose_app.analysis.report import write_csv, write_html, write_json
from pose_app.analysis.signal import Cycle
from pose_app.analysis.stations import STATIONS
from pose_app.analysis.stations.rowerg import RowErg, crossing, rhythm_reference
from pose_app.analysis.telemetry import clean, load_telemetry
from pose_app.analysis.tracking import AthleteTracker
from pose_app.analysis.video import VideoReader
from pose_app.pm5 import load_pm5_session
from pose_app.pm5.protocol import LBF_TO_N

FAULT = {"arms": "arms_early", "shoot": "shooting_slide", "body": "body_opens_early",
         "knees": "knees_early"}
PERIODS = [period_of(n) for n in range(len(KINDS))]
LOG_OFFSET = 7.43                    # PM5 log seconds at video second 0


def _extraction(d):
    video = os.path.join(d, "row.mp4")
    write_video(video)
    reader = VideoReader(video)
    ex = extract(reader, FakeEstimator().detect,
                 AthleteTracker("largest", frame_size=(960, 720)), "fake")
    return ex, reader


def test_rowerg_pipeline():
    with tempfile.TemporaryDirectory() as d:
        ex, reader = _extraction(d)
        res = analyze(ex, STATIONS["rowerg"])

        assert res.body.side == "left" and res.body.facing == 1
        assert len(res.reps) == len(KINDS), len(res.reps)
        for rep, kind, start, period in zip(res.reps, KINDS, STROKE_STARTS, PERIODS):
            m, ids = rep.metrics, {f.rule_id for f in rep.faults}
            if kind == "good":
                assert not ids, (rep.number, ids)
                assert m["seq_legs_ms"] < m["seq_trunk_ms"] < m["seq_arms_ms"], (rep.number, m)
                assert 0.85 < m["slide_lead"] < 1.05 and m["legs_at_arm_bend"] > 90, m
                assert abs(m["knee_catch"] - 51) < 5 and abs(m["trunk_catch"] - 25) < 3, m
                assert abs(m["trunk_finish"] - 25) < 3 and abs(m["shin_catch"]) < 4, m
            for k, rule in FAULT.items():
                assert (rule in ids) == (kind == k), (rep.number, kind, ids)
            # stroke timing: the first stroke starts when the handle leaves the still catch
            assert abs(m["t_start"] - start) < 0.1, (rep.number, m["t_start"], start)
            # one stroke's ends sit on a rounded, noisy peak: a frame or two either way
            assert abs(m["rate_spm"] - 60 / period) < 1.5, (rep.number, m["rate_spm"], period)
            assert abs(m["drive_s"] - DRIVE_S) < 0.1, (rep.number, m["drive_s"])
            shares = m["legs_pct"] + m["trunk_pct"] + m["arms_pct"]
            assert abs(shares - 100) < 1e-6 and m["legs_pct"] > m["arms_pct"], m
        mean_rate = np.mean([r.metrics["rate_spm"] for r in res.reps])
        assert abs(mean_rate - np.mean([60 / p for p in PERIODS])) < 0.3, mean_rate
        drift = {x.rule_id: x.triggered for x in res.drift}
        assert drift["drift_rate"] and not drift["drift_length"], drift

        write_csv(res, os.path.join(d, "reps.csv"))
        write_json(res, os.path.join(d, "summary.json"))
        write_html(res, os.path.join(d, "report.html"), reader)
        summary = json.load(open(os.path.join(d, "summary.json")))
        assert summary["station"]["key"] == "rowerg" and summary["station"]["name"] == "RowErg"
        charts = {c["metric"] for c in summary["charts"]}
        assert "slide_lead" in charts and "heel_lift_catch" not in charts, charts   # no feet


def test_rowerg_with_track_my_indoor_workout():
    with tempfile.TemporaryDirectory() as d:
        ex, _ = _extraction(d)
        path = os.path.join(d, "row.csv")
        make_tmiw_csv(path, STROKE_STARTS, PERIODS, KINDS, log_start=STROKE_STARTS[0] - 0.3)
        res = analyze(ex, STATIONS["rowerg"], telemetry=clean(load_telemetry(path)))
        al = res.machine.alignment
        assert al.ok and abs(al.offset + (STROKE_STARTS[0] - 0.3)) < 1.0, al
        assert all(np.isfinite(r.metrics["m_distance"]) for r in res.reps)


def _pm5_log(path):
    """Stroke data as a PM5 sends it on the rower: at the end of each drive (and again with
    the recovery time at the next catch), timed on the video clock moved by LOG_OFFSET."""
    u24 = lambda v: int(round(v)).to_bytes(3, "little")
    u16 = lambda v: int(round(v)).to_bytes(2, "little")
    zero = STROKE_STARTS[0] + LOG_OFFSET
    el = lambda lt: max(0.0, round(lt - zero + 0.02, 2))
    rows, dist, rec_prev = [], 0.0, 0.0
    for k, (start, period) in enumerate(zip(STROKE_STARTS, PERIODS)):
        end = start + DRIVE_S
        dist += 9.0
        work, peak = 420.0, 700.0
        def data(lt, rec):
            return (u24(el(lt) * 100) + u24(dist * 10) + bytes([140, int(round(DRIVE_S * 100))]) +
                    u16(rec * 100) + u16(900) + u16(peak / LBF_TO_N * 10) +
                    u16(work / 1.4 / LBF_TO_N * 10) + u16(work * 10) + u16(k + 1))
        rows.append((end + LOG_OFFSET + 0.035, 0x0035, data(end + LOG_OFFSET, rec_prev)))
        rows.append((end + LOG_OFFSET + 0.038, 0x0036,
                     u24(el(end + LOG_OFFSET) * 100) + u16(work / period) + u16(900) + u16(k + 1)))
        if k + 1 < len(STROKE_STARTS):
            nxt = STROKE_STARTS[k + 1] + LOG_OFFSET
            rows.append((nxt + 0.03, 0x0035, data(nxt, period - DRIVE_S)))
        rec_prev = period - DRIVE_S
    # status twice a second, from the moment the log starts
    ends = [st + DRIVE_S for st in STROKE_STARTS]
    for lt in np.arange(0.2, ends[-1] + LOG_OFFSET + 2.0, 0.5):
        done = sum(e + LOG_OFFSET <= lt for e in ends)
        rows.append((lt, 0x0031, u24(el(lt) * 100) + u24(done * 90) +
                     bytes([1, 0, 1, 1, 1]) + u24(0) + u24(0) + bytes([0, 110])))
        rows.append((lt + 0.05, 0x0032, u24(el(lt) * 100) + u16(4000 if done else 0) +
                     bytes([24 if done else 0, 150]) + u16(12500) + u16(12500) + u16(0) +
                     u24(0)))
    with open(path, "w") as fh:
        fh.write(json.dumps({"t": 1_790_000_000.0, "device": {"name": "PM5 431234567 Row"}}) + "\n")
        for t, short, payload in sorted(rows, key=lambda r: r[0]):
            fh.write(json.dumps({"t": round(1_790_000_000.0 + t, 3), "uuid": f"{short:04x}",
                                 "hex": payload.hex()}) + "\n")


def test_rowerg_with_a_pm5_log():
    """The PM5 log as the machine data: strokes matched on the handle's catch and finish."""
    with tempfile.TemporaryDirectory() as d:
        ex, _ = _extraction(d)
        path = os.path.join(d, "row.pm5.jsonl")
        _pm5_log(path)
        res = analyze(ex, STATIONS["rowerg"], pm5=load_pm5_session(path))
        sync = res.pm5_sync
        assert sync is not None and sync.ok and sync.events == "hands", sync
        assert len(sync.pairs) >= len(KINDS) - 1, len(sync.pairs)
        # the log's clock starts at its first notification, 0.2 s in
        assert abs(sync.offset - (LOG_OFFSET - 0.2)) < 0.15, sync.offset
        assert res.machine is not None and res.machine.alignment.method == "strokes"


def test_rowerg_has_no_force_analysis_yet():
    assert not STATIONS["rowerg"].force
    th = default_thresholds("rowerg")
    assert th["shooting_slide"] == 1.2 and "drift_catch_lean" in th


def test_rhythm_reference_follows_the_rate():
    assert abs(rhythm_reference(20.7) - 41.7) < 1e-9
    assert abs(rhythm_reference(32.1) - 51.96) < 0.01        # Kleshnev 2005: 51.9
    assert rhythm_reference(5) == 35.0 and rhythm_reference(60) == 60.0


def test_crossing_interpolates_both_ways():
    t = np.arange(0, 1.0, 0.04)                               # 25 fps
    x = 100 * t
    when, i = crossing(t, x, 0, len(t) - 1, 30.0, rising=True)
    assert abs(when - 0.30) < 1e-9 and t[i] >= 0.3
    when, _ = crossing(t, -x, 0, len(t) - 1, -50.0, rising=False)
    assert abs(when - 0.50) < 1e-9
    assert math.isnan(crossing(t, x, 0, len(t) - 1, 500.0, rising=True)[0])


def test_pause_at_the_catch_is_not_part_of_the_stroke():
    t = np.arange(0, 6, 1 / 30)
    def handle(tt):            # still at the catch for 2 s, then one 2.5 s stroke, then still
        u = np.clip((tt - 2.0) / 2.5, 0, 1)
        return 1.0 + 0.9 * np.cos(2 * np.pi * u)
    hx = handle(t)
    rep = Cycle(10, int(round(3.25 * 30)), len(t) - 5)       # peaks somewhere in the pauses
    s, mid, e = RowErg.moving(hx, t, rep)
    assert abs(t[s] - 2.0) < 0.2 and abs(t[e] - 4.5) < 0.2, (t[s], t[e])
    busy = Cycle(int(2.0 * 30), int(3.25 * 30), int(4.5 * 30))   # no pause: left alone
    assert RowErg.moving(hx, t, busy) == (busy.start, busy.mid, busy.end)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
