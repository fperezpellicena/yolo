"""Machine telemetry: reader, cleaning, sync and fusion with the synthetic clip.

Run: python tests/test_telemetry.py (or pytest). No model or GPU needed.
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from synthetic_skierg import FakeEstimator, KINDS, STROKE_STARTS, period_of, write_video
from synthetic_telemetry import make_tmiw_csv
from pose_app.analysis.pipeline import analyze, extract
from pose_app.analysis.report import write_csv, write_html, write_json
from pose_app.analysis.stations import STATIONS
from pose_app.analysis.telemetry import Telemetry, align, clean, load_telemetry
from pose_app.analysis.tracking import AthleteTracker
from pose_app.analysis.video import VideoReader

PERIODS = [int(period_of(n) * 30) / 30 for n in range(len(KINDS))]


def _extraction(d):
    video = os.path.join(d, "ski.mp4")
    write_video(video)
    reader = VideoReader(video)
    ex = extract(reader, FakeEstimator().detect,
                 AthleteTracker("largest", frame_size=(960, 720)), "fake", progress=False)
    return ex, reader


def test_reader_and_clean():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.csv")
        make_tmiw_csv(path, STROKE_STARTS, PERIODS, KINDS, log_start=STROKE_STARTS[0] - 0.3)
        raw = load_telemetry(path)
        assert abs(np.nanmedian(np.diff(raw.t)) - 0.48) < 0.01
        tel = clean(raw)
        a, b = tel.active_span
        last_stroke_end = STROKE_STARTS[-1] + PERIODS[-1] - (STROKE_STARTS[0] - 0.3)
        assert abs(b - last_stroke_end) < 2.5, (b, last_stroke_end)     # held tail cut
        assert any("held values" in n for n in tel.notes), tel.notes
        assert np.isnan(tel.power[-1]) and np.isnan(tel.speed[-1])


def test_clean_gap_and_flatline():
    t = np.concatenate([np.arange(0, 120, 0.5), np.arange(400, 520, 0.5)])
    dist = np.concatenate([np.arange(240) * 1.5, 360 + np.arange(240) * 1.5])
    hr = np.where(t > 20, 136.0, 100 + t)                  # strap dropped, value held
    tel = clean(Telemetry("test", 0, t, np.full(len(t), 150.0) + np.sin(t), np.full(len(t), 25.0) + (t // 30) % 2,
                          dist, np.full(len(t), 3.0), hr))
    notes = " ".join(tel.notes)
    assert "Recording gap of 280 s" in notes, notes
    assert "HR stuck at 136" in notes, notes
    assert np.isnan(tel.at("hr", 300.0)[0]) and np.isnan(tel.at("power", 300.0)[0])
    assert "SPM stuck" not in notes, notes             # runs do not span the gap
    assert abs(tel.strokes_between(0, 520) - 240 * 25.5 / 60) < 3   # 240 s active


def _rewrite_columns(path, drop=(), blank=()):
    """Remove or empty columns of a TMIW CSV, as other app settings/devices would."""
    lines = open(path).read().splitlines()
    i = lines.index("RIDE DATA")
    header = lines[i + 1].split(",")
    keep = [k for k, h in enumerate(header) if h not in drop]
    out = lines[:i + 1]
    for line in lines[i + 1:]:
        cells = line.split(",")
        cells = ["" if (header[k] in blank and line is not lines[i + 1]) else cells[k] for k in keep]
        out.append(",".join(cells))
    open(path, "w").write("\n".join(out) + "\n")


def test_missing_heart_rate_and_distance():
    for drop, blank in ((("HR",), ()), ((), ("HR",)), (("DISTANCE",), ())):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.csv")
            make_tmiw_csv(path, STROKE_STARTS, PERIODS, KINDS, log_start=STROKE_STARTS[0] - 0.3)
            _rewrite_columns(path, drop, blank)
            tel = clean(load_telemetry(path))
            if "DISTANCE" in drop:
                assert any("integrated from speed" in n for n in tel.notes), tel.notes
                assert np.nanmax(tel.distance) > 100 and np.isfinite(tel.hr).any()
            else:
                assert not np.isfinite(tel.hr).any()
                assert any("No heart-rate" in n for n in tel.notes), tel.notes
            al = align(tel, np.array(STROKE_STARTS), np.array(PERIODS), video_from_rest=True)
            assert al.ok and abs(al.offset + STROKE_STARTS[0] - 0.3) < 1.0, (drop, blank, al)


def test_video_machine_line():
    from pose_app.analysis.render import _machine_line
    full = {"m_hr": 142.4, "m_distance": 350.0, "m_power": 112.2, "m_pace": 134.04}
    assert _machine_line(full) == "HR 142  350 m  112 W  2:14.0 /500m"
    assert _machine_line({**full, "m_hr": np.nan}) == "350 m  112 W  2:14.0 /500m"
    assert _machine_line({**full, "m_power": np.nan, "m_pace": np.nan}) == "HR 142  350 m"
    assert _machine_line({}) == ""


def test_sync_recovers_offset():
    starts = np.array(STROKE_STARTS)
    for log_start in (STROKE_STARTS[0] - 0.3, -2.0):          # app started after / before video
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.csv")
            make_tmiw_csv(path, STROKE_STARTS, PERIODS, KINDS, log_start=log_start)
            tel = clean(load_telemetry(path))
            al = align(tel, starts, np.array(PERIODS), video_from_rest=True)
            assert al.ok and abs(al.offset + log_start) < 1.0, (log_start, al)
            manual = align(tel, starts, np.array(PERIODS), manual=1.5)
            assert manual.method == "manual" and manual.offset == 1.5


def test_pipeline_with_telemetry():
    with tempfile.TemporaryDirectory() as d:
        ex, reader = _extraction(d)
        path = os.path.join(d, "t.csv")
        make_tmiw_csv(path, STROKE_STARTS, PERIODS, KINDS, log_start=STROKE_STARTS[0] - 0.3)
        res = analyze(ex, STATIONS["skierg"], telemetry=clean(load_telemetry(path)))
        mc = res.machine
        assert mc.alignment.ok and mc.alignment.confidence in ("high", "medium"), mc.alignment
        assert abs(mc.alignment.offset + (STROKE_STARTS[0] - 0.3)) < 1.0, mc.alignment
        settled = [np.isfinite(r.metrics["m_power"]) for r in res.reps]
        assert not settled[0] and all(settled[3:]), settled   # monitor start-up ignored
        # heart rate and distance don't wait for the monitor to settle
        assert all(np.isfinite(r.metrics["m_hr"]) for r in res.reps)
        dist = [r.metrics["m_distance"] for r in res.reps]
        assert all(np.isfinite(dist)) and all(np.diff(dist) > 0), dist
        assert not mc.checks, mc.checks                    # rate and stroke count agree
        drift = {x.rule_id: x.triggered for x in res.drift}
        assert drift["drift_power"], drift                 # squat strokes lose power
        # technique unchanged by telemetry
        for rep, kind in zip(res.reps, KINDS):
            ids = {f.rule_id for f in rep.faults}
            assert ("squat_pattern" in ids) == (kind == "squat"), (rep.number, ids)
        assert mc.splits and sum(s.strokes for s in mc.splits) >= len(res.reps) - 1
        write_csv(res, os.path.join(d, "reps.csv"))
        write_json(res, os.path.join(d, "summary.json"))
        write_html(res, os.path.join(d, "report.html"), reader)
        assert "m_power" in open(os.path.join(d, "reps.csv")).readline()
        assert "Machine data" in open(os.path.join(d, "report.html")).read()


def test_pipeline_without_sync_degrades():
    with tempfile.TemporaryDirectory() as d:
        ex, reader = _extraction(d)
        path = os.path.join(d, "t.csv")
        # machine log from a different session: 10 minutes of steady rowing
        make_tmiw_csv(path, list(np.arange(0, 600, 2.5)), [2.5] * 240, ["good"] * 240,
                      log_start=0.0)
        res = analyze(ex, STATIONS["skierg"], telemetry=clean(load_telemetry(path)))
        assert not res.machine.alignment.ok, res.machine.alignment
        assert all(np.isnan(r.metrics["m_power"]) for r in res.reps)
        assert any("synchronise" in w for w in res.warnings), res.warnings
        write_html(res, os.path.join(d, "report.html"), reader)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
