"""Outputs: per-rep CSV, JSON summary and a self-contained HTML report."""

import base64
import csv
import html
import json
import math
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .force import FORCE_CHARTS
from .force.analysis import mean_curves
from .force.model import REPORTED_JOINTS
from .force.sync import curve_profile
from .pipeline import AnalysisResult, RepResult
from .render import snapshot
from .rules import Fault
from .stations import ChartSpec
from .video import VideoReader

# summary.json format; bump when a consumer (the web app) would need to change.
SCHEMA_VERSION = 1

MACHINE_CHARTS = (ChartSpec("m_power", "Machine power", "W"),
                  ChartSpec("m_dps", "Distance per stroke", "m"))


def _num(v) -> Optional[float]:
    return round(float(v), 3) if v is not None and math.isfinite(v) else None


def write_csv(res: AnalysisResult, path: str) -> None:
    keys = [k for k in (res.reps[0].metrics if res.reps else {}) if not k.startswith("frame_")]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([res.station.rep_word, *keys, "faults"])
        for r in res.reps:
            w.writerow([r.number, *[_num(r.metrics[k]) for k in keys],
                        ";".join(f.rule_id for f in r.faults)])


def chart_specs(res: AnalysisResult) -> List[ChartSpec]:
    """Per-rep metrics worth charting: the station's, plus machine output when synced, plus
    the force measures when the force analysis ran."""
    specs = list(res.station.charts)
    if res.machine is not None and res.machine.alignment.ok:
        specs += MACHINE_CHARTS
    if res.force is not None and res.force.strokes:
        specs += FORCE_CHARTS
    return specs


def worst_faults(res: AnalysisResult) -> List[Tuple[RepResult, Fault]]:
    """For each fault type, the rep where it was furthest past its threshold; majors first."""
    worst: Dict[str, tuple] = {}
    for r in res.reps:
        for f in r.faults:
            margin = abs(f.value - res.thresholds[f.rule_id])
            if f.rule_id not in worst or margin > worst[f.rule_id][0]:
                worst[f.rule_id] = (margin, r, f)
    return [(r, f) for _, r, f in sorted(worst.values(), key=lambda x: x[2].severity != "major")]


def _clock(res: AnalysisResult) -> Callable[[int], Dict[str, Optional[float]]]:
    """Source frame index -> seconds into the source video ("t") and into
    annotated.mp4 ("video_t"), which holds every analysed frame at `res.fps`."""
    pos = {int(f): i for i, f in enumerate(res.body.frame_index)}

    def at(frame: int) -> Dict[str, Optional[float]]:
        i = pos.get(int(frame))
        if i is None:
            return {"t": None, "video_t": None}
        return {"t": _num(res.body.t[i]), "video_t": _num(i / res.fps) if res.fps else None}
    return at


def _fault_dict(f: Fault, clock) -> Dict:
    return {"rule": f.rule_id, "value": _num(f.value), **clock(f.frame)}


def summary_dict(res: AnalysisResult) -> Dict:
    reps, n = res.reps, len(res.reps)
    counts = Counter(f.rule_id for r in reps for f in r.faults)
    rules = {r.id: r for r in res.rule_list()}
    st, clock = res.station, _clock(res)

    def mean(key):
        vals = [r.metrics[key] for r in reps if math.isfinite(r.metrics.get(key, math.nan))]
        return _num(sum(vals) / len(vals)) if vals else None

    return {
        "schema_version": SCHEMA_VERSION,
        "station": {"key": st.key, "name": st.name, "rep_word": st.rep_word, "view": st.view,
                    "phases": list(st.phases), "filming_tips": list(st.filming_tips)},
        "reps": n,
        "clean_reps": sum(1 for r in reps if not r.faults),
        "clean_pct": _num(100 * sum(1 for r in reps if not r.faults) / n) if n else None,
        "averages": {c.metric: mean(c.metric) for c in res.station.charts},
        "faults": [{"rule": rid, "title": rules[rid].title, "severity": rules[rid].severity,
                    "count": c, "pct": _num(100 * c / n), "cue": rules[rid].cue}
                   for rid, c in counts.most_common()],
        "rules": [{"id": r.id, "title": r.title, "severity": r.severity, "metric": r.metric,
                   "op": r.op, "threshold": res.thresholds[r.id], "cue": r.cue}
                  for r in rules.values()],
        "charts": [{"metric": c.metric, "label": c.label, "unit": c.unit}
                   for c in chart_specs(res)],
        "examples": [{"rep": r.number, **_fault_dict(f, clock)} for r, f in worst_faults(res)],
        "drift": [{"rule": d.rule_id, "title": d.title, "metric": d.metric,
                   "early": _num(d.early), "late": _num(d.late), "change": _num(d.change),
                   "triggered": d.triggered, "cue": d.cue} for d in res.drift],
        "camera": {"side": res.body.side, "facing": res.body.facing,
                   "view_ratio": _num(res.body.view_ratio), "coverage": _num(res.body.coverage)},
        "thresholds": res.thresholds,
        "warnings": res.warnings,
        "machine": _machine_dict(res),
        "force": _force_dict(res, clock),
        "pm5": _pm5_dict(res),
    }


def _machine_dict(res: AnalysisResult) -> Optional[Dict]:
    mc = res.machine
    if mc is None:
        return None
    al = mc.alignment
    return {
        "source": mc.telemetry.source,
        "sync": {"method": al.method, "confidence": al.confidence, "offset_s": _num(al.offset),
                 "rate_r": _num(al.rate_r), "rate_offset_s": _num(al.rate_offset),
                 "onset_offset_s": _num(al.onset_offset), "notes": al.notes},
        "summary": {k: _num(v) for k, v in mc.summary.items()},
        "splits": [{k: (_num(v) if isinstance(v, float) else v) for k, v in vars(sp).items()}
                   for sp in mc.splits],
        "technique_vs_output": [{k: (_num(v) if isinstance(v, float) else v)
                                 for k, v in vars(a).items()} for a in mc.associations],
        "checks": mc.checks,
        "data_quality": mc.notes,
    }


def _example_stroke(res: AnalysisResult):
    """The force stroke closest to the median body-weight share (a typical stroke)."""
    fa = res.force
    strokes = [s for s in fa.strokes if s.profile is not None] if fa is not None else []
    if not strokes:
        return None
    bw = np.array([s.metrics.get("f_bw_share", math.nan) for s in strokes])
    if not np.isfinite(bw).any():
        return strokes[len(strokes) // 2]
    return strokes[int(np.nanargmin(np.abs(bw - np.nanmedian(bw))))]


def _force_dict(res: AnalysisResult, clock) -> Optional[Dict]:
    fa = res.force
    if fa is None:
        return None
    a, sync, pm5 = fa.setup.athlete, fa.sync, fa.pm5
    cal = {"cord_exit_px": list(fa.setup.calibration.cord_exit)}
    if fa.frame is not None:
        cal.update(scale_px_per_m=_num(fa.frame.scale), scale_source=fa.frame.source,
                   origin_px=[_num(v) for v in fa.frame.origin], facing=fa.frame.facing,
                   cord_exit_m=[_num(v) for v in fa.exit_m])
    example = _example_stroke(res)
    ex = None
    if example is not None:
        ex = {"rep": example.rep + 1, **clock(res.body.frame_index[example.peak])}
    return {
        "athlete": {"mass_kg": a.mass_kg, "height_m": a.height_m, "sex": a.sex},
        "calibration": cal,
        "pm5": {"source": pm5.source, "device": {k: str(v) for k, v in pm5.device.items()},
                "strokes": len(pm5.strokes), "curves": pm5.curves,
                "curve_spacing": pm5.curve_basis, "notes": pm5.notes},
        "sync": {"method": sync.method, "confidence": sync.confidence,
                 "offset_s": _num(sync.offset), "matched": len(sync.pairs),
                 "residual_s": _num(sync.residual_s), "margin": _num(sync.margin),
                 "events": sync.events, "notes": sync.notes},
        "summary": {k: _num(v) for k, v in fa.summary.items()},
        "curve": fa.curves,
        "reported_joints": list(REPORTED_JOINTS),
        "example": ex,
        "checks": fa.checks,
    }


def _pm5_dict(res: AnalysisResult) -> Optional[Dict]:
    """Every stroke the PM5 logged, with its force curve. Strokes are placed on the video
    (`t_start` / `t_end`, seconds in the source video) only when the log's strokes were
    matched to the video's one by one; `rep` is the video rep a stroke was matched to."""
    pm5, sync = res.pm5, res.pm5_sync
    if pm5 is None:
        return None
    linked = sync is not None and sync.ok
    rep_of = {p: w + 1 for w, p in sync.pairs.items()} if linked else {}
    spacing = pm5.curve_basis
    strokes, profiles = [], []
    for k, s in enumerate(pm5.strokes):
        prof = curve_profile(s.curve_n) if s.curve_n is not None else None
        if prof is not None and s.curve_basis == spacing:
            profiles.append(prof)
        strokes.append({
            "n": k + 1, "piece": s.piece + 1, "count": s.count, "rep": rep_of.get(k),
            "t_start": _num(s.t_start - sync.offset) if linked else None,
            "t_end": _num(s.t_end - sync.offset) if linked else None,
            "distance_m": _num(s.distance_m), "drive_time_s": _num(s.drive_time_s),
            "drive_length_m": _num(s.drive_length_m), "recovery_time_s": _num(s.recovery_time_s),
            "peak_n": _num(s.peak_force_n), "avg_n": _num(s.avg_force_n),
            "work_j": _num(s.work_j), "power_w": _num(s.power_w),
            "spacing": s.curve_basis if prof is not None else None,
            "peak_pct": (_num(100.0 * int(np.argmax(prof)) / (len(prof) - 1))
                         if prof is not None else None),
            "curve": [int(round(float(v))) for v in prof] if prof is not None else None,
        })
    sync_d = None
    if sync is not None:
        sync_d = {"method": sync.method, "confidence": sync.confidence,
                  "offset_s": _num(sync.offset), "matched": len(sync.pairs),
                  "residual_s": _num(sync.residual_s), "events": sync.events,
                  "linked": linked, "notes": sync.notes}
    return {
        "source": pm5.source,
        "device": {k: str(v) for k, v in pm5.device.items()},
        "curves": pm5.curves,
        "curve_spacing": spacing,
        "sync": sync_d,
        "mean_curve": mean_curves(profiles) or None,
        "strokes": strokes,
        "notes": pm5.notes,
    }


def write_json(res: AnalysisResult, path: str) -> None:
    data = summary_dict(res)
    clock = _clock(res)

    def video_t(i: int) -> Optional[float]:
        return _num(i / res.fps) if res.fps else None
    data["per_rep"] = [{"n": r.number, **{k: _num(v) for k, v in r.metrics.items()},
                        "video_t_start": video_t(r.cycle.start),
                        "video_t_end": video_t(r.cycle.end),
                        "faults": [_fault_dict(f, clock) for f in r.faults]} for r in res.reps]
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


# ------------------------------------------------------------------ HTML

def _chart_svg(res: AnalysisResult, metric: str, label: str, unit: str) -> str:
    pts = [(r.number, r.metrics.get(metric, math.nan)) for r in res.reps]
    pts = [(x, y) for x, y in pts if math.isfinite(y)]
    if len(pts) < 2:
        return ""
    rules = [r for r in res.rule_list() if r.metric == metric]
    flagged = {r.number for r in res.reps for f in r.faults if f.metric == metric}
    ys = [y for _, y in pts] + [res.thresholds[r.id] for r in rules]
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.1 or 1.0
    lo, hi = lo - pad, hi + pad
    W, H, L, R, T, B = 640, 180, 44, 12, 12, 24
    x0, x1 = pts[0][0], pts[-1][0]

    def sx(x):
        return L + (x - x0) / max(x1 - x0, 1) * (W - L - R)

    def sy(y):
        return T + (hi - y) / (hi - lo) * (H - T - B)
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(label)}">']
    for frac in (0, 0.5, 1):
        v = lo + frac * (hi - lo)
        parts.append(f'<line class="grid" x1="{L}" x2="{W-R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>'
                     f'<text class="ax" x="{L-6}" y="{sy(v)+4:.1f}" text-anchor="end">{v:.3g}</text>')
    for r in rules:
        y = sy(res.thresholds[r.id])
        parts.append(f'<line class="thr" x1="{L}" x2="{W-R}" y1="{y:.1f}" y2="{y:.1f}"/>'
                     f'<text class="ax thr-t" x="{W-R}" y="{y-4:.1f}" text-anchor="end">'
                     f'{html.escape(r.title)}</text>')
    path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(pts))
    parts.append(f'<path class="series" d="{path}"/>')
    for x, y in pts:
        cls = "pt bad" if x in flagged else "pt"
        parts.append(f'<circle class="{cls}" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5">'
                     f'<title>#{x}: {y:.3g}{unit}</title></circle>')
    parts.append(f'<text class="ax" x="{L}" y="{H-6}">{res.station.rep_word} {x0}</text>'
                 f'<text class="ax" x="{W-R}" y="{H-6}" text-anchor="end">{x1}</text></svg>')
    return "".join(parts)


def _img_tag(img) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(buf).decode()}" alt="">' if ok else ""


def _examples(res: AnalysisResult, reader: Optional[VideoReader], per_rule: int = 1) -> str:
    """One snapshot per fault type, taken from the rep where it was worst."""
    if reader is None:
        return ""
    cards = []
    for r, f in worst_faults(res):
        img = snapshot(reader, res, f.frame) if f.frame >= 0 else None
        cards.append(
            f'<figure class="card">{_img_tag(img) if img is not None else ""}'
            f'<figcaption><b class="{f.severity}">{html.escape(f.title)}</b> '
            f'&middot; {res.station.rep_word} {r.number} at {r.metrics["t_start"]:.1f}s '
            f'&middot; {html.escape(f.metric)} = {f.value:.3g} ({html.escape(f.condition)})'
            f'<br>{html.escape(f.cue)}</figcaption></figure>')
    return "".join(cards)


CSS = """
:root{--bg:#fff;--fg:#1b1b1b;--mut:#6b6b6b;--line:#e3e3e3;--acc:#0a8a8a;--bad:#d43c3c;--warn:#e08a00;--ok:#3a9a3a}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#ececec;--mut:#9a9a9a;--line:#333;--acc:#3cc9c9}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}
h1{margin:0 0 4px}h2{margin:36px 0 10px;font-size:18px}
.mut{color:var(--mut)}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:18px 0}
.kpi{border:1px solid var(--line);border-radius:10px;padding:12px}.kpi b{display:block;font-size:24px}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top}
.scroll{overflow-x:auto}.major{color:var(--bad)}.minor{color:var(--warn)}.yes{color:var(--bad);font-weight:600}
.warn{border-left:4px solid var(--warn);padding:8px 12px;margin:8px 0;background:rgba(224,138,0,.08)}
svg{width:100%;height:auto}.grid{stroke:var(--line)}.ax{fill:var(--mut);font-size:11px}
.series{fill:none;stroke:var(--acc);stroke-width:2}.pt{fill:var(--acc)}.pt.bad{fill:var(--bad)}
.thr{stroke:var(--bad);stroke-dasharray:5 4;opacity:.7}.thr-t{fill:var(--bad)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}
.card{margin:0;border:1px solid var(--line);border-radius:10px;overflow:hidden}.card img{width:100%;display:block}
.card figcaption{padding:10px 12px;font-size:14px}
h3{font-size:15px;margin:22px 0 6px}.cover{fill:var(--acc);opacity:.08}
.tick{stroke-width:2}.major-t{stroke:var(--bad)}.minor-t{stroke:var(--warn)}
.series.early{stroke:var(--mut);stroke-dasharray:6 4}.series.late{stroke:var(--warn)}
.all-t{fill:var(--acc)}.early-t{fill:var(--mut)}.late-t{fill:var(--warn)}
.stroke{fill:none;stroke:var(--fg);stroke-width:1;opacity:.12}
"""


def _pace(sec: float) -> str:
    if sec is None or not math.isfinite(sec):
        return "–"
    return f"{int(sec // 60)}:{sec % 60:04.1f}"


def _f(v, fmt: str = "{:.0f}", unit: str = "") -> str:
    return (fmt.format(v) + unit) if v is not None and math.isfinite(v) else "–"


def _timeline_svg(res: AnalysisResult) -> str:
    """Machine power over the piece; ticks mark strokes where the video found a fault."""
    mc = res.machine
    tel, al = mc.telemetry, mc.alignment
    from .telemetry.fusion import SETTLE_S
    on = tel.active & (tel.power == tel.power) & (tel.t >= tel.active_span[0] + SETTLE_S)
    if on.sum() < 2:
        return ""
    t, p = tel.t, tel.power
    x0, x1 = float(t[on][0]), float(t[on][-1])
    lo, hi = float(p[on].min()), float(p[on].max())
    pad = (hi - lo) * 0.15 or 5.0
    lo, hi = max(0.0, lo - pad), hi + pad
    W, H, L, R, T, B = 640, 200, 44, 12, 12, 34
    sx = lambda x: L + (x - x0) / max(x1 - x0, 1e-6) * (W - L - R)
    sy = lambda y: T + (hi - y) / (hi - lo) * (H - T - B)
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Machine power over time">']
    if al.ok and res.reps:
        a = res.reps[0].metrics["t_start"] + al.offset
        b = res.reps[-1].metrics["t_start"] + res.reps[-1].metrics["duration_s"] + al.offset
        a, b = max(a, x0), min(b, x1)
        if b > a:
            parts.append(f'<rect class="cover" x="{sx(a):.1f}" y="{T}" width="{sx(b) - sx(a):.1f}" '
                         f'height="{H - T - B}"><title>on video</title></rect>')
    for frac in (0, 0.5, 1):
        v = lo + frac * (hi - lo)
        parts.append(f'<line class="grid" x1="{L}" x2="{W-R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>'
                     f'<text class="ax" x="{L-6}" y="{sy(v)+4:.1f}" text-anchor="end">{v:.0f}</text>')
    segs, pen = [], False
    for ti, pi, ok in zip(t, p, on):
        if ok and x0 <= ti <= x1:
            segs.append(f"{'L' if pen else 'M'}{sx(ti):.1f},{sy(pi):.1f}")
        pen = bool(ok)
    parts.append(f'<path class="series" d="{" ".join(segs)}"/>')
    if al.ok:
        for r in res.reps:
            if r.faults:
                x = r.metrics["t_start"] + r.metrics["duration_s"] + al.offset
                if x0 <= x <= x1:
                    cls = "tick major-t" if r.worst == "major" else "tick minor-t"
                    parts.append(f'<line class="{cls}" x1="{sx(x):.1f}" x2="{sx(x):.1f}" '
                                 f'y1="{H-B+2}" y2="{H-B+12}"><title>{res.station.rep_word} '
                                 f'{r.number}: {html.escape(", ".join(f.title for f in r.faults))}'
                                 f'</title></line>')
    parts.append(f'<text class="ax" x="{L}" y="{H-4}">{_pace(x0)}</text>'
                 f'<text class="ax" x="{W-R}" y="{H-4}" text-anchor="end">{_pace(x1)} '
                 f'machine time &middot; W</text></svg>')
    return "".join(parts)


def _machine_html(res: AnalysisResult) -> str:
    mc = res.machine
    if mc is None:
        return ""
    esc, sm, al, rw = html.escape, mc.summary, mc.alignment, res.station.rep_word
    kpis = [(_f(sm["distance_m"], "{:.0f}", " m"), "distance"),
            (_pace(sm["active_s"]), "active time"),
            (_pace(sm["pace_500"]), "avg pace /500 m"),
            (_f(sm["power_w"], "{:.0f}", " W"), "avg power"),
            (_f(sm["spm"], "{:.1f}"), "avg spm (machine)"),
            (f'{_f(sm["hr_avg"])} / {_f(sm["hr_max"])}', "HR avg / max")]
    sync = (f'Synchronised by <b>{esc(al.method)}</b>, confidence <b>{esc(al.confidence)}</b>'
            + (f': video 0 s = machine {al.offset:+.2f} s.' if al.ok else '.')
            + "".join(f" {esc(n)}" for n in al.notes if not n.startswith("Video 0 s")))
    split_rows = "".join(
        f'<tr><td>{sp.start_m:.0f}–{sp.end_m:.0f} m</td><td>{_pace(sp.time_s)}</td>'
        f'<td>{_pace(sp.pace)}</td><td>{_f(sp.power, "{:.0f}", " W")}</td>'
        f'<td>{_f(sp.spm, "{:.1f}")}</td><td>{_f(sp.hr)}</td><td>{sp.strokes or "–"}</td>'
        f'<td>{_f(sp.fault_pct, "{:.0f}", "%") if sp.strokes else "–"}</td>'
        f'<td>{esc(sp.top_fault)}</td></tr>' for sp in mc.splits)
    if mc.associations:
        assoc = ("<p class=\"mut\">Machine output on strokes where the video found each fault, "
                 "compared with strokes without it. Machine values are smoothed over several "
                 "strokes and faults tend to cluster when tired, so read this as an "
                 "association to discuss with a coach, not proof of cause.</p>"
                 "<div class=\"scroll\"><table><tr><th>Fault</th><th>" + rw.title() + "s with / "
                 "without</th><th>Power</th><th>Distance per stroke</th><th>Chance it's noise"
                 "</th></tr>" + "".join(
                     f'<tr><td>{esc(a.title)}</td><td>{a.n_fault} / {a.n_clean}</td>'
                     f'<td>{a.power_diff:+.1%}</td><td>{a.dps_diff:+.1%}</td>'
                     f'<td>{"low" if a.p_value < 0.05 else "high"} (p={a.p_value:.2f})</td></tr>'
                     for a in mc.associations) + "</table></div>")
    elif al.ok:
        assoc = (f'<p class="mut">No fault occurred on enough {rw}s (and differed enough in '
                 "output) to compare machine output with and without it.</p>")
    else:
        assoc = ""
    quality = "".join(f"<li>{esc(n)}</li>" for n in mc.notes) or "<li>No issues found.</li>"
    return f"""<h2>Machine data</h2>
<p class="mut">{esc(mc.telemetry.source)}. {sync}</p>
<div class="kpis">{"".join(f'<div class="kpi"><b>{v}</b><span class="mut">{esc(l)}</span></div>' for v, l in kpis)}</div>
<h3>Power over the piece</h3><p class="mut">Shaded: the part on video. Ticks: {rw}s with a
fault (red major, orange minor).</p>{_timeline_svg(res)}
<h3>Splits</h3><div class="scroll"><table><tr><th>Split</th><th>Time</th><th>Pace /500 m</th>
<th>Power</th><th>Rate</th><th>HR</th><th>{rw.title()}s on video</th><th>With faults</th>
<th>Most common fault</th></tr>{split_rows}</table></div>
<h3>Technique vs output</h3>{assoc}
<h3>Machine data quality</h3><ul>{quality}</ul>"""


def _curve_svg(curves: Dict[str, List[float]], strokes: Sequence[List[float]] = ()) -> str:
    """Mean force curve over the drive; early and late thirds when the piece is long enough.
    `strokes`: single strokes' curves, drawn faintly underneath."""
    if not curves:
        return ""
    x = curves["x"]
    series = [("all", "series", "all strokes")]
    if "early" in curves:
        series += [("early", "series early", "first third"), ("late", "series late", "last third")]
    top = max([max(curves[k]) for k, _, _ in series] + [max(c) for c in strokes])
    hi = max(100.0, math.ceil(top * 1.05 / 100.0) * 100.0)
    W, H, L, R, T, B = 640, 220, 44, 12, 12, 34
    sx = lambda v: L + v * (W - L - R)
    sy = lambda v: T + (hi - v) / hi * (H - T - B)
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Mean force curve">']
    for frac in (0, 0.5, 1):
        v = frac * hi
        parts.append(f'<line class="grid" x1="{L}" x2="{W-R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>'
                     f'<text class="ax" x="{L-6}" y="{sy(v)+4:.1f}" text-anchor="end">{v:.0f}</text>')
    for c in strokes:
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(i / (len(c) - 1)):.1f},{sy(v):.1f}"
                     for i, v in enumerate(c))
        parts.append(f'<path class="stroke" d="{d}"/>')
    legend = []
    for key, cls, label in series:
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(a):.1f},{sy(b):.1f}"
                     for i, (a, b) in enumerate(zip(x, curves[key])))
        parts.append(f'<path class="{cls}" d="{d}"><title>{label}</title></path>')
        legend.append(f'<tspan class="{key}-t">{"&#9472;" * 2} {label}</tspan>')
    parts.append(f'<text class="ax" x="{L}" y="{H-6}">start of drive</text>'
                 f'<text class="ax" x="{W-R}" y="{H-6}" text-anchor="end">end of drive &middot; N</text>'
                 f'<text class="ax" x="{(W + L) / 2:.0f}" y="{H-6}" text-anchor="middle">'
                 f'{" ".join(legend)}</text></svg>')
    return "".join(parts)


def _pm5_html(res: AnalysisResult) -> str:
    """Every stroke's PM5 force curve, faint, under the piece's mean curves."""
    d = _pm5_dict(res)
    if d is None:
        return ""
    esc, rw = html.escape, res.station.rep_word
    sync = d["sync"]
    if sync is not None and sync["linked"]:
        matched = sum(1 for s in d["strokes"] if s["rep"] is not None)
        where = f"{matched} of its strokes matched to {rw}s in the video, one by one."
    else:
        where = "Its strokes could not be matched to the video one by one, so they stand alone."
    head = f'<h2>PM5 force curves</h2><p class="mut">{esc(res.pm5.describe())}. {where}</p>'
    curves = [s["curve"] for s in d["strokes"] if s["curve"] and s["spacing"] == d["curve_spacing"]]
    if not curves:
        return head + "<ul>" + "".join(f"<li>{esc(n)}</li>" for n in d["notes"]) + "</ul>"
    basis = "handle travel" if d["curve_spacing"] == "travel" else "time"
    return (head + f'<p class="mut">Force on the handles in every stroke (faint) and the mean '
            f'of the piece, from the first force to the release, spaced by {basis}.</p>'
            + _curve_svg(d["mean_curve"], curves))


def _force_html(res: AnalysisResult, reader: Optional[VideoReader]) -> str:
    fa = res.force
    if fa is None:
        return ""
    esc, sm, rw = html.escape, fa.summary, res.station.rep_word
    a = fa.setup.athlete
    scale = (f"scale from {'the calibration length' if fa.frame.source == 'stick' else 'height'}"
             if fa.frame is not None else "no image scale")
    head = (f'<p class="mut">{esc(fa.pm5.describe())}. Athlete {a.mass_kg:g} kg, {a.height_m:.2f} m, '
            f'{esc(a.sex)}; {scale}. {esc(fa.sync.notes[0]) if fa.sync.notes else ""}</p>')
    if not fa.strokes:
        checks = "".join(f"<li>{esc(c)}</li>" for c in fa.checks) or "<li>No force data.</li>"
        return f"<h2>Force analysis</h2>{head}<ul>{checks}</ul>"
    kpis = [(_f(sm.get("f_peak_n"), "{:.0f}", " N"), "avg peak cord force"),
            (_f(sm.get("f_bw_share"), "{:.0f}", "%"), "of the work from body weight"),
            (_f(sm.get("f_cord_angle"), "{:.0f}", "°"), "cord angle at peak force"),
            (_f(sm.get("f_capacity_n"), "{:.0f}", " N"), "static force capacity"),
            (_f(sm.get("f_shoulder_ext_nm"), "{:.0f}", " N·m"), "shoulder moment at peak, per side"),
            (_f(sm.get("f_peak_pos"), "{:.0f}", "%"), "of the cord travel before force peaks")]
    example = _example_stroke(res)
    card = ""
    if example is not None and reader is not None:
        frame = int(res.body.frame_index[example.peak])
        img = snapshot(reader, res, frame)
        m = example.metrics
        card = (f'<figure class="card" style="max-width:640px">'
                f'{_img_tag(img) if img is not None else ""}<figcaption>'
                f'{rw.title()} {example.rep + 1} at peak force: {m.get("f_peak_n", 0):.0f} N on the '
                f'cords, {m.get("f_bw_share", math.nan):.0f}% of the work from body weight, '
                f'shoulder moment {m.get("f_shoulder_ext_nm", math.nan):.0f} N·m per side.'
                f'</figcaption></figure>')
    checks = "".join(f"<li>{esc(c)}</li>" for c in fa.checks) or "<li>No issues found.</li>"
    jit = sm.get("jitter_mm", math.nan)
    quality = (f"Stroke match: {esc(fa.sync.confidence)}, {len(fa.sync.pairs)} strokes. "
               f"Energy check: joint work vs PM5 work {_f(sm.get('energy_err_median'), '{:.0f}', '%')} "
               f"(median). Keypoint jitter at the ankle {_f(jit, '{:.0f}', ' mm')}.")
    return f"""<h2>Force analysis</h2>{head}
<div class="kpis">{"".join(f'<div class="kpi"><b>{v}</b><span class="mut">{esc(l)}</span></div>' for v, l in kpis)}</div>
<p class="mut">Force on the cords from the PM5 (both cords together), applied along the cord
seen in the video. A hinge-led stroke takes about 40% of its work from the body's drop and pulls
with the cords about 78° from horizontal. Moments are per side; only the shoulder and elbow are
shown, as one side-on camera cannot measure hip, knee and ankle loads reliably. These describe
technique, not injury risk.</p>
<h3>Force curve</h3>{_curve_svg(fa.curves)}
<div class="cards">{card}</div>
<h3>Force data quality</h3><p class="mut">{quality}</p><ul>{checks}</ul>"""


def write_html(res: AnalysisResult, path: str, reader: Optional[VideoReader] = None,
               title: str = "") -> None:
    s = summary_dict(res)
    st, esc = res.station, html.escape
    rep_word = st.rep_word
    kpis = [(s["reps"], f"{rep_word}s detected"),
            (f'{s["clean_pct"]:.0f}%' if s["clean_pct"] is not None else "–", f"clean {rep_word}s")]
    for c in st.charts[:2]:
        v = s["averages"].get(c.metric)
        kpis.append((f"{v:.3g} {c.unit}" if v is not None else "–", f"avg {c.label.lower()}"))

    fault_rows = "".join(
        f'<tr><td><b class="{f["severity"]}">{esc(f["title"])}</b></td><td>{f["count"]} '
        f'({f["pct"]:.0f}%)</td><td>{esc(f["cue"])}</td></tr>' for f in s["faults"]
    ) or f'<tr><td colspan="3">No faults detected in any {rep_word}.</td></tr>'
    drift_rows = "".join(
        f'<tr><td>{esc(d["title"])}</td><td>{d["early"]:.3g} &rarr; {d["late"]:.3g}</td>'
        f'<td class="{"yes" if d["triggered"] else "mut"}">{"yes" if d["triggered"] else "no"}</td>'
        f'<td>{esc(d["cue"]) if d["triggered"] else ""}</td></tr>' for d in s["drift"]
    ) or '<tr><td colspan="4" class="mut">Not enough reps to compare early and late.</td></tr>'
    specs = chart_specs(res)
    charts = "".join(f"<h3>{esc(c.label)}{f' ({esc(c.unit)})' if c.unit else ''}</h3>"
                     f"{_chart_svg(res, c.metric, c.label, c.unit)}" for c in specs)
    cols = [c.metric for c in specs]
    per_rep = "".join(
        f'<tr><td>{r.number}</td><td>{r.metrics["t_start"]:.1f}</td>'
        + "".join(f'<td>{r.metrics[c]:.3g}</td>' if math.isfinite(r.metrics[c]) else "<td>–</td>"
                  for c in cols)
        + f'<td>{", ".join(f"<span class={chr(34)}{f.severity}{chr(34)}>{esc(f.title)}</span>" for f in r.faults) or "✓"}</td></tr>'
        for r in res.reps)
    warnings = "".join(f'<div class="warn">{esc(w)}</div>' for w in s["warnings"])
    tips = "".join(f"<li>{esc(t)}</li>" for t in st.filming_tips)

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(st.name)} technique report</title><style>{CSS}</style></head><body><main>
<h1>{esc(st.name)} technique report</h1>
<div class="mut">{esc(title)}</div>
{warnings}
<div class="kpis">{"".join(f'<div class="kpi"><b>{v}</b><span class="mut">{esc(l)}</span></div>' for v, l in kpis)}</div>
<h2>Faults</h2><div class="scroll"><table><tr><th>Fault</th><th>{rep_word.title()}s</th><th>Cue</th></tr>{fault_rows}</table></div>
<h2>Fatigue: first third vs last third</h2><div class="scroll"><table>
<tr><th>Check</th><th>Early &rarr; late</th><th>Flagged</th><th>Cue</th></tr>{drift_rows}</table></div>
{_machine_html(res)}
{_pm5_html(res)}
{_force_html(res, reader)}
<h2>Examples</h2><div class="cards">{_examples(res, reader)}</div>
<h2>Per-{rep_word} trends</h2><p class="mut">Red points broke a rule; dashed lines are the thresholds.</p>{charts}
<h2>All {rep_word}s</h2><div class="scroll"><table><tr><th>#</th><th>t (s)</th>
{"".join(f"<th>{esc(c.label)}</th>" for c in specs)}<th>Faults</th></tr>{per_rep}</table></div>
<h2>Filming checklist</h2><ul>{tips}</ul>
<p class="mut">Thresholds are coaching heuristics, not standards. Tune them per athlete with
<code>--print-thresholds</code> / <code>--thresholds</code>. Camera: near side {esc(res.body.side)},
side-view ratio {res.body.view_ratio:.2f}, athlete measured in {res.body.coverage:.0%} of frames.</p>
</main></body></html>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
