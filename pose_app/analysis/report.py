"""Outputs: per-rep CSV, JSON summary and a self-contained HTML report."""

import base64
import csv
import html
import json
import math
from collections import Counter
from typing import Dict, List, Optional

import cv2

from .pipeline import AnalysisResult, RepResult
from .render import snapshot
from .stations import ChartSpec
from .video import VideoReader


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


def summary_dict(res: AnalysisResult) -> Dict:
    reps, n = res.reps, len(res.reps)
    counts = Counter(f.rule_id for r in reps for f in r.faults)
    rules = {r.id: r for r in res.station.rules()}

    def mean(key):
        vals = [r.metrics[key] for r in reps if math.isfinite(r.metrics.get(key, math.nan))]
        return _num(sum(vals) / len(vals)) if vals else None

    return {
        "station": res.station.key,
        "reps": n,
        "clean_reps": sum(1 for r in reps if not r.faults),
        "clean_pct": _num(100 * sum(1 for r in reps if not r.faults) / n) if n else None,
        "averages": {c.metric: mean(c.metric) for c in res.station.charts},
        "faults": [{"rule": rid, "title": rules[rid].title, "severity": rules[rid].severity,
                    "count": c, "pct": _num(100 * c / n), "cue": rules[rid].cue}
                   for rid, c in counts.most_common()],
        "drift": [{"rule": d.rule_id, "title": d.title, "metric": d.metric,
                   "early": _num(d.early), "late": _num(d.late), "change": _num(d.change),
                   "triggered": d.triggered, "cue": d.cue} for d in res.drift],
        "camera": {"side": res.body.side, "facing": res.body.facing,
                   "view_ratio": _num(res.body.view_ratio), "coverage": _num(res.body.coverage)},
        "thresholds": res.thresholds,
        "warnings": res.warnings,
        "machine": _machine_dict(res),
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


def write_json(res: AnalysisResult, path: str) -> None:
    data = summary_dict(res)
    data["per_rep"] = [{"n": r.number, **{k: _num(v) for k, v in r.metrics.items()},
                        "faults": [f.rule_id for f in r.faults]} for r in res.reps]
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


# ------------------------------------------------------------------ HTML

def _chart_svg(res: AnalysisResult, metric: str, label: str, unit: str) -> str:
    pts = [(r.number, r.metrics.get(metric, math.nan)) for r in res.reps]
    pts = [(x, y) for x, y in pts if math.isfinite(y)]
    if len(pts) < 2:
        return ""
    rules = [r for r in res.station.rules() if r.metric == metric]
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
    worst: Dict[str, tuple] = {}
    for r in res.reps:
        for f in r.faults:
            thr = res.thresholds[f.rule_id]
            margin = abs(f.value - thr)
            if f.rule_id not in worst or margin > worst[f.rule_id][0]:
                worst[f.rule_id] = (margin, r, f)
    cards = []
    for margin, r, f in sorted(worst.values(), key=lambda x: x[2].severity != "major"):
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
"""


MACHINE_CHARTS = (ChartSpec("m_power", "Machine power", "W"),
                  ChartSpec("m_dps", "Distance per stroke", "m"))


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
    specs = list(st.charts)
    if res.machine is not None and res.machine.alignment.ok:
        specs += MACHINE_CHARTS
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
