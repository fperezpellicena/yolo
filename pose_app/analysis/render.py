"""Annotated output video: skeleton, live metrics, rep counter and cues."""

import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..skeleton import KP
from ..ui.text import darken, text_size, thickness
from ..ui.theme import AA, ACCENT, FONT, TEXT
from .pipeline import AnalysisResult
from .video import VideoReader

GOOD = (90, 200, 90)
MINOR = (0, 180, 255)
MAJOR = (60, 60, 235)
SEVERITY_COLOR = {None: GOOD, "minor": MINOR, "major": MAJOR}

EDGES = [(a, b) for a, b in (
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
)]
ANGLE_JOINTS = (("elbow", "elbow"), ("hip", "hip"), ("knee", "knee"))


def draw_skeleton(img: np.ndarray, kp: np.ndarray, near: str, metrics_row: dict) -> None:
    s = max(0.6, img.shape[0] / 720)
    lw = max(2, int(3 * s))
    for a, b in EDGES:
        pa, pb = kp[KP[a]], kp[KP[b]]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            far = not (a.startswith(near) and b.startswith(near))
            color = (110, 110, 110) if far else ACCENT
            cv2.line(img, tuple(pa.astype(int)), tuple(pb.astype(int)), color,
                     max(1, lw - 1) if far else lw, AA)
    fs = 0.55 * s
    for metric, joint in ANGLE_JOINTS:
        p, v = kp[KP[f"{near}_{joint}"]], metrics_row.get(metric, np.nan)
        if np.isfinite(p).all():
            cv2.circle(img, tuple(p.astype(int)), lw + 2, ACCENT, -1, AA)
            if np.isfinite(v):
                x, y = (p + [10 * s, -10 * s]).astype(int)
                text = f"{v:.0f}"
                tw, th, _ = text_size(text, fs, thickness(fs))
                darken(img, x - 3, y - th - 4, x + tw + 4, y + 5, 0.25)
                cv2.putText(img, text, (x, y), FONT, fs, TEXT, thickness(fs), AA)


def _wrap(text: str, width: float, fs: float) -> List[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if text_size(trial, fs, thickness(fs))[0] > width and line:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + [line] if line else lines


class Annotator:
    def __init__(self, res: AnalysisResult):
        self.res = res
        self.rep_of = res.rep_at_frame()
        self.by_frame = {int(f): i for i, f in enumerate(res.body.frame_index)}
        t = res.body.t
        self.span = (t[0], t[-1]) if len(t) else (0.0, 1.0)

    def draw(self, img: np.ndarray, frame_index: int) -> np.ndarray:
        res, body = self.res, self.res.body
        i = self.by_frame.get(frame_index)
        h, w = img.shape[:2]
        s = max(0.6, h / 720)
        if i is not None:
            row = {k: v[i] for k, v in body.metrics.items()}
            draw_skeleton(img, body.keypoints[i], body.side, row)
        k = self.rep_of[i] if i is not None else -1
        rep = res.reps[k] if k >= 0 else None

        # Header: station, rep counter, rate, phase
        fs = 0.7 * s
        lines = [f"{res.station.name}"]
        if rep is not None:
            c = rep.cycle
            phase = res.station.phases[0] if i < c.mid else res.station.phases[1]
            lines.append(f"{res.station.rep_word.title()} {rep.number}/{len(res.reps)}"
                         f"  {rep.metrics.get('rate_spm', np.nan):.0f} spm  {phase}")
        pad = int(10 * s)
        lh = int(30 * s)
        box_w = max(text_size(l, fs, thickness(fs))[0] for l in lines) + 2 * pad
        darken(img, 0, 0, box_w, 2 * pad + lh * len(lines), 0.3)
        for n, line in enumerate(lines):
            cv2.putText(img, line, (pad, pad + lh * (n + 1) - int(4 * s)), FONT, fs,
                        TEXT if n else ACCENT, thickness(fs), AA)

        # Cues for the current rep
        if rep is not None:
            self._cues(img, rep, s)
        self._timeline(img, i, s)
        return img

    def _cues(self, img, rep, s) -> None:
        h, w = img.shape[:2]
        fs = 0.6 * s
        pad = int(12 * s)
        bottom = h - int(26 * s)
        if not rep.faults:
            items = [("Good " + self.res.station.rep_word, "", GOOD)]
        else:
            items = [(f.title, f.cue, SEVERITY_COLOR[f.severity])
                     for f in sorted(rep.faults, key=lambda f: f.severity != "major")[:2]]
        rows: List[Tuple[str, tuple, float]] = []
        for title, cue, color in items:
            rows.append((title, color, fs * 1.1))
            rows += [(l, TEXT, fs) for l in _wrap(cue, w - 2 * pad, fs)]
        lh = int(28 * s)
        top = bottom - lh * len(rows) - pad
        darken(img, 0, top, w, bottom, 0.3)
        for n, (text, color, f) in enumerate(rows):
            cv2.putText(img, text, (pad, top + pad + lh * n + int(16 * s)), FONT, f,
                        color, thickness(f), AA)

    def _timeline(self, img, i, s) -> None:
        h, w = img.shape[:2]
        y0, y1 = h - int(20 * s), h - int(6 * s)
        t0, t1 = self.span
        body = self.res.body
        darken(img, 0, y0 - int(4 * s), w, h, 0.3)

        def x(t):
            return int((t - t0) / max(t1 - t0, 1e-6) * (w - 1))
        for rep in self.res.reps:
            c = rep.cycle
            cv2.rectangle(img, (x(body.t[c.start]) + 1, y0), (x(body.t[c.end]) - 1, y1),
                          SEVERITY_COLOR[rep.worst], -1)
        if i is not None:
            cx = x(body.t[i])
            cv2.line(img, (cx, y0 - int(4 * s)), (cx, h), TEXT, max(1, int(2 * s)))


def render_video(reader: VideoReader, res: AnalysisResult, path: str,
                 max_width: int = 1280, progress: bool = True) -> None:
    info = reader.info
    scale = min(1.0, max_width / info.width)
    size = (int(info.width * scale) // 2 * 2, int(info.height * scale) // 2 * 2)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), info.fps, size)
    if not writer.isOpened():
        print(f"Could not write '{path}'.", file=sys.stderr)
        return
    ann = Annotator(res)
    try:
        for n, (index, _, frame) in enumerate(reader.frames(), 1):
            out = ann.draw(frame, index)
            if scale != 1.0:
                out = cv2.resize(out, size, interpolation=cv2.INTER_AREA)
            writer.write(out)
            if progress and n % 50 == 0:
                print(f"\r  video: frame {index + 1}/{info.frame_count}", end="",
                      file=sys.stderr, flush=True)
    finally:
        writer.release()
        if progress:
            print(file=sys.stderr)


def snapshot(reader: VideoReader, res: AnalysisResult, frame_index: int,
             max_width: int = 720) -> Optional[np.ndarray]:
    frame = reader.read_frame(frame_index)
    if frame is None:
        return None
    img = Annotator(res).draw(frame, frame_index)
    scale = min(1.0, max_width / img.shape[1])
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale < 1 else img
