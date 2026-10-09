#!/usr/bin/env python3
"""
bench_pose.py - benchmark pose models for the SkiErg worker, on CPU or GPU.

For every video, it runs each combination of model, export format, mode (full
frame or cropped to the athlete) and frame stride, and measures:

  speed       ms per analysed frame (decode, inference, end to end), how many
              times slower than real time the pose pass runs, and the projected
              pose-pass time for a 4.5-minute piece at this video's resolution
  detection   share of analysed frames where the athlete was found
  confidence  mean keypoint confidence over the 12 body joints
  jitter      per-frame keypoint noise of the body joints, as a % of torso
              length, estimated from the residual of the worker's own zero-lag
              filter (lowpass() in pose_app/analysis/force/kinematics.py) and
              corrected for frame rate so strides compare
  filtered    the noise left after that filter, as a % of torso length: what
              the force model's accelerations are built from. Analysing fewer
              frames leaves more of it (and lowpass() caps the cutoff at
              0.3 x frame rate, 4.5 Hz at 15 fps).
  agreement   mean keypoint distance to a reference run, as a % of torso
              length, and PCK@10% (share of joints within 10% of torso length)

No ground truth is needed. The reference run is the largest model listed (or
--reference), on the full frame, every frame; agreement with it stands in for
accuracy, and jitter measures the noise the force model would see.

Setup (from the repo root):
  pip install -r requirements.txt
  pip install onnxruntime        # for --formats onnx
  pip install openvino           # for --formats openvino (best on Intel CPUs)

Defaults match the worker: 960 px input on the full frame (WORKER_IMGSZ),
the 6 Hz force filter, and yolo11m-pose.pt among the models compared.

Examples:
  # Quick look: the first 60 s of each video, YOLO11 vs YOLO26, s and m sizes
  python tools/bench_pose.py videos/*.mp4 --device cpu

  # Full grid on the target server
  python tools/bench_pose.py session1.mp4 session2.mp4 \\
      --models yolo11s-pose.pt yolo11m-pose.pt yolo26n-pose.pt yolo26s-pose.pt yolo26m-pose.pt \\
      --reference yolo26x-pose.pt --modes full crop --strides 1 2 \\
      --formats pt openvino --max-seconds 90 --out bench_results

  # GPU comparison
  python tools/bench_pose.py session1.mp4 --device 0 --formats pt engine

Outputs (in --out): results.csv, results.json (with machine details), and with
--save-keypoints one .npz of keypoints per run, for feeding into the worker.

Notes:
  - Times are for one process. For throughput on a many-core server, run one
    process per video in parallel with --threads set to cores / processes.
  - Projected times scale this video's measured rate to 270 s; they hold for
    videos of the same resolution and frame rate.
  - OpenCV applies the phone's rotation metadata when it decodes.
  - The athlete is picked as the person overlapping the previous frame's box,
    else the largest one: close to, but not, the worker's AthleteTracker.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repo root, for pose_app
from pose_app.analysis.force.kinematics import CUTOFF_HZ, lowpass  # noqa: E402

BODY = list(range(5, 17))  # COCO: shoulders, elbows, wrists, hips, knees, ankles
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12
SIZE_RANK = {"n": 0, "s": 1, "m": 2, "l": 3, "x": 4}
TARGET_SECONDS = 270.0  # a 4.5-minute SkiErg piece
WARMUP = 3


# --------------------------------------------------------------------------- args

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Benchmark pose models (speed, jitter, agreement) on real videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("videos", nargs="+", help="video files to analyse")
    p.add_argument("--models", nargs="+",
                   default=["yolo11s-pose.pt", "yolo11m-pose.pt", "yolo26s-pose.pt", "yolo26m-pose.pt"],
                   help="pose model weights (downloaded on first use)")
    p.add_argument("--reference", default=None,
                   help="model for the reference run (default: the largest of --models)")
    p.add_argument("--modes", nargs="+", default=["full", "crop"], choices=["full", "crop"])
    p.add_argument("--strides", nargs="+", type=int, default=[1, 2],
                   help="analyse every Nth frame (1 = every frame)")
    p.add_argument("--formats", nargs="+", default=["pt"], choices=["pt", "onnx", "openvino", "engine"],
                   help="pt = PyTorch; onnx, openvino and engine (TensorRT) are exported once and cached")
    p.add_argument("--device", default="cpu", help="cpu, or a GPU index such as 0")
    p.add_argument("--imgsz", type=int, default=960,
                   help="model input size on the full frame (the worker's default is 960)")
    p.add_argument("--crop-imgsz", type=int, default=640, help="model input size on the crop")
    p.add_argument("--crop-margin", type=float, default=0.25,
                   help="margin added on each side of the athlete box, as a share of its longer side")
    p.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    p.add_argument("--kpt-conf", type=float, default=0.3,
                   help="keypoints below this confidence count as missing")
    p.add_argument("--cutoff-hz", type=float, default=CUTOFF_HZ,
                   help="low-pass cutoff for the jitter metric (the force model's, by default)")
    p.add_argument("--start-seconds", type=float, default=0.0, help="skip this much of each video")
    p.add_argument("--max-seconds", type=float, default=60.0,
                   help="analyse at most this much of each video (0 = all of it)")
    p.add_argument("--threads", type=int, default=None, help="CPU threads for inference")
    p.add_argument("--out", default="bench_results", help="output folder")
    p.add_argument("--export-dir", default="bench_exports", help="where exported models are cached")
    p.add_argument("--save-keypoints", action="store_true", help="save each run's keypoints as .npz")
    return p.parse_args(argv)


# ------------------------------------------------------------------ environment

def machine_info():
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor() or "",
        "logical_cpus": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    for mod in ("torch", "ultralytics", "cv2", "onnxruntime", "openvino"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "installed")
        except Exception:
            info[mod] = None
    try:
        import torch
        info["torch_threads"] = torch.get_num_threads()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


# ----------------------------------------------------------------------- models

def model_rank(name):
    m = re.search(r"(\d+)([nsmlx])-pose", Path(name).stem)
    if not m:
        return (-1, -1)
    return (SIZE_RANK[m.group(2)], int(m.group(1)))


class ModelCache:
    """Loads each (weights, format, input size) once; exports non-PyTorch formats on first use."""

    def __init__(self, export_dir, device):
        self.export_dir = Path(export_dir)
        self.device = device
        self.models = {}

    def get(self, weights, fmt, imgsz):
        key = (weights, fmt, imgsz if fmt != "pt" else 0)
        if key not in self.models:
            self.models[key] = self._load(weights, fmt, imgsz)
        return self.models[key]

    def _load(self, weights, fmt, imgsz):
        from ultralytics import YOLO
        if fmt == "pt":
            return YOLO(weights)
        stem = Path(weights).stem
        self.export_dir.mkdir(parents=True, exist_ok=True)
        dest = {
            "onnx": self.export_dir / f"{stem}-{imgsz}.onnx",
            "openvino": self.export_dir / f"{stem}-{imgsz}_openvino_model",
            "engine": self.export_dir / f"{stem}-{imgsz}.engine",
        }[fmt]
        if not dest.exists():
            print(f"  exporting {weights} to {fmt} at {imgsz} px ...", flush=True)
            exported = Path(YOLO(weights).export(
                format=fmt, imgsz=imgsz, half=(fmt == "engine"), device=self.device, verbose=False))
            shutil.move(str(exported), str(dest))
        return YOLO(str(dest), task="pose")


# ------------------------------------------------------------- detection helpers

def extract(result, dx=0, dy=0):
    """Boxes, box confidences, keypoints (n,17,2) and keypoint confidences (n,17), shifted by dx, dy."""
    if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        return None
    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float64)
    bconf = result.boxes.conf.cpu().numpy().astype(np.float64)
    kxy = result.keypoints.xy.cpu().numpy().astype(np.float64)
    kc = result.keypoints.conf
    kc = kc.cpu().numpy().astype(np.float64) if kc is not None else np.ones(kxy.shape[:2])
    if dx or dy:
        boxes[:, [0, 2]] += dx
        boxes[:, [1, 3]] += dy
        kxy[..., 0] += dx
        kxy[..., 1] += dy
    return boxes, bconf, kxy, kc


def iou(boxes, box):
    x0 = np.maximum(boxes[:, 0], box[0]); y0 = np.maximum(boxes[:, 1], box[1])
    x1 = np.minimum(boxes[:, 2], box[2]); y1 = np.minimum(boxes[:, 3], box[3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    b = (box[2] - box[0]) * (box[3] - box[1])
    return inter / np.maximum(a + b - inter, 1e-9)


def pick_athlete(det, prev_box):
    """The detection that continues the previous box, else the largest confident person."""
    boxes, bconf = det[0], det[1]
    if prev_box is not None:
        ov = iou(boxes, prev_box)
        i = int(np.argmax(ov))
        if ov[i] > 0.3:
            return i
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.argmax(areas * bconf))


def make_crop(box, W, H, margin):
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    w, h = min(side, W), min(side, H)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = min(max(cx - w / 2, 0), W - w)
    top_ = min(max(cy - h / 2, 0), H - h)
    return (int(left), int(top_), int(math.ceil(left + w)), int(math.ceil(top_ + h)))


def crop_needs_update(box, crop, W, H):
    """Re-centre when the athlete nears an inner edge of the crop, or the crop is far too loose."""
    cx0, cy0, cx1, cy1 = crop
    edge = 0.08 * max(cx1 - cx0, cy1 - cy0)
    x0, y0, x1, y1 = box
    near = ((x0 - cx0 < edge and cx0 > 0) or (cx1 - x1 < edge and cx1 < W) or
            (y0 - cy0 < edge and cy0 > 0) or (cy1 - y1 < edge and cy1 < H))
    # Longer sides, not areas: a tall, narrow athlete in a square crop is not "loose".
    loose = max(x1 - x0, y1 - y0) < 0.5 * max(cx1 - cx0, cy1 - cy0)
    return near or loose


# ------------------------------------------------------------------------- runs

def predict(model, img, imgsz, args):
    return model.predict(img, imgsz=imgsz, device=args.device, conf=args.conf, verbose=False)[0]


def run_config(video, cfg, cache, args):
    import cv2
    fmt, weights, mode, stride = cfg["format"], cfg["model"], cfg["mode"], cfg["stride"]
    m_full = cache.get(weights, fmt, args.imgsz)
    m_crop = cache.get(weights, fmt, args.crop_imgsz) if mode == "crop" else None

    # Warm up on the first frame, outside the timed loop.
    cap = cv2.VideoCapture(video)
    if args.start_seconds:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_seconds * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("cannot read the video")
    H, W = frame.shape[:2]
    for _ in range(WARMUP):
        predict(m_full, frame, args.imgsz, args)
        if m_crop is not None:
            predict(m_crop, np.ascontiguousarray(frame[: H // 2, : W // 2]), args.crop_imgsz, args)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if args.start_seconds:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_seconds * 1000)
    max_src = int(args.max_seconds * fps) if args.max_seconds else None

    records = {}  # source frame index -> (t, kxy (17,2), kc (17,))
    kept_idx, kept_t = [], []
    t_decode = t_infer = 0.0
    crop, prev_box, misses, fallbacks = None, None, 0, 0
    src = 0
    wall0 = time.perf_counter()
    while max_src is None or src < max_src:
        td = time.perf_counter()
        if src % stride == 0:
            ok, frame = cap.read()
        else:
            ok, frame = cap.grab(), None
        t_decode += time.perf_counter() - td
        if not ok:
            break
        if frame is None:
            src += 1
            continue
        t = src / fps
        kept_idx.append(src); kept_t.append(t)

        ti = time.perf_counter()
        det = None
        if mode == "crop" and crop is not None:
            x0, y0, x1, y1 = crop
            det = extract(predict(m_crop, np.ascontiguousarray(frame[y0:y1, x0:x1]), args.crop_imgsz, args), x0, y0)
            if det is None:
                fallbacks += 1
        if det is None:
            det = extract(predict(m_full, frame, args.imgsz, args))
        t_infer += time.perf_counter() - ti

        if det is None:
            misses += 1
            crop = None
            if misses >= 5:
                prev_box = None
        else:
            misses = 0
            i = pick_athlete(det, prev_box)
            box = det[0][i]
            records[src] = (t, det[2][i], det[3][i])
            prev_box = box
            if mode == "crop" and (crop is None or crop_needs_update(box, crop, W, H)):
                crop = make_crop(box, W, H, args.crop_margin)
        src += 1
    wall = time.perf_counter() - wall0
    cap.release()

    n = len(kept_idx)
    if n == 0:
        raise RuntimeError("no frames analysed")
    video_seconds = src / fps
    rt = wall / video_seconds
    out = {
        "frames_analysed": n,
        "video_seconds": round(video_seconds, 2),
        "resolution": f"{W}x{H}",
        "fps": round(fps, 2),
        "detect_pct": round(100 * len(records) / n, 1),
        "crop_fallbacks": fallbacks if mode == "crop" else None,
        "decode_ms_per_frame": round(1000 * t_decode / n, 1),
        "infer_ms_per_frame": round(1000 * t_infer / n, 1),
        "total_ms_per_frame": round(1000 * wall / n, 1),
        "x_realtime": round(rt, 2),
        "projected_min_4m30": round(rt * TARGET_SECONDS / 60, 1),
    }
    out.update(quality(records, kept_idx, fps / stride, args))
    return out, records, (np.array(kept_idx), np.array(kept_t))


# ---------------------------------------------------------------------- quality

def _lowpass_coeffs(fs, fc):
    """The single-pass coefficients pose_app.analysis.force.kinematics.lowpass uses."""
    wc = np.tan(np.pi * fc / (0.802 * fs))
    k1, k2 = np.sqrt(2.0) * wc, wc * wc
    a0 = 1.0 + k1 + k2
    return (np.array([k2, 2.0 * k2, k2]) / a0,
            np.array([1.0, 2.0 * (k2 - 1.0) / a0, (1.0 - k1 + k2) / a0]))


def torso_length(kxy, kc, thr):
    lens = []
    for a, b in ((L_SH, L_HIP), (R_SH, R_HIP)):
        if kc[a] >= thr and kc[b] >= thr:
            lens.append(float(np.linalg.norm(kxy[a] - kxy[b])))
    return float(np.mean(lens)) if lens else float("nan")


def quality(records, kept_idx, fs, args):
    if not records:
        return {"kpt_conf": None, "torso_px": None, "jitter_pct": None, "jitter_note": "no detections"}
    thr = args.kpt_conf
    conf = np.mean([r[2][BODY].mean() for r in records.values()])
    torso = np.nanmedian([torso_length(r[1], r[2], thr) for r in records.values()])

    # Positions on the analysed-frame grid, NaN where missing or low confidence.
    xy = np.full((len(kept_idx), 17, 2), np.nan)
    for row, idx in enumerate(kept_idx):
        rec = records.get(idx)
        if rec is not None:
            ok = rec[2] >= thr
            xy[row, ok] = rec[1][ok]

    # The worker's own zero-lag filter, so "filtered" is what the force model really sees.
    fc = min(args.cutoff_hz, 0.3 * fs)                  # lowpass() caps the cutoff the same way
    note = f"filter at {fc:.1f} Hz ({fs:.1f} fps)" if fc < args.cutoff_hz else ""
    # Two passes apply |H|^2. For white noise, the share of its power left in the residual and
    # in the filtered signal; dividing by them makes the noise estimate independent of frame rate.
    b, a = _lowpass_coeffs(fs, fc)
    z = np.exp(-1j * np.linspace(0, np.pi, 4096))
    h2 = np.abs((b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)) ** 2
    resid_share = float(np.mean((1 - h2) ** 2))
    kept_share = float(np.mean(h2 ** 2))
    sq = []
    for j in BODY:
        valid = ~np.isnan(xy[:, j, 0])
        edges = np.flatnonzero(np.diff(np.concatenate(([0], valid.astype(int), [0]))))
        for s, e in zip(edges[::2], edges[1::2]):
            if e - s < 15:
                continue
            seg = xy[s:e, j]
            smooth = lowpass(seg, fs, fc)
            sq.append(np.sum((seg - smooth) ** 2, axis=1))
    if not sq or not np.isfinite(torso) or torso <= 0:
        return {"kpt_conf": round(float(conf), 3), "torso_px": None, "jitter_pct": None,
                "filtered_noise_pct": None, "jitter_note": note or "too few continuous frames"}
    resid_px = float(np.sqrt(np.mean(np.concatenate(sq))))
    noise_px = resid_px / math.sqrt(resid_share)       # per-frame keypoint noise
    left_px = noise_px * math.sqrt(kept_share)          # noise still there after the low-pass
    return {"kpt_conf": round(float(conf), 3), "torso_px": round(float(torso), 1),
            "jitter_pct": round(100 * noise_px / torso, 2),
            "filtered_noise_pct": round(100 * left_px / torso, 2), "jitter_note": note}


def agreement(records, ref, thr):
    d = []
    for idx, (_, kxy, kc) in records.items():
        r = ref.get(idx)
        if r is None:
            continue
        torso = torso_length(r[1], r[2], thr)
        if not np.isfinite(torso) or torso <= 0:
            continue
        ok = (kc[BODY] >= thr) & (r[2][BODY] >= thr)
        if ok.any():
            d.extend(np.linalg.norm(kxy[BODY][ok] - r[1][BODY][ok], axis=1) / torso)
    if not d:
        return {"err_pct_vs_ref": None, "pck10_vs_ref": None, "joints_compared": 0}
    d = np.array(d)
    return {"err_pct_vs_ref": round(100 * float(d.mean()), 2),
            "pck10_vs_ref": round(100 * float((d < 0.10).mean()), 1),
            "joints_compared": int(d.size)}


# ------------------------------------------------------------------------- main

COLUMNS = ["video", "model", "format", "mode", "stride", "is_reference", "resolution", "fps",
           "frames_analysed", "video_seconds", "detect_pct", "crop_fallbacks",
           "decode_ms_per_frame", "infer_ms_per_frame", "total_ms_per_frame", "x_realtime",
           "projected_min_4m30", "kpt_conf", "torso_px", "jitter_pct", "filtered_noise_pct", "jitter_note",
           "err_pct_vs_ref", "pck10_vs_ref", "joints_compared", "error"]


def print_table(rows):
    head = ["video", "model", "fmt", "mode", "str", "det%", "inf ms", "tot ms", "xRT",
            "4m30 min", "conf", "jit%", "filt%", "err%", "PCK10"]
    keys = ["video", "model", "format", "mode", "stride", "detect_pct", "infer_ms_per_frame",
            "total_ms_per_frame", "x_realtime", "projected_min_4m30", "kpt_conf", "jitter_pct",
            "filtered_noise_pct", "err_pct_vs_ref", "pck10_vs_ref"]
    table = [head]
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k)
            if k == "model" and r.get("is_reference"):
                v = f"{v} (ref)"
            if k == "video":
                v = Path(v).name[:18]
            cells.append("ERROR" if r.get("error") and k == "detect_pct" else ("" if v is None else str(v)))
        table.append(cells)
    widths = [max(len(row[i]) for row in table) for i in range(len(head))]
    for i, row in enumerate(table):
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))
        if i == 0:
            print("  ".join("-" * w for w in widths))


def main(argv=None):
    args = parse_args(argv)
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
    try:
        import cv2  # noqa: F401
        from ultralytics import YOLO  # noqa: F401
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}. Install with: pip install -r requirements.txt")
    if args.threads:
        try:
            import torch
            torch.set_num_threads(args.threads)
        except ImportError:
            pass

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_model = args.reference or max(args.models, key=model_rank)
    ref_fmt = "pt" if "pt" in args.formats else args.formats[0]
    ref_cfg = {"model": ref_model, "format": ref_fmt, "mode": "full", "stride": 1}
    grid = [{"model": m, "format": f, "mode": mo, "stride": s}
            for f in args.formats for m in args.models for mo in args.modes for s in args.strides]
    grid = [c for c in grid if c != ref_cfg]

    info = machine_info()
    print(f"Machine: {info.get('cpu')} ({info.get('logical_cpus')} logical CPUs)"
          + (f", GPU {info['gpu']}" if info.get("gpu") else "") + f"; device {args.device}")
    print(f"Reference: {ref_model}, {ref_fmt}, full frame, every frame\n")

    cache = ModelCache(args.export_dir, args.device)
    rows = []
    for video in args.videos:
        ref_records = None
        for cfg in [ref_cfg] + grid:
            is_ref = cfg is ref_cfg
            label = f"{Path(video).name}: {cfg['model']} {cfg['format']} {cfg['mode']} stride {cfg['stride']}"
            print(f"> {label}{' (reference)' if is_ref else ''}", flush=True)
            row = {"video": video, **cfg, "is_reference": is_ref, "error": None}
            try:
                stats, records, (idx, ts) = run_config(video, cfg, cache, args)
                row.update(stats)
                if is_ref:
                    ref_records = records
                    row.update({"err_pct_vs_ref": 0.0, "pck10_vs_ref": 100.0,
                                "joints_compared": sum(int((r[2][BODY] >= args.kpt_conf).sum())
                                                       for r in records.values())})
                elif ref_records is not None:
                    row.update(agreement(records, ref_records, args.kpt_conf))
                if args.save_keypoints:
                    keys = sorted(records)
                    np.savez_compressed(
                        out_dir / f"{Path(video).stem}__{Path(cfg['model']).stem}_{cfg['format']}_{cfg['mode']}_s{cfg['stride']}.npz",
                        analysed_frames=idx, analysed_t=ts, frame=np.array(keys),
                        t=np.array([records[k][0] for k in keys]),
                        kxy=np.array([records[k][1] for k in keys]).reshape(-1, 17, 2),
                        kconf=np.array([records[k][2] for k in keys]).reshape(-1, 17))
                print(f"  {row['total_ms_per_frame']} ms/frame, {row['x_realtime']}x real time, "
                      f"~{row['projected_min_4m30']} min per 4.5-min piece, jitter {row.get('jitter_pct')}%, "
                      f"filtered {row.get('filtered_noise_pct')}%, PCK10 {row.get('pck10_vs_ref')}%", flush=True)
            except Exception as e:  # keep going with the other runs
                row["error"] = f"{type(e).__name__}: {e}"
                print(f"  failed: {row['error']}", flush=True)
                if os.environ.get("BENCH_DEBUG"):
                    traceback.print_exc()
            rows.append(row)

    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "results.json", "w") as f:
        json.dump({"machine": info, "args": vars(args), "runs": rows}, f, indent=2, default=str)

    print()
    print_table(rows)
    print(f"\nSaved {out_dir / 'results.csv'} and {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
