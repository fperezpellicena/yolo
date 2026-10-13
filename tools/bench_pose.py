#!/usr/bin/env python3
"""
bench_pose.py - benchmark pose models for the SkiErg worker, on CPU or GPU.

It compares Ultralytics YOLO pose models (yolo11m-pose.pt, ...) with RTMPose
through rtmlib (rtmpose-s, rtmpose-m, rtmpose-x: the worker's model names).
For every video, it runs each combination of model, export format (rtmlib
backend for RTMPose), mode and frame stride, and measures:

  speed       ms per analysed frame (decode, inference, end to end), how many
              times slower than real time the pose pass runs, and the projected
              pose-pass time for a 4.5-minute piece at this video's resolution.
              For RTMPose, inference is split between the person detector and
              the pose model, and detector % is how many frames ran the detector.
  detection   share of analysed frames where the athlete was found
  confidence  mean keypoint confidence over the 12 body joints (YOLO and RTMPose
              scores are on different scales: compare within a family)
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

Modes:
  full   as the worker runs today. YOLO: one pass over the whole frame.
         RTMPose: the person detector (YOLOX) over the whole frame, then the
         pose model on every person it finds.
  crop   follow the athlete. YOLO: the model on a crop around the athlete's
         last box, at --crop-imgsz, and on the full frame when the crop finds
         no one. RTMPose: the pose model alone, on a box around the athlete's
         keypoints in the previous analysed frame (as rtmlib's PoseTracker
         does); the detector runs again every --rtm-redetect analysed frames
         and whenever the athlete is lost. The detector is most of RTMPose's
         cost, so this is the mode that would make it affordable.

No ground truth is needed. Each family has its own reference run: the largest
YOLO model listed and the largest RTMPose model listed, on the full frame,
every frame. Agreement with it stands in for accuracy, and jitter measures the
noise the force model would see. --reference sets one reference for all runs;
across families, agreement then also counts the models' different keypoint
definitions (the hips above all), not only error.

Setup (from the repo root):
  pip install -r requirements.txt
  pip install -r requirements-rtm.txt  # RTMPose: rtmlib and onnxruntime
  pip install onnxruntime        # for --formats onnx
  pip install openvino           # for --formats openvino and --rtm-backends openvino
                                 # (best on Intel CPUs)

Defaults match the worker: 960 px YOLO input on the full frame (WORKER_IMGSZ),
the 6 Hz force filter, yolo11m-pose.pt and rtmpose-m among the models compared.
RTMPose models have fixed input sizes; rtmlib downloads them on first use.
Without rtmlib installed, the default run leaves RTMPose out (with a note).

Examples:
  # Quick look: the first 60 s of each video, YOLO11 vs YOLO26 (s and m), RTMPose-m
  python tools/bench_pose.py videos/*.mp4 --device cpu

  # YOLO vs RTMPose on CPU; crop runs RTMPose's detector on 1 frame in 10
  python tools/bench_pose.py videos/*.mp4 --device cpu \\
      --models yolo11m-pose.pt yolo26m-pose.pt rtmpose-s rtmpose-m --modes full crop

  # RTMPose on each rtmlib backend
  python tools/bench_pose.py session1.mp4 --models rtmpose-m \\
      --rtm-backends onnxruntime openvino opencv

  # Full grid on the target server
  python tools/bench_pose.py session1.mp4 session2.mp4 \\
      --models yolo11s-pose.pt yolo11m-pose.pt yolo26n-pose.pt yolo26s-pose.pt yolo26m-pose.pt \\
      --reference yolo26x-pose.pt --modes full crop --strides 1 2 \\
      --formats pt openvino --max-seconds 90 --out bench_results

  # GPU comparison
  python tools/bench_pose.py session1.mp4 --device 0 --formats pt engine

Outputs (in --out): results.csv, results.json (with machine details), and with
--save-keypoints one .npz of keypoints per run (17 COCO points for YOLO, the 26
Halpe points for RTMPose), for feeding into the worker.

Notes:
  - Times are for one process. For throughput on a many-core server, run one
    process per video in parallel with --threads set to cores / processes.
  - Projected times scale this video's measured rate to 270 s; they hold for
    videos of the same resolution and frame rate.
  - OpenCV applies the phone's rotation metadata when it decodes.
  - The athlete is picked as the person overlapping the previous frame's box,
    else the largest one: close to, but not, the worker's AthleteTracker.
  - --threads also sets ONNX Runtime's and OpenVINO's threads for RTMPose on CPU.
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
from pose_app.rtm_estimator import RTM_MODELS, is_rtm_model, rtm_device  # noqa: E402

BODY = list(range(5, 17))  # COCO: shoulders, elbows, wrists, hips, knees, ankles
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12
SIZE_RANK = {"n": 0, "s": 1, "m": 2, "l": 3, "x": 4}
TARGET_SECONDS = 270.0  # a 4.5-minute SkiErg piece
WARMUP = 3
DEFAULT_MODELS = ["yolo11s-pose.pt", "yolo11m-pose.pt", "yolo26s-pose.pt", "yolo26m-pose.pt", "rtmpose-m"]
TRACK_EXPAND = 1.25  # RTMPose crop mode: box around the last keypoints, grown as rtmlib's PoseTracker does


# --------------------------------------------------------------------------- args

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Benchmark pose models (speed, jitter, agreement) on real videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("videos", nargs="+", help="video files to analyse")
    p.add_argument("--models", nargs="+", default=None,
                   help="YOLO pose weights and/or RTMPose models (rtmpose-s, rtmpose-m, rtmpose-x), "
                        "downloaded on first use (default: " + " ".join(DEFAULT_MODELS) + ")")
    p.add_argument("--reference", default=None,
                   help="model for one reference run for all (default: the largest of each family)")
    p.add_argument("--modes", nargs="+", default=["full", "crop"], choices=["full", "crop"])
    p.add_argument("--strides", nargs="+", type=int, default=[1, 2],
                   help="analyse every Nth frame (1 = every frame)")
    p.add_argument("--formats", nargs="+", default=["pt"], choices=["pt", "onnx", "openvino", "engine"],
                   help="YOLO: pt = PyTorch; onnx, openvino and engine (TensorRT) are exported once "
                        "and cached")
    p.add_argument("--rtm-backends", nargs="+", default=["onnxruntime"],
                   choices=["onnxruntime", "openvino", "opencv"],
                   help="RTMPose: rtmlib backends to run (the worker uses onnxruntime)")
    p.add_argument("--rtm-redetect", type=int, default=10,
                   help="RTMPose crop mode: run the person detector every Nth analysed frame "
                        "(0 = only when the athlete is lost)")
    p.add_argument("--device", default="cpu", help="cpu, or a GPU index such as 0")
    p.add_argument("--imgsz", type=int, default=960,
                   help="YOLO input size on the full frame (the worker's default is 960); "
                        "RTMPose models have fixed sizes")
    p.add_argument("--crop-imgsz", type=int, default=640, help="YOLO input size on the crop")
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
    args = p.parse_args(argv)
    args.default_models = args.models is None
    if args.models is None:
        args.models = list(DEFAULT_MODELS)
    return args


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
    for mod in ("torch", "ultralytics", "cv2", "rtmlib", "onnxruntime", "openvino"):
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

def family(name):
    return "rtm" if is_rtm_model(name) else "yolo"


def model_rank(name):
    if is_rtm_model(name):
        size = name.strip().lower().partition("-")[2] or "m"   # plain "rtmpose" is rtmpose-m
        return (SIZE_RANK[size], 0)
    m = re.search(r"(\d+)([nsmlx])-pose", Path(name).stem)
    if not m:
        return (-1, -1)
    return (SIZE_RANK[m.group(2)], int(m.group(1)))


def rtm_backend_device(backend, device):
    """Our --device as each rtmlib backend names it."""
    dev = rtm_device(device)                     # cpu, cuda, cuda:N or mps (onnxruntime's names)
    if backend == "opencv":
        return "cuda" if dev.startswith("cuda") else "cpu"
    if backend == "openvino":
        return "gpu" if dev.startswith("cuda") else ("cpu" if dev == "mps" else dev)
    return dev


def set_rtm_threads(model, threads, device):
    """Rebuild an rtmlib model's CPU session with `threads` threads (OpenCV: cv2.setNumThreads)."""
    if model.backend == "onnxruntime":
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        model.session = ort.InferenceSession(model.onnx_model, sess_options=so,
                                             providers=model.session.get_providers())
    elif model.backend == "openvino":
        from openvino import Core
        core = Core()
        net = core.read_model(model=model.onnx_model)
        compiled = core.compile_model(model=net, device_name=device.upper(), config={
            "PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": threads})
        model.compiled_model = compiled
        model.input_layer = compiled.input(0)
        if hasattr(model, "_ov_outputs"):
            model._ov_outputs = [compiled.output(i) for i in range(len(net.outputs))]
        if hasattr(model, "output_layer"):
            model.output_layer = compiled.output(0)


class ModelCache:
    """Loads each (weights, format, input size) once; exports non-PyTorch formats on first use.
    RTMPose models load once per (model, backend), as (detector, pose model)."""

    def __init__(self, export_dir, device, conf=0.25, threads=None):
        self.export_dir = Path(export_dir)
        self.device = device
        self.conf = conf
        self.threads = threads
        self.models = {}

    def get(self, weights, fmt, imgsz):
        key = (weights, fmt, imgsz if fmt != "pt" else 0)
        if key not in self.models:
            self.models[key] = self._load(weights, fmt, imgsz)
        return self.models[key]

    def get_rtm(self, name, backend):
        key = ("rtm", name.strip().lower(), backend)
        if key not in self.models:
            from rtmlib import BodyWithFeet
            dev = rtm_backend_device(backend, self.device)
            solution = BodyWithFeet(mode=RTM_MODELS[key[1]], backend=backend, device=dev)
            det, pose = solution.det_model, solution.pose_model
            det.score_thr = self.conf                    # as the worker sets it
            if self.threads and dev == "cpu":
                for m in (det, pose):
                    try:
                        set_rtm_threads(m, self.threads, dev)
                    except Exception as e:               # keep the backend's own default
                        print(f"  could not set {backend} threads: {e}", flush=True)
            self.models[key] = (det, pose)
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


def pose_box(kxy, kc, thr, W, H, expand=TRACK_EXPAND):
    """The box around the confident keypoints, grown by `expand` about its centre and kept in
    the frame: where RTMPose looks for the athlete in the next frame. None if too few points."""
    ok = kc >= thr
    if ok.sum() < 4:
        return None
    (x0, y0), (x1, y1) = kxy[ok].min(0), kxy[ok].max(0)
    cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2 * expand, (y1 - y0) / 2 * expand
    box = np.array([max(cx - hw, 0.0), max(cy - hh, 0.0), min(cx + hw, W), min(cy + hh, H)])
    return box if box[2] - box[0] >= 8 and box[3] - box[1] >= 8 else None


def athlete_lost(kc, thr):
    """Fewer than half the body joints found: the tracked box no longer holds the athlete."""
    return int((kc[BODY] >= thr).sum()) < len(BODY) // 2


# ------------------------------------------------------------------------- runs

def predict(model, img, imgsz, args):
    return model.predict(img, imgsz=imgsz, device=args.device, conf=args.conf, verbose=False)[0]


class YoloRunner:
    """One Ultralytics model, full frame or cropped to the athlete. Call it on each analysed
    frame: (keypoints, confidences) of the athlete, or None."""

    def __init__(self, cfg, cache, args, W, H):
        self.args, self.mode, self.W, self.H = args, cfg["mode"], W, H
        self.m_full = cache.get(cfg["model"], cfg["format"], args.imgsz)
        self.m_crop = cache.get(cfg["model"], cfg["format"], args.crop_imgsz) if self.mode == "crop" else None
        self.crop, self.prev_box, self.misses = None, None, 0
        self.full_passes = self.fallbacks = 0
        self.t_det = self.t_pose = None              # one model: no split

    def warmup(self, frame):
        for _ in range(WARMUP):
            predict(self.m_full, frame, self.args.imgsz, self.args)
            if self.m_crop is not None:
                predict(self.m_crop, np.ascontiguousarray(frame[: self.H // 2, : self.W // 2]),
                        self.args.crop_imgsz, self.args)

    def __call__(self, frame):
        args, det = self.args, None
        if self.mode == "crop" and self.crop is not None:
            x0, y0, x1, y1 = self.crop
            det = extract(predict(self.m_crop, np.ascontiguousarray(frame[y0:y1, x0:x1]),
                                  args.crop_imgsz, args), x0, y0)
            if det is None:
                self.fallbacks += 1
        if det is None:
            self.full_passes += 1
            det = extract(predict(self.m_full, frame, args.imgsz, args))
        if det is None:
            self.misses += 1
            self.crop = None
            if self.misses >= 5:
                self.prev_box = None
            return None
        self.misses = 0
        i = pick_athlete(det, self.prev_box)
        box = det[0][i]
        self.prev_box = box
        if self.mode == "crop" and (self.crop is None or crop_needs_update(box, self.crop, self.W, self.H)):
            self.crop = make_crop(box, self.W, self.H, args.crop_margin)
        return det[2][i], det[3][i]


class RtmRunner:
    """RTMPose through rtmlib: a person detector, then the pose model on a box.

    full: the detector on every analysed frame, then the pose model on every person found,
          as the worker's RtmPoseEstimator does.
    crop: the pose model alone on the box around the athlete's previous keypoints; the
          detector again every --rtm-redetect analysed frames, and whenever the box loses
          the athlete (a fallback, on the same frame).
    """

    def __init__(self, cfg, cache, args, W, H):
        self.args, self.mode, self.W, self.H = args, cfg["mode"], W, H
        self.det, self.pose = cache.get_rtm(cfg["model"], cfg["format"])
        self.track, self.since_det = None, 0
        self.prev_box, self.misses = None, 0
        self.full_passes = self.fallbacks = 0
        self.t_det = self.t_pose = 0.0

    def warmup(self, frame):
        box = np.array([[0, 0, self.W / 2, self.H]], float)
        for _ in range(WARMUP):
            self.det(frame)
            self.pose(frame, bboxes=box)

    def _detect(self, frame):
        t = time.perf_counter()
        boxes = np.asarray(self.det(frame), dtype=np.float64).reshape(-1, 4)
        self.t_det += time.perf_counter() - t
        self.full_passes += 1
        return boxes

    def _pose(self, frame, boxes):
        t = time.perf_counter()
        kxy, kc = self.pose(frame, bboxes=boxes)
        self.t_pose += time.perf_counter() - t
        n = len(boxes)
        return (np.asarray(kxy, dtype=np.float64).reshape(n, -1, 2),
                np.asarray(kc, dtype=np.float64).reshape(n, -1))

    def _from_detector(self, frame):
        boxes = self._detect(frame)
        if len(boxes) == 0:
            return None
        i = pick_athlete((boxes, np.ones(len(boxes))), self.prev_box)
        if self.mode == "full":
            kxy, kc = self._pose(frame, boxes)
            return boxes[i], kxy[i], kc[i]
        kxy, kc = self._pose(frame, boxes[i:i + 1])
        return boxes[i], kxy[0], kc[0]

    def __call__(self, frame):
        thr, res = self.args.kpt_conf, None
        redetect = self.args.rtm_redetect
        if self.mode == "crop" and self.track is not None and (redetect <= 0 or self.since_det < redetect):
            kxy, kc = self._pose(frame, self.track[None])
            if athlete_lost(kc[0], thr):
                self.fallbacks += 1
            else:
                res = (self.track, kxy[0], kc[0])
                self.since_det += 1
        if res is None:
            res = self._from_detector(frame)
            self.since_det = 1
        if res is None:
            self.misses += 1
            self.track = None
            if self.misses >= 5:
                self.prev_box = None
            return None
        self.misses = 0
        box, kxy, kc = res
        self.prev_box = box
        if self.mode == "crop":
            self.track = pose_box(kxy, kc, thr, self.W, self.H)
        return kxy, kc


def run_config(video, cfg, cache, args):
    import cv2
    stride = cfg["stride"]

    # Warm up on the first frame, outside the timed loop.
    cap = cv2.VideoCapture(video)
    if args.start_seconds:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_seconds * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("cannot read the video")
    H, W = frame.shape[:2]
    runner = (RtmRunner if family(cfg["model"]) == "rtm" else YoloRunner)(cfg, cache, args, W, H)
    runner.warmup(frame)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if args.start_seconds:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_seconds * 1000)
    max_src = int(args.max_seconds * fps) if args.max_seconds else None

    records = {}  # source frame index -> (t, kxy (K,2), kc (K,)); K = 17 (YOLO) or 26 (RTMPose)
    kept_idx, kept_t = [], []
    t_decode = t_infer = 0.0
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
        found = runner(frame)
        t_infer += time.perf_counter() - ti
        if found is not None:
            records[src] = (t, found[0], found[1])
        src += 1
    wall = time.perf_counter() - wall0
    cap.release()

    n = len(kept_idx)
    if n == 0:
        raise RuntimeError("no frames analysed")
    video_seconds = src / fps
    rt = wall / video_seconds
    ms = lambda s: None if s is None else round(1000 * s / n, 1)  # noqa: E731
    out = {
        "frames_analysed": n,
        "video_seconds": round(video_seconds, 2),
        "resolution": f"{W}x{H}",
        "fps": round(fps, 2),
        "keypoints": len(next(iter(records.values()))[1]) if records else None,
        "detect_pct": round(100 * len(records) / n, 1),
        "crop_fallbacks": runner.fallbacks if cfg["mode"] == "crop" else None,
        "detector_pct": round(100 * runner.full_passes / n, 1),
        "decode_ms_per_frame": ms(t_decode),
        "infer_ms_per_frame": ms(t_infer),
        "det_ms_per_frame": ms(runner.t_det),
        "pose_ms_per_frame": ms(runner.t_pose),
        "total_ms_per_frame": ms(wall),
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

    # Positions on the analysed-frame grid, NaN where missing or low confidence. The COCO 17
    # come first in RTMPose's Halpe 26 too, so both families are measured on the same joints.
    xy = np.full((len(kept_idx), 17, 2), np.nan)
    for row, idx in enumerate(kept_idx):
        rec = records.get(idx)
        if rec is not None:
            ok = rec[2][:17] >= thr
            xy[row, ok] = rec[1][:17][ok]

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

COLUMNS = ["video", "model", "format", "mode", "stride", "is_reference", "reference", "resolution", "fps",
           "frames_analysed", "video_seconds", "keypoints", "detect_pct", "crop_fallbacks", "detector_pct",
           "decode_ms_per_frame", "infer_ms_per_frame", "det_ms_per_frame", "pose_ms_per_frame",
           "total_ms_per_frame", "x_realtime",
           "projected_min_4m30", "kpt_conf", "torso_px", "jitter_pct", "filtered_noise_pct", "jitter_note",
           "err_pct_vs_ref", "pck10_vs_ref", "joints_compared", "error"]


def print_table(rows):
    head = ["video", "model", "fmt", "mode", "str", "found%", "det run%", "inf ms", "det ms", "pose ms",
            "tot ms", "xRT", "4m30 min", "conf", "jit%", "filt%", "err%", "PCK10"]
    keys = ["video", "model", "format", "mode", "stride", "detect_pct", "detector_pct", "infer_ms_per_frame",
            "det_ms_per_frame", "pose_ms_per_frame", "total_ms_per_frame", "x_realtime",
            "projected_min_4m30", "kpt_conf", "jitter_pct", "filtered_noise_pct", "err_pct_vs_ref",
            "pck10_vs_ref"]
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


def check_dependencies(args):
    """Exit with an install hint when a listed model's library is missing. Without rtmlib, the
    default model list just leaves RTMPose out, with a note."""
    def missing(module):
        try:
            __import__(module)
            return None
        except ImportError as e:
            return str(e)

    if missing("cv2"):
        sys.exit("Missing dependency: OpenCV. Install with: pip install -r requirements.txt")
    used = list(args.models) + ([args.reference] if args.reference else [])
    if any(family(m) == "rtm" for m in used):
        problems = [p for p in [missing("rtmlib")] + [missing(b) for b in args.rtm_backends if b != "opencv"] if p]
        if problems:
            hint = "pip install -r requirements-rtm.txt" + (
                " (and pip install openvino)" if "openvino" in args.rtm_backends else "")
            if args.default_models and not (args.reference and family(args.reference) == "rtm"):
                args.models = [m for m in args.models if family(m) != "rtm"]
                print(f"Note: leaving RTMPose out ({'; '.join(problems)}). To include it: {hint}\n")
            else:
                sys.exit(f"Missing dependency for RTMPose: {'; '.join(problems)}. Install with: {hint}")
    used = list(args.models) + ([args.reference] if args.reference else [])
    if any(family(m) == "yolo" for m in used) and missing("ultralytics"):
        sys.exit("Missing dependency: ultralytics. Install with: pip install -r requirements.txt")


def reference_cfgs(args):
    """{family: its reference run}: the largest model of each family listed, full frame, every
    frame; with --reference, that one run for both."""
    def cfg(model):
        if family(model) == "rtm":
            fmt = args.rtm_backends[0]
        else:
            fmt = "pt" if "pt" in args.formats else args.formats[0]
        return {"model": model, "format": fmt, "mode": "full", "stride": 1}

    if args.reference:
        one = cfg(args.reference)
        return {"yolo": one, "rtm": one}
    refs = {}
    for fam in ("yolo", "rtm"):
        listed = [m for m in args.models if family(m) == fam]
        if listed:
            refs[fam] = cfg(max(listed, key=model_rank))
    return refs


def run_grid(args):
    """Every model x format (rtmlib backend for RTMPose) x mode x stride."""
    grid = []
    for m in args.models:
        fmts = args.rtm_backends if family(m) == "rtm" else args.formats
        grid += [{"model": m, "format": f, "mode": mo, "stride": s}
                 for f in fmts for mo in args.modes for s in args.strides]
    return grid


def main(argv=None):
    args = parse_args(argv)
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
    check_dependencies(args)
    if args.threads:
        import cv2
        cv2.setNumThreads(args.threads)
        try:
            import torch
            torch.set_num_threads(args.threads)
        except ImportError:
            pass

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = reference_cfgs(args)
    ref_list = []
    for c in refs.values():
        if c not in ref_list:
            ref_list.append(c)
    grid = [c for c in run_grid(args) if c not in ref_list]
    name = lambda c: f"{c['model']} {c['format']}"  # noqa: E731

    info = machine_info()
    print(f"Machine: {info.get('cpu')} ({info.get('logical_cpus')} logical CPUs)"
          + (f", GPU {info['gpu']}" if info.get("gpu") else "") + f"; device {args.device}")
    print(f"Reference: {'; '.join(name(c) for c in ref_list)} (full frame, every frame)\n")

    cache = ModelCache(args.export_dir, args.device, conf=args.conf, threads=args.threads)
    rows = []
    for video in args.videos:
        ref_records = {}
        for cfg in ref_list + grid:
            is_ref = cfg in ref_list
            ref = cfg if is_ref else refs[family(cfg["model"])]
            label = f"{Path(video).name}: {cfg['model']} {cfg['format']} {cfg['mode']} stride {cfg['stride']}"
            print(f"> {label}{' (reference)' if is_ref else ''}", flush=True)
            row = {"video": video, **cfg, "is_reference": is_ref, "reference": name(ref), "error": None}
            try:
                stats, records, (idx, ts) = run_config(video, cfg, cache, args)
                row.update(stats)
                if is_ref:
                    ref_records[name(cfg)] = records
                    row.update({"err_pct_vs_ref": 0.0, "pck10_vs_ref": 100.0,
                                "joints_compared": sum(int((r[2][BODY] >= args.kpt_conf).sum())
                                                       for r in records.values())})
                elif name(ref) in ref_records:
                    row.update(agreement(records, ref_records[name(ref)], args.kpt_conf))
                if args.save_keypoints:
                    keys = sorted(records)
                    npts = len(records[keys[0]][1]) if keys else 17
                    np.savez_compressed(
                        out_dir / f"{Path(video).stem}__{Path(cfg['model']).stem}_{cfg['format']}_{cfg['mode']}_s{cfg['stride']}.npz",
                        analysed_frames=idx, analysed_t=ts, frame=np.array(keys),
                        t=np.array([records[k][0] for k in keys]),
                        kxy=np.array([records[k][1] for k in keys]).reshape(-1, npts, 2),
                        kconf=np.array([records[k][2] for k in keys]).reshape(-1, npts),
                        layout="halpe26" if npts == 26 else "coco17")
                split = (f" (detector {row['det_ms_per_frame']} ms on {row['detector_pct']}% of frames, "
                         f"pose {row['pose_ms_per_frame']} ms)") if row.get("det_ms_per_frame") is not None else ""
                print(f"  {row['total_ms_per_frame']} ms/frame{split}, {row['x_realtime']}x real time, "
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
