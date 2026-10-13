"""tools/bench_pose.py: the metrics on hand-made keypoints, and the RTMPose runs end to end on
a small synthetic video through a stand-in rtmlib. Needs neither a model nor rtmlib.
Run: python tests/test_bench_pose.py (or pytest).
"""
import csv, os, sys, tempfile, types
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import cv2
import numpy as np

import bench_pose as bp

ARGS = SimpleNamespace(kpt_conf=0.3, cutoff_hz=6.0)
FPS = 30
SECONDS = 60
# A skeleton about 120 px from shoulder to hip, pulling at 0.5 Hz like a SkiErg stroke.
OFFS = np.array([[0, -90], [-5, -95], [5, -95], [-10, -92], [10, -92], [-18, -60], [18, -60],
                 [-25, -20], [25, -20], [-30, 15], [30, 15], [-12, 60], [12, 60],
                 [-14, 120], [14, 120], [-15, 175], [15, 175]], float)


def _records(sigma, stride, seed=0):
    """Keypoint records for every `stride`-th frame, with Gaussian noise of `sigma` px per axis."""
    rng = np.random.default_rng(seed)
    records, kept = {}, []
    for i in range(0, FPS * SECONDS, stride):
        t = i / FPS
        centre = np.array([400 + 2 * t, 330 + 60 * np.sin(np.pi * t)])
        kxy = OFFS + centre + rng.normal(0, sigma, OFFS.shape)
        records[i] = (t, kxy, np.full(17, 0.9))
        kept.append(i)
    return records, kept


def test_noise_estimate_does_not_depend_on_frame_stride():
    expected = 100 * 2.0 * np.sqrt(2) / 120          # per-joint noise as % of torso length
    one = bp.quality(*_records(2.0, 1), FPS, ARGS)
    two = bp.quality(*_records(2.0, 2), FPS / 2, ARGS)
    assert abs(one["jitter_pct"] - expected) < 0.15 * expected
    assert abs(two["jitter_pct"] - expected) < 0.15 * expected


def test_fewer_frames_leave_more_noise_after_the_force_filter():
    one = bp.quality(*_records(2.0, 1), FPS, ARGS)
    two = bp.quality(*_records(2.0, 2), FPS / 2, ARGS)
    assert two["filtered_noise_pct"] > 1.2 * one["filtered_noise_pct"]
    assert "4.5 Hz" in two["jitter_note"]             # lowpass() caps the cutoff at 0.3 x fps


def test_agreement_with_itself_is_perfect_and_drops_with_noise():
    ref, _ = _records(0.0, 1)
    assert bp.agreement(ref, ref, 0.3)["pck10_vs_ref"] == 100.0
    noisy, _ = _records(15.0, 2, seed=1)
    result = bp.agreement(noisy, ref, 0.3)
    assert result["err_pct_vs_ref"] > 10
    assert result["pck10_vs_ref"] < 80


def test_crop_stays_inside_the_frame_and_contains_the_athlete():
    box = (1800, 100, 1900, 400)                     # near the right edge of a 1920 x 1080 frame
    x0, y0, x1, y1 = bp.make_crop(box, 1920, 1080, 0.25)
    assert 0 <= x0 and x1 <= 1920 and 0 <= y0 and y1 <= 1080
    assert x0 <= box[0] and y0 <= box[1] and box[2] <= x1 and box[3] <= y1


def test_crop_is_recentred_when_the_athlete_nears_an_inner_edge():
    crop = bp.make_crop((800, 300, 900, 600), 1920, 1080, 0.25)
    assert not bp.crop_needs_update((800, 300, 900, 600), crop, 1920, 1080)
    assert bp.crop_needs_update((crop[0] + 5, 300, crop[0] + 105, 600), crop, 1920, 1080)


# ------------------------------------------------------------ RTMPose, stand-in rtmlib

VW, VH, VFPS, NFRAMES, JUMP = 480, 360, 30, 90, 65   # the jump falls between re-detections
# 26 Halpe points as fractions of the athlete's box: COCO 17 (shoulders 5-6, hips 11-12), then
# head, neck, mid-hip, big toes, small toes, heels. They span the whole box, as a body does.
FRACS = np.array([[.5, .08], [.45, .06], [.55, .06], [.4, .08], [.6, .08], [.2, .22], [.8, .22],
                  [.05, .4], [.95, .4], [0, .55], [1, .55], [.3, .55], [.7, .55], [.3, .75],
                  [.7, .75], [.3, .93], [.7, .93], [.5, 0], [.5, .17], [.5, .55], [.15, 1],
                  [.85, 1], [.2, 1], [.8, 1], [.4, .97], [.6, .97]])


def athlete_x(i):
    """Centre of the athlete block in frame i: a slow sway, then a jump the tracker must survive."""
    return 150 + 30 * np.sin(2 * np.pi * i / (3 * VFPS)) + (180 if i >= JUMP else 0)


def write_block_video(path):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), VFPS, (VW, VH))
    for i in range(NFRAMES):
        img = np.full((VH, VW, 3), 30, np.uint8)
        x = int(round(athlete_x(i)))
        img[80:280, x - 30:x + 30] = 230
        vw.write(img)
    vw.release()


def bright_box(frame, region=None):
    """Box of the bright pixels (within `region`), or None."""
    g = frame[..., 0] > 128
    x0 = y0 = 0
    if region is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in region]
        x0, y0 = max(x0, 0), max(y0, 0)
        g = g[y0:max(y1, y0), x0:max(x1, x0)]
    ys, xs = np.nonzero(g)
    if len(xs) < 50:
        return None
    return np.array([xs.min() + x0, ys.min() + y0, xs.max() + 1 + x0, ys.max() + 1 + y0], float)


class BlockDetector:
    def __init__(self):
        self.calls, self.score_thr, self.backend = 0, 0.7, "opencv"

    def __call__(self, frame):
        self.calls += 1
        box = bright_box(frame)
        return np.zeros((0, 4)) if box is None else box[None]


class BlockPose:
    """Like RTMPose: sees only its box, grown by 1.25; finds no one if the athlete left it."""
    def __init__(self):
        self.calls, self.backend = 0, "opencv"

    def __call__(self, frame, bboxes=()):
        kps, scs = [], []
        for b in np.asarray(bboxes, float).reshape(-1, 4):
            self.calls += 1
            c, half = (b[:2] + b[2:]) / 2, (b[2:] - b[:2]) / 2 * 1.25
            found = bright_box(frame, np.concatenate([c - half, c + half]))
            if found is None:
                kps.append(np.tile(c, (26, 1))); scs.append(np.full(26, 0.05))
            else:
                kps.append(found[:2] + FRACS * (found[2:] - found[:2])); scs.append(np.full(26, 0.9))
        return np.array(kps), np.array(scs)


def install_stub_rtmlib():
    made = {}

    class BodyWithFeet:
        def __init__(self, mode="balanced", backend="onnxruntime", device="cpu", **_):
            made.update(mode=mode, backend=backend, device=device)
            self.det_model, self.pose_model = BlockDetector(), BlockPose()
            made["solution"] = self

    sys.modules["rtmlib"] = types.SimpleNamespace(BodyWithFeet=BodyWithFeet)
    return made


_VIDEO = []


def block_video():
    if not _VIDEO:
        path = os.path.join(tempfile.mkdtemp(), "block.mp4")
        write_block_video(path)
        _VIDEO.append(path)
    return _VIDEO[0]


def rtm_run(mode, stride=1, redetect=10):
    made = install_stub_rtmlib()
    args = bp.parse_args([block_video(), "--models", "rtmpose-m", "--rtm-backends", "opencv",
                          "--rtm-redetect", str(redetect), "--max-seconds", "0"])
    cache = bp.ModelCache(args.export_dir, args.device, conf=args.conf)
    cfg = {"model": "rtmpose-m", "format": "opencv", "mode": mode, "stride": stride}
    stats, records, _ = bp.run_config(block_video(), cfg, cache, args)
    return stats, records, made["solution"], args


def test_rtm_full_runs_the_detector_on_every_frame_and_keeps_26_points():
    stats, records, sol, _ = rtm_run("full")
    assert stats["frames_analysed"] == NFRAMES and stats["detect_pct"] == 100.0
    assert stats["detector_pct"] == 100.0 and stats["crop_fallbacks"] is None
    assert sol.det_model.calls == NFRAMES + bp.WARMUP
    assert stats["keypoints"] == 26 and all(r[1].shape == (26, 2) for r in records.values())
    assert stats["det_ms_per_frame"] is not None and stats["pose_ms_per_frame"] is not None
    # The keypoints follow the block: its centre is the mid-hip point's x.
    for i in (0, 30, JUMP + 5):
        assert abs(records[i][1][19, 0] - athlete_x(i)) < 2


def test_rtm_crop_skips_the_detector_but_finds_the_same_keypoints():
    full, ref, _, args = rtm_run("full")
    crop, records, sol, _ = rtm_run("crop", redetect=10)
    assert crop["detect_pct"] == 100.0
    assert crop["crop_fallbacks"] >= 1                    # the jump: the box lost the athlete
    # The first frame, one frame in ten after it, and the fallback at the jump.
    assert crop["detector_pct"] <= 100 * (NFRAMES // 10 + 1 + crop["crop_fallbacks"]) / NFRAMES
    assert sol.det_model.calls - bp.WARMUP == round(crop["detector_pct"] * NFRAMES / 100)
    agree = bp.agreement(records, ref, args.kpt_conf)
    assert agree["pck10_vs_ref"] == 100.0 and agree["err_pct_vs_ref"] < 2


def test_rtm_crop_without_redetection_runs_the_detector_only_when_lost():
    crop, _, sol, _ = rtm_run("crop", stride=2, redetect=0)
    assert crop["detect_pct"] == 100.0 and crop["crop_fallbacks"] >= 1
    assert sol.det_model.calls - bp.WARMUP == 1 + crop["crop_fallbacks"]


def test_pose_box_grows_around_the_confident_points_and_stays_in_the_frame():
    kxy = np.array([[100, 100], [140, 300], [120, 200], [5, 150], [400, 150]], float)
    kc = np.array([.9, .9, .9, .9, .1])                  # the far-right point is not trusted
    box = bp.pose_box(kxy, kc, 0.3, 480, 360)
    assert box[0] == 0 and box[2] < 400                   # clipped at the left edge, ignores x=400
    assert box[1] < 100 and box[3] > 300
    assert bp.pose_box(kxy[:3], kc[:3], 0.3, 480, 360) is None   # fewer than 4 points


def test_each_family_has_its_own_reference_unless_one_is_given():
    args = bp.parse_args(["v.mp4", "--models", "yolo11m-pose.pt", "yolo26s-pose.pt", "rtmpose-s",
                          "rtmpose-x", "--rtm-backends", "openvino", "onnxruntime"])
    refs = bp.reference_cfgs(args)
    assert refs["yolo"] == {"model": "yolo11m-pose.pt", "format": "pt", "mode": "full", "stride": 1}
    assert refs["rtm"] == {"model": "rtmpose-x", "format": "openvino", "mode": "full", "stride": 1}
    grid = bp.run_grid(args)
    assert {c["format"] for c in grid if c["model"].startswith("rtm")} == {"openvino", "onnxruntime"}
    assert {c["format"] for c in grid if c["model"].startswith("yolo")} == {"pt"}
    one = bp.reference_cfgs(bp.parse_args(["v.mp4", "--models", "rtmpose-m", "yolo11s-pose.pt",
                                           "--reference", "yolo11m-pose.pt"]))
    assert one["rtm"] is one["yolo"] and one["rtm"]["model"] == "yolo11m-pose.pt"
    assert bp.model_rank("rtmpose") == bp.model_rank("rtmpose-m") < bp.model_rank("rtmpose-x")
    assert bp.rtm_backend_device("opencv", "0") == "cuda" and bp.rtm_backend_device("onnxruntime", "0") == "cuda:0"


def test_without_rtmlib_the_default_run_leaves_rtmpose_out():
    saved = {m: sys.modules.get(m) for m in ("rtmlib", "ultralytics")}
    sys.modules["rtmlib"] = None                        # import rtmlib -> ImportError
    sys.modules["ultralytics"] = types.ModuleType("ultralytics")
    try:
        args = bp.parse_args(["v.mp4"])
        bp.check_dependencies(args)
        assert args.models == [m for m in bp.DEFAULT_MODELS if not m.startswith("rtmpose")]
        try:
            bp.check_dependencies(bp.parse_args(["v.mp4", "--models", "rtmpose-m"]))
            raise AssertionError("asked-for RTMPose without rtmlib should stop")
        except SystemExit as e:
            assert "requirements-rtm.txt" in str(e)
    finally:
        for m, mod in saved.items():
            if mod is None:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = mod


class _T:
    """Just enough of a torch tensor for bench_pose.extract()."""
    def __init__(self, a): self.a = np.asarray(a, np.float32)
    def cpu(self): return self
    def numpy(self): return self.a
    def __len__(self): return len(self.a)


class _Boxes(SimpleNamespace):
    def __len__(self): return len(self.xyxy)


class BlockYolo:
    """A stand-in Ultralytics pose model: the bright block in whatever image it is given."""
    def __init__(self): self.calls = []

    def predict(self, img, imgsz=None, **_):
        self.calls.append(img.shape[:2])
        box = bright_box(img)
        if box is None:
            return [SimpleNamespace(boxes=None, keypoints=None)]
        kxy = box[:2] + FRACS[:17] * (box[2:] - box[:2])
        return [SimpleNamespace(boxes=_Boxes(xyxy=_T([box]), conf=_T([0.9])),
                                keypoints=SimpleNamespace(xy=_T([kxy]), conf=_T([np.full(17, 0.9)])))]


def test_yolo_full_and_crop_runs_are_unchanged_by_the_rtmpose_runner():
    args = bp.parse_args([block_video(), "--models", "yolo11m-pose.pt", "--max-seconds", "0"])
    model = BlockYolo()
    cache = SimpleNamespace(get=lambda weights, fmt, imgsz: model)
    full, ref, _ = bp.run_config(block_video(), {"model": "yolo11m-pose.pt", "format": "pt",
                                                 "mode": "full", "stride": 1}, cache, args)
    assert full["detect_pct"] == 100.0 and full["detector_pct"] == 100.0 and full["keypoints"] == 17
    assert full["det_ms_per_frame"] is None and full["crop_fallbacks"] is None
    model.calls.clear()
    crop, records, _ = bp.run_config(block_video(), {"model": "yolo11m-pose.pt", "format": "pt",
                                                     "mode": "crop", "stride": 1}, cache, args)
    assert crop["detect_pct"] == 100.0 and crop["detector_pct"] < 10
    assert sum(shape == (VH, VW) for shape in model.calls) == bp.WARMUP + round(crop["detector_pct"] * NFRAMES / 100)
    # At the jump the crop sees the athlete half inside its edge for one frame, then re-centres.
    after = {i: r for i, r in records.items() if i != JUMP}
    assert bp.agreement(after, ref, 0.3)["pck10_vs_ref"] == 100.0


def test_main_benchmarks_rtmpose_and_saves_all_26_points():
    made = install_stub_rtmlib()
    out = tempfile.mkdtemp()
    bp.main([block_video(), "--models", "rtmpose-m", "--rtm-backends", "opencv", "--modes", "full",
             "crop", "--strides", "1", "2", "--max-seconds", "0", "--out", out, "--save-keypoints"])
    assert made["mode"] == "balanced" and made["backend"] == "opencv" and made["device"] == "cpu"
    with open(os.path.join(out, "results.csv")) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 and not any(r["error"] for r in rows)
    assert [r["is_reference"] for r in rows] == ["True", "False", "False", "False"]
    assert all(r["reference"] == "rtmpose-m opencv" for r in rows)
    assert all(float(r["pck10_vs_ref"]) == 100.0 for r in rows)
    crop = [r for r in rows if r["mode"] == "crop"]
    assert all(float(r["detector_pct"]) < 30 for r in crop)
    saved = np.load(os.path.join(out, "block__rtmpose-m_opencv_crop_s1.npz"))
    assert saved["kxy"].shape == (NFRAMES, 26, 2) and str(saved["layout"]) == "halpe26"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
