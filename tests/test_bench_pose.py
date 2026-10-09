"""tools/bench_pose.py helpers on hand-made keypoints; no video or model needed.
Run: python tests/test_bench_pose.py (or pytest).
"""
import os, sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
